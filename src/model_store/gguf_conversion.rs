//! Explicit, local-only repacking of supported prequantized HF NVFP4 checkpoints.
use super::*;
use std::collections::BTreeSet;

const NVFP4_REVISION: &str = "fc343a84bbd925b37dde3219de35ea0bed50d630";

#[derive(Debug, Clone)]
pub struct GgufConversionOptions {
    pub name: String,
    pub converter: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub verbose: bool,
}

impl ModelStore {
    /// Preserve the source and publish a new manifest only after validating every
    /// output. No dependency installation, model download, or lossy requantization.
    pub fn convert_gguf(
        &self,
        model_id: &str,
        options: &GgufConversionOptions,
    ) -> Result<ModelManifest> {
        validate_id(&options.name)?;
        let source = self.get_existing(model_id)?;
        if !matches!(
            source.format,
            ModelFormat::SafeTensors | ModelFormat::PyTorch
        ) {
            bail!(
                "GGUF conversion requires an installed Hugging Face checkpoint, not {:?}",
                source.format
            );
        }
        let destination = self.model_dir(&options.name);
        if destination.exists() {
            bail!(
                "destination model already exists: {}; choose a new --name",
                options.name
            );
        }
        let profile = self.quantization_profile(&source)?
            .filter(|p| p.is_authoritative() && p.has_nvfp4())
            .context("GGUF conversion currently requires declared NVFP4 metadata; filename hints are insufficient")?;
        if !matches!(
            profile.method.as_deref(),
            Some("modelopt" | "compressed-tensors")
        ) {
            bail!(
                "unsupported NVFP4 serialization {:?}; supported converter inputs are ModelOpt and compressed-tensors",
                profile.method
            );
        }
        let source_dir = self.model_files_dir(&source).canonicalize()?;
        let config_path = source_dir.join("config.json");
        let config: Value = serde_json::from_reader(
            fs::File::open(&config_path)
                .with_context(|| format!("missing HF config: {}", config_path.display()))?,
        )?;
        let has_vision = source.supports_task(InferenceTask::ImageUnderstanding)
            || config.get("vision_config").is_some_and(|v| !v.is_null());
        let converter = discover_converter(self, options.converter.as_deref())?;
        let python = options
            .python
            .clone()
            .unwrap_or_else(|| PathBuf::from("python3"));
        self.ensure()?;
        let staging = tempfile::Builder::new()
            .prefix("gguf-convert-")
            .tempdir_in(self.tmp_dir())?;
        let files = staging.path().join("files");
        fs::create_dir(&files)?;
        let main_path = files.join("model.gguf");
        run_converter(
            &python,
            &converter,
            &source_dir,
            &main_path,
            false,
            options.verbose,
        )?;
        let main =
            inspect_gguf(&main_path).context("converter did not produce a complete valid GGUF")?;
        if options.verbose {
            eprintln!("Validated {} GGUF tensors", main.tensor_count);
        }
        if !main.tensor_types.contains(&40) {
            bail!(
                "converter output contains no native NVFP4 tensors; refusing a dequantized/requantized replacement"
            );
        }
        if main
            .architecture
            .as_deref()
            .is_none_or(|v| v.is_empty() || v == "clip")
        {
            bail!("converter output has no valid language-model architecture");
        }
        if has_vision {
            // Official converter uses a separate pass and prepends mmproj-.
            run_converter(
                &python,
                &converter,
                &source_dir,
                &files.join("vision.gguf"),
                true,
                options.verbose,
            )
            .context("vision checkpoint requires a supported multimodal projector conversion")?;
            let projector = inspect_gguf(&files.join("mmproj-vision.gguf"))
                .context("converter did not produce a complete multimodal projector")?;
            if projector.architecture.as_deref() != Some("clip") {
                bail!("multimodal projector has an unexpected GGUF architecture");
            }
        }
        // The invocation does not request sharding or side artifacts. Refuse
        // unexpected files instead of registering unvalidated converter output.
        for entry in fs::read_dir(&files)? {
            let entry = entry?;
            let name = entry.file_name();
            if name != "model.gguf" && !(has_vision && name == "mmproj-vision.gguf") {
                bail!(
                    "converter produced an unexpected artifact: {}",
                    entry.path().display()
                );
            }
        }
        // Only our owned staging directory is moved. create_dir is the atomic
        // no-overwrite reservation, including sanitized model-id collisions.
        fs::create_dir(&destination).with_context(|| {
            format!(
                "cannot reserve new model {} (it may already exist)",
                options.name
            )
        })?;
        let result = (|| {
            fs::rename(&files, destination.join("files"))?;
            let mut manifest = self.build_manifest(
                &options.name,
                ModelSource::LocalPath {
                    path: source_dir.to_string_lossy().into_owned(),
                },
                &destination,
            )?;
            manifest.architecture = main.architecture;
            manifest.model_path = Some("files/model.gguf".into());
            self.write_manifest(&manifest)?;
            Ok(manifest)
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&destination);
        }
        result
    }
}

fn discover_converter(store: &ModelStore, explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if !path.is_file() {
            bail!("converter script does not exist: {}", path.display());
        }
        return path.canonicalize().context("cannot resolve converter path");
    }
    let root = store.home().join("backends").join("llama-cuda");
    let mut pinned = Vec::new();
    if let Ok(entries) = fs::read_dir(&root) {
        for entry in entries.flatten() {
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with(&format!("nvfp4-{NVFP4_REVISION}"))
            {
                pinned.push(entry.path().join("llama.cpp/convert_hf_to_gguf.py"));
            }
        }
    }
    pinned.sort();
    pinned.reverse();
    pinned.push(root.join("llama.cpp/convert_hf_to_gguf.py"));
    for candidate in pinned {
        if candidate.is_file() {
            let base = candidate.parent().unwrap().join("conversion/base.py");
            if fs::read_to_string(base).is_ok_and(|s| {
                s.contains("_generate_nvfp4_tensors") && s.contains("GGMLQuantizationType.NVFP4")
            }) {
                return candidate
                    .canonicalize()
                    .context("cannot resolve managed converter");
            }
        }
    }
    bail!(
        "no managed NVFP4-capable llama.cpp converter found; install the CUDA NVFP4 runtime profile or provide --converter /path/to/convert_hf_to_gguf.py and --python /path/to/venv/bin/python (converter dependencies are not installed automatically)"
    )
}

fn run_converter(
    python: &Path,
    converter: &Path,
    source: &Path,
    output: &Path,
    mmproj: bool,
    verbose: bool,
) -> Result<()> {
    let mut cmd = Command::new(python);
    cmd.arg(converter)
        .arg(source)
        .arg("--outfile")
        .arg(output)
        .arg("--outtype")
        .arg("auto")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if mmproj {
        cmd.arg("--mmproj");
    }
    if verbose {
        cmd.arg("--verbose");
    }
    let mut child = cmd.spawn().with_context(|| format!("cannot start converter using {}; supply --python with the converter's dependencies installed", python.display()))?;
    let stdout = child.stdout.take().context("missing converter stdout")?;
    let stderr = child.stderr.take().context("missing converter stderr")?;
    // Drain both pipes concurrently and retain bounded diagnostics even for a
    // checkpoint with hundreds of thousands of tensors.
    let out = thread::spawn(move || capture_tail(stdout, verbose));
    let err = thread::spawn(move || capture_tail(stderr, verbose));
    let status = child.wait();
    let out = out
        .join()
        .map_err(|_| anyhow!("converter stdout reader panicked"))??;
    let err = err
        .join()
        .map_err(|_| anyhow!("converter stderr reader panicked"))??;
    let status = status?;
    if !status.success() {
        bail!(
            "GGUF converter failed ({status}); no model registered. Check architecture/quantization support and Python dependencies in the selected interpreter.\n{}\n{}",
            String::from_utf8_lossy(&out),
            String::from_utf8_lossy(&err)
        );
    }
    Ok(())
}

fn capture_tail(mut reader: impl Read, verbose: bool) -> std::io::Result<Vec<u8>> {
    const LIMIT: usize = 32 * 1024;
    let mut tail = Vec::new();
    let mut chunk = [0; 8192];
    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            return Ok(tail);
        }
        if verbose {
            let _ = std::io::stderr().write_all(&chunk[..n]);
        }
        tail.extend_from_slice(&chunk[..n]);
        if tail.len() > LIMIT {
            tail.drain(..tail.len() - LIMIT);
        }
    }
}

#[derive(Debug)]
pub(super) struct GgufInspection {
    pub tensor_types: Vec<u32>,
    pub tensor_count: u64,
    pub architecture: Option<String>,
}

/// Read only metadata and tensor descriptors. Tensor payloads are never loaded.
/// Bounded header counts/strings, checked arithmetic and payload bounds make this
/// usable for imported files as well as freshly converted output.
pub(super) fn inspect_gguf(path: &Path) -> Result<GgufInspection> {
    if not_regular_gguf(path)? {
        bail!("GGUF output must be a regular file, not a symlink");
    }
    let file =
        fs::File::open(path).with_context(|| format!("cannot open GGUF {}", path.display()))?;
    let size = file.metadata()?.len();
    let mut r = GgufReader {
        reader: BufReader::new(file),
        size,
        position: 0,
    };
    if r.bytes::<4>()? != *b"GGUF" {
        bail!("invalid GGUF magic");
    }
    if !matches!(r.u32()?, 2 | 3) {
        bail!("unsupported GGUF version/endian order");
    }
    let count = r.u64()?;
    let metadata_count = r.u64()?;
    if count == 0 || count > 1_000_000 || metadata_count > 1_000_000 {
        bail!("invalid GGUF descriptor counts");
    }
    let mut architecture = None;
    let mut alignment = 32u64;
    for _ in 0..metadata_count {
        let key = r.string()?;
        let kind = r.u32()?;
        if key == "general.architecture" {
            if kind != 8 {
                bail!("invalid GGUF architecture metadata");
            }
            architecture = Some(r.string()?);
        } else if key == "general.alignment" {
            if kind != 4 {
                bail!("invalid GGUF alignment metadata");
            }
            alignment = r.u32()? as u64;
            if !alignment.is_power_of_two() || alignment > 1_048_576 {
                bail!("invalid GGUF alignment");
            }
        } else {
            r.skip_value(kind, 0)?;
        }
    }
    let mut types = BTreeSet::new();
    let mut names = BTreeSet::new();
    let mut intervals = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name = r.string()?;
        if name.is_empty() || !names.insert(name) {
            bail!("empty or duplicate GGUF tensor name");
        }
        let dimensions = r.u32()?;
        if dimensions == 0 || dimensions > 4 {
            bail!("invalid GGUF tensor dimensions");
        }
        let mut elements = 1u64;
        let mut row = 0;
        for i in 0..dimensions {
            let dimension = r.u64()?;
            if dimension == 0 {
                bail!("zero-sized GGUF tensor");
            }
            if i == 0 {
                row = dimension;
            }
            elements = elements
                .checked_mul(dimension)
                .context("GGUF tensor size overflow")?;
        }
        let kind = r.u32()?;
        let offset = r.u64()?;
        types.insert(kind);
        let (block, bytes) = tensor_block_size(kind).with_context(|| {
            format!("unsupported GGUF tensor type {kind}; validation needs an updated type table")
        })?;
        if row % block != 0 || offset % alignment != 0 {
            bail!("misaligned GGUF tensor");
        }
        let bytes = (elements / block)
            .checked_mul(bytes)
            .context("GGUF tensor byte size overflow")?;
        intervals.push((
            offset,
            offset
                .checked_add(bytes)
                .context("GGUF tensor offset overflow")?,
        ));
    }
    let start = r
        .position
        .checked_add(alignment - 1)
        .context("GGUF data offset overflow")?
        / alignment
        * alignment;
    intervals.sort_unstable();
    let mut end = 0;
    for (offset, next) in intervals {
        if offset < end || start.checked_add(next).is_none_or(|v| v > size) {
            bail!("overlapping or truncated GGUF tensor payload");
        }
        end = next;
    }
    Ok(GgufInspection {
        tensor_types: types.into_iter().collect(),
        tensor_count: count,
        architecture,
    })
}

fn tensor_block_size(kind: u32) -> Option<(u64, u64)> {
    // Official gguf-py GGML_QUANT_SIZES, pinned converter revision above.
    Some(match kind {
        0 => (1, 4),
        1 | 25 | 30 => (1, 2),
        2 | 20 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        26 => (1, 4),
        27 | 28 => (1, 8),
        29 => (256, 56),
        34 => (256, 54),
        35 => (256, 66),
        39 => (32, 17),
        40 => (64, 36),
        41 => (128, 18),
        42 => (64, 18),
        _ => return None,
    })
}

struct GgufReader {
    reader: BufReader<fs::File>,
    size: u64,
    position: u64,
}
impl GgufReader {
    fn advance(&mut self, bytes: u64) -> Result<()> {
        self.position = self
            .position
            .checked_add(bytes)
            .context("GGUF header overflow")?;
        if self.position > self.size || self.position > 256 * 1024 * 1024 {
            bail!("truncated or oversized GGUF header");
        }
        Ok(())
    }
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.advance(N as u64)?;
        let mut bytes = [0; N];
        self.reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn skip(&mut self, n: u64) -> Result<()> {
        self.advance(n)?;
        self.reader.seek_relative(i64::try_from(n)?)?;
        Ok(())
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > 16 * 1024 * 1024 {
            bail!("oversized GGUF string");
        }
        self.advance(n)?;
        let mut bytes = vec![0; n as usize];
        self.reader.read_exact(&mut bytes)?;
        String::from_utf8(bytes).context("invalid GGUF UTF-8 string")
    }
    fn skip_value(&mut self, kind: u32, depth: u32) -> Result<()> {
        match kind {
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4 | 5 | 6 => self.skip(4),
            10..=12 => self.skip(8),
            8 => {
                let n = self.u64()?;
                self.skip(n)
            }
            9 if depth == 0 => {
                let kind = self.u32()?;
                let count = self.u64()?;
                if count > 16_000_000 {
                    bail!("oversized GGUF metadata array");
                }
                for _ in 0..count {
                    self.skip_value(kind, depth + 1)?;
                }
                Ok(())
            }
            _ => bail!("invalid GGUF metadata type {kind}"),
        }
    }
}

fn not_regular_gguf(path: &Path) -> Result<bool> {
    Ok(!fs::symlink_metadata(path)?.file_type().is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    fn gguf(kind: u32, architecture: &str) -> Vec<u8> {
        let mut bytes = Vec::from(*b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        string(&mut bytes, architecture);
        string(&mut bytes, "blk.0.ffn_gate.weight");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&64u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        let (block, size) = tensor_block_size(kind).unwrap();
        bytes.resize(bytes.len() + (64 / block * size) as usize, 0);
        bytes
    }
    struct Fixture {
        _tmp: tempfile::TempDir,
        store: ModelStore,
        options: GgufConversionOptions,
    }
    fn fixture(kind: u32, vision: bool, failure: bool) -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(tmp.path().join("store"))).unwrap();
        let source = tmp.path().join("source");
        fs::create_dir(&source).unwrap();
        let mut config = serde_json::json!({"model_type":"llama", "architectures":["LlamaForCausalLM"],
            "quantization_config":{"quant_method":"modelopt", "quant_algo":"NVFP4", "group_size":16}});
        if vision {
            config["vision_config"] = serde_json::json!({"model_type":"clip"});
        }
        fs::write(
            source.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        fs::write(
            source.join("model.safetensors"),
            b"fixture source remains unchanged",
        )
        .unwrap();
        store.import_path(&source, "source").unwrap();
        let payload = tmp.path().join("expected.gguf");
        fs::write(&payload, gguf(kind, "llama")).unwrap();
        let projector = tmp.path().join("projector.gguf");
        fs::write(&projector, gguf(1, "clip")).unwrap();
        let converter = tmp.path().join("convert_hf_to_gguf.py");
        let script = format!(
            r#"import argparse, os, pathlib, shutil, sys
p=argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--outfile', required=True)
p.add_argument('--outtype', choices=['auto'], required=True)
p.add_argument('--mmproj', action='store_true')
p.add_argument('--verbose', action='store_true')
a=p.parse_args()
assert os.environ['HF_HUB_OFFLINE']=='1'
out=pathlib.Path(a.outfile)
if {failure}:
    out.write_bytes(b'partial output')
    print('unsupported quantization layout', file=sys.stderr)
    sys.exit(7)
if a.mmproj:
    out=out.with_name('mmproj-'+out.name)
    shutil.copyfile({projector},out)
else:
    shutil.copyfile({payload},out)
"#,
            failure = if failure { "True" } else { "False" },
            projector = serde_json::to_string(&projector.to_string_lossy()).unwrap(),
            payload = serde_json::to_string(&payload.to_string_lossy()).unwrap()
        );
        fs::write(&converter, script).unwrap();
        Fixture {
            _tmp: tmp,
            store,
            options: GgufConversionOptions {
                name: "converted".into(),
                converter: Some(converter),
                python: Some("python3".into()),
                verbose: false,
            },
        }
    }
    #[test]
    fn conversion_keeps_nvfp4_and_source_and_refuses_overwrite() {
        let f = fixture(40, false, false);
        let before = fs::read(f.store.model_dir("source").join(MANIFEST_FILE)).unwrap();
        let source_bytes =
            fs::read(f.store.model_dir("source").join("files/model.safetensors")).unwrap();
        let manifest = f.store.convert_gguf("source", &f.options).unwrap();
        assert_eq!(manifest.id, "converted");
        assert_eq!(manifest.format, ModelFormat::Gguf);
        assert_eq!(manifest.architecture.as_deref(), Some("llama"));
        assert_eq!(
            inspect_gguf(&f.store.model_dir("converted").join("files/model.gguf"))
                .unwrap()
                .tensor_types,
            vec![40]
        );
        assert_eq!(
            before,
            fs::read(f.store.model_dir("source").join(MANIFEST_FILE)).unwrap()
        );
        assert_eq!(
            source_bytes,
            fs::read(f.store.model_dir("source").join("files/model.safetensors")).unwrap()
        );
        assert!(
            f.store
                .convert_gguf("source", &f.options)
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
        assert_eq!(fs::read_dir(f.store.tmp_dir()).unwrap().count(), 0);
    }
    #[test]
    fn converter_failure_removes_partial_output_without_registering() {
        let f = fixture(40, false, true);
        let error = f
            .store
            .convert_gguf("source", &f.options)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsupported quantization layout"), "{error}");
        assert!(!f.store.model_dir("converted").exists());
        assert_eq!(fs::read_dir(f.store.tmp_dir()).unwrap().count(), 0);
        assert!(f.store.get_existing("source").is_ok());
    }
    #[test]
    fn conversion_rejects_dequantized_output_and_unknown_source() {
        let f = fixture(1, false, false);
        assert!(
            f.store
                .convert_gguf("source", &f.options)
                .unwrap_err()
                .to_string()
                .contains("no native NVFP4")
        );
        assert!(f.store.convert_gguf("missing", &f.options).is_err());
        assert!(!f.store.model_dir("converted").exists());
        assert_eq!(fs::read_dir(f.store.tmp_dir()).unwrap().count(), 0);
    }
    #[test]
    fn vision_conversion_requires_and_registers_projector() {
        let f = fixture(40, true, false);
        let manifest = f.store.convert_gguf("source", &f.options).unwrap();
        assert_eq!(manifest.model_path.as_deref(), Some("files/model.gguf"));
        assert!(
            manifest
                .files
                .iter()
                .any(|f| f.path == "files/mmproj-vision.gguf")
        );
        assert!(manifest.supports_task(InferenceTask::ImageUnderstanding));
    }
    #[test]
    fn gguf_inspection_rejects_truncation_and_oversized_headers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.gguf");
        let mut bytes = gguf(40, "llama");
        fs::write(&path, &bytes).unwrap();
        assert_eq!(inspect_gguf(&path).unwrap().tensor_count, 1);
        bytes.pop();
        fs::write(&path, &bytes).unwrap();
        assert!(
            inspect_gguf(&path)
                .unwrap_err()
                .to_string()
                .contains("truncated")
        );
        bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&path, bytes).unwrap();
        assert!(
            inspect_gguf(&path)
                .unwrap_err()
                .to_string()
                .contains("counts")
        );
    }
    #[test]
    fn conversion_refuses_source_name_collision() {
        let mut f = fixture(40, false, false);
        f.options.name = "source".into();
        assert!(
            f.store
                .convert_gguf("source", &f.options)
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
        assert!(f.store.get_existing("source").is_ok());
    }
    #[test]
    fn conversion_rejects_unsupported_source_metadata() {
        let f = fixture(40, false, false);
        let config = f.store.model_dir("source").join("files/config.json");
        fs::write(
            &config,
            br#"{"model_type":"llama","quantization_config":{"quant_method":"gptq","bits":4}}"#,
        )
        .unwrap();
        let error = f
            .store
            .convert_gguf("source", &f.options)
            .unwrap_err()
            .to_string();
        assert!(error.contains("declared NVFP4"), "{error}");
        assert!(!f.store.model_dir("converted").exists());
    }
    #[test]
    fn missing_projector_fails_without_registering_language_only_output() {
        let f = fixture(40, true, false);
        let script = f.options.converter.as_ref().unwrap();
        let text = fs::read_to_string(script)
            .unwrap()
            .replace("if a.mmproj:\n", "if a.mmproj:\n    sys.exit(0)\n");
        fs::write(script, text).unwrap();
        let error = f
            .store
            .convert_gguf("source", &f.options)
            .unwrap_err()
            .to_string();
        assert!(error.contains("multimodal projector"), "{error}");
        assert!(!f.store.model_dir("converted").exists());
        assert_eq!(fs::read_dir(f.store.tmp_dir()).unwrap().count(), 0);
    }
    #[test]
    fn managed_converter_discovery_selects_pinned_nvfp4_profile() {
        let f = fixture(40, false, false);
        let source = f
            .store
            .home()
            .join("backends/llama-cuda")
            .join(format!("nvfp4-{NVFP4_REVISION}-0123456789ab/llama.cpp"));
        fs::create_dir_all(source.join("conversion")).unwrap();
        fs::write(source.join("convert_hf_to_gguf.py"), "# fixture").unwrap();
        fs::write(
            source.join("conversion/base.py"),
            "_generate_nvfp4_tensors GGMLQuantizationType.NVFP4",
        )
        .unwrap();
        assert_eq!(
            discover_converter(&f.store, None).unwrap(),
            source.join("convert_hf_to_gguf.py")
        );
    }
}

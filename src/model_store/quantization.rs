//! Quantization evidence from installed metadata. Describes storage and activation
//! schemes, never whether a runtime or GPU can execute them.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationFormat {
    Nvfp4,
    Mxfp4,
    Mixed,
    Other,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationScheme {
    Nvfp4,
    WeightOnly,
    Other,
    Unknown,
    Mixed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationEvidence {
    Configuration,
    TensorMetadata,
    Filename,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationGroup {
    pub name: String,
    pub format: QuantizationFormat,
    pub algorithm: Option<String>,
    pub activation_scheme: ActivationScheme,
    pub group_size: Option<u64>,
    pub scale_dtype: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationProfile {
    pub format: QuantizationFormat,
    pub method: Option<String>,
    pub algorithm: Option<String>,
    pub activation_scheme: ActivationScheme,
    pub group_size: Option<u64>,
    pub mixed_precision: bool,
    pub per_layer: bool,
    pub evidence: QuantizationEvidence,
    pub groups: Vec<QuantizationGroup>,
}
impl QuantizationProfile {
    pub fn has_nvfp4(&self) -> bool {
        self.format == QuantizationFormat::Nvfp4
            || self
                .groups
                .iter()
                .any(|g| g.format == QuantizationFormat::Nvfp4)
    }
    pub fn is_authoritative(&self) -> bool {
        self.evidence != QuantizationEvidence::Filename
    }
    pub fn label(&self) -> String {
        match self.format {
            QuantizationFormat::Nvfp4 => "nvfp4".into(),
            QuantizationFormat::Mxfp4 => "mxfp4".into(),
            QuantizationFormat::Mixed => {
                let mut formats = Vec::new();
                for group in &self.groups {
                    let label = match group.format {
                        QuantizationFormat::Nvfp4 => "nvfp4".into(),
                        QuantizationFormat::Mxfp4 => "mxfp4".into(),
                        _ => group
                            .algorithm
                            .as_deref()
                            .unwrap_or("unknown")
                            .to_ascii_lowercase(),
                    };
                    if !formats.contains(&label) {
                        formats.push(label);
                    }
                }
                format!("mixed_precision[{}]", formats.join(","))
            }
            QuantizationFormat::Other => self
                .algorithm
                .as_ref()
                .or(self.method.as_ref())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_else(|| "unknown".into()),
        }
    }
}
impl ModelStore {
    /// Read declared local metadata, including ModelOpt sidecars. File-name hints
    /// remain explicitly unverified and must not enable native execution alone.
    pub fn quantization_profile(
        &self,
        manifest: &ModelManifest,
    ) -> Result<Option<QuantizationProfile>> {
        if manifest.format == ModelFormat::Gguf {
            return gguf_profile(&self.model_dir(&manifest.id), manifest);
        }
        let config_path = manifest
            .config_path
            .clone()
            .or_else(|| find_root_repository_file(manifest, "config.json"));
        let config = config_path
            .as_deref()
            .map(|p| read_json(self, manifest, p))
            .transpose()?;
        let sidecar = find_root_repository_file(manifest, "hf_quant_config.json")
            .map(|p| read_json(self, manifest, &p))
            .transpose()?;
        Ok(infer_profile(
            config.as_ref(),
            sidecar.as_ref(),
            &selection_sensitive_weight_paths(manifest),
        ))
    }
}
const GGUF_PROFILE_CACHE_SIZE: usize = 32;
#[derive(Clone, PartialEq, Eq)]
struct GgufFileIdentity {
    path: PathBuf,
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    change: (u64, u64, i64, i64),
    #[cfg(windows)]
    change: (u64, u64),
}
type GgufCacheKey = Vec<GgufFileIdentity>;
type GgufProfileCache = VecDeque<(GgufCacheKey, Option<QuantizationProfile>)>;
fn gguf_cache() -> &'static std::sync::Mutex<GgufProfileCache> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<GgufProfileCache>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(VecDeque::new()))
}
fn gguf_cache_key(paths: &[PathBuf]) -> Option<GgufCacheKey> {
    paths
        .iter()
        .map(|path| {
            let metadata = fs::metadata(path).ok()?;
            #[cfg(unix)]
            let change = {
                use std::os::unix::fs::MetadataExt;
                (
                    metadata.dev(),
                    metadata.ino(),
                    metadata.ctime(),
                    metadata.ctime_nsec(),
                )
            };
            #[cfg(windows)]
            let change = {
                use std::os::windows::fs::MetadataExt;
                (metadata.creation_time(), metadata.last_write_time())
            };
            Some(GgufFileIdentity {
                path: path.clone(),
                len: metadata.len(),
                modified: metadata.modified().ok()?,
                #[cfg(any(unix, windows))]
                change,
            })
        })
        .collect()
}
fn cached_gguf_profile(key: &GgufCacheKey) -> Option<Option<QuantizationProfile>> {
    let mut cache = gguf_cache().lock().ok()?;
    let index = cache.iter().position(|entry| &entry.0 == key)?;
    let entry = cache.remove(index)?;
    let profile = entry.1.clone();
    cache.push_back(entry);
    Some(profile)
}
fn cache_gguf_profile(key: GgufCacheKey, profile: Option<QuantizationProfile>) {
    if let Ok(mut cache) = gguf_cache().lock() {
        cache.retain(|entry| entry.0 != key);
        cache.push_back((key, profile));
        while cache.len() > GGUF_PROFILE_CACHE_SIZE {
            cache.pop_front();
        }
    }
}

// GGUF tensor descriptors take precedence over source HF config or filenames.
// Ordinary legacy catalog fixtures may contain no usable GGUF header; that is
// insufficient evidence, while a model claiming NVFP4 must fail closed.
pub(super) fn gguf_profile(
    model_dir: &Path,
    manifest: &ModelManifest,
) -> Result<Option<QuantizationProfile>> {
    let Some(selected) = manifest.model_path.as_deref() else {
        return Ok(None);
    };
    let must_verify = manifest
        .metadata
        .quantization
        .as_deref()
        .is_some_and(|value| value.to_ascii_lowercase().contains("nvfp4"))
        || filename_profile(&[selected.to_owned()]).is_some_and(|profile| profile.has_nvfp4());
    let relative_paths = gguf_shard_paths(selected)?.unwrap_or_else(|| vec![selected.to_owned()]);
    let paths = relative_paths
        .into_iter()
        .map(|relative| resolve_model_file(model_dir, &manifest.storage, &relative).canonicalize())
        .collect::<std::io::Result<Vec<_>>>();
    let paths = match paths {
        Ok(paths) => paths,
        Err(error) if must_verify => {
            return Err(anyhow::Error::from(error)
                .context("cannot verify declared NVFP4 GGUF tensor metadata"));
        }
        Err(_) => return Ok(None),
    };
    let key = gguf_cache_key(&paths);
    if let Some(key) = &key
        && let Some(profile) = cached_gguf_profile(key)
    {
        return Ok(profile);
    }
    let mut types = Vec::new();
    for path in &paths {
        let inspected = match super::gguf_conversion::inspect_gguf(path) {
            Ok(value) => value,
            Err(error) if must_verify => {
                return Err(error.context("cannot verify declared NVFP4 GGUF tensor metadata"));
            }
            Err(_) => return Ok(None),
        };
        for kind in inspected.tensor_types {
            if !types.contains(&kind) {
                types.push(kind);
            }
        }
    }
    let mut groups = Vec::new();
    for kind in types {
        // Auxiliary full-precision/ordinary integer tensors are normal in a
        // uniformly quantized GGUF. Other quantized types do make it mixed.
        if matches!(kind, 0 | 1 | 24..=28 | 30) {
            continue;
        }
        let (format, group_size, scale_dtype) = match kind {
            40 => (
                QuantizationFormat::Nvfp4,
                Some(16),
                Some("float8_e4m3fn".into()),
            ),
            39 => (QuantizationFormat::Mxfp4, Some(32), Some("uint8".into())),
            _ => (QuantizationFormat::Other, None, None),
        };
        groups.push(QuantizationGroup {
            name: format!("ggml_type_{kind}"),
            format,
            algorithm: Some(format!("ggml_type_{kind}")),
            activation_scheme: ActivationScheme::Unknown,
            group_size,
            scale_dtype,
        });
    }
    let profile = if groups.is_empty() {
        None
    } else {
        let mut profile = aggregate(Some("gguf".into()), None, groups, false, false);
        profile.evidence = QuantizationEvidence::TensorMetadata;
        Some(profile)
    };
    // Inspect outside the cache lock, and never publish a result for changing
    // files. No tensor payload hashing is needed for ordinary request routing.
    if let Some(key) = key {
        if gguf_cache_key(&paths).as_ref() != Some(&key) {
            bail!("GGUF model changed while reading quantization metadata; retry the request");
        }
        cache_gguf_profile(key, profile.clone());
    }
    Ok(profile)
}

fn read_json(store: &ModelStore, manifest: &ModelManifest, relative: &str) -> Result<Value> {
    let path = store.absolute_model_file(manifest, relative);
    let file = fs::File::open(&path)
        .with_context(|| format!("cannot read quantization metadata {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(16 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        bail!("quantization metadata exceeds 16 MiB: {}", path.display());
    }
    serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid quantization JSON: {}", path.display()))
}

pub(super) fn infer_profile(
    config: Option<&Value>,
    sidecar: Option<&Value>,
    paths: &[String],
) -> Option<QuantizationProfile> {
    let mut profiles = Vec::new();
    if let Some(config) = config {
        for candidate in [
            config.get("quantization_config"),
            config.pointer("/text_config/quantization_config"),
        ] {
            if let Some(candidate) = candidate.and_then(parse_config) {
                profiles.push(candidate);
            }
        }
    }
    if let Some(sidecar) = sidecar {
        if let Some(mut profile) = parse_config(sidecar.get("quantization").unwrap_or(sidecar)) {
            if profile.method.is_none() {
                profile.method = sidecar
                    .pointer("/producer/name")
                    .and_then(Value::as_str)
                    .map(str::to_ascii_lowercase);
            }
            profiles.push(profile);
        }
    }
    if profiles.is_empty() {
        return filename_profile(paths);
    }
    let mut profile = profiles.remove(0);
    for next in profiles {
        profile = merge(profile, next);
    }
    Some(profile)
}
fn string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_owned)
}
fn algorithm(value: &Value) -> Option<String> {
    string(value, "quant_algo")
}
fn algo_format(algo: Option<&str>) -> QuantizationFormat {
    match algo.unwrap_or("").to_ascii_uppercase().as_str() {
        "NVFP4" | "W4A16_NVFP4" | "W4A8_NVFP4_FP8" => QuantizationFormat::Nvfp4,
        "MXFP4" | "W4A16_MXFP4" => QuantizationFormat::Mxfp4,
        "MIXED_PRECISION" => QuantizationFormat::Mixed,
        _ => QuantizationFormat::Other,
    }
}
fn algo_activation(algo: Option<&str>) -> ActivationScheme {
    match algo.unwrap_or("").to_ascii_uppercase().as_str() {
        "NVFP4" => ActivationScheme::Nvfp4,
        "W4A16_NVFP4" | "W4A16_MXFP4" => ActivationScheme::WeightOnly,
        "MIXED_PRECISION" => ActivationScheme::Mixed,
        "" => ActivationScheme::Unknown,
        _ => ActivationScheme::Other,
    }
}
fn tensor_format(value: &Value, format: Option<&str>) -> QuantizationFormat {
    let declared = match format.unwrap_or("").to_ascii_lowercase().as_str() {
        "nvfp4-pack-quantized" => Some((QuantizationFormat::Nvfp4, 16)),
        "mxfp4-pack-quantized" => Some((QuantizationFormat::Mxfp4, 32)),
        _ => None,
    };
    if let Some((format, expected_group)) = declared {
        // An explicit label cannot turn contradictory integer or group metadata
        // into a recognized floating-point layout.
        if value
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind != "float")
            || value
                .get("num_bits")
                .and_then(Value::as_u64)
                .is_some_and(|bits| bits != 4)
            || value
                .get("group_size")
                .and_then(Value::as_u64)
                .is_some_and(|group| group != expected_group)
        {
            return QuantizationFormat::Other;
        }
        return format;
    }
    if value.get("type").and_then(Value::as_str) != Some("float")
        || value.get("num_bits").and_then(Value::as_u64) != Some(4)
    {
        return QuantizationFormat::Other;
    }
    let scale = value
        .get("scale_dtype")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim_start_matches("torch.");
    match (value.get("group_size").and_then(Value::as_u64), scale) {
        (Some(16), "float8_e4m3fn" | "float8_e4m3") => QuantizationFormat::Nvfp4,
        (Some(32), "uint8" | "float8_e8m0fnu") => QuantizationFormat::Mxfp4,
        _ => QuantizationFormat::Other,
    }
}
fn parse_config(value: &Value) -> Option<QuantizationProfile> {
    let method = string(value, "quant_method")
        .or_else(|| string(value, "quantization_method"))
        .map(|s| s.to_ascii_lowercase());
    let algo = algorithm(value);
    let layers = value.get("quantized_layers").and_then(Value::as_object);
    let config_groups = value.get("config_groups").and_then(Value::as_object);
    if method.is_none() && algo.is_none() && layers.is_none() && config_groups.is_none() {
        return None;
    }
    let mut groups = Vec::new();
    if let Some(layers) = layers {
        for (name, layer) in layers {
            let algo = algorithm(layer);
            groups.push(QuantizationGroup {
                name: name.clone(),
                format: algo_format(algo.as_deref()),
                activation_scheme: algo_activation(algo.as_deref()),
                algorithm: algo,
                group_size: layer
                    .get("group_size")
                    .and_then(Value::as_u64)
                    .or_else(|| value.get("group_size").and_then(Value::as_u64)),
                scale_dtype: None,
            });
        }
    } else if let Some(config_groups) = config_groups {
        for (name, group) in config_groups {
            let weights = &group["weights"];
            let format = group
                .get("format")
                .or_else(|| value.get("format"))
                .and_then(Value::as_str);
            let activation_scheme = match group.get("input_activations") {
                Some(Value::Null) => ActivationScheme::WeightOnly,
                Some(activation)
                    if tensor_format(activation, None) == QuantizationFormat::Nvfp4 =>
                {
                    ActivationScheme::Nvfp4
                }
                Some(_) => ActivationScheme::Other,
                None => ActivationScheme::WeightOnly,
            };
            groups.push(QuantizationGroup {
                name: name.clone(),
                format: tensor_format(weights, format),
                algorithm: format.map(str::to_owned),
                activation_scheme,
                group_size: weights.get("group_size").and_then(Value::as_u64),
                scale_dtype: string(weights, "scale_dtype"),
            });
        }
    }
    if groups.is_empty() {
        groups.push(QuantizationGroup {
            name: "default".into(),
            format: algo_format(algo.as_deref()),
            activation_scheme: algo_activation(algo.as_deref()),
            algorithm: algo.clone(),
            group_size: value.get("group_size").and_then(Value::as_u64),
            scale_dtype: None,
        });
    }
    let explicit_mixed = algo_format(algo.as_deref()) == QuantizationFormat::Mixed;
    Some(aggregate(
        method,
        algo,
        groups,
        explicit_mixed,
        layers.is_some(),
    ))
}
fn aggregate(
    method: Option<String>,
    algorithm: Option<String>,
    groups: Vec<QuantizationGroup>,
    explicit_mixed: bool,
    per_layer: bool,
) -> QuantizationProfile {
    let first = &groups[0];
    let same_format = groups.iter().all(|g| g.format == first.format);
    let same_activation = groups
        .iter()
        .all(|g| g.activation_scheme == first.activation_scheme);
    let same_group = groups.iter().all(|g| g.group_size == first.group_size);
    let mixed_precision = explicit_mixed || !same_format || !same_activation || !same_group;
    QuantizationProfile {
        format: if mixed_precision {
            QuantizationFormat::Mixed
        } else {
            first.format
        },
        method,
        algorithm,
        activation_scheme: if same_activation {
            first.activation_scheme
        } else {
            ActivationScheme::Mixed
        },
        group_size: if same_group { first.group_size } else { None },
        mixed_precision,
        per_layer,
        evidence: QuantizationEvidence::Configuration,
        groups,
    }
}
fn merge(mut first: QuantizationProfile, mut next: QuantizationProfile) -> QuantizationProfile {
    // Older HF configs only identify the exporter. The sidecar supplies layout.
    if first.format == QuantizationFormat::Other
        && first.algorithm.is_none()
        && first.groups.len() == 1
    {
        if next.method.is_none() {
            next.method = first.method;
        }
        return next;
    }
    if next.format == QuantizationFormat::Other
        && next.algorithm.is_none()
        && next.groups.len() == 1
    {
        return first;
    }
    if first.format == next.format
        && first.activation_scheme == next.activation_scheme
        && (first.group_size == next.group_size
            || first.group_size.is_none()
            || next.group_size.is_none())
    {
        if next.groups.len() > first.groups.len() {
            std::mem::swap(&mut first, &mut next);
        }
        if first.group_size.is_none() {
            first.group_size = next.group_size;
        }
        if first.method.is_none() {
            first.method = next.method;
        }
        return first;
    }
    first.groups.extend(next.groups);
    aggregate(
        first.method.or(next.method),
        Some("MIXED_PRECISION".into()),
        first.groups,
        true,
        first.per_layer || next.per_layer,
    )
}
fn filename_profile(paths: &[String]) -> Option<QuantizationProfile> {
    let mut groups = Vec::new();
    for path in paths {
        let filename = Path::new(path)
            .file_name()?
            .to_string_lossy()
            .to_ascii_lowercase();
        let format = if filename.contains("nvfp4") {
            QuantizationFormat::Nvfp4
        } else if filename.contains("mxfp4") {
            QuantizationFormat::Mxfp4
        } else {
            continue;
        };
        groups.push(QuantizationGroup {
            name: path.clone(),
            format,
            algorithm: None,
            activation_scheme: ActivationScheme::Unknown,
            group_size: None,
            scale_dtype: None,
        });
    }
    if groups.is_empty() {
        return None;
    }
    let mut profile = aggregate(None, None, groups, false, false);
    profile.evidence = QuantizationEvidence::Filename;
    Some(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn profile(config: Value) -> QuantizationProfile {
        infer_profile(Some(&config), None, &[]).unwrap()
    }
    #[test]
    fn modelopt_preserves_nvfp4_and_weight_only_activation_schemes() {
        for (algo, activation) in [
            ("NVFP4", ActivationScheme::Nvfp4),
            ("W4A16_NVFP4", ActivationScheme::WeightOnly),
        ] {
            let p = profile(
                json!({"quantization_config":{"quant_method":"modelopt","quant_algo":algo,"group_size":16}}),
            );
            assert_eq!(p.format, QuantizationFormat::Nvfp4);
            assert_eq!(p.activation_scheme, activation);
            assert_eq!(p.group_size, Some(16));
            assert!(p.is_authoritative());
            assert_eq!(p.label(), "nvfp4");
        }
    }
    #[test]
    fn nested_text_config_and_modelopt_sidecar_refine_exporter_only_metadata() {
        let config = json!({"text_config":{"quantization_config":{"quant_method":"modelopt"}}});
        let sidecar = json!({"producer":{"name":"modelopt"},"quantization":{"quant_algo":"NVFP4","group_size":16}});
        let p = infer_profile(Some(&config), Some(&sidecar), &[]).unwrap();
        assert_eq!(p.format, QuantizationFormat::Nvfp4);
        assert_eq!(p.method.as_deref(), Some("modelopt"));
        assert_eq!(p.group_size, Some(16));
    }
    #[test]
    fn mixed_precision_layers_retain_activation_and_group_details() {
        let p = profile(
            json!({"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{
                "model.experts":{"quant_algo":"W4A16_NVFP4","group_size":16},"model.attention":{"quant_algo":"FP8"}
            }}}),
        );
        assert_eq!(p.format, QuantizationFormat::Mixed);
        assert!(p.mixed_precision && p.per_layer && p.has_nvfp4());
        assert_eq!(p.activation_scheme, ActivationScheme::Mixed);
        assert_eq!(p.group_size, None);
        assert!(p.label().contains("nvfp4"));
        assert!(p.label().contains("fp8"));
    }
    #[test]
    fn compressed_tensors_distinguishes_float4_formats_and_integer4() {
        for (bits_type, size, scale, expected) in [
            (
                "float",
                16,
                "torch.float8_e4m3fn",
                QuantizationFormat::Nvfp4,
            ),
            ("float", 32, "torch.uint8", QuantizationFormat::Mxfp4),
            ("int", 16, "torch.float8_e4m3fn", QuantizationFormat::Other),
            ("float", 16, "torch.float32", QuantizationFormat::Other),
        ] {
            let p = profile(
                json!({"quantization_config":{"quant_method":"compressed-tensors","config_groups":{"group_0":{
                    "weights":{"type":bits_type,"num_bits":4,"group_size":size,"scale_dtype":scale},"input_activations":null
                }}}}),
            );
            assert_eq!(p.format, expected);
            assert_eq!(p.activation_scheme, ActivationScheme::WeightOnly);
        }
    }
    #[test]
    fn compressed_tensors_explicit_format_and_activation_quantization_are_kept() {
        let p = profile(
            json!({"quantization_config":{"quant_method":"compressed-tensors","format":"nvfp4-pack-quantized","config_groups":{"g":{
                "weights":{"type":"float","num_bits":4,"group_size":16},
                "input_activations":{"type":"float","num_bits":4,"group_size":16,"scale_dtype":"torch.float8_e4m3fn"}
            }}}}),
        );
        assert_eq!(p.format, QuantizationFormat::Nvfp4);
        assert_eq!(p.activation_scheme, ActivationScheme::Nvfp4);
    }
    #[test]
    fn filenames_are_only_hints_and_parent_model_names_do_not_supply_evidence() {
        assert!(infer_profile(None, None, &["NVFP4-brand/model.safetensors".into()]).is_none());
        assert!(infer_profile(None, None, &["model-INT4.safetensors".into()]).is_none());
        let p = infer_profile(None, None, &["model-NVFP4.gguf".into()]).unwrap();
        assert_eq!(p.format, QuantizationFormat::Nvfp4);
        assert!(!p.is_authoritative());
        assert_eq!(p.activation_scheme, ActivationScheme::Unknown);
        let p = infer_profile(None, None, &["model-MXFP4.gguf".into()]).unwrap();
        assert_eq!(p.format, QuantizationFormat::Mxfp4);
    }
    #[test]
    fn unknown_declared_algorithms_are_not_overridden_by_filename_hints() {
        let config =
            json!({"quantization_config":{"quant_method":"modelopt","quant_algo":"FUTURE_FORMAT"}});
        let p = infer_profile(
            Some(&config),
            None,
            &["misleading-NVFP4.safetensors".into()],
        )
        .unwrap();
        assert_eq!(p.format, QuantizationFormat::Other);
        assert!(!p.has_nvfp4());
    }
    #[test]
    fn installed_profile_loads_sidecar_and_refines_legacy_exporter_label() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(dir.path().into())).unwrap();
        let mut manifest = ModelManifest {
            storage: ModelStorage::Managed,
            id: "local-quant".into(),
            source: ModelSource::LocalPath {
                path: "source".into(),
            },
            format: ModelFormat::SafeTensors,
            architecture: Some("qwen3".into()),
            tokenizer_path: None,
            config_path: Some("files/config.json".into()),
            model_path: Some("files/model.safetensors".into()),
            backend: "test".into(),
            created_unix: 1,
            files: ["config.json", "hf_quant_config.json", "model.safetensors"]
                .into_iter()
                .map(|name| ModelFile {
                    path: format!("files/{name}"),
                    size: 0,
                    checksum: String::new(),
                })
                .collect(),
            artifacts: vec![],
            metadata: ModelMetadata::default(),
        };
        let files = store.model_dir(&manifest.id).join("files");
        fs::create_dir_all(&files).unwrap();
        fs::write(
            files.join("config.json"),
            json!({"model_type":"qwen3","quantization_config":{"quant_method":"modelopt"}})
                .to_string(),
        )
        .unwrap();
        fs::write(files.join("hf_quant_config.json"), json!({"producer":{"name":"modelopt"},"quantization":{"quant_algo":"W4A16_NVFP4","group_size":16}}).to_string()).unwrap();
        let profile = store.quantization_profile(&manifest).unwrap().unwrap();
        assert_eq!(profile.format, QuantizationFormat::Nvfp4);
        assert_eq!(profile.activation_scheme, ActivationScheme::WeightOnly);
        manifest.metadata.quantization = Some("modelopt".into());
        enrich_manifest_metadata(&store.model_dir(&manifest.id), &mut manifest);
        assert_eq!(manifest.metadata.quantization.as_deref(), Some("nvfp4"));
        manifest.metadata.quantization = Some("curated-value".into());
        enrich_manifest_metadata(&store.model_dir(&manifest.id), &mut manifest);
        assert_eq!(
            manifest.metadata.quantization.as_deref(),
            Some("curated-value")
        );
        fs::write(files.join("hf_quant_config.json"), "{broken").unwrap();
        assert!(
            store
                .quantization_profile(&manifest)
                .unwrap_err()
                .to_string()
                .contains("invalid quantization JSON")
        );
    }

    #[test]
    fn existing_integer_quantization_labels_remain_compatible() {
        let config = json!({"quantization_config":{"quant_method":"gptq","bits":4}});
        assert_eq!(
            infer_quantization_from_paths_and_json(&[], Some(&config)).as_deref(),
            Some("gptq-4bit")
        );
        assert_eq!(
            infer_quantization_from_paths_and_json(&["files/model-Q4_K_M.gguf".into()], None)
                .as_deref(),
            Some("q4_k_m")
        );
        assert_eq!(
            infer_quantization_from_paths_and_json(&["NVFP4-brand/model.safetensors".into()], None),
            None
        );
    }
    fn write_gguf(path: &Path, kinds: &[u32]) {
        fn text(bytes: &mut Vec<u8>, text: &str) {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(kinds.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        text(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        text(&mut bytes, "qwen3");
        let mut end = 0u64;
        for (index, kind) in kinds.iter().enumerate() {
            let (elements, size) = match kind {
                40 | 2 => (64u64, 36u64),
                39 => (32, 17),
                12 => (256, 144),
                1 => (64, 128),
                _ => panic!("fixture type"),
            };
            let offset = end.div_ceil(32) * 32;
            text(&mut bytes, &format!("weight.{index}"));
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&elements.to_le_bytes());
            bytes.extend_from_slice(&kind.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            end = offset + size;
        }
        bytes.resize(bytes.len().div_ceil(32) * 32 + end as usize, 0);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
    fn gguf_manifest(path: &str) -> ModelManifest {
        ModelManifest {
            storage: ModelStorage::Managed,
            id: "gguf-profile".into(),
            source: ModelSource::LocalPath {
                path: "source".into(),
            },
            format: ModelFormat::Gguf,
            architecture: None,
            tokenizer_path: None,
            config_path: None,
            model_path: Some(path.into()),
            backend: "llama-server".into(),
            created_unix: 1,
            files: vec![ModelFile {
                path: path.into(),
                size: 0,
                checksum: String::new(),
            }],
            artifacts: vec![],
            metadata: Default::default(),
        }
    }
    #[test]
    fn gguf_tensor_metadata_identifies_nvfp4_and_mixed_layout_without_name_hints() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(tmp.path().into())).unwrap();
        let mut manifest = gguf_manifest("files/model.gguf");
        let path = store.absolute_model_file(&manifest, "files/model.gguf");
        write_gguf(&path, &[40, 1]);
        let p = store.quantization_profile(&manifest).unwrap().unwrap();
        assert_eq!(p.format, QuantizationFormat::Nvfp4);
        assert!(!p.mixed_precision);
        assert_eq!(p.evidence, QuantizationEvidence::TensorMetadata);
        assert_eq!(p.group_size, Some(16));
        assert_eq!(p.activation_scheme, ActivationScheme::Unknown);
        assert_eq!(
            detect_gguf_architecture(&path).unwrap().as_deref(),
            Some("qwen3")
        );
        enrich_manifest_metadata(&store.model_dir(&manifest.id), &mut manifest);
        assert_eq!(manifest.metadata.quantization.as_deref(), Some("nvfp4"));
        write_gguf(&path, &[40, 12]);
        let p = store.quantization_profile(&manifest).unwrap().unwrap();
        assert_eq!(p.format, QuantizationFormat::Mixed);
        assert!(p.has_nvfp4() && p.mixed_precision);
        write_gguf(&path, &[12]);
        let p = store.quantization_profile(&manifest).unwrap().unwrap();
        assert_eq!(p.format, QuantizationFormat::Other);
        assert!(!p.has_nvfp4());
    }
    #[test]
    fn gguf_metadata_checks_every_selected_shard_and_fails_closed_for_declared_nvfp4() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(tmp.path().into())).unwrap();
        let mut manifest = gguf_manifest("files/model-00001-of-00002.gguf");
        let first = store.absolute_model_file(&manifest, "files/model-00001-of-00002.gguf");
        let second = store.absolute_model_file(&manifest, "files/model-00002-of-00002.gguf");
        write_gguf(&first, &[1]);
        write_gguf(&second, &[40]);
        let p = store.quantization_profile(&manifest).unwrap().unwrap();
        assert!(p.has_nvfp4() && p.is_authoritative());
        fs::write(&second, b"not a real GGUF fixture").unwrap();
        assert!(store.quantization_profile(&manifest).unwrap().is_none());
        manifest.metadata.quantization = Some("nvfp4".into());
        assert!(
            store
                .quantization_profile(&manifest)
                .unwrap_err()
                .to_string()
                .contains("cannot verify declared NVFP4")
        );
    }
    #[cfg(unix)]
    #[test]
    fn gguf_profile_can_read_registered_symlinked_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(tmp.path().into())).unwrap();
        let manifest = gguf_manifest("files/model.gguf");
        let path = store.absolute_model_file(&manifest, "files/model.gguf");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let actual = tmp.path().join("weights.gguf");
        write_gguf(&actual, &[40]);
        std::os::unix::fs::symlink(&actual, &path).unwrap();
        assert!(
            store
                .quantization_profile(&manifest)
                .unwrap()
                .unwrap()
                .has_nvfp4()
        );
    }
    #[test]
    fn compressed_tensor_labels_do_not_override_conflicting_layout_fields() {
        for weights in [
            json!({"type":"int","num_bits":4,"group_size":16}),
            json!({"type":"float","num_bits":8,"group_size":16}),
            json!({"type":"float","num_bits":4,"group_size":32}),
        ] {
            let p = profile(
                json!({"quantization_config":{"quant_method":"compressed-tensors","format":"nvfp4-pack-quantized",
                "config_groups":{"g":{"weights":weights}}}}),
            );
            assert_eq!(p.format, QuantizationFormat::Other);
            assert!(!p.has_nvfp4());
        }
    }
    #[cfg(unix)]
    #[test]
    fn gguf_profile_cache_invalidates_same_size_rewrite_even_with_restored_mtime() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(tmp.path().into())).unwrap();
        let manifest = gguf_manifest("files/model.gguf");
        let path = store.absolute_model_file(&manifest, "files/model.gguf");
        write_gguf(&path, &[40]);
        let original = fs::metadata(&path).unwrap();
        let profile = store.quantization_profile(&manifest).unwrap().unwrap();
        assert!(profile.has_nvfp4());
        // Populate then reuse the same descriptor identity.
        assert_eq!(
            store.quantization_profile(&manifest).unwrap(),
            Some(profile)
        );
        write_gguf(&path, &[2]);
        assert_eq!(fs::metadata(&path).unwrap().len(), original.len());
        fs::File::open(&path)
            .unwrap()
            .set_modified(original.modified().unwrap())
            .unwrap();
        let changed = store.quantization_profile(&manifest).unwrap().unwrap();
        assert_eq!(changed.format, QuantizationFormat::Other);
        assert!(!changed.has_nvfp4());
    }
}

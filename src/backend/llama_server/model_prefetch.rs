//! Metadata-only inventory of explicitly CPU-placed MoE expert ranges.
//!
//! This is a positive subset of model weights, not an estimate of every native
//! allocation. Callers must abstain when custom tensor overrides make placement
//! ambiguous. It performs no prefetch, eviction, tensor loading or inference.
//!
//! Counts and bytes read are bounded. Candle's parser still allocates some
//! length-prefixed values before reading them; these limits do not make that
//! existing parser a complete defense against hostile allocation declarations.
//! Buffered metadata reads may read ahead by at most one 64-KiB buffer into
//! adjacent payload; no tensor payload is interpreted or deliberately traversed.
//!
//! Candle reads the metadata through a view advertising zero tensors. Its
//! complete tensor reader rejects integer routing tables, so the actual bounded
//! tensor directory is inventoried separately, preserving native dtype IDs.

use anyhow::{Context, Result, anyhow, ensure};
use candle_core::quantized::{
    GgmlDType,
    gguf_file::{Content, Value},
};
use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

pub(super) use crate::backend::model_file_cache::ModelFileIdentity;

const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSORS: u64 = 100_000;
const MAX_METADATA_VALUES: u64 = 16_384;
const MAX_SHARDS: usize = 256;
const MAX_TENSOR_NAME_BYTES: u64 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CpuMoePlacement {
    All,
    FirstLayers(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Range {
    /// Absolute byte offset in this shard, including the GGUF header.
    pub(super) offset: u64,
    pub(super) length: u64,
}

#[derive(Debug)]
pub(super) struct ModelFileRegions {
    pub(super) path: PathBuf,
    pub(super) file_size: u64,
    /// Identity of the original file whose header produced these ranges.
    pub(super) identity: ModelFileIdentity,
    /// Sorted, disjoint ranges; padding and non-expert tensors are excluded.
    pub(super) ranges: Vec<Range>,
}

#[derive(Debug)]
pub(super) struct Plan {
    pub(super) files: Vec<ModelFileRegions>,
    pub(super) total_bytes: u64,
}

/// Errors mean that this optional optimization is unavailable. They must not
/// prevent llama.cpp from trying to load a model it supports independently.
pub(super) fn inventory_cpu_experts(model_path: &Path, placement: CpuMoePlacement) -> Result<Plan> {
    let name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("GGUF filename is unavailable")?;
    let names =
        crate::model_store::gguf_shard_paths(name)?.unwrap_or_else(|| vec![name.to_string()]);
    ensure!(
        names.len() <= MAX_SHARDS,
        "too many GGUF shards to inventory"
    );
    let directory = model_path.parent().context("GGUF parent is unavailable")?;
    let mut plan = Plan {
        files: Vec::new(),
        total_bytes: 0,
    };
    for name in names {
        let path = directory.join(name);
        let mut file = File::open(&path).context("cannot open GGUF shard metadata")?;
        let before = file.metadata()?;
        ensure!(before.is_file(), "GGUF shard is not a regular file");
        let identity = ModelFileIdentity::from_metadata(&before);
        let ranges = {
            // Tokenizer metadata contains many tiny fields. Coalesce those
            // reads, then release the buffer before validating the file again.
            let mut reader = BufReader::with_capacity(64 * 1024, &mut file);
            inventory_shard(&mut reader, before.len(), placement)?
        };
        let after = file.metadata()?;
        ensure!(
            identity.matches(&after),
            "GGUF shard changed during metadata inspection"
        );
        for region in &ranges {
            plan.total_bytes = plan
                .total_bytes
                .checked_add(region.length)
                .context("CPU expert inventory size overflow")?;
        }
        if !ranges.is_empty() {
            plan.files.push(ModelFileRegions {
                path,
                file_size: before.len(),
                identity,
                ranges,
            });
        }
    }
    Ok(plan)
}

fn inventory_shard<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    placement: CpuMoePlacement,
) -> Result<Vec<Range>> {
    let mut prefix = [0_u8; 24];
    reader.read_exact(&mut prefix[..8])?;
    ensure!(&prefix[..4] == b"GGUF", "unsupported GGUF magic");
    let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
    // The existing native/runtime formats use v2/v3. Leave older versions to
    // native loading instead of adding a separate compatibility parser here.
    ensure!(
        matches!(version, 2 | 3),
        "unsupported GGUF inventory version"
    );
    reader.read_exact(&mut prefix[8..])?;
    let tensors = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
    let metadata_values = u64::from_le_bytes(prefix[16..24].try_into().unwrap());
    ensure!(
        tensors <= MAX_TENSORS,
        "GGUF tensor count exceeds inventory limit"
    );
    ensure!(
        metadata_values <= MAX_METADATA_VALUES,
        "GGUF metadata count exceeds inventory limit"
    );
    reader.seek(SeekFrom::Start(0))?;
    let mut limited = MetadataReader {
        inner: reader,
        position: 0,
        remaining: MAX_METADATA_BYTES.min(file_size),
    };
    // Malformed alignment metadata can panic in Candle's offset calculation.
    // A failed optional inventory must leave the native loading path available.
    let content = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Content::read(&mut MetadataOnlyHeader {
            inner: &mut limited,
            position: 0,
        })
    }))
    .map_err(|_| anyhow!("GGUF metadata parser rejected invalid metadata"))?
    .context("GGUF metadata is unsupported or invalid for expert inventory")?;
    let alignment = match content.metadata.get("general.alignment") {
        None => 32,
        Some(Value::U8(value)) => u64::from(*value),
        Some(Value::U16(value)) => u64::from(*value),
        Some(Value::U32(value)) => u64::from(*value),
        Some(Value::U64(value)) => *value,
        Some(_) => return Err(anyhow!("unsupported GGUF alignment metadata")),
    };
    ensure!(
        alignment.is_power_of_two(),
        "GGUF alignment must be a nonzero power of two"
    );

    // A metadata-only split shard has no payload boundary to validate. Its
    // metadata may end at EOF without otherwise unnecessary alignment padding.
    if tensors == 0 {
        return Ok(Vec::new());
    }

    // This directory contains no tensor payload. Read bounded names and at
    // most four dimensions; retain raw type IDs rather than reinterpreting I32
    // routing data as a floating-point tensor just to satisfy another parser.
    let mut regions = Vec::with_capacity(tensors as usize);
    let mut names = HashSet::with_capacity(tensors as usize);
    for _ in 0..tensors {
        let name_length = read_u64(&mut limited)?;
        ensure!(
            name_length > 0
                && name_length <= MAX_TENSOR_NAME_BYTES
                && name_length <= limited.remaining,
            "GGUF tensor name exceeds inventory limit"
        );
        let mut name_bytes = vec![0; name_length as usize];
        limited.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes).context("invalid GGUF tensor name")?;
        ensure!(
            names.insert(name.clone()),
            "GGUF contains duplicate tensor names"
        );
        let dimension_count = read_u32(&mut limited)?;
        ensure!(
            (1..=4).contains(&dimension_count),
            "GGUF tensor dimensions are invalid"
        );
        let mut dimensions = [0_u64; 4];
        for dimension in &mut dimensions[..dimension_count as usize] {
            *dimension = read_u64(&mut limited)?;
        }
        let dtype = read_u32(&mut limited)?;
        let length = tensor_bytes(dtype, &dimensions[..dimension_count as usize])?;
        let offset = read_u64(&mut limited)?;
        ensure!(offset % alignment == 0, "GGUF tensor offset is not aligned");
        regions.push((Range { offset, length }, matches_expert(&name, placement)));
    }
    let data_offset = aligned_data_offset(limited.position, alignment)?;
    ensure!(
        data_offset <= file_size,
        "GGUF data offset is outside the shard"
    );
    for (region, _) in &mut regions {
        region.offset = data_offset
            .checked_add(region.offset)
            .context("GGUF tensor offset overflow")?;
        ensure!(
            region
                .offset
                .checked_add(region.length)
                .is_some_and(|end| end <= file_size),
            "GGUF tensor range is outside the shard"
        );
    }
    regions.sort_by_key(|(region, _)| region.offset);
    for pair in regions.windows(2) {
        ensure!(
            pair[0].0.offset + pair[0].0.length <= pair[1].0.offset,
            "GGUF tensor ranges overlap"
        );
    }
    Ok(regions
        .into_iter()
        .filter_map(|(region, selected)| selected.then_some(region))
        .collect())
}

fn aligned_data_offset(position: u64, alignment: u64) -> Result<u64> {
    ensure!(
        alignment.is_power_of_two(),
        "GGUF alignment must be a nonzero power of two"
    );
    position
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .context("GGUF data alignment overflow")
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn tensor_bytes(dtype: u32, dimensions: &[u64]) -> Result<u64> {
    ensure!(
        (1..=4).contains(&dimensions.len()) && dimensions.iter().all(|&dim| dim > 0),
        "GGUF tensor dimensions are invalid"
    );
    let elements = dimensions.iter().try_fold(1_u64, |product, &dim| {
        product
            .checked_mul(dim)
            .context("GGUF tensor dimension overflow")
    })?;
    // Known scalar routing tables need only their storage width. Quantized
    // layouts retain Candle's existing authoritative block/type sizes.
    let (block_size, type_size) = if dtype == 26 {
        (1, 4) // Native GGML_TYPE_I32.
    } else {
        let known = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            30 => GgmlDType::BF16,
            _ => return Err(anyhow!("unsupported GGUF inventory tensor dtype {dtype}")),
        };
        (known.block_size() as u64, known.type_size() as u64)
    };
    ensure!(
        dimensions[0] % block_size == 0,
        "GGUF quantized tensor row is not block aligned"
    );
    (elements / block_size)
        .checked_mul(type_size)
        .context("GGUF tensor byte length overflow")
}

/// Candle owns metadata parsing; this changes only the count in the in-memory
/// header view, never bytes on disk. Its read then stops before the directory.
struct MetadataOnlyHeader<R> {
    inner: R,
    position: u64,
}

impl<R: Read> Read for MetadataOnlyHeader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(buffer)?;
        for (index, byte) in buffer[..count].iter_mut().enumerate() {
            if (8..16).contains(&(self.position + index as u64)) {
                *byte = 0;
            }
        }
        self.position += count as u64;
        Ok(count)
    }
}

impl<R: Seek> Seek for MetadataOnlyHeader<R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        match position {
            SeekFrom::Current(0) => Ok(self.position),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "metadata header view cannot seek",
            )),
        }
    }
}

fn matches_expert(name: &str, placement: CpuMoePlacement) -> bool {
    // Upstream common/common.h uses regex_search with
    // \.ffn_(up|down|gate|gate_up)_(ch|)exps, optionally preceded by blk\.<N>.
    // These literal alternatives preserve its unanchored/suffix semantics.
    [
        ".ffn_up_exps",
        ".ffn_up_chexps",
        ".ffn_down_exps",
        ".ffn_down_chexps",
        ".ffn_gate_exps",
        ".ffn_gate_chexps",
        ".ffn_gate_up_exps",
        ".ffn_gate_up_chexps",
    ]
    .into_iter()
    .any(|pattern| {
        name.match_indices(pattern)
            .any(|(start, _)| match placement {
                CpuMoePlacement::All => true,
                CpuMoePlacement::FirstLayers(count) => {
                    let prefix = &name[..start];
                    prefix.rsplit_once("blk.").is_some_and(|(_, number)| {
                        number
                            .parse::<u32>()
                            .is_ok_and(|layer| layer < count && number == layer.to_string())
                    })
                }
            })
    })
}

struct MetadataReader<R> {
    inner: R,
    position: u64,
    remaining: u64,
}

impl<R: Read> Read for MetadataReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.len() as u64 > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF metadata read limit exceeded",
            ));
        }
        let count = self.inner.read(buffer)?;
        self.remaining -= count as u64;
        self.position += count as u64;
        Ok(count)
    }
}

impl<R: Seek> Seek for MetadataReader<R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        // Content::read only queries the position; do not allow a parser change
        // to silently seek into model weights or around the read budget.
        match position {
            SeekFrom::Current(0) => Ok(self.position),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "metadata inventory cannot seek",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        io::Cursor,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn fixture(tensors: &[(&str, u32, &[u64], u64)], data_bytes: u64) -> (Vec<u8>, u64) {
        fixture_with_metadata(tensors, data_bytes, &[])
    }

    fn fixture_with_metadata(
        tensors: &[(&str, u32, &[u64], u64)],
        data_bytes: u64,
        metadata: &[(&str, u32, Vec<u8>)],
    ) -> (Vec<u8>, u64) {
        let mut bytes = Vec::from(&b"GGUF"[..]);
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend((tensors.len() as u64).to_le_bytes());
        bytes.extend((metadata.len() as u64).to_le_bytes());
        for (name, dtype, value) in metadata {
            bytes.extend((name.len() as u64).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.extend(dtype.to_le_bytes());
            bytes.extend(value);
        }
        for (name, dtype, dimensions, offset) in tensors {
            bytes.extend((name.len() as u64).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.extend((dimensions.len() as u32).to_le_bytes());
            for dim in *dimensions {
                bytes.extend(dim.to_le_bytes());
            }
            bytes.extend(dtype.to_le_bytes());
            bytes.extend(offset.to_le_bytes());
        }
        let data_offset = (bytes.len() as u64).div_ceil(32) * 32;
        // Leave weights absent: inventory must only consume metadata. The
        // supplied file size describes where synthetic tensor payloads end.
        (bytes, data_offset + data_bytes)
    }

    fn parse(bytes: Vec<u8>, size: u64, placement: CpuMoePlacement) -> Result<Vec<Range>> {
        inventory_shard(&mut Cursor::new(bytes), size, placement)
    }

    #[test]
    fn all_experts_excludes_dense_and_lazy_embedding_tensors() {
        let (bytes, size) = fixture(
            &[
                ("blk.0.ffn_up_exps.weight", 0, &[8], 0),
                ("blk.1.ffn_gate_up_chexps.weight", 0, &[8], 32),
                ("blk.1.ffn_up.weight", 0, &[8], 64),
                ("per_layer_token_embd.weight", 0, &[8], 96),
            ],
            128,
        );
        let first = parse(bytes.clone(), size, CpuMoePlacement::FirstLayers(1)).unwrap();
        assert_eq!(first.len(), 1);
        let parsed = parse(bytes, size, CpuMoePlacement::All).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.iter().map(|r| r.length).sum::<u64>(), 64);
    }

    #[test]
    fn first_layers_matches_upstream_search_without_prefix_or_suffix_anchors() {
        for (name, selected) in [
            ("blk.0.ffn_down_exps.weight", true),
            ("prefix.blk.1.ffn_gate_chexps.scale", true),
            ("blk.2.ffn_up_exps.weight", false),
            ("blk.01.ffn_up_exps.weight", false),
            ("blk.1.ffn_gate_inp.weight", false),
            ("blk.1.per_layer_token_embd.weight", false),
        ] {
            assert_eq!(
                matches_expert(name, CpuMoePlacement::FirstLayers(2)),
                selected,
                "{name}"
            );
        }
        let (bytes, size) = fixture(&[("blk.0.ffn_up_exps.weight", 0, &[8], 0)], 32);
        assert!(
            parse(bytes, size, CpuMoePlacement::FirstLayers(0))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn quantized_lengths_are_computed_without_reading_weights() {
        let (bytes, size) = fixture(&[("blk.0.ffn_up_exps.weight", 10, &[256, 2], 0)], 168);
        let parsed = parse(bytes, size, CpuMoePlacement::All).unwrap();
        assert_eq!(parsed[0].length, 168); // Two Q2_K blocks.
    }

    #[test]
    fn integer_routing_table_does_not_disable_quantized_expert_inventory() {
        let table_bytes = 6 * 129_280 * 4;
        let mut label = 4_u64.to_le_bytes().to_vec();
        label.extend(b"test");
        let (bytes, size) = fixture_with_metadata(
            &[
                ("blk.0.ffn_gate_tid2eid.weight", 26, &[6, 129_280], 0),
                ("blk.0.ffn_up_exps.weight", 10, &[256, 2], table_bytes),
            ],
            table_bytes + 168,
            &[("general.name", 8, label)],
        );
        let parsed = parse(bytes, size, CpuMoePlacement::All).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].length, 168);
        assert_eq!(parsed[0].offset + parsed[0].length, size);
        assert_eq!(tensor_bytes(26, &[6, 129_280]).unwrap(), table_bytes);
    }

    #[test]
    fn invalid_ranges_and_overflow_abstain() {
        for tensors in [
            vec![("blk.0.ffn_up_exps.weight", 0, vec![9], 0)],
            vec![("blk.0.ffn_up_exps.weight", 0, vec![8], u64::MAX)],
            vec![("blk.0.ffn_up_exps.weight", 0, vec![u64::MAX, 2], 0)],
            vec![("blk.0.ffn_up_exps.weight", 10, vec![255], 0)],
            vec![("blk.0.ffn_up_exps.weight", 0, vec![0], 0)],
        ] {
            let refs = tensors
                .iter()
                .map(|(n, t, d, o)| (*n, *t, d.as_slice(), *o))
                .collect::<Vec<_>>();
            let (bytes, size) = fixture(&refs, 32);
            assert!(parse(bytes, size, CpuMoePlacement::All).is_err());
        }
        let (bytes, size) = fixture(
            &[
                ("blk.0.ffn_up_exps.weight", 0, &[8], 0),
                ("blk.0.ffn_down_exps.weight", 0, &[8], 0),
            ],
            64,
        );
        assert!(parse(bytes, size, CpuMoePlacement::All).is_err());
    }

    #[test]
    fn unsupported_dtype_and_excessive_counts_abstain() {
        let (bytes, size) = fixture(&[("blk.0.ffn_up_exps.weight", u32::MAX, &[8], 0)], 32);
        assert!(parse(bytes, size, CpuMoePlacement::All).is_err());
        let (mut bytes, size) = fixture(&[], 0);
        bytes[8..16].copy_from_slice(&(MAX_TENSORS + 1).to_le_bytes());
        assert!(parse(bytes, size, CpuMoePlacement::All).is_err());
        let (mut bytes, size) = fixture(&[], 0);
        bytes[16..24].copy_from_slice(&(MAX_METADATA_VALUES + 1).to_le_bytes());
        assert!(parse(bytes, size, CpuMoePlacement::All).is_err());
    }

    #[test]
    fn directory_name_dimensions_and_alignment_are_bounded() {
        let name = "blk.0.ffn_up_exps.weight";
        let (mut bytes, size) = fixture(&[(name, 0, &[8], 0)], 32);
        bytes[24..32].copy_from_slice(&(MAX_TENSOR_NAME_BYTES + 1).to_le_bytes());
        assert!(parse(bytes, size, CpuMoePlacement::All).is_err());
        let (mut bytes, size) = fixture(&[(name, 0, &[8], 0)], 32);
        let dims_at = 32 + name.len();
        bytes[dims_at..dims_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse(bytes, size, CpuMoePlacement::All).is_err());
        assert!(aligned_data_offset(24, 0).is_err());
        assert!(aligned_data_offset(24, 3).is_err());
        assert!(aligned_data_offset(u64::MAX, 32).is_err());
        assert_eq!(aligned_data_offset(33, 32).unwrap(), 64);
        let (bytes, size) = fixture_with_metadata(
            &[],
            0,
            &[("general.alignment", 4, 3_u32.to_le_bytes().to_vec())],
        );
        assert!(parse(bytes, size, CpuMoePlacement::All).is_err());
    }

    #[test]
    fn metadata_reader_rejects_reads_beyond_budget() {
        let mut reader = MetadataReader {
            inner: Cursor::new(vec![0; 8]),
            position: 0,
            remaining: 4,
        };
        assert!(reader.read_exact(&mut [0; 5]).is_err());
        assert_eq!(reader.position, 0);
        assert!(reader.seek(SeekFrom::Start(4)).is_err());
    }

    #[test]
    fn inventories_selected_shards_and_requires_complete_family() {
        let root = std::env::temp_dir().join(format!(
            "werk-expert-inventory-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let first = root.join("model-00001-of-00002.gguf");
        let second = root.join("model-00002-of-00002.gguf");
        for path in [&first, &second] {
            let (bytes, size) = fixture(&[("blk.0.ffn_up_exps.weight", 0, &[8], 0)], 32);
            fs::write(path, bytes).unwrap();
            File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_len(size)
                .unwrap();
        }
        let plan = inventory_cpu_experts(&first, CpuMoePlacement::All).unwrap();
        assert_eq!(plan.files.len(), 2);
        assert_eq!(plan.total_bytes, 64);
        assert_eq!(plan.files[0].path, first);
        assert!(plan.files.iter().all(|shard| shard.file_size >= 32));
        // Split files may put all tokenizer metadata in a zero-tensor first
        // shard. The second shard still contributes its expert ranges.
        let (bytes, _) =
            fixture_with_metadata(&[], 0, &[("split.no", 2, 0_u16.to_le_bytes().to_vec())]);
        fs::write(&first, bytes).unwrap();
        let plan = inventory_cpu_experts(&first, CpuMoePlacement::All).unwrap();
        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.files[0].path, second);
        assert_eq!(plan.total_bytes, 32);
        fs::remove_file(&second).unwrap();
        assert!(inventory_cpu_experts(&first, CpuMoePlacement::All).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn inventory_identity_rejects_same_size_same_mtime_replacement() {
        let root = std::env::temp_dir().join(format!(
            "werk-expert-identity-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("model.gguf");
        let replacement = root.join("replacement.gguf");
        let (bytes, size) = fixture(&[("blk.0.ffn_up_exps.weight", 0, &[8], 0)], 32);
        fs::write(&path, &bytes).unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(size)
            .unwrap();
        let plan = inventory_cpu_experts(&path, CpuMoePlacement::All).unwrap();
        let original = fs::metadata(&path).unwrap();
        assert!(plan.files[0].identity.matches(&original));

        fs::write(&replacement, bytes).unwrap();
        let file = File::options().write(true).open(&replacement).unwrap();
        file.set_len(size).unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(original.modified().unwrap()))
            .unwrap();
        drop(file);
        fs::rename(&replacement, &path).unwrap();
        let changed = File::open(&path).unwrap().metadata().unwrap();
        assert_eq!(original.len(), changed.len());
        assert_eq!(original.modified().unwrap(), changed.modified().unwrap());
        assert!(!plan.files[0].identity.matches(&changed));
        fs::remove_dir_all(root).unwrap();
    }
}

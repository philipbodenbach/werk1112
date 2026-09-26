//! Passive Linux residency of explicitly CPU-placed GGUF expert tensors.
//! cachestat / passive mincore only: never fault in, pin, prefetch, or read tensor payload.
use super::{
    SupportedArgs,
    model_prefetch::{CpuMoePlacement, Plan, inventory_cpu_experts},
};
use crate::observability::BackendSnapshot;
use anyhow::{Context, Result, ensure};
use std::{
    fs::File,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    path::Path,
    sync::OnceLock,
    time::{Duration, Instant},
};

#[derive(Default)]
pub(super) struct State {
    placement: Option<CpuMoePlacement>,
    plan: OnceLock<Option<Plan>>,
}
impl State {
    pub(super) fn new(model: &Path, args: &[String], supported: &SupportedArgs) -> Self {
        let placement = super::model_prefetch_policy::observed_placement(args, model).filter(|p| {
            supported.help_succeeded
                && match p {
                    CpuMoePlacement::All => supported.cpu_moe,
                    CpuMoePlacement::FirstLayers(_) => supported.n_cpu_moe,
                }
        });
        Self {
            placement,
            ..Default::default()
        }
    }
    pub(super) fn sample(&self, model: &Path, sample: &mut BackendSnapshot) {
        let Some(placement) = self.placement else {
            return;
        };
        // Read the bounded tensor directory once per worker, not on every scrape.
        let Some(plan) = self
            .plan
            .get_or_init(|| inventory_cpu_experts(model, placement).ok())
        else {
            return;
        };
        sample
            .gauges
            .insert("cpu_expert_weight_bytes".into(), plan.total_bytes as f64);
        if let Ok(resident) = resident_bytes(plan) {
            sample
                .gauges
                .insert("cpu_expert_resident_bytes".into(), resident as f64);
        }
    }
}

struct Mapping {
    address: *mut libc::c_void,
    length: usize,
}
impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this owns the successful mmap range below.
        unsafe {
            libc::munmap(self.address, self.length);
        }
    }
}

// Linux UAPI: https://man7.org/linux/man-pages/man2/cachestat.2.html
// The supported 64-bit x86/ARM Linux ABIs assign syscall number 451. Older
// kernels and other ABIs use the passive mincore fallback below.
#[cfg(all(
    target_pointer_width = "64",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn cached_pages(file: &File, offset: u64, length: u64) -> Option<u64> {
    #[repr(C)]
    struct Range {
        off: u64,
        len: u64,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Stats {
        cached: u64,
        dirty: u64,
        writeback: u64,
        evicted: u64,
        recent: u64,
    }
    let range = Range {
        off: offset,
        len: length,
    };
    let mut stats = Stats::default();
    // SAFETY: live descriptor; valid pointers to the exact kernel UAPI layouts;
    // flags are zero. cachestat observes file cache and never reads file data.
    let result = unsafe {
        libc::syscall(
            451 as libc::c_long,
            file.as_raw_fd() as libc::c_uint,
            &range as *const Range,
            &mut stats as *mut Stats,
            0 as libc::c_uint,
        )
    };
    (result == 0).then_some(stats.cached)
}
#[cfg(not(all(
    target_pointer_width = "64",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
fn cached_pages(_file: &File, _offset: u64, _length: u64) -> Option<u64> {
    None
}

fn cached_overlap(
    file: &File,
    offset: u64,
    length: u64,
    page: u64,
    start: u64,
    end: u64,
) -> Option<u64> {
    let count = cached_pages(file, offset, length)?;
    if count > length / page {
        return None;
    }
    let mut bytes = count * page;
    if start > offset && cached_pages(file, offset, 1)? > 0 {
        bytes = bytes.saturating_sub(start - offset);
    }
    let high = offset + length;
    if end < high && cached_pages(file, high - page, 1)? > 0 {
        bytes = bytes.saturating_sub(high - end);
    }
    // Boundary pages can change residency between calls; never exceed the
    // actual tensor-byte intersection or include dense weights/padding.
    Some(bytes.min(end.min(high).saturating_sub(start.max(offset))))
}

fn resident_bytes(plan: &Plan) -> Result<u64> {
    let started = Instant::now();
    // SAFETY: sysconf and geteuid take no pointers.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let uid = unsafe { libc::geteuid() };
    ensure!(
        page > 0 && (page as u64).is_power_of_two(),
        "invalid page size"
    );
    let page = page as u64;
    let chunk = 32 * 1024 * 1024 / page * page;
    ensure!(chunk > 0, "page too large");
    let mut resident = 0;
    let mut total = 0_u64;
    for shard in &plan.files {
        let file = File::open(&shard.path)?;
        let metadata = file.metadata()?;
        ensure!(shard.identity.matches(&metadata), "expert shard changed");
        // Linux may report every page resident to callers without ownership or
        // write permission. Abstain rather than turn that restriction into 100%.
        ensure!(metadata.uid() == uid, "residency requires file ownership");
        // One passive virtual mapping per shard avoids thousands of mmap/munmap
        // calls and their cross-thread TLB invalidations on large MoE models.
        let length = usize::try_from(shard.file_size)?;
        ensure!(length > 0, "empty expert shard");
        // SAFETY: regular owned file, no access permissions or MAP_POPULATE.
        // This reserves virtual addresses only; no model pages are touched.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_NONE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        ensure!(address != libc::MAP_FAILED, "residency mmap failed");
        let mapping = Mapping { address, length };
        let mut previous_end = 0;
        for range in &shard.ranges {
            let end = range
                .offset
                .checked_add(range.length)
                .context("range overflow")?;
            ensure!(
                range.offset >= previous_end && end <= shard.file_size,
                "invalid expert range"
            );
            previous_end = end;
            total = total
                .checked_add(range.length)
                .context("inventory overflow")?;
            ensure!(total <= plan.total_bytes, "inventory changed");
            if range.length == 0 {
                continue;
            }
            let aligned_end =
                end.checked_add(page - 1).context("alignment overflow")? / page * page;
            let mut offset = range.offset / page * page;
            while offset < aligned_end {
                ensure!(
                    started.elapsed() < Duration::from_secs(2),
                    "residency sample deadline after {:.2} GiB resident in {:.3}s",
                    resident as f64 / 1073741824.,
                    started.elapsed().as_secs_f64()
                );
                let length = usize::try_from((aligned_end - offset).min(chunk))?;
                if let Some(bytes) =
                    cached_overlap(&file, offset, length as u64, page, range.offset, end)
                {
                    resident += bytes;
                    offset += length as u64;
                    continue;
                }
                let mut pages = vec![0_u8; length / page as usize];
                // SAFETY: offset is aligned inside the live shard mapping. mmap
                // maps the final partial file page in full; the query may cover
                // that page, but byte accounting clips it to the tensor range.
                let result = unsafe {
                    libc::mincore(
                        mapping
                            .address
                            .cast::<u8>()
                            .add(usize::try_from(offset)?)
                            .cast(),
                        length,
                        pages.as_mut_ptr(),
                    )
                };
                ensure!(result == 0, "residency query failed");
                resident += resident_overlap(&pages, offset, page, range.offset, end);
                offset += length as u64;
            }
        }
        ensure!(
            shard.identity.matches(&file.metadata()?),
            "expert shard changed during sample"
        );
    }
    ensure!(total == plan.total_bytes, "inventory changed");
    Ok(resident)
}

// Count only expert bytes on shared boundary pages, excluding padding and dense
// weights. Disjoint tensors may share a page without double-counting its bytes.
fn resident_overlap(pages: &[u8], offset: u64, page: u64, start: u64, end: u64) -> u64 {
    // Whole resident/absent chunks are common. Byte-slice equality uses memcmp,
    // so even unoptimized monitoring avoids walking millions of pages in Rust.
    const FULL: [u8; 8192] = [1; 8192];
    const EMPTY: [u8; 8192] = [0; 8192];
    let count = if pages.len() <= FULL.len() && pages == &FULL[..pages.len()] {
        pages.len() as u64
    } else if pages.len() <= EMPTY.len() && pages == &EMPTY[..pages.len()] {
        0
    } else {
        // Mixed chunks: count eight status bytes at once, only their low bits.
        let mut groups = pages.chunks_exact(8);
        let count: u64 = groups
            .by_ref()
            .map(|bytes| {
                (u64::from_ne_bytes(bytes.try_into().unwrap()) & 0x0101_0101_0101_0101).count_ones()
                    as u64
            })
            .sum();
        count
            + groups
                .remainder()
                .iter()
                .map(|byte| u64::from(byte & 1))
                .sum::<u64>()
    };
    let mut bytes = count * page;
    if start > offset && pages.first().is_some_and(|v| v & 1 != 0) {
        bytes = bytes.saturating_sub((start - offset).min(page));
    }
    let mapped_end = offset + pages.len() as u64 * page;
    if end < mapped_end && pages.last().is_some_and(|v| v & 1 != 0) {
        bytes = bytes.saturating_sub((mapped_end - end).min(page));
    }
    bytes
}

#[cfg(test)]
mod observability_tests {
    use super::super::model_prefetch::{ModelFileIdentity, ModelFileRegions, Range};
    use super::*;
    use std::io::Write;

    #[test]
    fn expert_residency_counts_only_tensor_bytes_on_shared_pages() {
        assert_eq!(resident_overlap(&[1, 0, 1], 0, 4096, 100, 9000), 3996 + 808);
        assert_eq!(resident_overlap(&[0, 0], 0, 4096, 10, 5000), 0);
        assert_eq!(resident_overlap(&[1], 4096, 4096, 4196, 4200), 4);
        assert_eq!(
            resident_overlap(
                &[0x81, 0x80, 1, 0, 1, 0, 1, 0, 1],
                0,
                4096,
                100,
                9 * 4096 - 200
            ),
            5 * 4096 - 300
        );
    }

    #[test]
    fn expert_residency_reads_cache_without_changing_file_and_rejects_replacement() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![42; 16384]).unwrap();
        file.as_file().sync_all().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let plan = Plan {
            total_bytes: 5000,
            files: vec![ModelFileRegions {
                path: file.path().into(),
                file_size: metadata.len(),
                identity: ModelFileIdentity::from_metadata(&metadata),
                ranges: vec![Range {
                    offset: 100,
                    length: 5000,
                }],
            }],
        };
        assert_eq!(resident_bytes(&plan).unwrap(), 5000);
        assert_eq!(std::fs::read(file.path()).unwrap(), vec![42; 16384]);
        file.as_file().set_len(8192).unwrap();
        assert!(resident_bytes(&plan).is_err());
    }

    #[test]
    #[ignore = "requires WERK_TEST_GGUF_PATH; only reads metadata and page residency"]
    fn observability_live_cpu_expert_ram() {
        let model = std::env::var("WERK_TEST_GGUF_PATH").unwrap();
        let started = Instant::now();
        let plan =
            inventory_cpu_experts(Path::new(&model), CpuMoePlacement::FirstLayers(38)).unwrap();
        let resident = resident_bytes(&plan).unwrap();
        assert!(resident <= plan.total_bytes);
        eprintln!(
            "CPU experts: {:.3} GiB resident / {:.3} GiB weights; {:.3}s",
            resident as f64 / 1073741824.,
            plan.total_bytes as f64 / 1073741824.,
            started.elapsed().as_secs_f64()
        );
    }
}

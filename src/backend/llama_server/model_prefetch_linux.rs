//! Optional, bounded Linux prefaulting and mapping of known CPU-weight pages.
//! No eviction, raw tensor reads, or lazy/non-selected tensor traversal.

use super::model_prefetch::{ModelFileIdentity, Plan};
use anyhow::{Context, Result, anyhow, ensure};
use std::{
    fs::{File, OpenOptions},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

const CHUNK_BYTES: u64 = 32 * 1024 * 1024;
const DEADLINE: Duration = Duration::from_secs(120);

#[derive(Debug, Default)]
pub(super) struct PrefetchStats {
    pub(super) checked_bytes: u64,
    pub(super) missing_bytes: u64,
    /// Previously missing pages successfully requested from the kernel; this
    /// does not guarantee that they remain resident after the call.
    pub(super) populated_bytes: u64,
    /// Page-aligned bytes whose read-only mappings remain owned by the result.
    /// These pages are not locked against reclamation under memory pressure.
    pub(super) retained_bytes: u64,
    pub(super) elapsed_seconds: f64,
}

#[derive(Debug, Default)]
pub(super) struct PreparedPages {
    pub(super) stats: PrefetchStats,
    // These mappings own installed PTEs until the model worker stops. Retaining
    // their ownership, rather than accessing their contents, is intentional.
    _mappings: Vec<Mapping>,
}

#[derive(Default)]
struct Counters {
    checked: AtomicU64,
    missing: AtomicU64,
    populated: AtomicU64,
}

struct Source {
    file: File,
    identity: ModelFileIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chunk {
    source: usize,
    offset: u64,
    length: usize,
}

/// Failure only disables this optional optimization. All scoped workers have
/// stopped and all partial mappings are dropped before returning an error.
/// The successful result must remain owned for the model worker's lifetime.
pub(super) fn prefetch(
    plan: &Plan,
    budget_bytes: u64,
    available_memory: &(impl Fn() -> Option<u64> + Sync),
) -> Result<PreparedPages> {
    let started = Instant::now();
    ensure!(
        plan.total_bytes <= budget_bytes,
        "CPU prefetch exceeds memory budget"
    );
    let page = page_size()?;
    let chunks = chunks(plan, page)?;
    if chunks.is_empty() {
        return Ok(PreparedPages::default());
    }
    // Unsupported advice returns EINVAL even for a zero-length request. This
    // probe touches no pages and avoids substituting blocking manual reads.
    ensure!(
        populate_supported(),
        "kernel does not support selective CPU prefaulting"
    );
    let sources = plan
        .files
        .iter()
        .map(|item| {
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&item.path)
                .context("cannot open CPU prefetch shard")?;
            let metadata = file.metadata()?;
            ensure!(
                metadata.is_file()
                    && metadata.len() == item.file_size
                    && item.identity.matches(&metadata),
                "CPU prefetch shard changed or is not a regular file"
            );
            Ok(Source {
                file,
                identity: ModelFileIdentity::from_metadata(&metadata),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let next = AtomicUsize::new(0);
    let stopped = AtomicBool::new(false);
    let counters = Counters::default();
    let reserve = (512 * 1024 * 1024).max(budget_bytes / 16);
    // MemAvailable includes reclaimable file cache, so prefaulting the model
    // should not itself consume this bound. Stop if unrelated allocations make
    // the entire selected footprint cease to fit, before cycling cached pages.
    let required_available = plan.total_bytes.saturating_add(reserve);
    let result = std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for _ in 0..2.min(chunks.len()) {
            workers.push(scope.spawn(|| -> Result<Vec<Mapping>> {
                let result = (|| {
                    let mut mappings = Vec::new();
                    while !stopped.load(Ordering::Relaxed) {
                        let Some(chunk) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) else {
                            break;
                        };
                        ensure!(
                            started.elapsed() < DEADLINE,
                            "CPU prefetch deadline reached"
                        );
                        ensure!(
                            available_memory().context("CPU prefetch available memory unknown")?
                                >= required_available,
                            "CPU prefetch stopped for memory pressure"
                        );
                        mappings.push(populate_chunk(
                            &sources[chunk.source],
                            *chunk,
                            page,
                            &counters,
                        )?);
                    }
                    Ok(mappings)
                })();
                if result.is_err() {
                    stopped.store(true, Ordering::Relaxed);
                }
                result
            }));
        }
        let mut first_error = None;
        let mut mappings = Vec::new();
        for worker in workers {
            let result = worker.join().unwrap_or_else(|_| {
                stopped.store(true, Ordering::Relaxed);
                Err(anyhow!("CPU prefetch worker stopped unexpectedly"))
            });
            match result {
                Ok(mut completed) => mappings.append(&mut completed),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(mappings),
        }
    });
    let mappings = result.with_context(|| {
        format!(
            "optional CPU prefetch stopped after populating {} bytes",
            counters.populated.load(Ordering::Relaxed)
        )
    })?;
    Ok(PreparedPages {
        stats: PrefetchStats {
            checked_bytes: counters.checked.load(Ordering::Relaxed),
            missing_bytes: counters.missing.load(Ordering::Relaxed),
            populated_bytes: counters.populated.load(Ordering::Relaxed),
            retained_bytes: mappings.iter().map(|mapping| mapping.length as u64).sum(),
            elapsed_seconds: started.elapsed().as_secs_f64(),
        },
        _mappings: mappings,
    })
}

fn page_size() -> Result<u64> {
    // SAFETY: sysconf does not access caller memory.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    ensure!(
        size > 0 && (size as u64).is_power_of_two(),
        "invalid system page size"
    );
    Ok(size as u64)
}

fn populate_supported() -> bool {
    // SAFETY: the range is empty; no memory is accessed.
    unsafe { libc::madvise(std::ptr::null_mut(), 0, libc::MADV_POPULATE_READ) == 0 }
}

fn chunks(plan: &Plan, page: u64) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    let mut total = 0_u64;
    let chunk_size = CHUNK_BYTES / page * page;
    ensure!(
        chunk_size > 0,
        "system page exceeds CPU prefetch chunk size"
    );
    for (source, file) in plan.files.iter().enumerate() {
        let mut previous_end = 0;
        for range in &file.ranges {
            let end = range
                .offset
                .checked_add(range.length)
                .context("CPU prefetch range overflow")?;
            ensure!(
                range.offset >= previous_end && end <= file.file_size,
                "CPU prefetch ranges overlap or exceed the shard"
            );
            previous_end = end;
            total = total
                .checked_add(range.length)
                .context("CPU prefetch size overflow")?;
            ensure!(
                total <= plan.total_bytes,
                "CPU prefetch inventory size changed"
            );
            // Shared boundary pages may contain excluded tensors. Leave them
            // to native demand paging instead of intentionally touching them.
            let mut offset = range
                .offset
                .checked_add(page - 1)
                .context("CPU prefetch alignment overflow")?
                / page
                * page;
            let end = end / page * page;
            while offset < end {
                let length = (end - offset).min(chunk_size);
                ensure!(
                    libc::off_t::try_from(offset).is_ok(),
                    "CPU prefetch offset is too large"
                );
                chunks.push(Chunk {
                    source,
                    offset,
                    length: usize::try_from(length)?,
                });
                offset += length;
            }
        }
    }
    ensure!(
        total == plan.total_bytes,
        "CPU prefetch inventory size changed"
    );
    Ok(chunks)
}

fn unchanged(source: &Source) -> Result<()> {
    let current = source.file.metadata()?;
    ensure!(
        source.identity.matches(&current),
        "CPU prefetch shard changed during preparation"
    );
    Ok(())
}

#[derive(Debug)]
struct Mapping {
    address: *mut libc::c_void,
    length: usize,
}

// SAFETY: Mapping uniquely owns a read-only, process-wide kernel mapping. It
// never dereferences the address or creates Rust references into the mapping;
// only kernel APIs receive it. Ownership may move between threads, including
// unmapping on the receiving thread, without exposing aliased Rust memory.
unsafe impl Send for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this owns precisely the successful mmap range below.
        unsafe {
            libc::munmap(self.address, self.length);
        }
    }
}

fn populate_chunk(
    source: &Source,
    chunk: Chunk,
    page: u64,
    counters: &Counters,
) -> Result<Mapping> {
    unchanged(source)?;
    // SAFETY: validated, page-aligned offset and length in a regular read-only
    // file. There is no MAP_POPULATE and the returned memory is never dereferenced.
    let address = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            chunk.length,
            libc::PROT_READ,
            libc::MAP_SHARED,
            source.file.as_raw_fd(),
            chunk.offset as libc::off_t,
        )
    };
    ensure!(
        address != libc::MAP_FAILED,
        "CPU prefetch mapping failed: {}",
        std::io::Error::last_os_error()
    );
    let mapping = Mapping {
        address,
        length: chunk.length,
    };
    let mut residency = vec![0_u8; chunk.length / page as usize];
    // SAFETY: mapping is live and residency has one byte for each mapped page.
    let result = unsafe { libc::mincore(mapping.address, mapping.length, residency.as_mut_ptr()) };
    ensure!(
        result == 0,
        "CPU prefetch residency check failed: {}",
        std::io::Error::last_os_error()
    );
    let missing = residency.iter().filter(|value| **value & 1 == 0).count() as u64 * page;
    counters
        .checked
        .fetch_add(chunk.length as u64, Ordering::Relaxed);
    counters.missing.fetch_add(missing, Ordering::Relaxed);
    // mincore reports file-cache residency, not this mapping's installed PTEs.
    // Populate already-cached chunks too so ownership of this mapping retains
    // their PTEs after preparation. Unlike userspace dereferencing, this reports
    // truncated-file faults through errno, not SIGBUS. One syscall cannot be
    // deadline-interrupted.
    // SAFETY: the mapping remains live and contains only selected CPU pages.
    let result =
        unsafe { libc::madvise(mapping.address, mapping.length, libc::MADV_POPULATE_READ) };
    ensure!(
        result == 0,
        "CPU prefetch population failed: {}",
        std::io::Error::last_os_error()
    );
    counters.populated.fetch_add(missing, Ordering::Relaxed);
    unchanged(source)?;
    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use super::super::model_prefetch::{ModelFileRegions, Range};
    use super::*;
    use std::{
        fs,
        io::Write,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct Fixture {
        path: PathBuf,
        contents: Vec<u8>,
    }
    impl Fixture {
        fn new() -> Self {
            Self::create(false)
        }
        fn create(sparse: bool) -> Self {
            let id = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("werk-cpu-prefetch-{}-{id}", std::process::id()));
            let contents =
                vec![if sparse { 0 } else { 0x5a }; page_size().unwrap() as usize * 8 + 3];
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            if sparse {
                file.set_len(contents.len() as u64).unwrap();
            } else {
                file.write_all(&contents).unwrap();
            }
            file.sync_all().unwrap();
            Self { path, contents }
        }
        fn plan(&self, ranges: Vec<Range>) -> Plan {
            Plan {
                total_bytes: ranges.iter().map(|range| range.length).sum(),
                files: vec![ModelFileRegions {
                    path: self.path.clone(),
                    file_size: self.contents.len() as u64,
                    identity: ModelFileIdentity::from_metadata(&fs::metadata(&self.path).unwrap()),
                    ranges,
                }],
            }
        }

        fn advise_away(&self) {
            let file = File::open(&self.path).unwrap();
            // SAFETY: advice is scoped to this fixture's read-only descriptor.
            // Fixture creation has already synchronized its file contents.
            assert_eq!(
                unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) },
                0
            );
        }

        fn resident_pages(&self, offset: u64, length: u64) -> Vec<bool> {
            let file = File::open(&self.path).unwrap();
            let page = page_size().unwrap();
            assert_eq!(offset % page, 0);
            assert_eq!(length % page, 0);
            assert!(offset + length <= self.contents.len() as u64);
            // SAFETY: this creates a passive, read-only mapping of the fixture's
            // validated complete pages. Nothing faults in or dereferences them.
            let address = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length as usize,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    offset as libc::off_t,
                )
            };
            assert_ne!(address, libc::MAP_FAILED);
            let mapping = Mapping {
                address,
                length: length as usize,
            };
            let mut residency = vec![0_u8; (length / page) as usize];
            // SAFETY: the output contains one byte for each live mapped page.
            assert_eq!(
                unsafe { libc::mincore(mapping.address, mapping.length, residency.as_mut_ptr()) },
                0
            );
            residency.into_iter().map(|value| value & 1 != 0).collect()
        }
    }

    // These tiny file tests exercise paging, not the host/container detector.
    // Production always passes the cgroup-aware policy's live memory check.
    fn sufficient_memory() -> Option<u64> {
        Some(u64::MAX)
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    #[test]
    fn selects_only_complete_pages_inside_cpu_regions_including_eof() {
        let fixture = Fixture::new();
        let p = page_size().unwrap();
        let plan = fixture.plan(vec![
            Range {
                offset: 1,
                length: p * 3 - 1,
            },
            Range {
                offset: p * 6 + 1,
                length: p * 2 + 2,
            },
        ]);
        let actual = chunks(&plan, p).unwrap();
        assert_eq!(
            actual,
            vec![
                Chunk {
                    source: 0,
                    offset: p,
                    length: (p * 2) as usize
                },
                Chunk {
                    source: 0,
                    offset: p * 7,
                    length: p as usize
                }
            ]
        );
        // The excluded middle tensor and shared boundary pages are not selected;
        // kernel readahead, independently, is allowed to cover additional pages.
        assert!(
            actual
                .iter()
                .all(|chunk| chunk.offset + chunk.length as u64 <= fixture.contents.len() as u64)
        );
        if populate_supported() {
            let prepared = prefetch(&plan, 512 * 1024 * 1024, &sufficient_memory).unwrap();
            assert_eq!(prepared.stats.retained_bytes, p * 3);
            assert!(prepared.stats.retained_bytes < plan.total_bytes);
        }
    }

    #[test]
    fn retains_pte_mappings_for_already_resident_pages_without_counting_new_io() {
        if !populate_supported() {
            return;
        }
        let fixture = Fixture::new();
        let p = page_size().unwrap();
        let plan = fixture.plan(vec![Range {
            offset: p,
            length: p * 2,
        }]);
        let first = prefetch(&plan, 512 * 1024 * 1024, &sufficient_memory).unwrap();
        let second = prefetch(&plan, 512 * 1024 * 1024, &sufficient_memory).unwrap();
        assert_eq!(first.stats.checked_bytes, p * 2);
        assert_eq!(second.stats.checked_bytes, p * 2);
        assert_eq!(second.stats.missing_bytes, 0);
        assert_eq!(second.stats.populated_bytes, 0);
        assert_eq!(second.stats.retained_bytes, p * 2);
        drop(first);
        fixture.advise_away();
        // This must be the second preparation's PTEs: the first is gone, and
        // merely observing already-cached pages with mincore would not retain
        // them against this fixture-scoped advice.
        assert_eq!(fixture.resident_pages(p, p * 2), vec![true; 2]);
        drop(second);
        fixture.advise_away();
        assert_eq!(fixture.resident_pages(p, p * 2), vec![false; 2]);
        assert_eq!(fs::read(&fixture.path).unwrap(), fixture.contents);
    }

    #[test]
    fn retains_prepared_pages_until_drop_without_mutating_the_file() {
        if !populate_supported() {
            return;
        }
        let fixture = Fixture::new();
        let p = page_size().unwrap();
        let plan = fixture.plan(vec![Range {
            offset: p,
            length: p * 2,
        }]);
        fixture.advise_away();
        assert_eq!(fixture.resident_pages(p, p * 2), vec![false; 2]);
        let prepared = prefetch(&plan, 512 * 1024 * 1024, &sufficient_memory).unwrap();
        assert_eq!(prepared.stats.checked_bytes, p * 2);
        assert_eq!(prepared.stats.missing_bytes, p * 2);
        assert_eq!(prepared.stats.populated_bytes, p * 2);
        assert_eq!(prepared.stats.retained_bytes, p * 2);
        fixture.advise_away();
        assert_eq!(fixture.resident_pages(p, p * 2), vec![true; 2]);
        drop(prepared);
        fixture.advise_away();
        assert_eq!(fixture.resident_pages(p, p * 2), vec![false; 2]);
        assert_eq!(fs::read(&fixture.path).unwrap(), fixture.contents);
    }

    #[test]
    fn prefaults_missing_sparse_pages_without_mutating_the_file() {
        if !populate_supported() {
            return;
        }
        let fixture = Fixture::create(true);
        let p = page_size().unwrap();
        let plan = fixture.plan(vec![Range {
            offset: p,
            length: p * 2,
        }]);
        let prepared = prefetch(&plan, 512 * 1024 * 1024, &sufficient_memory).unwrap();
        let stats = &prepared.stats;
        assert_eq!(stats.checked_bytes, p * 2);
        assert_eq!(stats.missing_bytes, p * 2);
        assert_eq!(stats.populated_bytes, p * 2);
        assert_eq!(stats.retained_bytes, p * 2);
        assert_eq!(fs::read(&fixture.path).unwrap(), fixture.contents);
    }

    #[test]
    fn over_budget_abstains_before_opening_or_prefaulting_files() {
        let fixture = Fixture::new();
        let mut plan = fixture.plan(vec![Range {
            offset: 0,
            length: 4096,
        }]);
        plan.files[0].path = PathBuf::from("/does-not-exist/werk-cpu-prefetch");
        assert!(
            prefetch(&plan, 4095, &sufficient_memory)
                .unwrap_err()
                .to_string()
                .contains("budget")
        );
    }

    #[test]
    fn prefault_stops_when_live_budget_drops_between_chunks() {
        if !populate_supported() {
            return;
        }
        let fixture = Fixture::create(true);
        let p = page_size().unwrap();
        let plan = fixture.plan(vec![
            Range {
                offset: p,
                length: p,
            },
            Range {
                offset: p * 6,
                length: p,
            },
        ]);
        let checks = AtomicUsize::new(0);
        let insufficient = 512 * 1024 * 1024 + plan.total_bytes - 1;
        let live_memory = || {
            Some(if checks.fetch_add(1, Ordering::Relaxed) == 0 {
                u64::MAX
            } else {
                // The reserve still fits, but the whole expert footprint no
                // longer does. Continuing would risk repeated cache eviction.
                insufficient
            })
        };
        let error = prefetch(&plan, 512 * 1024 * 1024, &live_memory).unwrap_err();
        assert!(format!("{error:#}").contains("memory pressure"));
        assert!(checks.load(Ordering::Relaxed) >= 2);
        fixture.advise_away();
        // Even when another scoped worker already populated a chunk, an error
        // must release every partial mapping before preparation returns.
        assert_eq!(fixture.resident_pages(p, p), vec![false]);
        assert_eq!(fixture.resident_pages(p * 6, p), vec![false]);
        assert_eq!(fs::read(&fixture.path).unwrap(), fixture.contents);
    }

    #[test]
    fn rejects_changed_shards_and_ranges_outside_eof() {
        let fixture = Fixture::new();
        let p = page_size().unwrap();
        let bad = fixture.plan(vec![Range {
            offset: p * 8,
            length: p,
        }]);
        assert!(chunks(&bad, p).is_err());
        let mut changed = fixture.plan(vec![Range {
            offset: 0,
            length: p,
        }]);
        changed.files[0].file_size += p;
        if populate_supported() {
            assert!(prefetch(&changed, 512 * 1024 * 1024, &sufficient_memory).is_err());
        }
        let overflow = fixture.plan(vec![Range {
            offset: u64::MAX,
            length: 1,
        }]);
        assert!(chunks(&overflow, p).is_err());
    }
}

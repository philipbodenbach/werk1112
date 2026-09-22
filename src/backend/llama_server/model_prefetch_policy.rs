//! Optional preparation of the current model's CPU expert pages. This never
//! changes native arguments, cache compatibility, or another model's files.

use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use super::{
    LlamaCppMode, SupportedArgs,
    model_prefetch::{CpuMoePlacement, inventory_cpu_experts},
    model_prefetch_linux::{PreparedPages, prefetch},
};

const GIB: u64 = 1024 * 1024 * 1024;

pub(super) fn prepare(
    mode: LlamaCppMode,
    model_path: &Path,
    args: &[String],
    supported: &SupportedArgs,
    cancelled: Option<&AtomicBool>,
) -> (Option<String>, Option<PreparedPages>) {
    let mut pages = None;
    let diagnostic = prepare_inner(mode, model_path, args, supported, cancelled, &mut pages);
    (diagnostic, pages)
}

fn prepare_inner(
    mode: LlamaCppMode,
    model_path: &Path,
    args: &[String],
    supported: &SupportedArgs,
    cancelled: Option<&AtomicBool>,
    retained: &mut Option<PreparedPages>,
) -> Option<String> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Some("llama.cpp CPU expert prefetch: cancelled during shutdown".into());
    }
    if mode != LlamaCppMode::Cuda {
        return None;
    }
    let enabled = env::var("WERK_LLAMA_PREFETCH").unwrap_or_else(|_| "auto".into());
    if enabled == "off" {
        return Some("llama.cpp CPU expert prefetch: disabled".into());
    }
    if enabled != "auto" {
        return Some(
            "llama.cpp CPU expert prefetch: skipped; expected WERK_LLAMA_PREFETCH=auto|off".into(),
        );
    }
    // Native environment overrides are outside this positive placement proof.
    // Do not guess their interaction with arbitrary user-supplied arguments.
    if [
        "LLAMA_ARG_CPU_MOE",
        "LLAMA_ARG_N_CPU_MOE",
        "LLAMA_ARG_OVERRIDE_TENSOR",
        "LLAMA_ARG_NO_HOST",
        "LLAMA_ARG_NUMA",
        "GGML_CUDA_NO_PINNED",
    ]
    .iter()
    .any(|key| env::var_os(key).is_some())
    {
        return Some(
            "llama.cpp CPU expert prefetch: skipped; native environment overrides placement".into(),
        );
    }
    let load_mode = env::var("LLAMA_ARG_LOAD_MODE").ok();
    let Some(placement) = placement(args, model_path, load_mode.as_deref()) else {
        return Some(
            "llama.cpp CPU expert prefetch: skipped; no unambiguous supported CPU placement".into(),
        );
    };
    if !supported.help_succeeded
        || !match placement {
            CpuMoePlacement::All => supported.cpu_moe,
            CpuMoePlacement::FirstLayers(_) => supported.n_cpu_moe,
        }
    {
        return Some(
            "llama.cpp CPU expert prefetch: skipped; runtime did not advertise the selected placement option".into(),
        );
    }
    let started = Instant::now();
    let result = (|| -> anyhow::Result<String> {
        let plan = inventory_cpu_experts(model_path, placement)?;
        if plan.total_bytes < GIB {
            return Ok("not needed for this CPU expert footprint".into());
        }
        let memory =
            MemoryBudget::detect().ok_or_else(|| anyhow::anyhow!("memory budget unavailable"))?;
        let available = memory
            .available()
            .ok_or_else(|| anyhow::anyhow!("memory budget unavailable"))?;
        // This is a reclaimable page-cache budget, not an eager allocation or
        // an estimate of the native worker's total RAM requirements.
        let budget = available.saturating_sub((available / 8).max(4 * GIB));
        if plan.total_bytes > budget {
            return Ok("skipped; CPU expert footprint exceeds the prefetch budget".into());
        }
        let pages = prefetch(&plan, budget, &|| {
            if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                None
            } else {
                memory.available()
            }
        })?;
        let stats = &pages.stats;
        let detail = format!(
            "checked {:.2} GiB; missing {:.2} GiB; prefaulted {:.2} GiB in {:.6}s; {:.2} GiB mapped for worker lifetime (reclaimable, not locked)",
            stats.checked_bytes as f64 / GIB as f64,
            stats.missing_bytes as f64 / GIB as f64,
            stats.populated_bytes as f64 / GIB as f64,
            stats.elapsed_seconds,
            stats.retained_bytes as f64 / GIB as f64,
        );
        *retained = Some(pages);
        Ok(detail)
    })();
    let detail = match result {
        Ok(detail) => detail,
        Err(error) => format!("unavailable; native loading continues ({error:#})"),
    };
    Some(format!(
        "llama.cpp CPU expert prefetch: {detail}; {:.6}s",
        started.elapsed().as_secs_f64()
    ))
}

fn option<'a>(args: &'a [String], names: &[&str]) -> Option<&'a str> {
    let mut value = None;
    for (i, arg) in args.iter().enumerate() {
        if names.contains(&arg.as_str()) {
            value = args.get(i + 1).map(String::as_str);
        } else if let Some((name, next)) = arg.split_once('=')
            && names.contains(&name)
        {
            value = Some(next);
        }
    }
    value
}

fn has(args: &[String], names: &[&str]) -> bool {
    args.iter()
        .any(|arg| names.contains(&arg.split('=').next().unwrap_or_default()))
}

fn placement(args: &[String], model: &Path, env_load: Option<&str>) -> Option<CpuMoePlacement> {
    if option(args, &["--model", "-m"]).map(Path::new) != Some(model)
        || has(
            args,
            &[
                "--override-tensor",
                "-ot",
                "--no-host",
                "--no-mmap",
                "--mlock",
                "--numa",
            ],
        )
        || !matches!(
            option(args, &["--load-mode", "-lm"])
                .or(env_load)
                .unwrap_or("auto"),
            "auto" | "mmap"
        )
    {
        return None;
    }
    let directives = args
        .iter()
        .filter(|arg| {
            ["--cpu-moe", "-cmoe", "--n-cpu-moe", "-ncmoe"]
                .contains(&arg.split('=').next().unwrap_or_default())
        })
        .count();
    if directives != 1 {
        return None;
    }
    if has(args, &["--cpu-moe", "-cmoe"]) {
        return Some(CpuMoePlacement::All);
    }
    let layers = option(args, &["--n-cpu-moe", "-ncmoe"])?
        .parse::<u32>()
        .ok()?;
    (layers > 0).then_some(CpuMoePlacement::FirstLayers(layers))
}

fn meminfo_value(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == key)
            .then(|| {
                value
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)
            })
            .flatten()
    })
}

/// Locate the relevant controller files once, then reread their changing values
/// while preparing pages. Unconstrained root hosts only reread /proc/meminfo.
struct MemoryBudget {
    meminfo: PathBuf,
    constraints: Vec<PathBuf>,
}

impl MemoryBudget {
    fn detect() -> Option<Self> {
        let cgroup = fs::read_to_string("/proc/self/cgroup").ok()?;
        let group = cgroup.lines().find_map(|line| line.strip_prefix("0::"))?;
        // Resolve only the standard complete cgroup-v2 hierarchy. Unknown/v1 or
        // remapped container hierarchies retain native loading without prefetch.
        let mounts = fs::read_to_string("/proc/self/mountinfo").ok()?;
        if !mounts.lines().any(|line| {
            line.contains(" - cgroup2 ")
                && line.split_whitespace().nth(3) == Some("/")
                && line.split_whitespace().nth(4) == Some("/sys/fs/cgroup")
        }) {
            return None;
        }
        if Path::new(group)
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return None;
        }
        let root = Path::new("/sys/fs/cgroup");
        let mut directory = root.join(group.trim_start_matches('/'));
        let mut constraints = Vec::new();
        loop {
            match fs::read_to_string(directory.join("memory.max")) {
                Ok(value) => {
                    if value.trim() != "max" {
                        value.trim().parse::<u64>().ok()?;
                    }
                    // Keep currently unlimited controllers too: an administrator
                    // can lower their limit while the preparation is running.
                    constraints.push(directory.clone());
                }
                Err(error) if directory == root && error.kind() == std::io::ErrorKind::NotFound => {
                }
                Err(_) => return None,
            }
            if directory == root {
                break;
            }
            if !directory.pop() || !directory.starts_with(root) {
                return None;
            }
        }
        Some(Self {
            meminfo: "/proc/meminfo".into(),
            constraints,
        })
    }

    fn available(&self) -> Option<u64> {
        let mut available =
            meminfo_value(&fs::read_to_string(&self.meminfo).ok()?, "MemAvailable")?;
        for directory in &self.constraints {
            let maximum = fs::read_to_string(directory.join("memory.max")).ok()?;
            if maximum.trim() == "max" {
                continue;
            }
            let limit = maximum.trim().parse::<u64>().ok()?;
            let current = fs::read_to_string(directory.join("memory.current"))
                .ok()?
                .trim()
                .parse::<u64>()
                .ok()?;
            let stat = fs::read_to_string(directory.join("memory.stat")).ok()?;
            let inactive = stat
                .lines()
                .find_map(|line| line.strip_prefix("inactive_file "))?
                .parse::<u64>()
                .ok()?;
            available = available.min(cgroup_available(limit, current, inactive));
        }
        Some(available)
    }
}

fn cgroup_available(limit: u64, current: u64, inactive_file: u64) -> u64 {
    limit.saturating_sub(current.saturating_sub(inactive_file))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn args(extra: &[&str]) -> Vec<String> {
        ["--model", "/tmp/model.gguf"]
            .into_iter()
            .chain(extra.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn explicit_cpu_placement_only_and_native_overrides_preserved() {
        let model = Path::new("/tmp/model.gguf");
        assert_eq!(
            placement(&args(&["--cpu-moe"]), model, None),
            Some(CpuMoePlacement::All)
        );
        assert_eq!(
            placement(&args(&["-ncmoe", "38"]), model, None),
            Some(CpuMoePlacement::FirstLayers(38))
        );
        assert_eq!(
            placement(&args(&["--n-cpu-moe=20"]), model, None),
            Some(CpuMoePlacement::FirstLayers(20))
        );
        for extra in [
            vec![],
            vec!["--cpu-moe-draft"],
            vec!["--n-cpu-moe", "0"],
            vec!["--n-cpu-moe", "-1"],
            vec!["--cpu-moe", "-ncmoe", "38"],
            vec!["--cpu-moe", "-ot", "x=CPU"],
            vec!["--cpu-moe", "--no-host"],
            vec!["--cpu-moe", "--numa", "distribute"],
            vec!["--cpu-moe", "-m", "/tmp/other.gguf"],
            vec!["--cpu-moe", "-lm", "none"],
        ] {
            assert_eq!(placement(&args(&extra), model, None), None, "{extra:?}");
        }
        assert_eq!(placement(&args(&["--cpu-moe"]), model, Some("dio")), None);
        assert_eq!(
            placement(
                &args(&["--cpu-moe", "--load-mode=mmap"]),
                model,
                Some("dio")
            ),
            Some(CpuMoePlacement::All)
        );
    }

    #[test]
    fn memory_budget_is_available_not_free_and_cgroup_bounded() {
        assert_eq!(
            meminfo_value("MemFree: 1 kB\nMemAvailable: 9000 kB\n", "MemAvailable"),
            Some(9000 * 1024)
        );
        assert_eq!(
            meminfo_value("MemAvailable: invalid kB", "MemAvailable"),
            None
        );
        assert_eq!(cgroup_available(100, 90, 20), 30);
        assert_eq!(cgroup_available(100, 120, 10), 0);
        assert_eq!(cgroup_available(100, 20, 30), 100);
    }

    #[test]
    fn live_budget_tracks_controller_pressure_and_changed_ancestor_limits() {
        let root = env::temp_dir().join(format!(
            "werk-prefetch-budget-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let child = root.join("parent/child");
        let parent = root.join("parent");
        fs::create_dir_all(&child).unwrap();
        let meminfo = root.join("meminfo");
        fs::write(&meminfo, "MemFree: 1 kB\nMemAvailable: 1000 kB\n").unwrap();
        for (path, limit, current, inactive) in [
            (&child, "600000", "300000", "100000"),
            (&parent, "250000", "50000", "0"),
        ] {
            fs::write(path.join("memory.max"), limit).unwrap();
            fs::write(path.join("memory.current"), current).unwrap();
            fs::write(
                path.join("memory.stat"),
                format!("inactive_file {inactive}\n"),
            )
            .unwrap();
        }
        let budget = MemoryBudget {
            meminfo,
            constraints: vec![child.clone(), parent.clone()],
        };
        assert_eq!(budget.available(), Some(200000));
        // No hierarchy reparsing: the existing detector sees a lowered parent
        // limit and additional memory charged by sibling processes immediately.
        fs::write(parent.join("memory.max"), "150000").unwrap();
        assert_eq!(budget.available(), Some(100000));
        fs::write(parent.join("memory.current"), "140000").unwrap();
        assert_eq!(budget.available(), Some(10000));
        fs::write(parent.join("memory.stat"), "inactive_file 5000\n").unwrap();
        assert_eq!(budget.available(), Some(15000));
        fs::write(child.join("memory.max"), "max").unwrap();
        fs::write(parent.join("memory.max"), "max").unwrap();
        assert_eq!(budget.available(), Some(1000 * 1024));
        // A previously unlimited controller can become constrained mid-scan.
        fs::write(parent.join("memory.max"), "150000").unwrap();
        assert_eq!(budget.available(), Some(15000));
        fs::write(parent.join("memory.current"), "invalid").unwrap();
        assert_eq!(budget.available(), None);
        fs::remove_dir_all(root).unwrap();
    }
}

use super::*;
use crate::model_store::{ModelMetadata, ModelSource};

pub(crate) fn manifest() -> ModelManifest {
    ModelManifest {
        id: "model".into(),
        source: ModelSource::LocalPath {
            path: "test".into(),
        },
        storage: Default::default(),
        format: ModelFormat::Gguf,
        architecture: Some("qwen3".into()),
        tokenizer_path: None,
        config_path: None,
        model_path: Some("model.gguf".into()),
        backend: "llama-server".into(),
        created_unix: 1,
        files: vec![],
        artifacts: vec![],
        metadata: ModelMetadata::default(),
    }
}
pub(crate) fn inventory() -> Inventory {
    let mut i = Inventory::from_csv(
        "0, GPU-a, large, 0000:01:00.0, 24000, 23000\n1, GPU-b, small, 0000:02:00.0, 12000, 11000",
    )
    .unwrap();
    i.host_available_bytes = Some(64 * 1024 * 1024 * 1024);
    i
}
pub(crate) fn profile(executable: PathBuf) -> Profile {
    Profile {
        id: "worker-a".into(),
        alias: "chat".into(),
        model: "model".into(),
        runtime: Runtime::LlamaCuda,
        executable,
        build: "test".into(),
        gpus: vec!["GPU-a".into()],
        strategy: Strategy::Single,
        tensor_parallel: 1,
        pipeline_parallel: 1,
        split: vec![],
        context: 1024,
        batch: 128,
        parallel: 1,
        cpu_threads: 2,
        offload: Offload::None,
        kv_cpu: false,
        memory: vec![DeviceBudget {
            weights: 1024 * 1024 * 1024,
            kv: 1024,
            compute: 1024,
            overhead: 1024,
            reserve: 1024,
            ..Default::default()
        }],
        host_bytes: 1024 * 1024 * 1024,
        native_args: vec![],
    }
}
#[cfg(unix)]
pub(crate) fn executable(dir: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join("runtime");
    fs::write(&path, "#!/bin/sh\necho '--device --split-mode --main-gpu --tensor-split --n-cpu-moe --moe-expert-cache-size --moe-expert-cache-host-pinned-mb --tensor-parallel-size --pipeline-parallel-size --distributed-executor-backend --max-model-len --max-num-seqs --max-num-batched-tokens --gpu-memory-utilization --cpu-offload-gb --enable-expert-parallel --no-warmup'\n").unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}
#[cfg(unix)]
pub(crate) fn plan() -> (tempfile::TempDir, Plan) {
    let dir = tempfile::tempdir().unwrap();
    let p = profile(executable(dir.path()))
        .resolve(&manifest(), &inventory())
        .unwrap();
    (dir, p)
}

#[test]
fn memory_overflow_rejected() {
    assert!(
        DeviceBudget {
            weights: u64::MAX,
            kv: 1,
            ..Default::default()
        }
        .total()
        .is_err()
    );
}

#[cfg(unix)]
#[test]
fn mixed_group_dry_run_is_deterministic_and_rejects_overlaps() {
    let dir = tempfile::tempdir().unwrap();
    let binary = executable(dir.path());
    let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
    store.ensure().unwrap();
    let m = manifest();
    fs::create_dir_all(store.model_dir(&m.id)).unwrap();
    fs::write(
        store.model_dir(&m.id).join("manifest.json"),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    let mut i = inventory();
    let mut third = i.devices[1].clone();
    third.id = "GPU-c".into();
    third.physical_index = 2;
    third.visible_index = Some(2);
    i.devices.push(third);
    let mut group = profile(binary.clone());
    group.strategy = Strategy::Layer;
    group.gpus.push("GPU-b".into());
    group.memory.push(group.memory[0].clone());
    let mut independent = profile(binary);
    independent.id = "independent".into();
    independent.alias = "second".into();
    independent.gpus = vec!["GPU-c".into()];
    let mut config = Configuration {
        version: 1,
        profiles: vec![group, independent],
    };
    let a = config.resolve(&store, &i).unwrap();
    let b = config.resolve(&store, &i).unwrap();
    assert_eq!(
        a.iter().map(|p| &p.fingerprint).collect::<Vec<_>>(),
        b.iter().map(|p| &p.fingerprint).collect::<Vec<_>>()
    );
    assert_eq!(a[0].devices.len(), 2);
    assert_eq!(a[1].devices[0].id, "GPU-c");
    config.profiles[1].gpus = vec!["GPU-b".into()];
    assert!(config.resolve(&store, &i).is_err());
}
#[test]
fn native_escape_hatches_rejected() {
    for flag in [
        "--device=CUDA2",
        "-ts",
        "--config",
        "--port",
        "--gpu-memory-utilization=1",
        "--tensor-parallel-size",
        "--env",
    ] {
        assert!(validate_native_args(Runtime::LlamaCuda, &[flag.into()]).is_err());
        assert!(validate_native_args(Runtime::VllmCuda, &[flag.into()]).is_err());
    }
}
#[cfg(unix)]
#[test]
fn single_reindex_and_auto_smallest_fit() {
    let (dir, p) = plan();
    assert_eq!(p.environment["CUDA_VISIBLE_DEVICES"], "GPU-a");
    assert!(p.args.windows(2).any(|a| a == ["--device", "CUDA0"]));
    let mut profile = profile(executable(dir.path()));
    profile.strategy = Strategy::Auto;
    profile.gpus.clear();
    let auto = profile.resolve(&manifest(), &inventory()).unwrap();
    assert_eq!(auto.devices[0].id, "GPU-b");
}
#[cfg(unix)]
#[test]
fn unequal_layer_groups_validate_each_device_and_order() {
    let (dir, _) = plan();
    let mut p = profile(executable(dir.path()));
    p.strategy = Strategy::Layer;
    p.gpus = vec!["GPU-b".into(), "GPU-a".into()];
    p.memory.push(p.memory[0].clone());
    p.split = vec![1.0, 2.0];
    let plan = p.resolve(&manifest(), &inventory()).unwrap();
    assert_eq!(plan.environment["CUDA_VISIBLE_DEVICES"], "GPU-b,GPU-a");
    assert!(plan.args.windows(2).any(|a| a == ["--tensor-split", "1,2"]));
    p.memory[0].weights = 12 * 1024 * 1024 * 1024;
    assert!(p.resolve(&manifest(), &inventory()).is_err());
    p.memory[0].weights = 1024;
    p.split[0] = f64::NAN;
    assert!(p.resolve(&manifest(), &inventory()).is_err());
}
#[cfg(unix)]
#[test]
fn unsupported_combinations_and_hidden_devices_fail() {
    let (dir, _) = plan();
    let mut p = profile(executable(dir.path()));
    p.strategy = Strategy::Tensor;
    assert!(p.resolve(&manifest(), &inventory()).is_err());
    p.strategy = Strategy::Single;
    p.offload = Offload::HelperGpu;
    assert!(p.resolve(&manifest(), &inventory()).is_err());
    p.offload = Offload::ExpertCache {
        slots: 8,
        pinned_mib: 512,
    };
    assert!(p.resolve(&manifest(), &inventory()).is_err());
    p.offload = Offload::None;
    let mut i = inventory();
    i.apply_visibility(Some("GPU-b"), None).unwrap();
    assert!(p.resolve(&manifest(), &i).is_err());
}
#[cfg(unix)]
#[test]
fn fingerprint_ignores_scrapes_but_tracks_placement_and_context() {
    let (_dir, a) = plan();
    let mut i = inventory();
    i.devices[0].free_bytes = Some(22 * 1024 * 1024 * 1024);
    let b = a.profile.resolve(&manifest(), &i).unwrap();
    assert_eq!(a.fingerprint, b.fingerprint);
    let mut p = a.profile.clone();
    p.gpus = vec!["GPU-b".into()];
    assert_ne!(
        a.fingerprint,
        p.resolve(&manifest(), &i).unwrap().fingerprint
    );
    p = a.profile.clone();
    p.context += 1;
    assert_ne!(
        a.fingerprint,
        p.resolve(&manifest(), &i).unwrap().fingerprint
    );
    fs::write(&a.profile.executable, "changed").unwrap();
    assert!(a.verify_binary().is_err());
}
#[cfg(unix)]
#[test]
fn vllm_tp_pp_and_per_gpu_fraction() {
    let (dir, _) = plan();
    let mut p = profile(executable(dir.path()));
    let mut m = manifest();
    m.format = ModelFormat::SafeTensors;
    p.runtime = Runtime::VllmCuda;
    p.strategy = Strategy::Tensor;
    p.tensor_parallel = 2;
    p.gpus.push("GPU-b".into());
    p.memory.push(p.memory[0].clone());
    let plan = p.resolve(&m, &inventory()).unwrap();
    assert!(
        plan.args
            .windows(2)
            .any(|a| a == ["--distributed-executor-backend", "mp"])
    );
    assert!(
        plan.args
            .windows(2)
            .any(|a| a == ["--tensor-parallel-size", "2"])
    );
    p.pipeline_parallel = 2;
    assert!(p.resolve(&m, &inventory()).is_err());
}
#[cfg(unix)]
#[test]
fn reservation_race_and_rollback() {
    use crate::inference_service::resources::Reservations;
    let (_dir, plan) = plan();
    let resources = Arc::new(Reservations::new(inventory().host_available_bytes.unwrap()));
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|n| {
            let resources = resources.clone();
            let barrier = barrier.clone();
            let mut plan = plan.clone();
            plan.profile.id = format!("worker-{n}");
            std::thread::spawn(move || {
                barrier.wait();
                resources.acquire(&plan, &inventory()).ok()
            })
        })
        .collect();
    let leases: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(leases.iter().filter(|l| l.is_some()).count(), 1);
    drop(leases);
    assert!(resources.acquire(&plan, &inventory()).is_ok());
    let mut bad = plan.clone();
    bad.devices.push(inventory().devices[1].clone());
    bad.profile.memory.push(DeviceBudget {
        weights: u64::MAX,
        ..Default::default()
    });
    assert!(resources.acquire(&bad, &inventory()).is_err());
    assert!(resources.acquire(&plan, &inventory()).is_ok());
}
#[cfg(unix)]
#[test]
fn disjoint_workers_admit_independently() {
    use crate::inference_service::resources::Reservations;
    let (_dir, plan) = plan();
    let resources = Arc::new(Reservations::new(inventory().host_available_bytes.unwrap()));
    let first = resources.acquire(&plan, &inventory()).unwrap();
    let mut second = plan.clone();
    second.profile.id = "other".into();
    second.devices[0] = inventory().devices[1].clone();
    let other = resources.acquire(&second, &inventory()).unwrap();
    drop(first);
    assert!(resources.acquire(&plan, &inventory()).is_ok());
    drop(other);
}

#[cfg(unix)]
#[test]
fn controlled_launcher_binds_visibility_reuses_restarts_and_rolls_back() {
    use crate::inference_service::resources::Reservations;
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let executable = dir.path().join("worker.py");
    fs::write(
        &executable,
        include_str!("../../tests/fixtures/deployment_worker.py"),
    )
    .unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
    let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
    store.ensure().unwrap();
    let m = manifest();
    fs::create_dir_all(store.model_dir(&m.id)).unwrap();
    let mut gguf = b"GGUF".to_vec();
    gguf.extend(3u32.to_le_bytes());
    gguf.extend(0u64.to_le_bytes());
    gguf.extend(0u64.to_le_bytes());
    fs::write(store.absolute_model_file(&m, "model.gguf"), gguf).unwrap();
    let p = profile(executable).resolve(&m, &inventory()).unwrap();
    let resources = Arc::new(Reservations::new(inventory().host_available_bytes.unwrap()));
    let execution = Execution {
        plan: Arc::new(p.clone()),
        resources: resources.clone(),
        inventory: Some(inventory()),
    };
    let backend = execution.backend(store.clone());
    backend.prepare(&m).unwrap();
    backend.prepare(&m).unwrap();
    let starts = || {
        fs::read_to_string(dir.path().join("starts.jsonl"))
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap())
            .collect::<Vec<_>>()
    };
    let retained_session = backend.start_chat_session(&m, None).unwrap().unwrap();
    let records = starts();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["cuda"], "GPU-a");
    assert_eq!(records[0]["threads"], "2");
    assert!(resources.acquire(&p, &inventory()).is_err());
    let pid = records[0]["pid"].as_i64().unwrap() as i32;
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
    for _ in 0..100 {
        if backend.start_chat_session(&m, None).is_ok() && starts().len() == 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(starts().len(), 2);
    assert!(!retained_session.is_available());
    drop(retained_session);
    drop(backend);
    assert!(resources.acquire(&p, &inventory()).is_ok());
    fs::write(dir.path().join("fail-start"), "").unwrap();
    let failed = execution.backend(store);
    assert!(failed.prepare(&m).is_err());
    assert!(resources.acquire(&p, &inventory()).is_ok());
}

#[cfg(unix)]
#[test]
fn controlled_disjoint_workers_start_concurrently_and_fail_independently() {
    use crate::inference_service::resources::Reservations;
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let executable = dir.path().join("worker.py");
    fs::write(
        &executable,
        include_str!("../../tests/fixtures/deployment_worker.py"),
    )
    .unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
    let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
    store.ensure().unwrap();
    let m = manifest();
    fs::create_dir_all(store.model_dir(&m.id)).unwrap();
    let mut gguf = b"GGUF".to_vec();
    gguf.extend(3u32.to_le_bytes());
    gguf.extend([0u8; 16]);
    fs::write(store.absolute_model_file(&m, "model.gguf"), gguf).unwrap();
    let resources = Arc::new(Reservations::new(inventory().host_available_bytes.unwrap()));
    let mut a = profile(executable);
    let mut b = a.clone();
    b.id = "worker-b".into();
    b.gpus = vec!["GPU-b".into()];
    a.alias = "replicas".into();
    b.alias = "replicas".into();
    let backends: Vec<_> = [a, b]
        .into_iter()
        .map(|p| {
            Execution {
                plan: Arc::new(p.resolve(&m, &inventory()).unwrap()),
                resources: resources.clone(),
                inventory: Some(inventory()),
            }
            .backend(store.clone())
        })
        .collect();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let threads: Vec<_> = backends
        .iter()
        .map(|backend| {
            let backend = backend.clone();
            let barrier = barrier.clone();
            let m = m.clone();
            std::thread::spawn(move || {
                barrier.wait();
                backend.prepare(&m).unwrap();
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    let records: Vec<serde_json::Value> = fs::read_to_string(dir.path().join("starts.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(records.len(), 2);
    assert_ne!(records[0]["cuda"], records[1]["cuda"]);
    let a = records.iter().find(|r| r["cuda"] == "GPU-a").unwrap();
    unsafe {
        libc::kill(a["pid"].as_i64().unwrap() as i32, libc::SIGKILL);
    }
    backends[1].prepare(&m).unwrap();
    assert_eq!(
        fs::read_to_string(dir.path().join("starts.jsonl"))
            .unwrap()
            .lines()
            .count(),
        2
    );
}

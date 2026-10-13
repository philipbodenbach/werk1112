use crate::{
    backend::{
        PreloadHandle,
        runtime_cache::{self, RuntimeCache},
    },
    model_store::{ModelManifest, ModelStore},
    model_support,
};
use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Read, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionRequest {
    pub model: String,
    pub task: String,
    pub state: [f32; 8],
    pub images: BTreeMap<String, String>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionResponse {
    pub actions: Vec<[f32; 8]>,
    pub embodiment: String,
    pub joint_units: String,
    pub gripper_units: String,
    pub action_space: String,
    pub executed: bool,
}
impl ActionRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.task.trim().is_empty() && self.state.iter().all(|x| x.is_finite()),
            "invalid_request: task and finite DROID state required"
        );
        ensure!(
            (0.0..=1.0).contains(&self.state[7]),
            "invalid_request: gripper fraction must be 0..1"
        );
        ensure!(
            self.images.len() == 3
                && ["wrist", "left", "right"].iter().all(|k| self
                    .images
                    .get(*k)
                    .is_some_and(|v| !v.is_empty() && v.len() <= 8 * 1024 * 1024)),
            "invalid_request: wrist, left and right base64 RGB images required (640x360)"
        );
        Ok(())
    }
}
struct Process {
    child: Child,
    stdin: ChildStdin,
    replies: mpsc::Receiver<Result<Value, String>>,
    healthy: bool,
}
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
struct Worker(Mutex<Process>);
pub struct ActionBackend {
    workers: RuntimeCache<String, Worker>,
    #[cfg(test)]
    python: Option<std::path::PathBuf>,
}
impl Default for ActionBackend {
    fn default() -> Self {
        Self {
            workers: RuntimeCache::bounded(1),
            #[cfg(test)]
            python: None,
        }
    }
}
impl Process {
    fn response(&mut self) -> Result<Value> {
        match self.replies.recv_timeout(Duration::from_secs(900)) {
            Ok(Ok(value)) if value.get("error").is_none() => Ok(value),
            _ => {
                self.healthy = false;
                let _ = self.child.kill();
                Err(anyhow!(
                    "action_runtime_failed: check FLUX Action environment, checkpoint and hardware; worker discarded"
                ))
            }
        }
    }
}
impl ActionBackend {
    fn worker(&self, store: &ModelStore, manifest: &ModelManifest) -> Result<Arc<Worker>> {
        ensure!(
            model_support::is_action(manifest),
            "unsupported_task: model is not a registered DROID action policy"
        );
        let configured_python = std::env::var_os("WERK_FLUX_ACTION_PYTHON");
        #[cfg(test)]
        let configured_python = self
            .python
            .as_ref()
            .map(|p| p.as_os_str().to_os_string())
            .or(configured_python);
        let python = configured_python.ok_or_else(|| anyhow!("missing_runtime: set WERK_FLUX_ACTION_PYTHON to the official flux-action environment's Python"))?;
        let checkpoint = store.model_files_dir(manifest);
        ensure!(
            checkpoint.is_dir(),
            "unavailable_weights: DROID checkpoint directory is missing"
        );
        let identity = crate::model_store::ModelRuntimeIdentity::from_manifest(manifest)?;
        let key = format!("{identity}:{python:?}");
        let (worker, _, _) = self.workers.get_or_try_init(
            key,
            |w| {
                w.0.lock()
                    .is_ok_and(|mut p| p.healthy && matches!(p.child.try_wait(), Ok(None)))
            },
            || {
                let mut child = Command::new(&python)
                    .arg("-u")
                    .arg("-c")
                    .arg(include_str!("../../runtime/werk_flux_action.py"))
                    .arg(&checkpoint)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .context("missing_runtime: could not start FLUX Action Python")?;
                let stdin = child.stdin.take().context("missing action worker stdin")?;
                let stdout = child
                    .stdout
                    .take()
                    .context("missing action worker stdout")?;
                let (tx, rx) = mpsc::sync_channel(1);
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stdout);
                    loop {
                        let mut line = String::new();
                        match Read::by_ref(&mut reader)
                            .take(1024 * 1024)
                            .read_line(&mut line)
                        {
                            Ok(0) | Err(_) => break,
                            _ => {
                                if tx
                                    .send(
                                        serde_json::from_str(&line)
                                            .map_err(|_| "invalid worker JSON".into()),
                                    )
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                    }
                });
                let mut process = Process {
                    child,
                    stdin,
                    replies: rx,
                    healthy: true,
                };
                ensure!(
                    process.response()?["ready"] == true,
                    "action_runtime_failed: worker failed to load"
                );
                Ok(Worker(Mutex::new(process)))
            },
        )?;
        Ok(worker)
    }
}
fn infer(worker: &Worker, value: Value) -> Result<ActionResponse> {
    let mut process = worker
        .0
        .lock()
        .map_err(|_| anyhow!("action worker lock poisoned"))?;
    ensure!(
        process.healthy,
        "action_runtime_failed: worker is unavailable"
    );
    let bytes = serde_json::to_vec(&value)?;
    if process
        .stdin
        .write_all(&bytes)
        .and_then(|_| process.stdin.write_all(b"\n"))
        .and_then(|_| process.stdin.flush())
        .is_err()
    {
        process.healthy = false;
        anyhow::bail!("action_runtime_failed: worker disconnected");
    }
    let response: ActionResponse = serde_json::from_value(process.response()?["result"].clone())?;
    ensure!(
        response.actions.len() == 32
            && response.actions.iter().flatten().all(|x| x.is_finite())
            && response.embodiment == "droid"
            && response.joint_units == "radians"
            && response.gripper_units == "closed_fraction"
            && response.action_space == "absolute_joint_targets"
            && !response.executed,
        "upstream_schema: invalid DROID trajectory"
    );
    Ok(response)
}
impl ActionBackend {
    pub fn execute(&self, store: &ModelStore, request: ActionRequest) -> Result<ActionResponse> {
        request.validate()?;
        let manifest = store.get(&request.model)?;
        let mut value = serde_json::to_value(&request)?;
        value.as_object_mut().unwrap().remove("model");
        infer(self.worker(store, &manifest)?.as_ref(), value)
    }
    pub fn preload(&self, store: &ModelStore, manifest: &ModelManifest) -> Result<PreloadHandle> {
        runtime_cache::preload(|| {
            infer(
                self.worker(store, manifest)?.as_ref(),
                json!({"preload":true}),
            )?;
            Ok(())
        })
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::model_store::{ModelFile, ModelFormat, ModelSource, ModelStorage};
    use std::os::unix::fs::PermissionsExt;
    fn manifest(store: &ModelStore, id: &str) -> ModelManifest {
        let path = store.home().join(id);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("config.json"), "{}").unwrap();
        ModelManifest {
            id: id.into(),
            source: ModelSource::LocalPath {
                path: path.display().to_string(),
            },
            storage: ModelStorage::External { path },
            format: ModelFormat::SafeTensors,
            architecture: Some("flux3_action_droid".into()),
            tokenizer_path: None,
            config_path: None,
            model_path: None,
            backend: "flux-action".into(),
            created_unix: 1,
            files: vec![ModelFile {
                path: "files/config.json".into(),
                size: 2,
                checksum: String::new(),
            }],
            artifacts: vec![],
            metadata: Default::default(),
        }
    }
    #[test]
    fn action_preload_pins_each_worker_rolls_back_failure_and_reaps_on_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(dir.path().into())).unwrap();
        let executable = dir.path().join("fake-python");
        let pidfile = dir.path().join("pids");
        let script = format!(
            "#!/usr/bin/env python3\nimport sys, json, os\nwith open({:?}, 'a') as f: f.write(str(os.getpid())+'\\n')\nprint(json.dumps({{'ready':True}}),flush=True)\nfor line in sys.stdin:\n req=json.loads(line)\n response={{'error':'failure'}} if sys.argv[-1].endswith('broken') else {{'result':{{'actions':[[0.0]*8]*32,'embodiment':'droid','joint_units':'radians','gripper_units':'closed_fraction','action_space':'absolute_joint_targets','executed':False}}}}\n print(json.dumps(response),flush=True)\n",
            pidfile.display().to_string()
        );
        std::fs::write(&executable, script).unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let backend = ActionBackend {
            workers: RuntimeCache::bounded(1),
            python: Some(executable),
        };
        let models: Vec<_> = (0..4)
            .map(|i| manifest(&store, &format!("droid-{i}")))
            .collect();
        let mut handles = Vec::new();
        for m in &models {
            handles.push(backend.preload(&store, m).unwrap());
        }
        assert!(
            backend
                .preload(&store, &manifest(&store, "broken"))
                .is_err()
        );
        for m in &models {
            let worker = backend.worker(&store, m).unwrap();
            assert_eq!(
                infer(&worker, json!({"preload":true}))
                    .unwrap()
                    .actions
                    .len(),
                32
            );
        }
        assert_eq!(backend.workers.snapshot().len(), 4);
        let pids: Vec<i32> = std::fs::read_to_string(pidfile)
            .unwrap()
            .lines()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(pids.len(), 5);
        assert_eq!(unsafe { libc::kill(*pids.last().unwrap(), 0) }, -1);
        drop(handles);
        drop(backend);
        for pid in pids {
            assert_eq!(unsafe { libc::kill(pid, 0) }, -1, "worker must be reaped");
        }
    }
    #[test]
    fn action_schema_requires_droid_dimensions_and_no_actuation() {
        let mut request = ActionRequest {
            model: "droid".into(),
            task: "hold".into(),
            state: [0.0; 8],
            images: BTreeMap::from([
                ("wrist".into(), "AA==".into()),
                ("left".into(), "AA==".into()),
                ("right".into(), "AA==".into()),
            ]),
        };
        request.validate().unwrap();
        request.state[7] = 2.0;
        assert!(request.validate().is_err());
        request.state[7] = 0.0;
        request.images.remove("left");
        assert!(request.validate().is_err());
        assert!(
            serde_json::from_value::<ActionRequest>(
                json!({"model":"x","task":"hold","state":[0,0],"images":{}})
            )
            .is_err()
        );
    }
}

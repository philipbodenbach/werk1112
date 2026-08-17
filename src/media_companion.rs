use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, VecDeque},
    env,
    ffi::OsString,
    fmt,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, ExitStatus, Stdio},
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[cfg(test)]
use std::fs;

const PROTOCOL_VERSION: u32 = 1;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_EXECUTE_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);
const POLL_INTERVAL: Duration = Duration::from_millis(20);
const STDERR_TAIL_CHARS: usize = 8_000;
const STDERR_TAIL_BYTES: usize = STDERR_TAIL_CHARS * 4;
const EMBEDDED_COMPANION: &str = include_str!("../runtime/werk_media_companion.py");
const EMBEDDED_BOOTSTRAP: &str = "import io,sys\n\
n=int(sys.stdin.buffer.readline())\n\
code=sys.stdin.buffer.read(n)\n\
payload=sys.stdin.buffer.read()\n\
sys.stdin=io.TextIOWrapper(io.BytesIO(payload),encoding='utf-8')\n\
exec(compile(code,'<werk_media_companion>','exec'))";
const EMBEDDED_RESIDENT_BOOTSTRAP: &str = "import sys\n\
n=int(sys.stdin.buffer.readline())\n\
code=sys.stdin.buffer.read(n)\n\
exec(compile(code,'<werk_media_companion>','exec'))";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionDependency {
    pub available: bool,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionAccelerator {
    pub available: bool,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionHealth {
    pub ok: bool,
    pub status: String,
    pub protocol_version: u32,
    #[serde(default)]
    pub companion_version: Option<String>,
    #[serde(default)]
    pub python_version: Option<String>,
    #[serde(default)]
    pub dependencies: BTreeMap<String, CompanionDependency>,
    #[serde(default)]
    pub accelerators: BTreeMap<String, CompanionAccelerator>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionOutput {
    pub path: String,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionExecution {
    pub ok: bool,
    pub task: String,
    #[serde(default)]
    pub outputs: Vec<CompanionOutput>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompanionDoctorCheck {
    pub name: String,
    pub available: bool,
    pub required: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompanionDoctorReport {
    /// True when the launcher and protocol are usable. Optional ML packages do
    /// not make the companion globally unhealthy.
    pub available: bool,
    pub launcher: Option<String>,
    pub summary: String,
    pub checks: Vec<CompanionDoctorCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompanionProtocolError {
    pub command: String,
    pub code: String,
    pub message: String,
    pub detail: Option<String>,
}

impl fmt::Display for CompanionProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "media companion {} failed [{}]: {}",
            self.command, self.code, self.message
        )?;
        if let Some(detail) = self.detail.as_deref().filter(|value| !value.is_empty()) {
            write!(f, " ({detail})")?;
        }
        Ok(())
    }
}

impl std::error::Error for CompanionProtocolError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LauncherKind {
    Executable,
    Python,
}

#[derive(Debug, Clone)]
struct CompanionLauncher {
    program: PathBuf,
    args: Vec<OsString>,
    source: String,
    kind: LauncherKind,
    embedded_script: bool,
}

impl CompanionLauncher {
    fn command(&self, operation: &str) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args).arg(operation);
        command
    }

    fn resident_command(&self) -> Command {
        let mut command = Command::new(&self.program);
        if self.embedded_script {
            command
                .arg("-c")
                .arg(EMBEDDED_RESIDENT_BOOTSTRAP)
                .arg("serve");
        } else {
            command.args(&self.args).arg("serve");
        }
        command
    }

    fn display(&self) -> String {
        let mut parts = vec![self.program.display().to_string()];
        parts.extend(
            self.args
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned()),
        );
        format!("{} ({})", parts.join(" "), self.source)
    }
}

enum ResidentReaderEvent {
    Line(Vec<u8>),
    Eof,
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentNegotiation {
    Unknown,
    Confirmed,
    OneShotFallback,
}

struct ResidentProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    responses: Receiver<ResidentReaderEvent>,
    stdout_reader: Option<JoinHandle<()>>,
    stderr_reader: Option<JoinHandle<()>>,
    stderr_tail: Arc<Mutex<VecDeque<u8>>>,
}

impl ResidentProcess {
    fn clear_stderr(&self) {
        self.stderr_tail
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    fn terminate(&mut self) -> Option<ExitStatus> {
        self.stdin.take();
        let status = match self.child.try_wait() {
            Ok(Some(status)) => Some(status),
            Ok(None) => {
                let _ = self.child.kill();
                self.child.wait().ok()
            }
            Err(_) => {
                let _ = self.child.kill();
                self.child.wait().ok()
            }
        };
        // Do not join pipe readers here: a grandchild spawned by a media
        // encoder may still own an inherited pipe after Python was killed.
        // Dropping the handles detaches the readers and keeps timeout cleanup
        // bounded while the OS closes those pipes naturally.
        drop(self.stdout_reader.take());
        drop(self.stderr_reader.take());
        status
    }
}

impl Drop for ResidentProcess {
    fn drop(&mut self) {
        let _ = self.terminate();
    }
}

struct ResidentWorkerState {
    negotiation: ResidentNegotiation,
    process: Option<ResidentProcess>,
}

struct ResidentTransport {
    launcher: CompanionLauncher,
    next_request_id: AtomicU64,
    state: Mutex<ResidentWorkerState>,
}

impl fmt::Debug for ResidentTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResidentTransport")
            .field("launcher", &self.launcher.display())
            .finish_non_exhaustive()
    }
}

enum ResidentRequestOutcome {
    Response(Value),
    OneShotFallback,
}

enum ResidentExchangeOutcome {
    Response(Value),
    LegacyResponse(Vec<u8>),
}

enum ResidentSupportNegotiation {
    Confirmed,
    OneShotFallback,
}

impl ResidentTransport {
    fn new(launcher: CompanionLauncher) -> Self {
        let negotiation = if launcher.embedded_script {
            ResidentNegotiation::Confirmed
        } else {
            ResidentNegotiation::Unknown
        };
        Self {
            launcher,
            next_request_id: AtomicU64::new(1),
            state: Mutex::new(ResidentWorkerState {
                negotiation,
                process: None,
            }),
        }
    }

    fn request(
        &self,
        operation: &str,
        request: &Value,
        started: Instant,
        timeout: Duration,
    ) -> Result<ResidentRequestOutcome> {
        let mut state = loop {
            match self.state.try_lock() {
                Ok(state) => break state,
                Err(TryLockError::Poisoned(_)) => {
                    bail!("media companion resident worker mutex is poisoned")
                }
                Err(TryLockError::WouldBlock) => {
                    let elapsed = started.elapsed();
                    if elapsed >= timeout {
                        bail!(
                            "media companion resident '{operation}' timed out after {:.3}s waiting for the active worker",
                            timeout.as_secs_f64()
                        );
                    }
                    thread::sleep(POLL_INTERVAL.min(timeout.saturating_sub(elapsed)));
                }
            }
        };
        if state.negotiation == ResidentNegotiation::OneShotFallback {
            return Ok(ResidentRequestOutcome::OneShotFallback);
        }

        if state.negotiation == ResidentNegotiation::Unknown {
            let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
            match negotiate_resident_support(&self.launcher, request_id, started, timeout)? {
                ResidentSupportNegotiation::Confirmed => {
                    state.negotiation = ResidentNegotiation::Confirmed;
                }
                ResidentSupportNegotiation::OneShotFallback => {
                    state.negotiation = ResidentNegotiation::OneShotFallback;
                    return Ok(ResidentRequestOutcome::OneShotFallback);
                }
            }
        }

        let process_exited = match state.process.as_mut() {
            Some(process) => process
                .child
                .try_wait()
                .context("failed to inspect media companion resident worker")?
                .is_some(),
            None => false,
        };
        if process_exited {
            state.process.take();
        }
        if state.process.is_none() {
            state.process = Some(spawn_resident_process(&self.launcher)?);
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            bail!(
                "media companion resident '{operation}' timed out after {:.3}s before the request could be sent",
                timeout.as_secs_f64()
            );
        }

        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let exchange = resident_exchange(
            state
                .process
                .as_mut()
                .context("media companion resident worker was not started")?,
            request_id,
            operation,
            request,
            remaining,
        );

        match exchange {
            Ok(ResidentExchangeOutcome::Response(value)) => {
                Ok(ResidentRequestOutcome::Response(value))
            }
            Ok(ResidentExchangeOutcome::LegacyResponse(line)) => {
                let error = anyhow!(
                    "returned a response outside the resident JSONL envelope{}",
                    output_detail("stdout", &line)
                );
                Err(reset_resident_after_error(&mut state, operation, error))
            }
            Err(error) => Err(reset_resident_after_error(&mut state, operation, error)),
        }
    }
}

impl Drop for ResidentTransport {
    fn drop(&mut self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if let Some(mut process) = state.process.take() {
            let _ = process.terminate();
        }
    }
}

fn spawn_resident_process(launcher: &CompanionLauncher) -> Result<ResidentProcess> {
    let mut command = launcher.resident_command();
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to start media companion resident worker using {}",
            launcher.display()
        )
    })?;
    let stdout = child
        .stdout
        .take()
        .context("failed to capture media companion resident stdout")?;
    let stderr = child
        .stderr
        .take()
        .context("failed to capture media companion resident stderr")?;
    let mut stdin = child
        .stdin
        .take()
        .context("failed to open media companion resident stdin")?;

    let (response_sender, responses) = mpsc::channel();
    let stdout_reader = thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = Vec::new();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) => {
                    let _ = response_sender.send(ResidentReaderEvent::Eof);
                    break;
                }
                Ok(_) => {
                    while matches!(line.last(), Some(b'\n' | b'\r')) {
                        line.pop();
                    }
                    if response_sender
                        .send(ResidentReaderEvent::Line(line))
                        .is_err()
                    {
                        break;
                    }
                }
                Err(error) => {
                    let _ = response_sender.send(ResidentReaderEvent::Error(error.to_string()));
                    break;
                }
            }
        }
    });
    let stderr_tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_BYTES)));
    let stderr_reader_tail = stderr_tail.clone();
    let stderr_reader = thread::spawn(move || {
        let mut stderr = stderr;
        let mut buffer = [0_u8; 4096];
        loop {
            match stderr.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => {
                    let mut tail = stderr_reader_tail
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    tail.extend(&buffer[..count]);
                    while tail.len() > STDERR_TAIL_BYTES {
                        tail.pop_front();
                    }
                }
                Err(_) => break,
            }
        }
    });

    if launcher.embedded_script {
        let script = EMBEDDED_COMPANION.as_bytes();
        if let Err(error) = (|| -> std::io::Result<()> {
            writeln!(stdin, "{}", script.len())?;
            stdin.write_all(script)?;
            stdin.flush()
        })() {
            let mut process = ResidentProcess {
                child,
                stdin: Some(stdin),
                responses,
                stdout_reader: Some(stdout_reader),
                stderr_reader: Some(stderr_reader),
                stderr_tail,
            };
            let _ = process.terminate();
            return Err(error)
                .context("failed to send embedded media companion to resident worker");
        }
    }

    Ok(ResidentProcess {
        child,
        stdin: Some(stdin),
        responses,
        stdout_reader: Some(stdout_reader),
        stderr_reader: Some(stderr_reader),
        stderr_tail,
    })
}

fn remaining_timeout(started: Instant, timeout: Duration) -> Option<Duration> {
    let remaining = timeout.saturating_sub(started.elapsed());
    (!remaining.is_zero()).then_some(remaining)
}

fn resident_request_frame(request_id: u64, operation: &str, request: &Value) -> Result<Vec<u8>> {
    serde_json::to_vec(&json!({
        "transport_version": 1,
        "request_id": request_id,
        "operation": operation,
        "payload": request,
    }))
    .with_context(|| format!("failed to serialize media companion {operation} request"))
}

fn negotiate_resident_support(
    launcher: &CompanionLauncher,
    request_id: u64,
    started: Instant,
    timeout: Duration,
) -> Result<ResidentSupportNegotiation> {
    remaining_timeout(started, timeout)
        .context("media companion resident negotiation timed out")?;
    let mut process = spawn_resident_process(launcher)?;
    let negotiated = (|| -> Result<ResidentSupportNegotiation> {
        process.clear_stderr();
        let frame = resident_request_frame(request_id, "transport-handshake", &json!({}))?;
        let mut stdin = process
            .stdin
            .take()
            .context("media companion resident negotiation stdin is closed")?;
        let write_result = (|| -> std::io::Result<()> {
            stdin.write_all(&frame)?;
            stdin.write_all(b"\n")?;
            stdin.flush()
        })();
        if let Err(error) = write_result {
            // Some legacy launchers reject the unknown `serve` operation and
            // close stdin before the synthetic handshake can be delivered.
            // The original operation has not run, so one-shot remains safe.
            if error.kind() == std::io::ErrorKind::BrokenPipe {
                return Ok(ResidentSupportNegotiation::OneShotFallback);
            }
            return Err(error).context("failed to write media companion resident negotiation");
        }
        // Closing stdin is essential for legacy companions that use
        // json.load(stdin) and therefore cannot answer until they observe EOF.
        drop(stdin);

        let remaining = remaining_timeout(started, timeout)
            .context("media companion resident negotiation timed out")?;
        let line = match process.responses.recv_timeout(remaining) {
            Ok(ResidentReaderEvent::Line(line)) => line,
            Ok(ResidentReaderEvent::Eof) => {
                return Ok(ResidentSupportNegotiation::OneShotFallback);
            }
            Ok(ResidentReaderEvent::Error(error)) => {
                bail!("failed while reading resident negotiation response: {error}")
            }
            Err(RecvTimeoutError::Timeout) => {
                bail!(
                    "media companion resident negotiation timed out after {:.3}s",
                    timeout.as_secs_f64()
                )
            }
            Err(RecvTimeoutError::Disconnected) => {
                bail!("media companion resident negotiation response channel closed")
            }
        };
        match decode_resident_response(&line, request_id)? {
            ResidentExchangeOutcome::Response(_) => Ok(ResidentSupportNegotiation::Confirmed),
            ResidentExchangeOutcome::LegacyResponse(_) => {
                Ok(ResidentSupportNegotiation::OneShotFallback)
            }
        }
    })();
    let _ = process.terminate();
    negotiated
}

fn resident_exchange(
    process: &mut ResidentProcess,
    request_id: u64,
    operation: &str,
    request: &Value,
    timeout: Duration,
) -> Result<ResidentExchangeOutcome> {
    // The worker is serialized by ResidentTransport, so clearing here gives
    // every exchange an independent diagnostic window. A failure in request N
    // must never report stderr emitted by a successful request N-1.
    process.clear_stderr();
    let frame = resident_request_frame(request_id, operation, request)?;
    let stdin = process
        .stdin
        .as_mut()
        .context("media companion resident stdin is closed")?;
    stdin
        .write_all(&frame)
        .context("failed to write media companion resident request")?;
    stdin
        .write_all(b"\n")
        .context("failed to terminate media companion resident request frame")?;
    stdin
        .flush()
        .context("failed to flush media companion resident request")?;

    let event = match process.responses.recv_timeout(timeout) {
        Ok(event) => event,
        Err(RecvTimeoutError::Timeout) => {
            bail!("timed out after {:.3}s", timeout.as_secs_f64())
        }
        Err(RecvTimeoutError::Disconnected) => bail!("closed its response channel"),
    };
    let line = match event {
        ResidentReaderEvent::Line(line) => line,
        ResidentReaderEvent::Eof => bail!("closed stdout before returning a response"),
        ResidentReaderEvent::Error(error) => {
            bail!("failed while reading stdout: {error}")
        }
    };
    decode_resident_response(&line, request_id)
}

fn decode_resident_response(line: &[u8], request_id: u64) -> Result<ResidentExchangeOutcome> {
    let value: Value = serde_json::from_slice(line).map_err(|error| {
        anyhow!(
            "resident worker returned invalid JSON: {error}{}",
            output_detail("stdout", line)
        )
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("resident worker response must be a JSON object"))?;
    let envelope_like = object.contains_key("transport_version")
        || object.contains_key("request_id")
        || object.contains_key("response");
    if !envelope_like {
        if object.get("ok").is_some_and(Value::is_boolean) {
            return Ok(ResidentExchangeOutcome::LegacyResponse(line.to_vec()));
        }
        bail!(
            "resident worker returned JSON outside both the resident and legacy response envelopes{}",
            output_detail("stdout", line)
        );
    }
    let version = object
        .get("transport_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("resident response has no integer transport_version"))?;
    if version != 1 {
        bail!("resident response transport version mismatch: expected 1, got {version}");
    }
    let response_id = object
        .get("request_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("resident response has no integer request_id"))?;
    if response_id != request_id {
        bail!("resident response request id mismatch: expected {request_id}, got {response_id}");
    }
    let response = object.get("response").cloned().unwrap_or_else(|| {
        let mut response = object.clone();
        response.remove("transport_version");
        response.remove("request_id");
        Value::Object(response)
    });
    Ok(ResidentExchangeOutcome::Response(response))
}

fn reset_resident_after_error(
    state: &mut ResidentWorkerState,
    operation: &str,
    error: anyhow::Error,
) -> anyhow::Error {
    let Some(mut process) = state.process.take() else {
        return error.context(format!("media companion resident '{operation}' failed"));
    };
    let status = process.terminate();
    anyhow!(
        "media companion resident '{operation}' failed: {error:#}{}",
        status
            .map(|status| format!("; terminated with {}", exit_status_detail(status)))
            .unwrap_or_default(),
    )
}

#[derive(Debug, Clone)]
pub struct CompanionClient {
    launcher: CompanionLauncher,
    request_timeout: Duration,
    execute_timeout: Duration,
    resident: Option<Arc<ResidentTransport>>,
}

impl CompanionClient {
    pub fn new() -> Result<Self> {
        Self::discover()
    }

    pub fn discover() -> Result<Self> {
        let launcher = discover_launcher()?;
        Ok(Self {
            launcher,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            execute_timeout: DEFAULT_EXECUTE_TIMEOUT,
            resident: None,
        })
    }

    /// Builds a client for an explicitly resolved process. `program` and
    /// `args` are passed directly to `std::process::Command`; no shell parsing
    /// or interpolation is performed.
    pub fn from_command(
        program: impl Into<PathBuf>,
        args: impl IntoIterator<Item = OsString>,
    ) -> Self {
        Self {
            launcher: CompanionLauncher {
                program: program.into(),
                args: args.into_iter().collect(),
                source: "explicit command".to_string(),
                kind: LauncherKind::Executable,
                embedded_script: false,
            },
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            execute_timeout: DEFAULT_EXECUTE_TIMEOUT,
            resident: None,
        }
    }

    /// Keeps one companion process alive and serializes requests through it.
    ///
    /// Compatible legacy companions that do not implement the resident
    /// `serve` JSONL transport are detected on the first request and continue
    /// to use the existing one-process-per-request protocol.
    pub fn with_resident_worker(mut self) -> Self {
        if self.resident.is_none() {
            self.resident = Some(Arc::new(ResidentTransport::new(self.launcher.clone())));
        }
        self
    }

    /// Uses the one-shot protocol for this clone without changing other
    /// clones that share the resident worker. This is intended for lightweight
    /// health/probe/estimate preflights so they never queue behind a long
    /// serialized execute call.
    pub fn without_resident_worker(mut self) -> Self {
        self.resident = None;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self.execute_timeout = timeout;
        self
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn with_execute_timeout(mut self, timeout: Duration) -> Self {
        self.execute_timeout = timeout;
        self
    }

    pub fn launcher_description(&self) -> String {
        self.launcher.display()
    }

    pub fn health(&self) -> Result<CompanionHealth> {
        let value = self.request_with_timeout("health", &json!({}), self.request_timeout)?;
        let health: CompanionHealth =
            serde_json::from_value(value).context("invalid media companion health response")?;
        if health.protocol_version != PROTOCOL_VERSION {
            bail!(
                "media companion protocol mismatch: Werk expects {}, companion reports {}",
                PROTOCOL_VERSION,
                health.protocol_version
            );
        }
        Ok(health)
    }

    pub fn capabilities(&self) -> Result<Value> {
        self.request_with_timeout("capabilities", &json!({}), self.request_timeout)
    }

    pub fn probe_model(&self, request: &Value) -> Result<Value> {
        self.request_with_timeout("probe-model", request, self.request_timeout)
    }

    pub fn estimate(&self, request: &Value) -> Result<Value> {
        self.request_with_timeout("estimate", request, self.request_timeout)
    }

    pub fn execute(&self, request: &Value) -> Result<CompanionExecution> {
        let value = self.request_with_timeout("execute", request, self.execute_timeout)?;
        serde_json::from_value(value).context("invalid media companion execute response")
    }

    pub fn request(&self, operation: &str, request: &Value) -> Result<Value> {
        let timeout = if operation == "execute" {
            self.execute_timeout
        } else {
            self.request_timeout
        };
        self.request_with_timeout(operation, request, timeout)
    }

    pub fn doctor(&self) -> CompanionDoctorReport {
        let launcher = self.launcher_description();
        let mut checks = vec![CompanionDoctorCheck {
            name: match self.launcher.kind {
                LauncherKind::Executable => "media companion executable".to_string(),
                LauncherKind::Python => "Python media companion".to_string(),
            },
            available: true,
            required: true,
            detail: launcher.clone(),
        }];

        match self.health() {
            Ok(health) => {
                checks.push(CompanionDoctorCheck {
                    name: "media companion protocol".to_string(),
                    available: health.status == "ok",
                    required: true,
                    detail: format!(
                        "protocol v{}; companion {}; Python {}",
                        health.protocol_version,
                        health.companion_version.as_deref().unwrap_or("unknown"),
                        health.python_version.as_deref().unwrap_or("unknown")
                    ),
                });
                for (name, accelerator) in health.accelerators {
                    let detail = match (accelerator.version, accelerator.detail) {
                        (Some(version), Some(detail)) => format!("{detail}; runtime {version}"),
                        (Some(version), None) => format!("runtime {version}"),
                        (None, Some(detail)) => detail,
                        (None, None) if accelerator.available => "available".to_string(),
                        (None, None) => "unavailable".to_string(),
                    };
                    checks.push(CompanionDoctorCheck {
                        name: format!("media accelerator {name}"),
                        available: accelerator.available,
                        required: false,
                        detail,
                    });
                }
                for (name, dependency) in health.dependencies {
                    checks.push(CompanionDoctorCheck {
                        name,
                        available: dependency.available,
                        required: false,
                        detail: dependency.detail.or(dependency.version).unwrap_or_else(|| {
                            if dependency.available {
                                "available".to_string()
                            } else {
                                "not installed (optional)".to_string()
                            }
                        }),
                    });
                }
                let available = checks
                    .iter()
                    .filter(|check| check.required)
                    .all(|check| check.available);
                CompanionDoctorReport {
                    available,
                    launcher: Some(launcher),
                    summary: if available {
                        "media companion is usable; missing optional runtimes only limit matching tasks"
                            .to_string()
                    } else {
                        "media companion launcher exists, but its protocol health check failed"
                            .to_string()
                    },
                    checks,
                }
            }
            Err(err) => {
                checks.push(CompanionDoctorCheck {
                    name: "media companion protocol".to_string(),
                    available: false,
                    required: true,
                    detail: err.to_string(),
                });
                CompanionDoctorReport {
                    available: false,
                    launcher: Some(launcher),
                    summary: "media companion health check failed".to_string(),
                    checks,
                }
            }
        }
    }

    pub fn discover_doctor_report() -> CompanionDoctorReport {
        match Self::discover() {
            Ok(client) => client.doctor(),
            Err(err) => CompanionDoctorReport {
                available: false,
                launcher: None,
                summary: "media companion is not configured".to_string(),
                checks: vec![CompanionDoctorCheck {
                    name: "media companion launcher".to_string(),
                    available: false,
                    required: true,
                    detail: err.to_string(),
                }],
            },
        }
    }

    fn request_with_timeout(
        &self,
        operation: &str,
        request: &Value,
        timeout: Duration,
    ) -> Result<Value> {
        validate_operation(operation)?;
        if !request.is_object() {
            bail!("media companion request for '{operation}' must be a JSON object");
        }
        if timeout.is_zero() {
            bail!("media companion timeout must be greater than zero");
        }
        let started = Instant::now();

        if let Some(resident) = &self.resident {
            match resident.request(operation, request, started, timeout)? {
                ResidentRequestOutcome::Response(value) => {
                    return parse_companion_response(operation, value);
                }
                ResidentRequestOutcome::OneShotFallback => {}
            }
        }

        self.request_one_shot_with_deadline(operation, request, started, timeout)
    }

    fn request_one_shot_with_deadline(
        &self,
        operation: &str,
        request: &Value,
        started: Instant,
        timeout: Duration,
    ) -> Result<Value> {
        if remaining_timeout(started, timeout).is_none() {
            bail!(
                "media companion '{operation}' timed out after {:.3}s before one-shot execution",
                timeout.as_secs_f64()
            );
        }

        let request_json = serde_json::to_vec(request)
            .with_context(|| format!("failed to serialize media companion {operation} request"))?;
        let input = if self.launcher.embedded_script {
            let script = EMBEDDED_COMPANION.as_bytes();
            let mut framed = format!("{}\n", script.len()).into_bytes();
            framed.extend_from_slice(script);
            framed.extend_from_slice(&request_json);
            framed
        } else {
            request_json
        };
        let mut command = self.launcher.command(operation);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().with_context(|| {
            format!(
                "failed to start media companion for '{operation}' using {}",
                self.launcher.display()
            )
        })?;

        let stdout = child
            .stdout
            .take()
            .context("failed to capture media companion stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("failed to capture media companion stderr")?;
        let stdout_reader = read_pipe(stdout);
        let stderr_reader = read_pipe(stderr);

        let write_result = child
            .stdin
            .take()
            .context("failed to open media companion stdin")
            .and_then(|mut stdin| {
                stdin
                    .write_all(&input)
                    .context("failed to write media companion request")?;
                stdin
                    .flush()
                    .context("failed to flush media companion request")
            });
        if let Err(err) = write_result {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(err);
        }

        let status = loop {
            if started.elapsed() >= timeout {
                let _ = child.kill();
                let status = child.wait().ok();
                let stdout = join_pipe(stdout_reader);
                let stderr = join_pipe(stderr_reader);
                bail!(
                    "media companion '{operation}' timed out after {:.3}s{}{}{}",
                    timeout.as_secs_f64(),
                    status
                        .map(|status| format!("; terminated with {status}"))
                        .unwrap_or_default(),
                    output_detail("stderr", &stderr),
                    output_detail("stdout", &stdout),
                );
            }
            if let Some(status) = child
                .try_wait()
                .context("failed while waiting for media companion")?
            {
                break status;
            }
            thread::sleep(POLL_INTERVAL.min(timeout.saturating_sub(started.elapsed())));
        };

        let stdout = join_pipe(stdout_reader);
        let stderr = join_pipe(stderr_reader);
        if !status.success() {
            bail!(
                "media companion '{operation}' exited with {}{}{}",
                exit_status_detail(status),
                output_detail("stderr", &stderr),
                output_detail("stdout", &stdout),
            );
        }

        let value: Value = serde_json::from_slice(&stdout).map_err(|err| {
            anyhow!(
                "media companion '{operation}' returned invalid JSON: {err}{}{}",
                output_detail("stdout", &stdout),
                output_detail("stderr", &stderr)
            )
        })?;
        parse_companion_response(operation, value)
    }
}

fn parse_companion_response(operation: &str, value: Value) -> Result<Value> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("media companion '{operation}' response must be a JSON object"))?;
    match object.get("ok").and_then(Value::as_bool) {
        Some(true) => Ok(value),
        Some(false) => Err(protocol_error(operation, object).into()),
        None => bail!("media companion '{operation}' response has no boolean 'ok' field"),
    }
}

fn protocol_error(operation: &str, object: &Map<String, Value>) -> CompanionProtocolError {
    let error = object.get("error").and_then(Value::as_object);
    let code = error
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("companion_error")
        .to_string();
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| object.get("message").and_then(Value::as_str))
        .unwrap_or("media companion rejected the request")
        .to_string();
    let detail = error
        .and_then(|error| error.get("detail"))
        .and_then(value_detail);
    CompanionProtocolError {
        command: operation.to_string(),
        code,
        message,
        detail,
    }
}

fn value_detail(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(value) => Some(value.clone()),
        value => serde_json::to_string(value).ok(),
    }
}

fn validate_operation(operation: &str) -> Result<()> {
    if operation.is_empty()
        || !operation
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        bail!("invalid media companion operation: {operation:?}");
    }
    Ok(())
}

fn read_pipe<R>(mut reader: R) -> thread::JoinHandle<std::io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
}

fn join_pipe(handle: thread::JoinHandle<std::io::Result<Vec<u8>>>) -> Vec<u8> {
    handle.join().ok().and_then(Result::ok).unwrap_or_default()
}

fn output_detail(label: &str, bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }
    let text = String::from_utf8_lossy(bytes);
    let text = tail_chars(text.trim(), STDERR_TAIL_CHARS);
    if text.is_empty() {
        String::new()
    } else {
        format!("; {label}: {text}")
    }
}

fn tail_chars(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    format!(
        "...{}",
        text.chars()
            .skip(count.saturating_sub(max_chars))
            .collect::<String>()
    )
}

fn exit_status_detail(status: ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("{status} (signal {signal})");
        }
    }
    status.to_string()
}

fn discover_launcher() -> Result<CompanionLauncher> {
    if let Some(configured) = env::var_os("WERK_MEDIA_COMPANION") {
        let path = resolve_program(&configured).ok_or_else(|| {
            anyhow!(
                "WERK_MEDIA_COMPANION does not resolve to an executable file: {}",
                PathBuf::from(&configured).display()
            )
        })?;
        return Ok(CompanionLauncher {
            program: path,
            args: Vec::new(),
            source: "env WERK_MEDIA_COMPANION".to_string(),
            kind: LauncherKind::Executable,
            embedded_script: false,
        });
    }

    let (python, python_source) = discover_python().ok_or_else(|| {
        anyhow!(
            "no media companion executable or Python found; set WERK_MEDIA_COMPANION or WERK_MEDIA_PYTHON"
        )
    })?;
    if let Some((script, source)) = discover_repo_script() {
        return Ok(CompanionLauncher {
            program: python,
            args: vec![script.into_os_string()],
            source: format!("{python_source}; {source}"),
            kind: LauncherKind::Python,
            embedded_script: false,
        });
    }

    Ok(CompanionLauncher {
        program: python,
        args: vec![OsString::from("-c"), OsString::from(EMBEDDED_BOOTSTRAP)],
        source: format!("{python_source}; embedded companion script"),
        kind: LauncherKind::Python,
        embedded_script: true,
    })
}

fn discover_python() -> Option<(PathBuf, String)> {
    if let Some(configured) = env::var_os("WERK_MEDIA_PYTHON") {
        return resolve_program(&configured)
            .map(|path| (path, "env WERK_MEDIA_PYTHON".to_string()));
    }
    for name in python_program_names() {
        if let Some(path) = find_in_path(name) {
            return Some((path, format!("PATH {name}")));
        }
    }
    None
}

fn discover_repo_script() -> Option<(PathBuf, String)> {
    if let Some(configured) = env::var_os("WERK_MEDIA_COMPANION_SCRIPT") {
        let path = PathBuf::from(configured);
        if path.is_file() {
            return Some((path, "env WERK_MEDIA_COMPANION_SCRIPT".to_string()));
        }
    }

    let mut candidates = Vec::new();
    if let Some(root) = option_env!("CARGO_MANIFEST_DIR") {
        candidates.push((
            PathBuf::from(root)
                .join("runtime")
                .join("werk_media_companion.py"),
            "repository runtime script".to_string(),
        ));
    }
    if let Ok(executable) = env::current_exe()
        && let Some(dir) = executable.parent()
    {
        candidates.push((
            dir.join("runtime").join("werk_media_companion.py"),
            "runtime script next to executable".to_string(),
        ));
        candidates.push((
            dir.join("werk_media_companion.py"),
            "script next to executable".to_string(),
        ));
        if let Some(parent) = dir.parent() {
            candidates.push((
                parent.join("runtime").join("werk_media_companion.py"),
                "runtime script next to installation".to_string(),
            ));
        }
    }
    candidates.into_iter().find(|(path, _)| path.is_file())
}

fn python_program_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["python.exe", "python3.exe", "python", "python3"]
    } else {
        &["python3", "python"]
    }
}

fn resolve_program(program: &OsString) -> Option<PathBuf> {
    let path = PathBuf::from(program);
    if path.components().count() > 1 || path.is_absolute() {
        return path.is_file().then_some(path);
    }
    find_in_path(path.to_str()?)
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = Path::new(name);
    if (path.components().count() > 1 || path.is_absolute()) && path.is_file() {
        return Some(path.to_path_buf());
    }
    let path_env = env::var_os("PATH")?;
    for directory in env::split_paths(&path_env) {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        if candidate.extension().is_none() {
            for extension in ["exe", "cmd", "bat"] {
                let candidate = candidate.with_extension(extension);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    const MOCK_COMPANION: &str = r#"
import json
import sys
import time

operation = sys.argv[1]
payload = json.load(sys.stdin)

if payload.get("sleep"):
    time.sleep(float(payload["sleep"]))
if payload.get("exit"):
    print("mock process failure", file=sys.stderr)
    raise SystemExit(17)
if payload.get("malformed"):
    print('{"ok":true}{"ok":true}')
    raise SystemExit(0)
if payload.get("reject"):
    print(json.dumps({
        "ok": False,
        "error": {
            "code": "unsupported_parameter",
            "message": "mock rejection",
            "detail": {"parameter": "steps"},
        },
    }))
    raise SystemExit(0)

if operation == "health":
    response = {
        "ok": True,
        "status": "ok",
        "protocol_version": 1,
        "companion_version": "mock",
        "python_version": sys.version.split()[0],
        "accelerators": {
            "cpu": {
                "available": True,
                "version": None,
                "detail": "mock CPU",
            },
            "cuda": {
                "available": False,
                "version": "12.4",
                "detail": "mock CUDA unavailable",
            },
        },
        "dependencies": {
            "torch": {
                "available": False,
                "version": None,
                "detail": "not installed in mock",
            },
            "PIL": {
                "available": True,
                "version": "mock",
                "detail": None,
            },
        },
    }
elif operation == "probe-model":
    response = {
        "ok": True,
        "supported": True,
        "echo": payload,
    }
elif operation == "estimate":
    response = {
        "ok": True,
        "confidence": "heuristic",
        "accelerator_peak_bytes": 123,
    }
elif operation == "execute":
    response = {
        "ok": True,
        "task": payload.get("task", "image_generation"),
        "outputs": [{
            "path": "/tmp/mock.png",
            "mime_type": "image/png",
            "size": 3,
            "width": 1,
            "height": 1,
            "duration": None,
            "metadata": {},
        }],
        "metadata": {"mock": True},
        "warnings": [],
    }
else:
    response = {
        "ok": False,
        "error": {"code": "unknown_command", "message": operation},
    }

print(json.dumps(response))
"#;

    const RESIDENT_MOCK_COMPANION: &str = r#"
import json
import os
import sys
import time

if len(sys.argv) != 2 or sys.argv[1] != "serve":
    raise SystemExit(64)

instance = f"{os.getpid()}-{time.time_ns()}"
sequence = 0
for raw in sys.stdin:
    frame = json.loads(raw)
    request_id = frame["request_id"]
    operation = frame["operation"]
    payload = frame["payload"]
    sequence += 1
    if payload.get("log"):
        with open(payload["log"], "a", encoding="utf-8") as handle:
            handle.write(f"{operation}\n")
            handle.flush()
    if payload.get("stderr"):
        print(str(payload["stderr"]), file=sys.stderr, flush=True)
        if payload.get("stderr_settle"):
            time.sleep(float(payload["stderr_settle"]))
    if payload.get("sleep"):
        time.sleep(float(payload["sleep"]))
    if payload.get("crash"):
        print("mock resident crash", file=sys.stderr, flush=True)
        os._exit(17)
    if payload.get("raw_response") is not None:
        print(str(payload["raw_response"]), flush=True)
        continue
    if payload.get("reject"):
        response = {
            "ok": False,
            "error": {
                "code": "unsupported_parameter",
                "message": "mock resident rejection",
                "detail": {"parameter": "steps"},
            },
        }
    elif operation == "health":
        response = {
            "ok": True,
            "status": "ok",
            "protocol_version": 1,
            "companion_version": "resident-mock",
            "python_version": sys.version.split()[0],
            "accelerators": {},
            "dependencies": {},
            "instance": instance,
            "sequence": sequence,
        }
    elif operation == "execute":
        response = {
            "ok": True,
            "task": payload.get("task", "image_generation"),
            "outputs": [],
            "metadata": {"instance": instance, "sequence": sequence},
            "warnings": [],
        }
    else:
        response = {
            "ok": True,
            "instance": instance,
            "sequence": sequence,
            "echo": payload,
        }
    envelope = {
        "transport_version": 1,
        "request_id": request_id,
        "response": response,
    }
    print(json.dumps(envelope, separators=(",", ":")), flush=True)
"#;

    const LEGACY_FALLBACK_COMPANION: &str = r#"
import json
import sys
from pathlib import Path

operation = sys.argv[1]
with open(Path(__file__).with_suffix(".invocations"), "a", encoding="utf-8") as handle:
    handle.write(f"{operation}\n")
payload = json.load(sys.stdin)
if operation == "serve":
    response = {
        "ok": False,
        "error": {"code": "unknown_command", "message": "serve"},
    }
else:
    response = {"ok": True, "operation": operation, "echo": payload}
print(json.dumps(response))
"#;

    const MALFORMED_NEGOTIATION_COMPANION: &str = r#"
import json
import sys
from pathlib import Path

operation = sys.argv[1]
with open(Path(__file__).with_suffix(".invocations"), "a", encoding="utf-8") as handle:
    handle.write(f"{operation}\n")
payload = json.load(sys.stdin)
print("not-a-json-response", flush=True)
"#;

    const INVALID_STARTUP_ENVELOPE_COMPANION: &str = r#"
import json
import sys
from pathlib import Path

operation = sys.argv[1]
with open(Path(__file__).with_suffix(".invocations"), "a", encoding="utf-8") as handle:
    handle.write(f"{operation}\n")
if operation == "serve":
    json.loads(sys.stdin.readline())
    print(json.dumps({
        "transport_version": 1,
        "request_id": None,
        "ok": False,
        "error": {"code": "invalid_configuration", "message": "bad cache size"},
    }), flush=True)
else:
    payload = json.load(sys.stdin)
    print(json.dumps({"ok": True, "operation": operation, "echo": payload}))
"#;

    const NONZERO_LEGACY_COMPANION: &str = r#"
import json
import sys
from pathlib import Path

operation = sys.argv[1]
with open(Path(__file__).with_suffix(".invocations"), "a", encoding="utf-8") as handle:
    handle.write(f"{operation}\n")
payload = json.load(sys.stdin)
if operation == "serve":
    print("serve is unsupported", file=sys.stderr)
    raise SystemExit(17)
print(json.dumps({"ok": True, "operation": operation, "echo": payload}))
"#;

    const SLOW_LEGACY_COMPANION: &str = r#"
import json
import sys
import time

operation = sys.argv[1]
payload = json.load(sys.stdin)
time.sleep(0.2)
if operation == "serve":
    print(json.dumps({
        "ok": False,
        "error": {"code": "unknown_command", "message": "serve"},
    }))
else:
    print(json.dumps({"ok": True, "operation": operation, "echo": payload}))
"#;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = env::temp_dir().join(format!(
                "werk-media-companion-{label}-{}-{timestamp}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn mock_client(timeout: Duration) -> Option<(TestDirectory, CompanionClient)> {
        let python = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))?;
        let directory = TestDirectory::new("mock");
        let script = directory.0.join("mock_companion.py");
        fs::write(&script, MOCK_COMPANION).unwrap();
        let client = CompanionClient::from_command(python, vec![script.into_os_string()])
            .with_timeout(timeout);
        Some((directory, client))
    }

    fn scripted_client(
        label: &str,
        script_source: &str,
        timeout: Duration,
        resident: bool,
    ) -> Option<(TestDirectory, CompanionClient)> {
        let python = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))?;
        let directory = TestDirectory::new(label);
        let script = directory.0.join(format!("{label}.py"));
        fs::write(&script, script_source).unwrap();
        let client = CompanionClient::from_command(python, vec![script.into_os_string()])
            .with_timeout(timeout);
        let client = if resident {
            client.with_resident_worker()
        } else {
            client
        };
        Some((directory, client))
    }

    fn resident_mock_client(timeout: Duration) -> Option<(TestDirectory, CompanionClient)> {
        scripted_client("resident-mock", RESIDENT_MOCK_COMPANION, timeout, true)
    }

    fn resident_process_id(client: &CompanionClient) -> Option<u32> {
        let resident = client.resident.as_ref()?;
        let mut state = resident.state.lock().ok()?;
        state.process.as_mut().map(|process| process.child.id())
    }

    #[test]
    fn mock_companion_successfully_handles_all_public_operations() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let health = client.health().unwrap();
        assert_eq!(health.status, "ok");
        assert_eq!(health.protocol_version, PROTOCOL_VERSION);
        assert!(!health.dependencies["torch"].available);
        assert!(health.accelerators["cpu"].available);
        assert!(!health.accelerators["cuda"].available);

        let probe = client
            .probe_model(&json!({"model_path": "/tmp/model"}))
            .unwrap();
        assert_eq!(probe["supported"], true);
        let estimate = client
            .estimate(&json!({"task": "image_generation"}))
            .unwrap();
        assert_eq!(estimate["accelerator_peak_bytes"], 123);
        let execution = client
            .execute(&json!({"task": "image_generation"}))
            .unwrap();
        assert_eq!(execution.task, "image_generation");
        assert_eq!(execution.outputs.len(), 1);
    }

    #[test]
    fn protocol_error_preserves_companion_code_and_detail() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let err = client.probe_model(&json!({"reject": true})).unwrap_err();
        let protocol = err.downcast_ref::<CompanionProtocolError>().unwrap();
        assert_eq!(protocol.code, "unsupported_parameter");
        assert!(protocol.detail.as_deref().unwrap().contains("steps"));
    }

    #[test]
    fn nonzero_exit_reports_status_and_stderr() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let err = client
            .probe_model(&json!({"exit": true}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("17") || err.contains("exit"));
        assert!(err.contains("mock process failure"));
    }

    #[test]
    fn timeout_terminates_mock_process() {
        let Some((_directory, client)) = mock_client(Duration::from_millis(100)) else {
            return;
        };
        let started = Instant::now();
        let err = client
            .probe_model(&json!({"sleep": 2}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn malformed_or_multiple_json_objects_are_rejected() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let err = client
            .probe_model(&json!({"malformed": true}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid JSON"));
    }

    #[test]
    fn missing_optional_dependencies_do_not_make_doctor_globally_fatal() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let report = client.doctor();
        assert!(report.available);
        let torch = report
            .checks
            .iter()
            .find(|check| check.name == "torch")
            .unwrap();
        assert!(!torch.required);
        assert!(!torch.available);
    }

    #[test]
    fn resident_worker_reuses_one_process_across_requests() {
        let Some((_directory, client)) = resident_mock_client(Duration::from_secs(3)) else {
            return;
        };

        let first = client.capabilities().unwrap();
        let first_process = resident_process_id(&client).unwrap();
        let second = client.capabilities().unwrap();
        let second_process = resident_process_id(&client).unwrap();

        assert_eq!(first["instance"], second["instance"]);
        assert_eq!(first_process, second_process);
        assert_eq!(first["sequence"], 1);
        assert_eq!(second["sequence"], 2);
    }

    #[test]
    fn resident_worker_serializes_requests_from_client_clones() {
        let Some((_directory, client)) = resident_mock_client(Duration::from_secs(5)) else {
            return;
        };
        let mut workers = Vec::new();
        for index in 0..6 {
            let client = client.clone();
            workers.push(thread::spawn(move || {
                client
                    .request("probe-model", &json!({"index": index, "sleep": 0.01}))
                    .unwrap()
            }));
        }
        let responses = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();

        let instance = responses[0]["instance"].clone();
        for (index, response) in responses.iter().enumerate() {
            assert_eq!(response["instance"], instance);
            assert_eq!(response["echo"]["index"], index as u64);
        }
    }

    #[test]
    fn resident_protocol_error_preserves_the_worker() {
        let Some((_directory, client)) = resident_mock_client(Duration::from_secs(3)) else {
            return;
        };
        let before = client.capabilities().unwrap();
        let before_process = resident_process_id(&client).unwrap();

        let error = client.probe_model(&json!({"reject": true})).unwrap_err();
        let protocol = error.downcast_ref::<CompanionProtocolError>().unwrap();
        assert_eq!(protocol.code, "unsupported_parameter");

        let after = client.capabilities().unwrap();
        assert_eq!(before["instance"], after["instance"]);
        assert_eq!(resident_process_id(&client), Some(before_process));
    }

    #[test]
    fn resident_stderr_is_never_exposed_in_client_errors() {
        let Some((_directory, client)) = resident_mock_client(Duration::from_secs(3)) else {
            return;
        };
        client
            .request(
                "probe-model",
                &json!({"stderr": "old-request-diagnostic", "stderr_settle": 0.05}),
            )
            .unwrap();

        let error = client
            .request(
                "probe-model",
                &json!({
                    "stderr": "current-request-diagnostic",
                    "stderr_settle": 0.05,
                    "raw_response": "not-json",
                }),
            )
            .unwrap_err()
            .to_string();

        assert!(!error.contains("current-request-diagnostic"), "{error}");
        assert!(!error.contains("old-request-diagnostic"), "{error}");
    }

    #[test]
    fn resident_timeout_kills_worker_and_next_request_restarts_it() {
        let Some((_directory, client)) = resident_mock_client(Duration::from_secs(3)) else {
            return;
        };
        let before = client.capabilities().unwrap();
        let before_process = resident_process_id(&client).unwrap();
        let impatient = client
            .clone()
            .with_request_timeout(Duration::from_millis(100));

        let started = Instant::now();
        let error = impatient
            .request("probe-model", &json!({"sleep": 2}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(resident_process_id(&client).is_none());

        let after = client.capabilities().unwrap();
        assert_ne!(before["instance"], after["instance"]);
        assert_ne!(resident_process_id(&client), Some(before_process));
    }

    #[test]
    fn resident_execute_crash_is_not_replayed_and_next_request_restarts() {
        let Some((directory, client)) = resident_mock_client(Duration::from_secs(3)) else {
            return;
        };
        let before = client.capabilities().unwrap();
        let log = directory.0.join("execute.log");

        let error = client
            .request(
                "execute",
                &json!({
                    "task": "image_generation",
                    "crash": true,
                    "log": log.display().to_string(),
                }),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("closed") || error.contains("failed"));
        assert_eq!(fs::read_to_string(&log).unwrap(), "execute\n");

        let after = client.capabilities().unwrap();
        assert_ne!(before["instance"], after["instance"]);
        assert_eq!(fs::read_to_string(&log).unwrap(), "execute\n");
    }

    #[test]
    fn resident_negotiation_falls_back_once_for_legacy_companion() {
        let Some((directory, client)) = scripted_client(
            "legacy-fallback",
            LEGACY_FALLBACK_COMPANION,
            Duration::from_secs(3),
            true,
        ) else {
            return;
        };
        let log = directory.0.join("legacy-fallback.invocations");
        let request = json!({});

        let first = client.request("capabilities", &request).unwrap();
        let second = client.request("capabilities", &request).unwrap();

        assert_eq!(first["operation"], "capabilities");
        assert_eq!(second["operation"], "capabilities");
        assert_eq!(
            fs::read_to_string(log).unwrap(),
            "serve\ncapabilities\ncapabilities\n"
        );
    }

    #[test]
    fn resident_negotiation_surfaces_invalid_preflight_without_one_shot() {
        let Some((directory, client)) = scripted_client(
            "malformed-negotiation",
            MALFORMED_NEGOTIATION_COMPANION,
            Duration::from_secs(3),
            true,
        ) else {
            return;
        };
        let log = directory.0.join("malformed-negotiation.invocations");

        let error = client
            .request("capabilities", &json!({}))
            .unwrap_err()
            .to_string();

        assert!(error.contains("invalid JSON"), "{error}");
        assert_eq!(fs::read_to_string(log).unwrap(), "serve\n");
    }

    #[test]
    fn resident_startup_envelope_error_is_not_downgraded_to_one_shot() {
        let Some((directory, client)) = scripted_client(
            "invalid-startup-envelope",
            INVALID_STARTUP_ENVELOPE_COMPANION,
            Duration::from_secs(3),
            true,
        ) else {
            return;
        };
        let log = directory.0.join("invalid-startup-envelope.invocations");

        let error = client
            .request("capabilities", &json!({}))
            .unwrap_err()
            .to_string();

        assert!(error.contains("integer request_id"), "{error}");
        assert_eq!(fs::read_to_string(log).unwrap(), "serve\n");
    }

    #[test]
    fn resident_unknown_serve_nonzero_runs_first_execute_exactly_once() {
        let Some((directory, client)) = scripted_client(
            "nonzero-first-execute",
            NONZERO_LEGACY_COMPANION,
            Duration::from_secs(3),
            true,
        ) else {
            return;
        };
        let log = directory.0.join("nonzero-first-execute.invocations");

        let response = client.request("execute", &json!({"value": 17})).unwrap();

        assert_eq!(response["operation"], "execute");
        assert_eq!(response["echo"]["value"], 17);
        assert_eq!(fs::read_to_string(log).unwrap(), "serve\nexecute\n");
    }

    #[test]
    fn resident_negotiation_and_one_shot_share_one_wall_clock_deadline() {
        let Some((_directory, client)) = scripted_client(
            "slow-legacy",
            SLOW_LEGACY_COMPANION,
            Duration::from_millis(300),
            true,
        ) else {
            return;
        };

        let started = Instant::now();
        let error = client.capabilities().unwrap_err().to_string();
        let elapsed = started.elapsed();

        assert!(error.contains("timed out after 0.300s"), "{error}");
        assert!(elapsed >= Duration::from_millis(250), "{elapsed:?}");
        assert!(elapsed < Duration::from_millis(600), "{elapsed:?}");
    }

    #[test]
    fn one_shot_clone_does_not_detach_other_resident_clients() {
        let Some((_directory, client)) = resident_mock_client(Duration::from_secs(3)) else {
            return;
        };

        let one_shot = client.clone().without_resident_worker();

        assert!(client.resident.is_some());
        assert!(one_shot.resident.is_none());
    }

    #[test]
    fn resident_queue_timeout_does_not_kill_the_active_worker() {
        let Some((directory, client)) = resident_mock_client(Duration::from_secs(3)) else {
            return;
        };
        let before = client.capabilities().unwrap();
        let before_process = resident_process_id(&client).unwrap();
        let log = directory.0.join("active.log");
        let active = client.clone();
        let active_log = log.clone();
        let worker = thread::spawn(move || {
            active.request(
                "probe-model",
                &json!({
                    "sleep": 0.4,
                    "log": active_log.display().to_string(),
                }),
            )
        });
        let wait_started = Instant::now();
        while !log.is_file() && wait_started.elapsed() < Duration::from_secs(1) {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(log.is_file(), "active request did not reach the worker");

        let impatient = client
            .clone()
            .with_request_timeout(Duration::from_millis(100));
        let started = Instant::now();
        let error = impatient.capabilities().unwrap_err().to_string();
        let elapsed = started.elapsed();
        assert!(error.contains("waiting for the active worker"), "{error}");
        assert!(elapsed >= Duration::from_millis(80), "{elapsed:?}");
        assert!(elapsed < Duration::from_millis(350), "{elapsed:?}");

        worker.join().unwrap().unwrap();
        let after = client.capabilities().unwrap();
        assert_eq!(before["instance"], after["instance"]);
        assert_eq!(resident_process_id(&client), Some(before_process));
    }

    #[test]
    fn embedded_script_transport_keeps_large_script_out_of_command_line() {
        let Some(python) = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))
        else {
            return;
        };
        let client = CompanionClient {
            launcher: CompanionLauncher {
                program: python,
                args: vec![OsString::from("-c"), OsString::from(EMBEDDED_BOOTSTRAP)],
                source: "embedded transport test".to_string(),
                kind: LauncherKind::Python,
                embedded_script: true,
            },
            request_timeout: Duration::from_secs(15),
            execute_timeout: Duration::from_secs(5),
            resident: None,
        };
        let health = client.health().unwrap();
        assert_eq!(health.protocol_version, PROTOCOL_VERSION);
        assert_eq!(health.status, "ok");
    }

    #[test]
    fn embedded_script_supports_resident_serve_transport() {
        let Some(python) = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))
        else {
            return;
        };
        let client = CompanionClient {
            launcher: CompanionLauncher {
                program: python,
                args: vec![OsString::from("-c"), OsString::from(EMBEDDED_BOOTSTRAP)],
                source: "embedded resident transport test".to_string(),
                kind: LauncherKind::Python,
                embedded_script: true,
            },
            request_timeout: Duration::from_secs(15),
            execute_timeout: Duration::from_secs(5),
            resident: None,
        }
        .with_resident_worker();

        let first = client.health().unwrap();
        let first_process = resident_process_id(&client).unwrap();
        let second = client.health().unwrap();

        assert_eq!(first.protocol_version, PROTOCOL_VERSION);
        assert_eq!(second.status, "ok");
        assert_eq!(resident_process_id(&client), Some(first_process));
    }

    #[test]
    fn real_companion_offload_contract_tests_pass() {
        let Some(python) = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))
        else {
            return;
        };
        let tests = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("runtime")
            .join("test_werk_media_companion.py");
        let output = Command::new(python).arg(tests).output().unwrap();

        assert!(
            output.status.success(),
            "media companion offload tests failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn real_companion_strict_policy_rejects_unsupported_explicit_parameter() {
        let Some(python) = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))
        else {
            return;
        };
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("runtime")
            .join("werk_media_companion.py");
        let client = CompanionClient::from_command(python, vec![script.into_os_string()])
            .with_timeout(Duration::from_secs(5));
        let model = TestDirectory::new("strict-model");
        fs::write(
            model.0.join("model_index.json"),
            br#"{"_class_name":"FixturePipeline"}"#,
        )
        .unwrap();
        let output = TestDirectory::new("strict-output");

        let err = client
            .execute(&json!({
                "model_path": model.0.display().to_string(),
                "output_dir": output.0.display().to_string(),
                "task": "image_generation",
                "prompt": "fixture",
                "effective_parameters": {
                    "image.sampler": "euler"
                },
                "explicit_parameters": ["image.sampler"],
                "parameter_policy": "strict"
            }))
            .unwrap_err();
        let protocol = err.downcast_ref::<CompanionProtocolError>().unwrap();
        assert_eq!(protocol.code, "unsupported_parameter");
        assert!(
            protocol
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("image.sampler"))
        );
    }

    #[test]
    fn real_companion_warn_policy_reports_ignored_parameter_during_estimate() {
        let Some(python) = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))
        else {
            return;
        };
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("runtime")
            .join("werk_media_companion.py");
        let client = CompanionClient::from_command(python, vec![script.into_os_string()])
            .with_timeout(Duration::from_secs(5));
        let model = TestDirectory::new("warn-model");
        fs::write(
            model.0.join("model_index.json"),
            br#"{"_class_name":"FixturePipeline"}"#,
        )
        .unwrap();
        fs::write(model.0.join("weights.bin"), b"fixture").unwrap();

        let estimate = client
            .estimate(&json!({
                "model_path": model.0.display().to_string(),
                "task": "image_generation",
                "effective_parameters": {
                    "image.sampler": "euler",
                    "image.width": 64,
                    "image.height": 64
                },
                "explicit_parameters": ["image.sampler", "image.width"],
                "parameter_policy": "warn"
            }))
            .unwrap();
        let warnings = estimate["warnings"].as_array().unwrap();
        assert!(
            warnings
                .iter()
                .filter_map(Value::as_str)
                .any(|warning| warning.contains("image.sampler"))
        );
        assert_eq!(
            estimate["parameter_support"]["unsupported_explicit_parameters"],
            json!(["image.sampler"])
        );
    }
}

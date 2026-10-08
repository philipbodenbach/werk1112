//! Operational events: content-free request metadata, JSONL files and terminal presentation.
use anyhow::{Context as _, Result, ensure};
use clap::{Args, ValueEnum};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    fs::{File, OpenOptions},
    io::{self, Write},
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, SystemTime},
};

#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
    Trace,
    Off,
}
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub enum Format {
    #[default]
    Auto,
    Text,
    Json,
}
#[derive(Clone, Debug, Args)]
pub struct LogArgs {
    /// Operational log threshold. --verbose/--verbose-pure default to debug.
    #[arg(long, global = true, env = "WERK_LOG_LEVEL", value_enum)]
    pub log_level: Option<Level>,
    /// stderr rendering; auto uses the terminal palette and plain text in pipes.
    #[arg(
        long,
        global = true,
        env = "WERK_LOG_FORMAT",
        value_enum,
        default_value = "auto"
    )]
    pub log_format: Format,
    /// Append JSONL events to this file, independently of terminal rendering.
    #[arg(long, global = true, env = "WERK_LOG_FILE")]
    pub log_file: Option<PathBuf>,
    /// Rotate the JSONL file after this many MiB (one writer per path).
    #[arg(long, global = true, env = "WERK_LOG_MAX_SIZE_MB", default_value = "10", value_parser = clap::value_parser!(u64).range(1..=4096))]
    pub log_max_size_mb: u64,
    /// Number of rotated files to retain, named PATH.1 through PATH.N.
    #[arg(long, global = true, env = "WERK_LOG_RETENTION", default_value = "5", value_parser = clap::value_parser!(u16).range(1..=100))]
    pub log_retention: u16,
}

impl Default for LogArgs {
    fn default() -> Self {
        Self {
            log_level: None,
            log_format: Format::Auto,
            log_file: None,
            log_max_size_mb: 10,
            log_retention: 5,
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct Context {
    pub request_id: Option<String>,
    logger: Option<Arc<Logger>>,
}
tokio::task_local! { static REQUEST: Context; }
thread_local! { static BLOCKING: RefCell<Option<Context>> = const { RefCell::new(None) }; }
static LOGGER: OnceLock<Arc<Logger>> = OnceLock::new();
static SECRETS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static IDS: AtomicU64 = AtomicU64::new(1);

pub(crate) fn capture() -> Context {
    BLOCKING
        .with(|c| c.borrow().clone())
        .or_else(|| REQUEST.try_with(Clone::clone).ok())
        .unwrap_or_else(|| Context {
            request_id: None,
            logger: LOGGER.get().cloned(),
        })
}
pub(crate) async fn scope<T>(context: Context, future: impl std::future::Future<Output = T>) -> T {
    REQUEST.scope(context, future).await
}
pub(crate) fn with_context<T>(context: Context, operation: impl FnOnce() -> T) -> T {
    struct Reset(Option<Context>);
    impl Drop for Reset {
        fn drop(&mut self) {
            BLOCKING.with(|c| *c.borrow_mut() = self.0.take());
        }
    }
    let _reset = Reset(BLOCKING.with(|c| c.replace(Some(context))));
    operation()
}
pub(crate) fn spawn_blocking<F, R>(operation: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let context = capture();
    tokio::task::spawn_blocking(move || with_context(context, operation))
}
pub(crate) fn spawn<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    tokio::spawn(scope(capture(), future))
}
pub(crate) fn request_context() -> Context {
    let mut context = capture();
    // Server-owned IDs: do not trust or log arbitrary client correlation headers.
    static INSTANCE: OnceLock<String> = OnceLock::new();
    let instance = INSTANCE.get_or_init(|| {
        let mut bytes = [0u8; 8];
        if getrandom::getrandom(&mut bytes).is_ok() {
            format!("{:016x}", u64::from_le_bytes(bytes))
        } else {
            format!("{}-{}", std::process::id(), crate::observability::now_ms())
        }
    });
    context.request_id = Some(format!(
        "req_werk_{instance}_{}",
        IDS.fetch_add(1, Ordering::Relaxed)
    ));
    context
}
pub(crate) fn register_secrets(values: impl IntoIterator<Item = String>) {
    let mut secrets = SECRETS.lock().unwrap_or_else(|e| e.into_inner());
    for value in values {
        if !value.is_empty() && !secrets.contains(&value) {
            secrets.push(value);
        }
    }
    secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
}
fn scrub(text: &str) -> String {
    let mut text = crate::terminal::clean(text);
    for secret in SECRETS.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        text = text.replace(secret, "[REDACTED]");
    }
    text.chars().take(16_384).collect()
}
fn scrub_fields(value: Value) -> Value {
    match value {
        Value::String(s) => Value::String(scrub(&s)),
        Value::Array(a) => Value::Array(a.into_iter().map(scrub_fields).collect()),
        Value::Object(o) => Value::Object(
            o.into_iter()
                .map(|(k, v)| {
                    let protected = matches!(
                        k.to_ascii_lowercase().as_str(),
                        "authorization"
                            | "api_key"
                            | "password"
                            | "token"
                            | "secret"
                            | "prompt"
                            | "content"
                            | "body"
                            | "query"
                            | "questions"
                    );
                    (
                        k,
                        if protected {
                            json!("[REDACTED]")
                        } else {
                            scrub_fields(v)
                        },
                    )
                })
                .collect(),
        ),
        v => v,
    }
}
#[derive(Serialize)]
struct Event<'a> {
    schema_version: u32,
    timestamp: String,
    level: Level,
    event: &'a str,
    message: String,
    pid: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    fields: Value,
}

pub(crate) struct Logger {
    level: Level,
    format: Format,
    server: bool,
    console: bool,
    file: Option<mpsc::SyncSender<WriteMessage>>,
    dropped: Arc<AtomicU64>,
    failures: Arc<AtomicU64>,
}
enum WriteMessage {
    Line(Vec<u8>),
    Flush(mpsc::Sender<()>),
}
impl Logger {
    fn new(
        args: &LogArgs,
        verbose: bool,
        pure: bool,
        server: bool,
        console: bool,
    ) -> Result<Arc<Self>> {
        let level = args.log_level.unwrap_or(if verbose || pure {
            Level::Debug
        } else {
            Level::Info
        });
        let dropped = Arc::new(AtomicU64::new(0));
        let failures = Arc::new(AtomicU64::new(0));
        let file = if let Some(path) = &args.log_file {
            let mut writer = RotatingFile::open(
                path.clone(),
                args.log_max_size_mb.max(1) * 1024 * 1024,
                args.log_retention.max(1),
            )?;
            let (tx, rx) = mpsc::sync_channel::<WriteMessage>(4096);
            let lost = dropped.clone();
            let failed = failures.clone();
            std::thread::Builder::new().name("werk-log-writer".into()).spawn(move || {
                while let Ok(message) = rx.recv() {
                    match message {
                        WriteMessage::Line(line) => {
                            if let Err(error) = writer.write(&line) {
                                if failed.fetch_add(1, Ordering::Relaxed) == 0 {
                                    emergency("logging.write_failed", &format!("Log file write failed: {error}"));
                                }
                            }
                        },
                        WriteMessage::Flush(done) => {
                            if let Err(error) = writer.flush() {
                                failed.fetch_add(1, Ordering::Relaxed);
                                emergency("logging.flush_failed", &format!("Log file flush failed: {error}"));
                            }
                            let count = lost.load(Ordering::Relaxed);
                            if count > 0 { emergency("logging.dropped", &format!("{count} log records dropped because the file queue was full")); }
                            let _ = done.send(());
                        },
                    }
                }
            })?;
            Some(tx)
        } else {
            None
        };
        Ok(Arc::new(Self {
            level,
            format: if pure { Format::Json } else { args.log_format },
            server,
            console,
            file,
            dropped,
            failures,
        }))
    }
    fn enabled(&self, level: Level) -> bool {
        self.level != Level::Off && level != Level::Off && level <= self.level
    }
    fn emit(&self, context: &Context, level: Level, name: &str, message: &str, fields: Value) {
        if !self.enabled(level) {
            return;
        }
        let event = Event {
            schema_version: 1,
            timestamp: chrono::DateTime::<chrono::Utc>::from(SystemTime::now())
                .to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            level,
            event: name,
            message: scrub(message),
            pid: std::process::id(),
            request_id: context.request_id.clone(),
            fields: scrub_fields(fields),
        };
        let mut bytes = serde_json::to_vec(&event).expect("operational event serialization");
        bytes.push(b'\n');
        if let Some(tx) = &self.file {
            if tx.try_send(WriteMessage::Line(bytes.clone())).is_err() {
                if self.dropped.fetch_add(1, Ordering::Relaxed) == 0 {
                    emergency(
                        "logging.queue_full",
                        "Log file queue full; subsequent dropped records are counted",
                    );
                }
            }
        }
        if self.console {
            if self.format == Format::Json {
                crate::terminal::write(
                    crate::terminal::Stream::Err,
                    std::str::from_utf8(&bytes).unwrap(),
                );
            } else {
                let suffix = event
                    .request_id
                    .as_ref()
                    .map(|id| format!(" request_id={id}"))
                    .unwrap_or_default();
                let fields = event
                    .fields
                    .as_object()
                    .filter(|o| !o.is_empty() && self.format != Format::Auto)
                    .map(|_| format!(" {}", event.fields))
                    .unwrap_or_default();
                let text = format!(
                    "{} {} {}{suffix} {}{fields}",
                    event.timestamp,
                    format!("{level:?}").to_uppercase(),
                    name,
                    event.message
                );
                if self.format == Format::Auto {
                    crate::terminal::line(crate::terminal::Stream::Err, format_args!("{text}"));
                } else {
                    crate::terminal::write(crate::terminal::Stream::Err, &format!("{text}\n"));
                }
            }
        }
    }
    pub(crate) fn flush(&self) {
        if let Some(tx) = &self.file {
            let (done, wait) = mpsc::channel();
            // Never hang shutdown on a wedged disk or full queue.
            let start = std::time::Instant::now();
            let mut message = WriteMessage::Flush(done);
            loop {
                match tx.try_send(message) {
                    Ok(()) => {
                        if wait
                            .recv_timeout(Duration::from_secs(3).saturating_sub(start.elapsed()))
                            .is_err()
                        {
                            emergency(
                                "logging.flush_timeout",
                                "Timed out flushing operational logs",
                            );
                        }
                        break;
                    }
                    Err(mpsc::TrySendError::Full(returned))
                        if start.elapsed() < Duration::from_secs(3) =>
                    {
                        message = returned;
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    _ => {
                        emergency("logging.flush_failed", "Could not flush operational logs");
                        break;
                    }
                }
            }
        }
    }
}
fn emergency(name: &str, message: &str) {
    let line = json!({"schema_version":1,"timestamp":chrono::DateTime::<chrono::Utc>::from(SystemTime::now()).to_rfc3339(),"level":"error","event":name,"message":scrub(message),"pid":std::process::id(),"fields":{}});
    let _ = writeln!(io::stderr().lock(), "{line}");
}
pub(crate) fn init(
    args: &LogArgs,
    verbose: bool,
    pure: bool,
    server: bool,
    console: bool,
) -> Result<()> {
    register_secrets(
        std::env::vars()
            .filter(|(name, _)| {
                name.starts_with("WERK_") && (name.ends_with("KEY") || name.ends_with("TOKEN"))
                    || matches!(name.as_str(), "HF_TOKEN" | "HUGGING_FACE_HUB_TOKEN")
            })
            .map(|(_, value)| value),
    );
    let logger = Logger::new(args, verbose, pure, server, console)?;
    ensure!(
        LOGGER.set(logger).is_ok(),
        "operational logging already initialized"
    );
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        emit(
            Level::Error,
            "process.panicked",
            "Rust panic",
            json!({
                "file":info.location().map(|l| l.file()), "line":info.location().map(|l| l.line())
            }),
        );
        // Panic payloads can contain request data; keep JSON logs metadata-only.
        if !json_console() {
            previous(info);
        }
        flush();
    }));
    Ok(())
}
pub(crate) fn emit(level: Level, name: &str, message: &str, fields: Value) {
    emit_in(&capture(), level, name, message, fields);
}
pub(crate) fn emit_in(context: &Context, level: Level, name: &str, message: &str, fields: Value) {
    if let Some(logger) = &context.logger {
        logger.emit(context, level, name, message, fields);
    }
}
pub(crate) fn json_console() -> bool {
    capture().logger.is_some_and(|l| l.format == Format::Json)
}
pub(crate) fn raw_console() -> bool {
    capture().logger.is_some_and(|l| l.format != Format::Auto)
}
pub(crate) fn enabled() -> bool {
    capture().logger.is_some()
}
pub(crate) fn enabled_for(level: Level) -> bool {
    capture().logger.is_none_or(|l| l.enabled(level))
}
pub(crate) fn flush() {
    if let Some(logger) = LOGGER.get() {
        logger.flush();
    }
}
pub(crate) fn counters() -> (u64, u64) {
    LOGGER.get().map_or((0, 0), |l| {
        (
            l.dropped.load(Ordering::Relaxed),
            l.failures.load(Ordering::Relaxed),
        )
    })
}
/// Bridge legacy diagnostic lines; command results on stdout remain command results.
#[doc(hidden)]
pub fn diagnostic(text: &str, stdout: bool) -> bool {
    let context = capture();
    let Some(logger) = &context.logger else {
        return false;
    };
    if stdout && !(logger.server && logger.format == Format::Json) {
        return false;
    }
    if text.trim().is_empty() {
        return true;
    }
    let lower = text.to_ascii_lowercase();
    let level =
        if lower.starts_with("error") || lower.contains("] error") || lower.contains("] failed") {
            Level::Error
        } else if lower.starts_with("warning") || lower.contains("] warning") {
            Level::Warn
        } else {
            Level::Info
        };
    logger.emit(&context, level, "diagnostic", text, json!({}));
    true
}

struct RotatingFile {
    path: PathBuf,
    file: Option<File>,
    _lock: File,
    bytes: u64,
    max: u64,
    keep: u16,
}
fn private_file(path: &std::path::Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}
impl RotatingFile {
    fn sibling(path: &std::path::Path, suffix: impl std::fmt::Display) -> PathBuf {
        let mut name = path.as_os_str().to_owned();
        name.push(format!(".{suffix}"));
        PathBuf::from(name)
    }
    fn open(path: PathBuf, max: u64, keep: u16) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let lock =
            private_file(&Self::sibling(&path, "lock")).context("cannot open log lock file")?;
        fs2::FileExt::try_lock_exclusive(&lock).context(
            "log file is already used by another Werk process; choose a different --log-file",
        )?;
        let file = private_file(&path)
            .with_context(|| format!("cannot open log file {}", path.display()))?;
        let bytes = file.metadata()?.len();
        Ok(Self {
            path,
            file: Some(file),
            _lock: lock,
            bytes,
            max,
            keep,
        })
    }
    fn write(&mut self, line: &[u8]) -> io::Result<()> {
        if self.bytes > 0 && self.bytes.saturating_add(line.len() as u64) > self.max {
            if let Some(mut file) = self.file.take() {
                file.flush()?;
            }
            let oldest = Self::sibling(&self.path, self.keep);
            if oldest.exists() {
                std::fs::remove_file(oldest)?;
            }
            for index in (1..self.keep).rev() {
                let from = Self::sibling(&self.path, index);
                if from.exists() {
                    std::fs::rename(from, Self::sibling(&self.path, index + 1))?;
                }
            }
            if self.path.exists() {
                std::fs::rename(&self.path, Self::sibling(&self.path, 1))?;
            }
            self.bytes = 0;
        }
        if self.file.is_none() {
            self.file = Some(private_file(&self.path)?);
        }
        self.file.as_mut().unwrap().write_all(line)?;
        self.bytes += line.len() as u64;
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = &mut self.file {
            file.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests;

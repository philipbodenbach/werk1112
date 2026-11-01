//! CLI liveness for work without native progress, with byte counts where known.
//! Library calls remain silent; redirected/JSON logs get periodic status events.
use super::{Stream, clean, interactive};
use crate::logging::{self, Level};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;
use std::{
    sync::{Arc, Mutex, atomic::Ordering, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Clone)]
struct Status {
    label: String,
    phase: String,
    bytes: Option<(u64, u64)>,
    started: Instant,
}

impl Status {
    fn emit(&self, event: &str, outcome: &str) {
        let elapsed = self.started.elapsed().as_secs_f64();
        let transfer = self.bytes.map_or(String::new(), |(position, total)| {
            format!("; {position}/{total} bytes")
        });
        let message = format!(
            "{}: {}{transfer} ({elapsed:.1}s; {outcome})",
            self.label, self.phase
        );
        let level = if outcome == "incomplete" {
            Level::Warn
        } else {
            Level::Info
        };
        if logging::enabled() {
            logging::emit(
                level,
                event,
                &message,
                json!({
                    "operation": self.label, "phase": self.phase,
                    "duration_seconds": elapsed, "status": outcome,
                    "bytes": self.bytes.map(|b| b.0), "total_bytes": self.bytes.map(|b| b.1),
                }),
            );
        } else {
            super::line(Stream::Err, format_args!("{message}"));
        }
    }
}

struct Running {
    state: Arc<Mutex<Status>>,
    bar: Option<ProgressBar>,
    stop: mpsc::Sender<()>,
    heartbeat: Option<JoinHandle<()>>,
    completed: bool,
}

pub(crate) struct Activity(Option<Running>);

fn style(download: bool) -> ProgressStyle {
    ProgressStyle::with_template(if download {
        "{spinner} {msg} [{elapsed_precise}] {bytes}/{total_bytes} {bytes_per_sec} ETA {eta}"
    } else {
        "{spinner} {msg} [{elapsed_precise}]"
    })
    .expect("static activity template")
    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
}

fn label(text: &str) -> String {
    clean(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(240)
        .collect()
}

impl Activity {
    pub(crate) fn start(message: impl AsRef<str>) -> Self {
        if !super::ENABLED.load(Ordering::Relaxed) || !logging::enabled_for(Level::Info) {
            return Self(None);
        }
        let status = Status {
            label: label(message.as_ref()),
            phase: "Preparing".into(),
            bytes: None,
            started: Instant::now(),
        };
        status.emit("operation.started", "running");
        let bar = interactive(Stream::Err).then(|| {
            let bar = super::progress(ProgressBar::new_spinner());
            bar.set_style(style(false));
            bar.set_message(format!("{}: {}", status.label, status.phase));
            bar.enable_steady_tick(Duration::from_millis(110));
            bar
        });
        let state = Arc::new(Mutex::new(status));
        let (stop, receiver) = mpsc::channel();
        let heartbeat = {
            let state = state.clone();
            let context = logging::capture();
            Some(thread::spawn(move || {
                logging::with_context(context, || {
                    while receiver.recv_timeout(Duration::from_secs(5))
                        == Err(mpsc::RecvTimeoutError::Timeout)
                    {
                        let snapshot = state.lock().unwrap_or_else(|e| e.into_inner()).clone();
                        snapshot.emit("operation.progress", "running");
                    }
                })
            }))
        };
        Self(Some(Running {
            state,
            bar,
            stop,
            heartbeat,
            completed: false,
        }))
    }

    pub(crate) fn phase(&self, phase: impl AsRef<str>) {
        let Some(running) = &self.0 else { return };
        let (previous, snapshot) = {
            let mut status = running.state.lock().unwrap_or_else(|e| e.into_inner());
            let phase = label(phase.as_ref());
            if status.phase == phase {
                return;
            }
            let previous = status.bytes.map(|_| status.clone());
            status.phase = phase;
            status.bytes = None;
            (previous, status.clone())
        };
        if let Some(previous) = previous {
            previous.emit("operation.progress", "running");
        }
        if let Some(bar) = &running.bar {
            bar.set_style(style(false));
            bar.set_message(format!("{}: {}", snapshot.label, snapshot.phase));
        }
        snapshot.emit("operation.phase", "running");
    }

    pub(crate) fn download(&self, total: u64) {
        self.phase("Downloading");
        let Some(running) = &self.0 else { return };
        running
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bytes = Some((0, total));
        if let Some(bar) = &running.bar {
            bar.set_length(total);
            bar.set_position(0);
            bar.reset_eta();
            bar.set_style(style(true));
        }
    }

    pub(crate) fn position(&self, bytes: u64) {
        let Some(running) = &self.0 else { return };
        if let Some((position, _)) = &mut running
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bytes
        {
            *position = bytes;
        }
        if let Some(bar) = &running.bar {
            bar.set_position(bytes);
        }
    }

    pub(crate) fn finish(mut self) {
        if let Some(running) = &mut self.0 {
            running.completed = true;
        }
    }
}

impl Drop for Activity {
    fn drop(&mut self) {
        let Some(mut running) = self.0.take() else {
            return;
        };
        let _ = running.stop.send(());
        if let Some(thread) = running.heartbeat.take() {
            let _ = thread.join();
        }
        if let Some(bar) = running.bar {
            bar.finish_and_clear();
        }
        let snapshot = running
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        snapshot.emit(
            "operation.finished",
            if running.completed {
                "completed"
            } else {
                "incomplete"
            },
        );
    }
}

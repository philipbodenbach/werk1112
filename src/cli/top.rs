//! Read-only terminal client. Rendering never waits for network I/O.
mod ui;
use crate::observability::{Rates, Snapshot};
use anyhow::{Context, Result, bail};
use clap::Args;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{
    collections::VecDeque,
    io::{self, IsTerminal, Read},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Args)]
pub struct TopArgs {
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub url: String,
    #[arg(long, env = "WERK_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    #[arg(long, default_value_t=2000, value_parser=clap::value_parser!(u64).range(500..=60000))]
    pub interval_ms: u64,
    /// Print one snapshot and exit, without entering terminal mode.
    #[arg(long)]
    pub once: bool,
    #[arg(long, requires = "once")]
    pub json: bool,
    #[arg(long)]
    pub no_animation: bool,
    /// Preview the dashboard with explicitly simulated data; no server connection.
    #[arg(long, conflicts_with_all=["once","json","api_key"])]
    pub demo: bool,
}

struct Client {
    http: reqwest::blocking::Client,
    url: reqwest::Url,
    key: Option<String>,
}
impl Client {
    fn new(args: &TopArgs) -> Result<Self> {
        let mut url = reqwest::Url::parse(&args.url).context("invalid Werk URL")?;
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            bail!("use an HTTP(S) server URL without embedded credentials, query or fragment");
        }
        if url.path() != "/" && !url.path().is_empty() {
            bail!("use the Werk server root URL, without /v1");
        }
        let local = matches!(
            url.host_str(),
            Some("127.0.0.1" | "localhost" | "[::1]" | "::1")
        );
        let key = args.api_key.clone().or_else(|| {
            if !local {
                return None;
            }
            crate::api_keys::default_api_keys_path()
                .ok()
                .filter(|p| p.exists())
                .and_then(|p| crate::api_keys::load_api_keys_file(&p).ok())
                .and_then(|keys| keys.into_iter().next().map(|k| k.key))
        });
        url.set_path("/werk/v1/observability");
        Ok(Self {
            http: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(5))
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
                .build()?,
            url,
            key,
        })
    }
    fn snapshot(&self) -> Result<Snapshot> {
        let mut request = self
            .http
            .get(self.url.clone())
            .header("x-werk-protocol-version", "1.0");
        if let Some(key) = &self.key {
            request = request.bearer_auth(key);
        }
        let response = request.send().map_err(|_| {
            anyhow::anyhow!("Werk unavailable: check server address and connection")
        })?;
        match response.status().as_u16() {
            401 | 403 => bail!("authentication failed: supply WERK_API_KEY"),
            404 => {
                bail!("this server has no observability endpoint; it needs the updated Werk build")
            }
            200 => {}
            code => bail!("Werk returned HTTP {code}"),
        }
        let mut body = Vec::new();
        response.take(2 * 1024 * 1024 + 1).read_to_end(&mut body)?;
        if body.len() > 2 * 1024 * 1024 {
            bail!("observability response exceeds 2 MiB");
        }
        let value: serde_json::Value =
            serde_json::from_slice(&body).context("invalid observability JSON")?;
        let version: crate::werk_protocol::ProtocolVersion =
            serde_json::from_value(value["protocol"].clone())
                .context("missing Werk protocol version")?;
        if !crate::werk_protocol::ProtocolVersion::V1.accepts(version) {
            bail!("unsupported Werk protocol version");
        }
        let snapshot: Snapshot = serde_json::from_value(
            value
                .get("data")
                .cloned()
                .context("missing protocol data")?,
        )
        .context("invalid observability snapshot")?;
        if snapshot.schema_version != 1 {
            bail!("unsupported observability schema");
        }
        Ok(snapshot)
    }
}

struct App {
    snapshot: Option<Snapshot>,
    previous: Option<Snapshot>,
    rates: Rates,
    history: VecDeque<Option<f64>>,
    hit_history: VecDeque<Option<f64>>,
    read_history: VecDeque<Option<f64>>,
    error: Option<String>,
    last_received: Option<Instant>,
    paused: bool,
    selected: usize,
    worker: usize,
    details: bool,
    demo: bool,
    animation: bool,
    tick: u64,
    target: String,
}
impl App {
    fn new(args: &TopArgs) -> Self {
        Self {
            snapshot: None,
            previous: None,
            rates: Rates::default(),
            history: VecDeque::new(),
            hit_history: VecDeque::new(),
            read_history: VecDeque::new(),
            error: None,
            last_received: None,
            paused: false,
            selected: 0,
            worker: 0,
            details: false,
            demo: args.demo,
            animation: !args.no_animation,
            tick: 0,
            target: args.url.clone(),
        }
    }
    fn update(&mut self, snapshot: Snapshot) {
        self.rates = Rates::default();
        if let Some(old) = &self.snapshot {
            if old.server_started_ms == snapshot.server_started_ms {
                if let (Some(a), Some(b)) = (
                    old.backends.get(self.worker),
                    snapshot.backends.get(self.worker),
                ) {
                    let elapsed = snapshot
                        .backend_observed_at_ms
                        .unwrap_or(snapshot.observed_at_ms)
                        .saturating_sub(old.backend_observed_at_ms.unwrap_or(old.observed_at_ms))
                        as f64
                        / 1000.;
                    self.rates = Rates::between(a, b, elapsed);
                }
            } else {
                self.history.clear();
                self.hit_history.clear();
                self.read_history.clear();
            }
        }
        if !self.demo {
            if let Some(b) = snapshot.backends.get(self.worker) {
                self.rates = Rates {
                    decode_estimate: b.gauges.get("decode_tokens_per_second_estimate").copied(),
                    expert_hit_ratio: b.gauges.get("expert_cache_hit_ratio").copied(),
                    read_bytes_per_second: b.gauges.get("expert_read_bytes_per_second").copied(),
                };
            }
        }
        for (history, value) in [
            (&mut self.history, self.rates.decode_estimate),
            (
                &mut self.hit_history,
                self.rates.expert_hit_ratio.map(|v| v * 100.),
            ),
            (
                &mut self.read_history,
                self.rates.read_bytes_per_second.map(|v| v / 1048576.),
            ),
        ] {
            if history.len() == 90 {
                history.pop_front();
            }
            history.push_back(value);
        }
        self.previous = self.snapshot.replace(snapshot);
        self.last_received = Some(Instant::now());
        self.error = None;
    }
}

struct TerminalGuard;
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
    }
}

pub fn run(args: TopArgs) -> Result<()> {
    if args.once {
        let snapshot = Client::new(&args)?.snapshot()?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&snapshot)?);
        } else {
            println!(
                "WERK TOP  up {:.0}s  active {}  completed {}  errors {}\n{}",
                snapshot.uptime_seconds,
                snapshot.totals.active,
                snapshot.totals.completed,
                snapshot.totals.errors,
                snapshot
                    .requests
                    .iter()
                    .take(12)
                    .map(|r| format!(
                        "{}  {}  {:.1}s  output {}",
                        clean(&r.model),
                        r.state,
                        r.elapsed_seconds,
                        r.output_tokens
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| "n/a".into())
                    ))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
        return Ok(());
    }
    if !io::stdout().is_terminal() || !io::stdin().is_terminal() {
        bail!("werk top needs an interactive terminal; use --once --json for scripts");
    }
    let client = if args.demo {
        None
    } else {
        Some(Client::new(&args)?)
    };
    let latest = Arc::new(Mutex::new(None::<Result<Snapshot, String>>));
    let stop = Arc::new(AtomicBool::new(false));
    let worker = client.map(|client| {
        let latest = latest.clone();
        let stop = stop.clone();
        let interval = Duration::from_millis(args.interval_ms);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let result = client.snapshot().map_err(|e| e.to_string());
                *latest.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
                std::thread::park_timeout(interval);
            }
        })
    });
    // Restore terminal state even if a widget panics. The hook contains no secrets.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
        previous_hook(info);
    }));
    let result = (|| -> Result<()> {
        enable_raw_mode()?;
        let _guard = TerminalGuard;
        execute!(io::stdout(), EnterAlternateScreen, crossterm::cursor::Hide)?;
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        let mut app = App::new(&args);
        let started = Instant::now();
        let mut demo_sample = Instant::now() - Duration::from_secs(3);
        loop {
            if !app.paused {
                if args.demo && demo_sample.elapsed() >= Duration::from_millis(args.interval_ms) {
                    app.update(demo(started.elapsed().as_secs_f64()));
                    demo_sample = Instant::now();
                }
                if let Some(result) = latest.lock().unwrap_or_else(|e| e.into_inner()).take() {
                    match result {
                        Ok(snapshot) => app.update(snapshot),
                        Err(error) => {
                            app.error = Some(error);
                            app.rates = Rates::default();
                        }
                    }
                }
            }
            terminal.draw(|frame| ui::draw(frame, &app))?;
            if event::poll(Duration::from_millis(if app.animation { 100 } else { 250 }))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            break;
                        }
                        KeyCode::Char(' ') => app.paused = !app.paused,
                        KeyCode::Char('a') => app.animation = !app.animation,
                        KeyCode::Char('b') => {
                            app.worker = (app.worker + 1)
                                % app.snapshot.as_ref().map_or(1, |s| s.backends.len().max(1));
                            app.history.clear();
                            app.hit_history.clear();
                            app.read_history.clear();
                            app.rates = Rates::default();
                        }
                        KeyCode::Tab | KeyCode::Enter => app.details = !app.details,
                        KeyCode::Up => app.selected = app.selected.saturating_sub(1),
                        KeyCode::Down => {
                            app.selected = (app.selected + 1).min(
                                app.snapshot
                                    .as_ref()
                                    .map_or(0, |s| s.requests.len().saturating_sub(1)),
                            )
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            if app.animation && !app.paused {
                app.tick = app.tick.wrapping_add(1);
            }
        }
        Ok(())
    })();
    stop.store(true, Ordering::Relaxed);
    if let Some(worker) = worker {
        worker.thread().unpark();
        let _ = worker.join();
    }
    result
}
fn clean(value: &str) -> String {
    value
        .chars()
        .filter(|c| !c.is_control())
        .take(256)
        .collect()
}

fn demo(seconds: f64) -> Snapshot {
    use crate::observability::{BackendSnapshot, RequestSnapshot, Totals};
    let started = 1_000_000;
    let time = started + (seconds * 1000.) as u64;
    let mut b = BackendSnapshot {
        backend: "omlx".into(),
        instance: "demo".into(),
        model: "Qwen Flash / simulated".into(),
        available: true,
        ..Default::default()
    };
    for (k, v) in [
        ("requests_completed_total", 18),
        ("expert_cache_hits_total", (seconds * 950.) as u64),
        ("expert_cache_misses_total", (seconds * 50.) as u64),
        (
            "expert_read_bytes_total",
            (seconds * 180. * 1048576.) as u64,
        ),
    ] {
        b.counters.insert(k.into(), v);
    }
    for (k, v) in [
        ("requests_active", 1.),
        ("requests_waiting", 0.),
        (
            "decode_context_tokens",
            13703. + seconds * 7. + 2. * seconds.sin(),
        ),
        ("expert_cache_resident_bytes", 21.4 * 1073741824.),
        ("expert_cache_budget_bytes", 22. * 1073741824.),
        ("expert_cache_effective_budget_bytes", 22. * 1073741824.),
        ("ngram_cache_resident_bytes", 52. * 1048576.),
    ] {
        b.gauges.insert(k.into(), v);
    }
    Snapshot {
        host_swap_used_bytes: Some(1610612736),
        schema_version: 1,
        observed_at_ms: time,
        server_started_ms: started,
        uptime_seconds: seconds + 3200.,
        totals: Totals {
            active: 1,
            completed: 18,
            started: 19,
            prompt_tokens: 91402,
            output_tokens: 2811,
            cached_tokens: 53248,
            ..Default::default()
        },
        requests: vec![
            RequestSnapshot {
                id: 19,
                model: b.model.clone(),
                state: "streaming".into(),
                started_ms: started,
                elapsed_seconds: seconds + 42.,
                ..Default::default()
            },
            RequestSnapshot {
                id: 18,
                model: b.model.clone(),
                state: "tool_calls".into(),
                prompt_tokens: Some(7144),
                cached_tokens: Some(2048),
                output_tokens: Some(55),
                elapsed_seconds: 119.5,
                decode_tokens_per_second: Some(5.9),
                ..Default::default()
            },
        ],
        backends: vec![b],
        memory: None,
        backend_observed_at_ms: Some(time),
    }
}

#[cfg(test)]
mod observability_tests {
    use super::*;
    use std::{io::Write, net::TcpListener};

    fn args(url: String) -> TopArgs {
        TopArgs {
            url,
            api_key: Some("fixture-key".into()),
            interval_ms: 2000,
            once: true,
            json: true,
            no_animation: true,
            demo: false,
        }
    }
    #[test]
    fn client_rejects_embedded_credentials_and_api_paths() {
        for url in [
            "http://user:secret@localhost:11434",
            "file:///tmp/anything",
            "http://localhost:11434/v1",
            "http://localhost:11434?api_key=secret",
        ] {
            assert!(Client::new(&args(url.into())).is_err());
        }
    }
    #[test]
    fn client_roundtrip_uses_werk_protocol_and_bearer_auth() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut bytes = [0; 8192];
            let mut request = Vec::new();
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = socket.read(&mut bytes).unwrap();
                assert!(n > 0);
                request.extend_from_slice(&bytes[..n]);
            }
            let request = String::from_utf8(request).unwrap().to_lowercase();
            assert!(request.starts_with("get /werk/v1/observability "));
            assert!(request.contains("authorization: bearer fixture-key"));
            assert!(request.contains("x-werk-protocol-version: 1.0"));
            let body = serde_json::to_string(&crate::werk_protocol::ProtocolEnvelope::v1(
                "fixture",
                demo(1.),
            ))
            .unwrap();
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
        });
        let snapshot = Client::new(&args(url)).unwrap().snapshot().unwrap();
        assert_eq!(snapshot.schema_version, 1);
        assert_eq!(snapshot.totals.completed, 18);
        thread.join().unwrap();
    }
    #[test]
    fn history_stays_bounded_and_resets_on_server_restart() {
        let args = args("http://localhost:11434".into());
        let mut app = App::new(&args);
        for n in 0..300 {
            app.update(demo(n as f64));
        }
        assert_eq!(app.history.len(), 90);
        let mut restarted = demo(1.);
        restarted.server_started_ms += 1;
        app.update(restarted);
        assert_eq!(app.history.len(), 1);
    }
}

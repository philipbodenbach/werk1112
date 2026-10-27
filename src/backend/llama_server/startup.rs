//! Cold-start limits are separate from generation timeouts. Health probes must
//! themselves be bounded, including a native server that accepts but never replies.
use anyhow::{Context, Result, anyhow, ensure};
use std::{
    env,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    time::{Duration, Instant},
};

const TIMEOUT_ENV: &str = "WERK_LLAMA_HEALTH_TIMEOUT_SECONDS";
const DEFAULT_TIMEOUT_SECONDS: u64 = 900;

pub(super) fn timeout() -> Result<Duration> {
    let value = match env::var(TIMEOUT_ENV) {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(_) => return Err(anyhow!("{TIMEOUT_ENV} must be a positive integer")),
    };
    parse_timeout(value.as_deref())
}

fn parse_timeout(value: Option<&str>) -> Result<Duration> {
    let seconds = value.map_or(Ok(DEFAULT_TIMEOUT_SECONDS), |value| {
        value
            .trim()
            .parse::<u64>()
            .with_context(|| format!("{TIMEOUT_ENV} must be a positive integer (seconds)"))
    })?;
    ensure!(seconds > 0, "{TIMEOUT_ENV} must be greater than zero");
    let duration = Duration::from_secs(seconds);
    ensure!(
        Instant::now().checked_add(duration).is_some(),
        "{TIMEOUT_ENV} is too large"
    );
    Ok(duration)
}

pub(super) fn healthy(url: &str, budget: Duration) -> Result<bool> {
    let started = Instant::now();
    let remaining = || {
        budget
            .checked_sub(started.elapsed())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| anyhow!("llama-server health probe timed out"))
    };
    let (_, host, port) = super::parse_local_url(url)?;
    // Startup owns this numeric loopback address; avoid an unbounded DNS lookup.
    let address = SocketAddr::new(host.parse()?, port);
    let mut socket = TcpStream::connect_timeout(&address, remaining()?)?;
    socket.set_write_timeout(Some(remaining()?))?;
    socket.write_all(
        format!("GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n")
            .as_bytes(),
    )?;
    // Read only a bounded status line. Loading responses are normally 503; their
    // body is not needed and must not postpone the overall startup deadline.
    let mut line = Vec::new();
    let mut buffer = [0u8; 1024];
    loop {
        socket.set_read_timeout(Some(remaining()?))?;
        let n = socket.read(&mut buffer)?;
        ensure!(
            n > 0,
            "llama-server closed its health response before the status line"
        );
        let end = buffer[..n].iter().position(|b| *b == b'\n');
        line.extend_from_slice(&buffer[..end.unwrap_or(n)]);
        ensure!(
            line.len() <= 1024,
            "llama-server health status line is too long"
        );
        if end.is_some() {
            let mut fields = std::str::from_utf8(&line)?.split_ascii_whitespace();
            ensure!(
                matches!(fields.next(), Some("HTTP/1.0" | "HTTP/1.1")),
                "invalid llama-server health response"
            );
            let code = fields
                .next()
                .context("missing health status")?
                .parse::<u16>()?;
            return Ok(code == 200);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{net::TcpListener, sync::mpsc, thread};

    #[test]
    fn startup_budget_allows_large_models_and_validates_overrides() {
        assert_eq!(parse_timeout(None).unwrap().as_secs(), 900);
        assert_eq!(parse_timeout(Some(" 1200 ")).unwrap().as_secs(), 1200);
        for value in ["", "0", "-1", "1.5", "forever", "18446744073709551615"] {
            assert!(parse_timeout(Some(value)).is_err(), "{value}");
        }
    }

    #[test]
    fn silent_health_endpoint_cannot_block_startup_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (done, wait) = mpsc::channel();
        let server = thread::spawn(move || {
            let (_socket, _) = listener.accept().unwrap();
            let _ = wait.recv_timeout(Duration::from_secs(2));
        });
        let start = Instant::now();
        assert!(healthy(&url, Duration::from_millis(100)).is_err());
        assert!(start.elapsed() < Duration::from_secs(1));
        let _ = done.send(());
        server.join().unwrap();
    }

    #[test]
    fn readiness_needs_200_and_does_not_wait_for_response_bodies() {
        for (status, expected) in [(200, true), (503, false), (401, false)] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let (done, wait) = mpsc::channel();
            let server = thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                // Deliberately leave the headers/body and connection unfinished.
                write!(socket, "HTTP/1.1 {status} test\r\n").unwrap();
                let _ = wait.recv_timeout(Duration::from_secs(2));
            });
            assert_eq!(healthy(&url, Duration::from_millis(250)).unwrap(), expected);
            let _ = done.send(());
            server.join().unwrap();
        }
    }

    #[test]
    fn slow_status_bytes_do_not_renew_the_probe_budget() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            for byte in b"HTTP/1.1 200 OK\r\n" {
                if socket.write_all(&[*byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(30));
            }
        });
        let start = Instant::now();
        assert!(healthy(&url, Duration::from_millis(100)).is_err());
        assert!(start.elapsed() < Duration::from_millis(400));
        server.join().unwrap();
    }
}

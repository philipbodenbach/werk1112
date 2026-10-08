use super::*;

pub(crate) fn fixture(path: PathBuf, level: Level) -> Context {
    Context {
        request_id: None,
        logger: Some(
            Logger::new(
                &LogArgs {
                    log_file: Some(path),
                    log_level: Some(level),
                    ..LogArgs::default()
                },
                false,
                false,
                true,
                false,
            )
            .unwrap(),
        ),
    }
}
pub(crate) fn records(context: &Context, path: &std::path::Path) -> Vec<Value> {
    context.logger.as_ref().unwrap().flush();
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn structured_records_filter_levels_redact_and_preserve_scalar_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    let context = fixture(path.clone(), Level::Info);
    register_secrets(["logging-test-credential-never-persist".into()]);
    emit_in(&context, Level::Debug, "hidden", "hidden", json!({}));
    emit_in(&context, Level::Off, "hidden", "hidden", json!({}));
    emit_in(
        &context,
        Level::Info,
        "test.event",
        "line one\nline two\x1b[31m logging-test-credential-never-persist",
        json!({
            "prompt":"private", "nested":{"authorization":"Bearer secret", "input_tokens":47}, "duration_seconds":0.058
        }),
    );
    let events = records(&context, &path);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["level"], "info");
    assert_eq!(events[0]["schema_version"], 1);
    assert_eq!(events[0]["fields"]["prompt"], "[REDACTED]");
    assert_eq!(events[0]["fields"]["nested"]["authorization"], "[REDACTED]");
    assert_eq!(events[0]["fields"]["nested"]["input_tokens"], 47);
    assert_eq!(events[0]["fields"]["duration_seconds"], 0.058);
    let bytes = std::fs::read_to_string(path).unwrap();
    assert!(!bytes.contains("logging-test-credential-never-persist"));
    assert!(!bytes.contains("private"));
    assert!(!bytes.contains("\\u001b"));
    assert_eq!(bytes.lines().count(), 1);
}

#[test]
fn rotation_retains_complete_records_appends_and_excludes_second_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    let mut file = RotatingFile::open(path.clone(), 12, 2).unwrap();
    assert!(RotatingFile::open(path.clone(), 12, 2).is_err());
    for i in 0..4 {
        file.write(format!("{{\"n\":{i}}}\n").as_bytes()).unwrap();
    }
    file.flush().unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"n\":3}\n");
    assert_eq!(
        std::fs::read_to_string(RotatingFile::sibling(&path, 1)).unwrap(),
        "{\"n\":2}\n"
    );
    assert_eq!(
        std::fs::read_to_string(RotatingFile::sibling(&path, 2)).unwrap(),
        "{\"n\":1}\n"
    );
    assert!(!RotatingFile::sibling(&path, 3).exists());
    drop(file);
    let mut file = RotatingFile::open(path.clone(), 100, 2).unwrap();
    file.write(b"{\"n\":4}\n").unwrap();
    file.flush().unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 2);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[test]
fn backpressure_is_bounded_and_counted_without_blocking() {
    let (tx, _rx) = mpsc::sync_channel(1);
    let logger = Arc::new(Logger {
        level: Level::Info,
        format: Format::Json,
        server: true,
        console: false,
        file: Some(tx),
        dropped: Arc::new(AtomicU64::new(0)),
        failures: Arc::new(AtomicU64::new(0)),
    });
    let context = Context {
        request_id: None,
        logger: Some(logger.clone()),
    };
    for _ in 0..3 {
        emit_in(&context, Level::Info, "test", "test", json!({}));
    }
    assert_eq!(logger.dropped.load(Ordering::Relaxed), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn correlation_survives_async_and_blocking_workers_without_leaking() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    let root = fixture(path.clone(), Level::Debug);
    let mut handles = vec![];
    for n in 0..12 {
        let mut context = root.clone();
        context.request_id = Some(format!("req_test_{n}"));
        handles.push(tokio::spawn(scope(context, async move {
            spawn(async move {
                spawn_blocking(move || {
                    emit(Level::Info, "worker", "worker", json!({"worker":n}));
                    let telemetry = Arc::new(crate::observability::Telemetry::default());
                    let _guard = telemetry.begin("model-fixture"); // drop records cancellation
                })
                .await
                .unwrap();
            })
            .await
            .unwrap();
        })));
    }
    for handle in handles {
        handle.await.unwrap();
    }
    assert!(capture().request_id.is_none());
    let events = records(&root, &path);
    assert_eq!(events.len(), 36);
    for event in events.iter().filter(|e| e["event"] == "worker") {
        assert_eq!(
            event["request_id"],
            format!("req_test_{}", event["fields"]["worker"])
        );
    }
    assert_eq!(
        events
            .iter()
            .filter(|e| e["event"] == "inference.cancelled" && e["level"] == "warn")
            .count(),
        12
    );
}

#[test]
fn pure_alias_conflicts_and_explicit_level_precedence() {
    use clap::Parser;
    for flag in ["--verbose-pure", "--verbose-lite"] {
        let parsed =
            crate::cli::Cli::try_parse_from(["werk", "serve", flag, "--log-level", "warn"])
                .unwrap();
        assert!(matches!(
            parsed.command,
            Some(crate::cli::Commands::Serve {
                verbose_pure: true,
                ..
            })
        ));
        let logger = Logger::new(&parsed.logging, false, true, true, false).unwrap();
        assert_eq!(logger.level, Level::Warn);
        assert_eq!(logger.format, Format::Json);
        assert!(!logger.enabled(Level::Info));
        assert!(crate::cli::Cli::try_parse_from(["werk", "serve", flag, "--verbose"]).is_err());
    }
    let off = Logger::new(
        &LogArgs {
            log_level: Some(Level::Off),
            ..LogArgs::default()
        },
        true,
        true,
        true,
        false,
    )
    .unwrap();
    assert!(!off.enabled(Level::Error));
}

#[test]
fn file_io_failure_is_reported_and_counted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    let logger = Logger::new(
        &LogArgs {
            log_file: Some(path.clone()),
            log_max_size_mb: 1,
            log_retention: 1,
            ..LogArgs::default()
        },
        false,
        false,
        true,
        false,
    )
    .unwrap();
    let context = Context {
        request_id: None,
        logger: Some(logger.clone()),
    };
    // A directory at the rotation destination produces a deterministic I/O error,
    // even when tests run as root. Ordinary records fit before rotation is due.
    std::fs::create_dir(RotatingFile::sibling(&path, 1)).unwrap();
    let message = "x".repeat(12_000);
    for _ in 0..100 {
        emit_in(&context, Level::Info, "test", &message, json!({}));
    }
    logger.flush();
    assert!(logger.failures.load(Ordering::Relaxed) > 0);
    assert_eq!(logger.dropped.load(Ordering::Relaxed), 0);
    assert!(!std::fs::read_to_string(&path).unwrap().is_empty());
}

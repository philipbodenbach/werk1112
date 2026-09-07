use super::*;

struct Fixture {
    root: PathBuf,
    store: ModelStore,
    model: PathBuf,
    invocation: OmlxInvocation,
}

impl Fixture {
    fn new(settings: Value) -> Self {
        let root = env::temp_dir().join(format!("werk-omlx-test-{}", random_id().unwrap()));
        let store = ModelStore::resolve(Some(root.clone())).unwrap();
        let model = root.join("models").join("owner_model").join("files");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), "{}").unwrap();
        fs::write(model.join("fixture.json"), settings.to_string()).unwrap();
        let launcher = root.join("omlx");
        let python = absolute_program(PathBuf::from("python3")).unwrap();
        fs::write(
            &launcher,
            format!("#!{}\n{}", python.display(), MOCK_SERVER),
        )
        .unwrap();
        let invocation = OmlxInvocation::from_launcher(launcher, Duration::from_secs(3)).unwrap();
        Self {
            root,
            store,
            model: model.canonicalize().unwrap(),
            invocation,
        }
    }
    fn start(&self) -> Result<OmlxProcess> {
        OmlxProcess::start(
            &self.store,
            &self.invocation,
            &self.model,
            ProbeReport {
                detail: "fixture".into(),
                version: "test-0.6.4".into(),
                tools: true,
                tool_calling_detail: None,
                runtime: json!({}),
            },
        )
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

const MOCK_SERVER: &str = r#"
import argparse, json, os, sys, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
p=argparse.ArgumentParser(); p.add_argument('serve'); p.add_argument('--model-dir'); p.add_argument('--base-path'); p.add_argument('--host'); p.add_argument('--port', type=int); p.add_argument('--api-key'); a=p.parse_args()
root=Path(a.model_dir); settings=json.loads((root/'fixture.json').read_text()); loaded=False
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self,value,status=200):
        raw=json.dumps(value).encode(); self.send_response(status); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path=='/health':
            if settings.get('stall_health'): time.sleep(2)
            return self.reply({'status':'healthy'})
        if self.headers.get('Authorization') != 'Bearer '+a.api_key: return self.reply({'error':'unauthorized'},401)
        if self.path=='/api/status': return self.reply({'version':settings.get('version','test-0.6.4')})
        if self.path=='/v1/models/status':
            entries=[{'id':'virtual-doc','model_type':'llm','model_path':'builtin://markitdown','loaded':True}, {'id':'physical / model','model_type':'llm','model_path':str(root),'loaded':loaded}]
            if settings.get('duplicate'): entries.append(dict(entries[-1],id='duplicate'))
            return self.reply({'models':entries})
        return self.reply({'error':'unknown path'},404)
    def do_POST(self):
        global loaded
        body=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))) or b'{}')
        if self.headers.get('Authorization') != 'Bearer '+a.api_key: return self.reply({'error':'unauthorized'},401)
        if self.path=='/v1/models/physical%20%2F%20model/load':
            if settings.get('load_fail'): return self.reply({'error':'model exceeds available memory'},507)
            loaded=True; return self.reply({'status':'ok','model_id':'physical / model'})
        if self.path != '/v1/chat/completions': return self.reply({'error':'unknown path'},404)
        (root/'request.json').write_text(json.dumps(body)); (root/'chat_started').write_text('yes')
        if settings.get('stall_headers'): time.sleep(2)
        if settings.get('chat_http_error'): return self.reply({'error':'quantization kernel failed'},500)
        if not body.get('stream'): return self.reply(settings.get('response',{'choices':[{'message':{'content':'answer'},'finish_reason':'stop'}], 'usage':{'prompt_tokens':17,'completion_tokens':4}}))
        self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.send_header('Transfer-Encoding','chunked'); self.end_headers()
        events=settings.get('events',['data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\r\n\r\n','data: [DONE]\r\n\r\n'])
        for event in events:
            raw=event.encode()
            for i in range(0,len(raw),7):
                part=raw[i:i+7]
                try: self.wfile.write(('%x\r\n'%len(part)).encode()+part+b'\r\n'); self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError): return
        self.wfile.write(b'0\r\n\r\n'); self.wfile.flush()
HTTPServer((a.host,a.port),Handler).serve_forever()
"#;

fn request() -> GenerateRequest {
    GenerateRequest {
        prompt: "hello".into(),
        messages: Vec::new(),
        image_urls: Vec::new(),
        max_tokens: 30,
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: vec!["end".into()],
        seed: Some(3),
        stream_granularity: super::super::StreamGranularity::Token,
        verbose: false,
        debug: false,
        tool_config: None,
    }
}

#[test]
fn encodes_served_model_id_as_one_path_segment() {
    assert_eq!(
        encode_path_segment("owner/model + 4"),
        "owner%2Fmodel%20%2B%204"
    );
}

#[test]
fn worker_loads_exact_physical_model_and_tears_down_owned_process() {
    let fixture = Fixture::new(json!({}));
    let server = fixture.start().unwrap();
    assert_eq!(server.model_name, "physical / model");
    assert!(server.is_running());
    let pid = server.child.lock().unwrap().id();
    let base_path = server.base_path.clone();
    let result = server.generate(&request(), None).unwrap();
    assert_eq!(result.text, "answer");
    assert_eq!(result.prompt_tokens, 17);
    assert_eq!(result.completion_tokens, 4);
    let sent: Value =
        serde_json::from_slice(&fs::read(fixture.model.join("request.json")).unwrap()).unwrap();
    assert_eq!(sent["model"], "physical / model");
    assert_eq!(sent["seed"], 3);
    assert_eq!(sent["stop"], json!(["end"]));
    drop(server);
    assert!(!base_path.exists());
    #[cfg(unix)]
    assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
}

#[test]
fn failed_load_preserves_http_cause_and_cleans_worker_directory() {
    let fixture = Fixture::new(json!({"load_fail":true}));
    let error = fixture.start().err().unwrap();
    let detail = format!("{error:#}");
    assert!(detail.contains("507"), "{detail}");
    assert!(
        detail.contains("model exceeds available memory"),
        "{detail}"
    );
    assert_eq!(
        fs::read_dir(fixture.root.join("backends/omlx/workers"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn startup_rejects_server_version_mismatch_and_duplicate_physical_ids() {
    for settings in [json!({"version":"other"}), json!({"duplicate":true})] {
        let fixture = Fixture::new(settings);
        let detail = format!("{:#}", fixture.start().err().unwrap());
        assert!(
            detail.contains("version does not match") || detail.contains("multiple IDs"),
            "{detail}"
        );
    }
}

#[test]
fn health_deadline_bounds_stalled_http_server() {
    let mut fixture = Fixture::new(json!({"stall_health":true}));
    fixture.invocation.health_timeout = Duration::from_millis(200);
    let started = Instant::now();
    let detail = format!("{:#}", fixture.start().err().unwrap());
    assert!(detail.contains("timed out"), "{detail}");
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn nonstream_preserves_nullable_tool_response_and_separates_reasoning() {
    let fixture = Fixture::new(
        json!({"response": {"choices":[{"message":{"content":null,"reasoning_content":"private","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"a\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":6,"completion_tokens":8}}}),
    );
    let server = fixture.start().unwrap();
    let result = server.generate(&request(), None).unwrap();
    assert_eq!(result.text, "");
    assert_eq!(result.finish_reason, "tool_calls");
    let assistant = result.assistant_message.unwrap();
    assert!(assistant.content.is_none());
    assert_eq!(
        assistant.tool_calls.unwrap()[0].function.arguments,
        "{\"a\":1}"
    );
    let fixture = Fixture::new(
        json!({"response":{"choices":[{"message":{"content":"","reasoning":"private"},"finish_reason":"length"}]}}),
    );
    let server = fixture.start().unwrap();
    assert!(
        format!("{:#}", server.generate(&request(), None).unwrap_err())
            .contains("hidden reasoning but no visible answer")
    );
}

#[tokio::test]
async fn stream_preserves_tool_fragments_and_usage_without_reasoning() {
    let events = vec![
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"private\"}}]}\r\n\r\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\"}}]}}]}\r\n\r\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"} \"}}]},\"finish_reason\":\"tool_calls\"}]}\r\n\r\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":11}}\r\n\r\n",
        "data: [DONE]\r\n\r\n",
    ];
    let fixture = Fixture::new(json!({"events":events}));
    let server = Arc::new(fixture.start().unwrap());
    let (tx, mut rx) = mpsc::channel(16);
    let handle = tokio::task::spawn_blocking(move || {
        let result = server.generate(&request(), Some(tx.clone()));
        send_stream_result(tx, result);
    });
    let mut deltas = Vec::new();
    let mut done = false;
    while let Some(event) = rx.recv().await {
        match event.unwrap() {
            GenerateStreamEvent::ToolCallDelta(calls) => deltas.extend(calls),
            GenerateStreamEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                assert_eq!(finish_reason, "tool_calls");
                assert_eq!(prompt_tokens, 9);
                assert_eq!(completion_tokens, 11);
                done = true;
            }
            other => panic!("unexpected event {other:?}"),
        }
    }
    handle.await.unwrap();
    assert!(done);
    assert_eq!(deltas.len(), 2);
    assert_eq!(
        deltas[0].function.as_ref().unwrap().arguments.as_deref(),
        Some("{")
    );
    assert_eq!(
        deltas[1].function.as_ref().unwrap().arguments.as_deref(),
        Some("} ")
    );
}

#[tokio::test]
async fn partial_stream_failures_never_emit_done() {
    for ending in [
        "event: error\ndata: {\"message\":\"load failed\"}\n\n",
        "data: {\"error\":\"kernel failed\"}\n\n",
        "",
    ] {
        let fixture = Fixture::new(
            json!({"events":["data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n", ending]}),
        );
        let server = Arc::new(fixture.start().unwrap());
        let (tx, mut rx) = mpsc::channel(16);
        let handle = tokio::task::spawn_blocking(move || {
            let result = server.generate(&request(), Some(tx.clone()));
            send_stream_result(tx, result);
        });
        assert!(
            matches!(rx.recv().await.unwrap().unwrap(),GenerateStreamEvent::TextChunk(text) if text=="partial")
        );
        assert!(rx.recv().await.unwrap().is_err());
        assert!(rx.recv().await.is_none());
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn disconnect_cancels_request_while_waiting_for_headers() {
    let fixture = Fixture::new(json!({"stall_headers":true}));
    let server = Arc::new(fixture.start().unwrap());
    let (tx, rx) = mpsc::channel(16);
    let handle = tokio::task::spawn_blocking(move || server.generate(&request(), Some(tx)));
    let started = Instant::now();
    while !fixture.model.join("chat_started").exists() {
        assert!(started.elapsed() < Duration::from_secs(1));
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    drop(rx);
    let result = tokio::time::timeout(Duration::from_millis(700), handle)
        .await
        .expect("cancelled HTTP request stayed blocked")
        .unwrap();
    assert!(result.is_err());
}

#[test]
fn malformed_model_status_and_virtual_models_cannot_select_a_model() {
    let fixture = Fixture::new(json!({}));
    for value in [
        json!({}),
        json!({"models":[{"id":"virtual","model_type":"llm","model_path":"builtin://markitdown"}]}),
        json!({"models":[{"id":"vision","model_type":"vlm","model_path":fixture.model}]}),
    ] {
        assert!(physical_model_name(&value, &fixture.model).is_err());
    }
}

#[test]
fn captured_environment_identity_is_opaque_and_command_uses_snapshot() {
    let fixture = Fixture::new(json!({}));
    let mut invocation = fixture.invocation.clone();
    invocation
        .environment
        .push(("PYTHONPATH".into(), "private-python-path".into()));
    let first = format!("{invocation:?}");
    let command = invocation.python_command();
    assert!(command.get_envs().any(|(key, value)| key == "PYTHONPATH"
        && value == Some(std::ffi::OsStr::new("private-python-path"))));
    invocation
        .environment
        .push(("PYTHONHOME".into(), "another-secret".into()));
    assert_ne!(first, format!("{invocation:?}"));
    assert!(!first.contains("private-python-path"));
    assert!(invocation.python.is_absolute());
}

fn fixture_manifest() -> ModelManifest {
    ModelManifest {
        id: "owner/model".into(),
        source: crate::model_store::ModelSource::LocalPath {
            path: "fixture".into(),
        },
        format: ModelFormat::Mlx,
        architecture: Some("deepseek_v4".into()),
        tokenizer_path: None,
        config_path: Some("files/config.json".into()),
        model_path: Some("files/model.safetensors".into()),
        backend: "omlx".into(),
        created_unix: 0,
        files: vec![],
        artifacts: vec![],
        metadata: Default::default(),
    }
}

fn fixture_backend(fixture: &Fixture) -> OmlxBackend {
    OmlxBackend {
        store: fixture.store.clone(),
        invocation: Ok(fixture.invocation.clone()),
        servers: Arc::new(Mutex::new(HashMap::new())),
        test_probe: Some(ProbeReport {
            detail: "fixture".into(),
            version: "test-0.6.4".into(),
            tools: true,
            tool_calling_detail: None,
            runtime: json!({"fixture":true}),
        }),
    }
}

#[test]
fn cache_reuses_worker_recreates_dead_child_and_never_reuses_changed_manifest() {
    let fixture = Fixture::new(json!({}));
    let backend = fixture_backend(&fixture);
    let manifest = fixture_manifest();
    // ModelStore uses its own sanitized ID convention.
    let actual = fixture.store.model_dir(&manifest.id).join("files");
    if actual != fixture.model {
        fs::create_dir_all(actual.parent().unwrap()).unwrap();
        fs::rename(&fixture.model, &actual).unwrap();
    }
    let (first, _) = backend.cached_server(&manifest).unwrap();
    let (second, reload) = backend.cached_server(&manifest).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(reload, 0.0);
    first.child.lock().unwrap().kill().unwrap();
    first.child.lock().unwrap().wait().unwrap();
    let (replacement, _) = backend.cached_server(&manifest).unwrap();
    assert!(!Arc::ptr_eq(&first, &replacement));
    assert!(replacement.is_running());
    let mut changed = manifest.clone();
    changed.created_unix = 1;
    let descriptor = backend
        .runtime_control_adapter_for(&changed)
        .unwrap()
        .descriptor();
    let residency = descriptor
        .capabilities
        .iter()
        .find(|cap| cap.id == crate::runtime_control::MODEL_RESIDENCY_CAPABILITY)
        .unwrap();
    assert_eq!(
        residency.status,
        crate::werk_protocol::CapabilityStatus::Unavailable
    );
    let (changed_server, _) = backend.cached_server(&changed).unwrap();
    assert!(!Arc::ptr_eq(&replacement, &changed_server));
    let descriptor = backend
        .runtime_control_adapter_for(&changed)
        .unwrap()
        .descriptor();
    assert_eq!(descriptor.instance_id, changed_server.instance_id);
}

#[test]
fn closing_parent_lifetime_pipe_stops_child_without_dropping_owner() {
    let fixture = Fixture::new(json!({}));
    let mut server = fixture.start().unwrap();
    let base = server.base_path.clone();
    drop(server.parent_pipe.take());
    let started = Instant::now();
    while server.is_running() {
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "worker ignored parent EOF"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert!(
        base.exists(),
        "test must observe stop before owner Drop cleanup"
    );
    drop(server);
    assert!(!base.exists());
}

#[test]
fn unsupported_tool_options_fail_before_any_worker_or_chat_starts() {
    let fixture = Fixture::new(json!({}));
    let backend = fixture_backend(&fixture);
    for config in [
        json!({"tool_choice":"required"}),
        json!({"tool_choice":{"type":"function","function":{"name":"lookup"}}}),
        json!({"parallel_tool_calls":false}),
        json!({"parallel_tool_calls":true}),
        json!({"tools":[{"type":"function","function":{"name":"lookup","strict":true}}]}),
    ] {
        let mut req = request();
        req.tool_config = Some(super::super::ToolCallingConfig {
            tools: config
                .get("tools")
                .map(|value| serde_json::from_value(value.clone()).unwrap()),
            tool_choice: config
                .get("tool_choice")
                .map(|value| serde_json::from_value(value.clone()).unwrap()),
            parallel_tool_calls: config.get("parallel_tool_calls").and_then(Value::as_bool),
        });
        assert!(
            backend
                .generate_inner(&fixture_manifest(), req, None)
                .is_err()
        );
        assert!(backend.servers.lock().unwrap().is_empty());
        assert!(!fixture.model.join("chat_started").exists());
    }
    for choice in ["auto", "none"] {
        let mut req = request();
        req.tool_config = Some(super::super::ToolCallingConfig {
            tools: None,
            tool_choice: Some(serde_json::from_value(json!(choice)).unwrap()),
            parallel_tool_calls: None,
        });
        validate_tool_options(&req).unwrap();
    }
}

#[test]
fn changing_selected_launcher_is_detected_before_execution() {
    let fixture = Fixture::new(json!({}));
    fs::write(
        &fixture.invocation.launcher,
        "#!/usr/bin/python3\nraise RuntimeError('changed')",
    )
    .unwrap();
    assert!(
        fixture
            .invocation
            .verify_launcher()
            .unwrap_err()
            .to_string()
            .contains("launcher changed")
    );
}

#[test]
fn console_bootstrap_excludes_caller_directory_before_stdlib_imports() {
    let fixture = Fixture::new(json!({}));
    fs::write(
        fixture.model.join("json.py"),
        "raise RuntimeError('caller model code executed')",
    )
    .unwrap();
    let mut command = fixture.invocation.python_command();
    let output = command
        .current_dir(&fixture.model)
        .args(["-c", &console_script_source("import json; print('safe')")])
        .arg(fixture.invocation.launcher.parent().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "safe");
}

#[test]
fn model_python_environment_is_rejected_before_probe_or_worker_startup() {
    let mut fixture = Fixture::new(json!({}));
    let marker = fixture.model.join("repository_code_executed");
    fs::write(
        fixture.model.join("sitecustomize.py"),
        format!(
            "open({:?}, 'w').write('executed')",
            marker.to_str().unwrap()
        ),
    )
    .unwrap();
    fixture.invocation.working_directory = fixture.model.clone();
    for value in [
        fixture.model.as_os_str(),
        std::ffi::OsStr::new("."),
        std::ffi::OsStr::new(":"),
    ] {
        fixture
            .invocation
            .environment
            .retain(|(key, _)| key != "PYTHONPATH");
        fixture
            .invocation
            .environment
            .push(("PYTHONPATH".into(), value.to_owned()));
        let error = fixture.invocation.probe(Some(&fixture.model)).unwrap_err();
        assert!(
            error.to_string().contains("include the model repository"),
            "{error:#}"
        );
        assert!(fixture.start().is_err());
        assert!(!marker.exists());
    }
    #[cfg(unix)]
    {
        let link = fixture.root.join("python-import-link");
        std::os::unix::fs::symlink(&fixture.model, &link).unwrap();
        fixture
            .invocation
            .environment
            .retain(|(key, _)| key != "PYTHONPATH");
        fixture
            .invocation
            .environment
            .push(("PYTHONPATH".into(), link.into_os_string()));
        assert!(
            fixture
                .invocation
                .verify_import_paths(&fixture.model)
                .is_err()
        );
    }
}

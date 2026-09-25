use super::*;

#[test]
fn text_worker_helpers_bootstrap_without_model_directory_imports() {
    let python = absolute_program(PathBuf::from("python3")).unwrap();
    for architecture in [
        None,
        Some("deepseek_v4"),
        Some("qwen4_exp"),
        Some("glm5_next"),
    ] {
        let source = "import json\nprint(json.dumps({name: name in sys.modules for name in ('_werk_omlx_decode', '_werk_omlx_glm_profile')}))\n";
        let mut command = Command::new(&python);
        command.arg("-I");
        let _script = attach_python_script(
            &mut command,
            &text_worker_script_source(source, architecture),
        )
        .unwrap();
        let output = command.arg("/unused").output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let modules: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            modules["_werk_omlx_decode"],
            matches!(architecture, Some("qwen4_exp" | "glm5_next"))
        );
        assert_eq!(
            modules["_werk_omlx_glm_profile"],
            architecture == Some("glm5_next")
        );
        if !matches!(architecture, Some("qwen4_exp" | "glm5_next")) {
            assert_eq!(
                text_worker_script_source(source, architecture),
                worker_script_source(source)
            );
        }
    }
}

#[test]
fn glm_profile_diagnostics_are_opt_in_bounded_and_allowlisted() {
    assert!(glm_profile_diagnostics(&json!({})).is_none());
    assert!(
        glm_profile_diagnostics(&json!({"glm_layer_profile":{"enabled":false,"layers":[]}}))
            .is_none()
    );
    let row = json!({"layer":3,"calls":2,"failures":0,"cache_hits":9,"cache_misses":7,
        "cache_evictions":1,"disk_bytes_read":1024,"wall_seconds":0.1,
        "forward_seconds":0.09,"routing_seconds":0.01,"disk_read_seconds":0.05,
        "api_key":"never print this"});
    let status = json!({"glm_layer_profile":{"enabled":true,"layers":[row.clone()]}});
    let line = glm_profile_diagnostics(&status).unwrap();
    assert!(line.contains("worker cumulative") && line.contains("\"layer\":3"));
    assert!(!line.contains("api_key") && !line.contains("never print this"));
    assert!(
        glm_profile_diagnostics(
            &json!({"glm_layer_profile":{"enabled":true,"layers":vec![row;1025]}})
        )
        .is_none()
    );
    let mut invalid = status;
    invalid["glm_layer_profile"]["layers"][0]["wall_seconds"] = json!(-1);
    assert!(glm_profile_diagnostics(&invalid).is_none());
}
#[test]
fn expert_cache_defaults_to_native_with_explicit_auto_small_and_large_overrides() {
    assert_eq!(expert_cache_bytes(None).unwrap(), None);
    assert_eq!(expert_cache_bytes(Some("auto".into())).unwrap(), Some(0));
    assert_eq!(
        expert_cache_bytes(Some("4194304".into())).unwrap(),
        Some(4 * 1024_u64.pow(4))
    );
    assert_eq!(expert_cache_bytes(Some("0".into())).unwrap(), None);
    assert_eq!(
        expert_cache_bytes(Some("8192".into())).unwrap(),
        Some(8 * 1024 * 1024 * 1024)
    );
    for value in ["", "-1", "1.5", "unlimited", "18446744073709551615"] {
        assert!(expert_cache_bytes(Some(value.into())).is_err());
    }
}

#[test]
fn thinking_override_preserves_defaults_and_rejects_invalid_values() {
    assert_eq!(thinking_enabled(None).unwrap(), None);
    assert_eq!(thinking_enabled(Some("0".into())).unwrap(), Some(false));
    assert_eq!(thinking_enabled(Some("1".into())).unwrap(), Some(true));
    for value in ["", "true", "false", "2", "-1", " 0", "1 "] {
        let error = thinking_enabled(Some(value.into())).unwrap_err();
        assert!(error.to_string().contains("WERK_OMLX_THINKING"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        assert!(thinking_enabled(Some(OsString::from_vec(vec![0xff]))).is_err());
    }
}

#[test]
fn expert_execution_mode_is_explicit_and_part_of_cache_identity() {
    assert_eq!(expert_execution(None).unwrap(), "grouped");
    assert_eq!(expert_execution(Some("serial".into())).unwrap(), "serial");
    for invalid in ["", "auto", "SERIAL", "grouped "] {
        assert!(expert_execution(Some(invalid.into())).is_err());
    }
    let fixture = Fixture::new(json!({}));
    let base = fixture_backend(&fixture);
    let mut other = base.clone();
    other.invocation.as_mut().unwrap().expert_execution =
        if base.invocation().unwrap().expert_execution == "grouped" {
            "serial"
        } else {
            "grouped"
        };
    assert_ne!(base.cache_identity(), other.cache_identity());
}

#[test]
fn expert_diagnostics_are_allowlisted_worker_intervals_and_reject_resets() {
    let before =
        json!({"cache_hits":10,"cache_misses":3,"disk_bytes_read":500,"disk_read_seconds":1.0});
    let after = json!({"cache_hits":14,"cache_misses":5,"disk_bytes_read":700,"disk_read_seconds":1.125,
        "resident_cache_bytes":100,"execution":"grouped","api_key":"never print this"});
    let line = expert_interval_diagnostics(&before, &after).unwrap();
    assert!(line.contains("worker interval"));
    assert!(line.contains("\"cache_hits\":4"));
    assert!(line.contains("\"disk_bytes_read\":200"));
    assert!(line.contains("\"disk_read_seconds\":0.125"));
    assert!(line.contains("\"resident_cache_bytes\":100"));
    assert!(!line.contains("api_key") && !line.contains("never print this"));
    assert!(expert_interval_diagnostics(&after, &before).is_none());
    assert!(expert_interval_diagnostics(&json!({}), &after).is_none());
}

#[test]
fn thinking_override_changes_only_explicit_omlx_template_kwargs() {
    let request = request();
    for stream in [false, true] {
        let baseline = chat_completion_body("model", &request, stream);
        assert_eq!(
            omlx_chat_completion_body("model", &request, stream, None, None),
            baseline
        );
        for thinking in [false, true] {
            let mut body =
                omlx_chat_completion_body("model", &request, stream, Some(thinking), None);
            assert_eq!(
                body.as_object_mut().unwrap().remove("chat_template_kwargs"),
                Some(json!({"enable_thinking": thinking}))
            );
            assert_eq!(body, baseline);
        }
    }
}

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
                runtime: json!({}),
                cache_paths: vec![],
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
p=argparse.ArgumentParser(); p.add_argument('serve'); p.add_argument('--model-dir'); p.add_argument('--base-path'); p.add_argument('--host'); p.add_argument('--port', type=int); p.add_argument('--api-key'); p.add_argument('--paged-ssd-cache-dir'); p.add_argument('--paged-ssd-cache-max-size'); a=p.parse_args()
root=Path(a.model_dir); settings=json.loads((root/'fixture.json').read_text()); loaded=False
with (root/'starts.jsonl').open('a') as log:
    log.write(json.dumps({'base':a.base_path,'cache':a.paged_ssd_cache_dir,'limit':a.paged_ssd_cache_max_size,'env_cache':os.environ.get('WERK_OMLX_PERSISTENCE_DIR'),'env_model':os.environ.get('WERK_OMLX_PERSISTENCE_MODEL_DIR')})+'\n')
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
        if self.path=='/werk/persistence/status': return self.reply({'installed':True,'active':settings.get('cache_active',True),'format':'omlx-exact-prefix-v1','model_id':'physical / model'})
        if self.path=='/werk/experts/status':
            if settings.get('native_experts'):
                return self.reply({'active':True,'cache_budget_bytes':0,'cache_budget_mode':'auto','experts_offloaded':False,'ngram_offload':'disabled','ngram_cache_budget_bytes':0})
            budget=int(os.environ.get('WERK_OMLX_EXPERT_CACHE_BYTES','0'))
            return self.reply({'active':True,'cache_budget_bytes':budget or 8*1024**3,'cache_budget_mode':'explicit' if budget else 'auto'})
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
        if self.path=='/werk/tokenize':
            (root/'tokenize.json').write_text(json.dumps(body))
            return self.reply({'input_tokens':settings.get('token_count',37)})
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
            // Stable lifetime lock inodes survive disposable worker bases.
            .filter(|entry| entry.as_ref().unwrap().file_name() != ".locks")
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

#[tokio::test]
async fn thinking_override_is_captured_by_worker_and_changes_cache_identity() {
    tokio::task::spawn_blocking(|| {
        let mut fixture = Fixture::new(json!({}));
        fixture.invocation.thinking = None;
        let default_identity = format!("{:?}", fixture.invocation);
        let mut previous_identity = default_identity.clone();
        for thinking in [false, true] {
            fixture.invocation.thinking = Some(thinking);
            let identity = format!("{:?}", fixture.invocation);
            assert_ne!(identity, default_identity);
            assert_ne!(identity, previous_identity);
            previous_identity = identity;
            let server = fixture.start().unwrap();
            // A worker keeps the selected setting even if later selection changes.
            fixture.invocation.thinking = Some(!thinking);
            for stream in [false, true] {
                let (tx, _rx) = mpsc::channel(8);
                server.generate(&request(), stream.then_some(tx)).unwrap();
                let sent: Value =
                    serde_json::from_slice(&fs::read(fixture.model.join("request.json")).unwrap())
                        .unwrap();
                assert_eq!(sent["stream"], stream);
                assert_eq!(
                    sent["chat_template_kwargs"],
                    json!({"enable_thinking": thinking})
                );
            }
        }
    })
    .await
    .unwrap();
}

#[test]
fn omlx_usage_timings_include_reasoning_and_override_visible_token_delay() {
    let mut completion = OpenAiCompletion::default();
    let mut timings = OmlxUsageTimings::default();
    update_omlx_completion_from_event(
        &mut completion,
        &mut timings,
        &json!({"choices": [{"delta": {"reasoning_content": "hidden"}}]}),
        Some(2.0),
    );
    assert_eq!(completion.first_token_seconds, 2.0);
    assert!(completion.text.is_empty());
    update_omlx_completion_from_event(
        &mut completion,
        &mut timings,
        &json!({"choices": [], "usage": {
            "prompt_tokens": 267, "completion_tokens": 173,
            "time_to_first_token": 1.9, "prompt_eval_duration": 1.8,
            "generation_duration": 148.1, "total_time": 150.0
        }}),
        Some(150.2),
    );
    assert!(
        finalize_omlx_completion_stats(&mut completion, &request(), 150.266, &timings).is_empty()
    );
    assert_eq!(completion.first_token_seconds, 1.9);
    assert_eq!(completion.prompt_seconds, 1.8);
    assert_eq!(completion.decode_seconds, 148.1);
    assert_eq!(completion.completion_tokens, 173);
}

#[test]
fn omlx_missing_phase_timings_use_reasoning_arrival_or_total_duration() {
    let mut completion = OpenAiCompletion::default();
    let mut timings = OmlxUsageTimings::default();
    update_omlx_completion_from_event(
        &mut completion,
        &mut timings,
        &json!({"choices": [{"delta": {"reasoning_content": "hidden"}}]}),
        Some(2.0),
    );
    finalize_omlx_completion_stats(&mut completion, &request(), 150.0, &timings);
    assert_eq!(completion.prompt_seconds, 2.0);
    assert_eq!(completion.decode_seconds, 148.0);

    let mut completion = OpenAiCompletion::default();
    update_omlx_completion_from_event(
        &mut completion,
        &mut timings,
        &json!({"usage": {"completion_tokens": 173, "total_time": 150.0}}),
        None,
    );
    let diagnostics =
        finalize_omlx_completion_stats(&mut completion, &request(), 150.266, &timings);
    assert_eq!(completion.first_token_seconds, 0.0);
    assert!(completion.prompt_seconds.is_nan());
    assert_eq!(completion.decode_seconds, 150.0);
    assert!(diagnostics[0].contains("includes prompt processing"));

    let mut completion = OpenAiCompletion::default();
    finalize_omlx_completion_stats(
        &mut completion,
        &request(),
        10.0,
        &OmlxUsageTimings::default(),
    );
    assert_eq!(completion.decode_seconds, 10.0);
    assert_eq!(completion.first_token_seconds, 0.0);
}

#[test]
fn omlx_usage_timings_ignore_malformed_fields_without_losing_valid_values() {
    let mut completion = OpenAiCompletion::default();
    let mut timings = OmlxUsageTimings::default();
    update_omlx_completion_from_event(
        &mut completion,
        &mut timings,
        &json!({"usage": {"time_to_first_token": 0.0, "generation_duration": 2.0}}),
        None,
    );
    update_omlx_completion_from_event(
        &mut completion,
        &mut timings,
        &json!({"usage": {
            "time_to_first_token": "NaN", "generation_duration": -1,
            "prompt_eval_duration": null, "total_time": false
        }}),
        None,
    );
    finalize_omlx_completion_stats(&mut completion, &request(), 3.0, &timings);
    assert_eq!(completion.first_token_seconds, 0.0);
    assert_eq!(completion.prompt_seconds, 0.0);
    assert_eq!(completion.decode_seconds, 2.0);
}

#[test]
fn native_cache_diagnostics_report_only_upstream_cached_token_counts() {
    let mut completion = OpenAiCompletion::default();
    let mut metadata = OmlxUsageTimings::default();
    for tokens in [0, 127] {
        update_omlx_completion_from_event(
            &mut completion,
            &mut metadata,
            &json!({"usage":{"prompt_tokens":128,"generation_duration":1.0,"prompt_tokens_details":{"cached_tokens":tokens}}}),
            None,
        );
        let diagnostics =
            finalize_omlx_completion_stats(&mut completion, &request(), 2.0, &metadata);
        assert_eq!(
            diagnostics,
            vec![format!("oMLX cached prompt tokens: {tokens}")]
        );
    }
    update_omlx_completion_from_event(
        &mut completion,
        &mut metadata,
        &json!({"usage":{"prompt_tokens_details":{"cached_tokens":-1}}}),
        None,
    );
    assert_eq!(metadata.cached_prompt_tokens, Some(127));
    assert!(
        finalize_omlx_completion_stats(
            &mut completion,
            &request(),
            2.0,
            &OmlxUsageTimings::default()
        )
        .iter()
        .all(|line| !line.contains("cached prompt tokens"))
    );
}

#[test]
fn model_directory_resolves_external_config_and_files_fallback() {
    let root = env::temp_dir().join(format!("werk-omlx-external-{}", random_id().unwrap()));
    let store = ModelStore::resolve(Some(root.join("store"))).unwrap();
    let external = root.join("raid/qwen");
    fs::create_dir_all(external.join("snapshot")).unwrap();
    fs::write(external.join("config.json"), b"{}").unwrap();
    fs::write(external.join("snapshot/config.json"), b"{}").unwrap();
    let mut manifest = fixture_manifest();
    manifest.storage = crate::model_store::ModelStorage::External {
        path: external.clone(),
    };
    manifest.config_path = Some("files/snapshot/config.json".to_string());

    assert_eq!(
        resolve_model_dir(&store, &manifest).unwrap(),
        external.join("snapshot").canonicalize().unwrap()
    );
    manifest.config_path = None;
    assert_eq!(
        resolve_model_dir(&store, &manifest).unwrap(),
        external.canonicalize().unwrap()
    );
    assert!(!store.model_dir(&manifest.id).join("files").exists());
    fs::remove_dir_all(root).unwrap();
}

fn fixture_manifest() -> ModelManifest {
    ModelManifest {
        storage: Default::default(),
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
        model_probes: Arc::new(Mutex::new(VecDeque::new())),
        request_thinking: None,
        request_reasoning_effort: None,
        test_probe: Some(ProbeReport {
            detail: "fixture".into(),
            version: "test-0.6.4".into(),
            tools: true,
            runtime: json!({"fixture":true}),
            cache_paths: vec![],
        }),
    }
}

#[cfg(unix)]
fn counting_probe_backend(fixture: &Fixture, dependency_inventory: bool) -> OmlxBackend {
    use std::os::unix::fs::PermissionsExt;
    let mut backend = fixture_backend(fixture);
    backend.test_probe = None;
    let runtime_dir = fixture.root.join("probe-runtime");
    fs::create_dir(&runtime_dir).unwrap();
    fs::write(runtime_dir.join("runtime.py"), "runtime = 1\n").unwrap();
    let interpreter = fixture.root.join("counting-probe-python");
    let source = format!(
        "#!{}\n{}",
        fixture.invocation.python.display(),
        r#"import json, pathlib, sys
root = pathlib.Path(__file__).parent
counter = root / 'probe-count.txt'
counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
payload = json.load(sys.stdin)
config = json.loads((pathlib.Path(payload['model_dir']) / 'config.json').read_text())
result = {'ok': not config.get('reject_probe', False), 'detail': 'counted probe',
          'runtime': {'omlx_version': '0.6.4'}, 'supports_tool_calling': True}
if INVENTORY:
    result['cache_paths'] = [str(root / 'probe-runtime'),
                             str(root / 'probe-runtime' / 'runtime.py'), __file__]
print(json.dumps(result))
sys.exit(0 if result['ok'] else 1)
"#
        .replace(
            "INVENTORY",
            if dependency_inventory {
                "True"
            } else {
                "False"
            }
        )
    );
    fs::write(&interpreter, source).unwrap();
    fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o700)).unwrap();
    backend.invocation.as_mut().unwrap().python = interpreter;
    backend
}

#[cfg(unix)]
fn probe_count(fixture: &Fixture) -> usize {
    fs::read_to_string(fixture.root.join("probe-count.txt"))
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
#[cfg(unix)]
fn probe_json_larger_than_stderr_limit_preserves_dependency_inventory() {
    let fixture = Fixture::new(json!({}));
    let backend = counting_probe_backend(&fixture, true);
    let interpreter = &backend.invocation().unwrap().python;
    let source = fs::read_to_string(interpreter).unwrap().replace(
        "print(json.dumps(result))",
        "result['cache_paths'] *= 1000\nprint('diagnostic' * 10000, file=sys.stderr)\nprint(json.dumps(result))",
    );
    fs::write(interpreter, source).unwrap();
    let manifest = fixture_model_for_backend(&fixture);
    let directory = resolve_model_dir(&fixture.store, &manifest).unwrap();
    let report = backend
        .invocation()
        .unwrap()
        .probe(Some(&directory))
        .unwrap();
    assert_eq!(report.cache_paths.len(), 3000);
    assert!(serde_json::to_vec(&report.cache_paths).unwrap().len() > MAX_PROBE_STDERR_BYTES);
    assert_eq!(report.version, "0.6.4");
    assert!(report.tools);
    let payload = vec![b'x'; MAX_PROBE_STDERR_BYTES + 123];
    assert_eq!(
        read_bounded(payload.as_slice(), MAX_PROBE_STDERR_BYTES).len(),
        MAX_PROBE_STDERR_BYTES
    );
    assert_eq!(
        read_bounded(payload.as_slice(), MAX_PROBE_JSON_BYTES),
        payload
    );
}

#[test]
#[cfg(unix)]
fn stable_model_probes_reuse_verified_runtime_for_chat_and_tools() {
    let fixture = Fixture::new(json!({}));
    let backend = counting_probe_backend(&fixture, true);
    let manifest = fixture_model_for_backend(&fixture);
    backend.probe_model(&manifest).unwrap();
    backend.probe_model(&manifest).unwrap();
    assert!(backend.probe_tool_calling(&manifest).unwrap());
    backend
        .configured_for_chat(&chat_options(Some(false), None))
        .unwrap()
        .probe_model(&manifest)
        .unwrap();
    assert_eq!(probe_count(&fixture), 1);
    assert_eq!(backend.model_probes.lock().unwrap().len(), 1);
    assert!(backend.servers.lock().unwrap().is_empty());
}

#[test]
#[cfg(unix)]
fn probe_cache_invalidates_metadata_shards_runtime_and_invocation_changes() {
    let fixture = Fixture::new(json!({}));
    let backend = counting_probe_backend(&fixture, true);
    let manifest = fixture_model_for_backend(&fixture);
    let directory = resolve_model_dir(&fixture.store, &manifest).unwrap();
    let mut expected = 0;
    let mut assert_new_probe = |backend: &OmlxBackend, manifest: &ModelManifest| {
        backend.probe_model(manifest).unwrap();
        expected += 1;
        assert_eq!(probe_count(&fixture), expected);
        backend.probe_model(manifest).unwrap();
        assert_eq!(probe_count(&fixture), expected);
    };
    assert_new_probe(&backend, &manifest);
    fs::write(directory.join("config.json"), r#"{"changed":true}"#).unwrap();
    assert_new_probe(&backend, &manifest);
    fs::write(directory.join("tokenizer_config.json"), "{}").unwrap();
    assert_new_probe(&backend, &manifest);
    let template = directory.join("chat_template.jinja");
    fs::write(&template, "first tool format").unwrap();
    assert_new_probe(&backend, &manifest);
    fs::write(&template, "changed tool format").unwrap();
    assert_new_probe(&backend, &manifest);
    fs::remove_file(&template).unwrap();
    assert_new_probe(&backend, &manifest);
    fs::create_dir(directory.join("chat_templates")).unwrap();
    assert_new_probe(&backend, &manifest);
    fs::remove_dir(directory.join("chat_templates")).unwrap();
    assert_new_probe(&backend, &manifest);
    let shard = directory.join("model.safetensors");
    fs::write(&shard, "first fixture header").unwrap();
    assert_new_probe(&backend, &manifest);
    fs::write(&shard, "replacement fixture header").unwrap();
    assert_new_probe(&backend, &manifest);
    fs::remove_file(shard).unwrap();
    assert_new_probe(&backend, &manifest);
    fs::write(
        fixture.root.join("probe-runtime/runtime.py"),
        "runtime = 22\n",
    )
    .unwrap();
    assert_new_probe(&backend, &manifest);
    fs::write(
        fixture.root.join("probe-runtime/new_module.py"),
        "new_module = True\n",
    )
    .unwrap();
    assert_new_probe(&backend, &manifest);
    let mut changed_manifest = manifest.clone();
    changed_manifest.created_unix += 1;
    assert_new_probe(&backend, &changed_manifest);
    let configured = backend
        .configured_for_chat(&chat_options(None, Some(7)))
        .unwrap();
    assert_new_probe(&configured, &manifest);
    let mut changed_environment = backend.clone();
    changed_environment
        .invocation
        .as_mut()
        .unwrap()
        .environment
        .push(("WERK_TEST_PROBE_RUNTIME".into(), "different".into()));
    assert_new_probe(&changed_environment, &manifest);
    fs::write(&fixture.invocation.launcher, "changed launcher").unwrap();
    assert!(
        backend
            .probe_model(&manifest)
            .unwrap_err()
            .to_string()
            .contains("launcher changed")
    );
    assert_eq!(probe_count(&fixture), expected);
}

#[test]
#[cfg(unix)]
fn probe_failures_and_missing_dependency_inventory_are_not_cached() {
    for inventory in [false, true] {
        let fixture = Fixture::new(json!({}));
        let backend = counting_probe_backend(&fixture, inventory);
        let manifest = fixture_model_for_backend(&fixture);
        let directory = resolve_model_dir(&fixture.store, &manifest).unwrap();
        if inventory {
            fs::write(directory.join("config.json"), r#"{"reject_probe":true}"#).unwrap();
        }
        for _ in 0..2 {
            assert_eq!(backend.probe_model(&manifest).is_ok(), !inventory);
        }
        assert_eq!(probe_count(&fixture), 2);
        assert!(backend.model_probes.lock().unwrap().is_empty());
    }
}

#[test]
#[cfg(unix)]
fn successful_probe_cache_has_a_bounded_lru() {
    let fixture = Fixture::new(json!({}));
    let backend = counting_probe_backend(&fixture, true);
    let manifest = fixture_model_for_backend(&fixture);
    for revision in 0..=MAX_CACHED_MODEL_PROBES {
        let mut revision_manifest = manifest.clone();
        revision_manifest.created_unix = revision as u64;
        backend.probe_model(&revision_manifest).unwrap();
    }
    assert_eq!(
        backend.model_probes.lock().unwrap().len(),
        MAX_CACHED_MODEL_PROBES
    );
    assert_eq!(probe_count(&fixture), MAX_CACHED_MODEL_PROBES + 1);
    backend.probe_model(&manifest).unwrap();
    assert_eq!(probe_count(&fixture), MAX_CACHED_MODEL_PROBES + 2);
}

fn chat_options(
    thinking: Option<bool>,
    expert_cache_mb: Option<u64>,
) -> crate::openai::ChatRuntimeOptions {
    crate::openai::ChatRuntimeOptions {
        documents: None,
        omlx: Some(crate::openai::OmlxChatOptions {
            reasoning_effort: None,
            thinking,
            expert_cache_mb,
            ngram_cache_mb: None,
        }),
    }
}

fn fixture_model_for_backend(fixture: &Fixture) -> ModelManifest {
    let manifest = fixture_manifest();
    let actual = fixture.store.model_dir(&manifest.id).join("files");
    if actual != fixture.model {
        fs::create_dir_all(actual.parent().unwrap()).unwrap();
        fs::rename(&fixture.model, &actual).unwrap();
    }
    manifest
}

// Exercise Rust's real HTTP/worker/configuration paths without importing MLX.
// The embedded supervisor/helper is covered separately by its Python tests.
#[cfg(unix)]
fn server_cache_fixture(settings: Value) -> (Fixture, OmlxBackend, ModelManifest) {
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new(settings);
    let mut backend = fixture_backend(&fixture).with_server_prefix_cache(true);
    backend.test_probe.as_mut().unwrap().version = "0.6.4".into();
    let interpreter = fixture.root.join("fixture-worker-python");
    fs::write(&interpreter, format!(
        "#!{}\nimport runpy, sys\nassert sys.argv[1] == '-c'\nsys.argv = sys.argv[4:]\nrunpy.run_path(sys.argv[0], run_name='__main__')\n",
        fixture.invocation.python.display()
    )).unwrap();
    fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o700)).unwrap();
    let invocation = backend.invocation.as_mut().unwrap();
    invocation.python = interpreter;
    invocation.thinking = None;
    invocation.expert_cache_bytes = None;
    let manifest = fixture_model_for_backend(&fixture);
    (fixture, backend, manifest)
}

#[test]
#[cfg(unix)]
fn native_count_reuses_worker_and_template_controls_without_generating() {
    let (fixture, backend, manifest) = server_cache_fixture(json!({"version":"0.6.4"}));
    backend.prepare(&manifest).unwrap();
    let configured = backend
        .with_chat_options(&manifest, &chat_options(Some(false), None))
        .unwrap();
    assert_eq!(configured.count_tokens(&manifest, request()).unwrap(), 37);
    assert_eq!(configured.count_tokens(&manifest, request()).unwrap(), 37);
    let model = resolve_model_dir(&fixture.store, &manifest).unwrap();
    let sent: Value =
        serde_json::from_slice(&fs::read(model.join("tokenize.json")).unwrap()).unwrap();
    assert_eq!(sent["model"], "physical / model");
    assert_eq!(sent["chat_template_kwargs"]["enable_thinking"], false);
    assert!(!model.join("chat_started").exists());
    assert_eq!(
        fs::read_to_string(model.join("starts.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    assert_eq!(backend.servers.lock().unwrap().len(), 1);
}

#[tokio::test]
#[cfg(unix)]
async fn server_prefix_cache_covers_prepare_sessions_stream_tools_and_request_options() {
    use tokio_stream::StreamExt;
    let (fixture, backend, manifest) = server_cache_fixture(json!({"version":"0.6.4"}));
    backend.prepare(&manifest).unwrap();
    let session = backend
        .start_chat_session(&manifest, None)
        .unwrap()
        .unwrap();
    assert_eq!(session.generate(request()).unwrap().text, "answer");
    assert_eq!(
        backend.generate(&manifest, request()).unwrap().text,
        "answer"
    );
    let mut req = request();
    req.tool_config = Some(super::super::ToolCallingConfig {
        tools: Some(serde_json::from_value(json!([{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}])).unwrap()),
        tool_choice: None,
        parallel_tool_calls: None,
    });
    let configured = backend
        .with_chat_options(&manifest, &chat_options(Some(false), None))
        .unwrap();
    let mut stream = configured.generate_stream(manifest.clone(), req);
    let mut done = false;
    while let Some(event) = stream.next().await {
        done |= matches!(event.unwrap(), GenerateStreamEvent::Done { .. });
    }
    assert!(done);
    let model = resolve_model_dir(&fixture.store, &manifest).unwrap();
    let starts = fs::read_to_string(model.join("starts.jsonl")).unwrap();
    assert_eq!(
        starts.lines().count(),
        1,
        "all paths must retain the eagerly loaded worker"
    );
    let start: Value = serde_json::from_str(starts.trim()).unwrap();
    assert_eq!(start["cache"], start["env_cache"]);
    assert_eq!(start["env_model"], model.to_str().unwrap());
    assert_eq!(start["limit"], "4GB");
    assert_eq!(
        Path::new(start["cache"].as_str().unwrap()),
        Path::new(start["base"].as_str().unwrap()).join("cache/prefix-cache")
    );
    let sent: Value =
        serde_json::from_slice(&fs::read(model.join("request.json")).unwrap()).unwrap();
    assert_eq!(sent["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(sent["tools"][0]["function"]["name"], "lookup");
    assert_eq!(backend.servers.lock().unwrap().len(), 1);
}

#[test]
#[cfg(unix)]
fn server_prefix_cache_expert_variants_have_independent_managed_cache_directories() {
    let (fixture, backend, manifest) = server_cache_fixture(json!({"version":"0.6.4"}));
    backend.prepare(&manifest).unwrap();
    let configured = backend
        .with_chat_options(&manifest, &chat_options(None, Some(8)))
        .unwrap();
    configured.generate(&manifest, request()).unwrap();
    backend.generate(&manifest, request()).unwrap();
    let starts = fs::read_to_string(
        resolve_model_dir(&fixture.store, &manifest)
            .unwrap()
            .join("starts.jsonl"),
    )
    .unwrap();
    let starts: Vec<Value> = starts
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(starts.len(), 2);
    assert_ne!(starts[0]["cache"], starts[1]["cache"]);
    assert!(starts.iter().all(|start| start["limit"] == "4GB"));
    assert_eq!(backend.servers.lock().unwrap().len(), 2);
}

#[test]
#[cfg(unix)]
fn server_prefix_cache_unavailable_or_disabled_keeps_one_ordinary_worker() {
    for (version, enabled, active, expected_cache) in [
        ("0.6.4", false, true, false),
        ("0.6.5", true, true, false),
        ("0.6.4", true, false, true),
    ] {
        let (fixture, mut backend, manifest) =
            server_cache_fixture(json!({"version":version,"cache_active":active}));
        backend = backend.with_server_prefix_cache(enabled);
        backend.test_probe.as_mut().unwrap().version = version.into();
        backend.prepare(&manifest).unwrap();
        assert_eq!(
            backend.generate(&manifest, request()).unwrap().text,
            "answer"
        );
        let starts = fs::read_to_string(
            resolve_model_dir(&fixture.store, &manifest)
                .unwrap()
                .join("starts.jsonl"),
        )
        .unwrap();
        assert_eq!(starts.lines().count(), 1);
        let start: Value = serde_json::from_str(starts.trim()).unwrap();
        assert_eq!(start["cache"].is_string(), expected_cache);
        assert_eq!(backend.servers.lock().unwrap().len(), 1);
    }
}

#[test]
fn ngram_budget_is_independent_and_changes_worker_and_persistence_identity() {
    assert_eq!(ngram_cache_bytes(None).unwrap(), Some(0));
    assert_eq!(ngram_cache_bytes(Some("auto".into())).unwrap(), None);
    assert_eq!(ngram_cache_bytes(Some("0".into())).unwrap(), Some(0));
    assert_eq!(
        ngram_cache_bytes(Some("1".into())).unwrap(),
        Some(1024 * 1024)
    );
    for value in ["-1", "1.5", "", "true", "18446744073709551615"] {
        assert!(ngram_cache_bytes(Some(value.into())).is_err());
    }
    let fixture = Fixture::new(json!({}));
    let base = fixture_backend(&fixture);
    let identity = base.cache_identity();
    for budget in [0, 1024, 1_048_576] {
        let mut options = chat_options(None, None);
        options.omlx.as_mut().unwrap().ngram_cache_mb = Some(budget.into());
        let configured = base.configured_for_chat(&options).unwrap();
        assert_eq!(
            configured.invocation().unwrap().ngram_cache_bytes,
            Some(budget * 1024 * 1024)
        );
        assert_eq!(
            configured.invocation().unwrap().expert_cache_bytes,
            base.invocation().unwrap().expert_cache_bytes
        );
        if budget == 0 {
            assert_eq!(configured.cache_identity(), identity);
        } else {
            assert_ne!(configured.cache_identity(), identity);
        }
    }
    assert_eq!(base.cache_identity(), identity);
}

#[test]
fn ngram_auto_request_overrides_fixed_server_budget_without_mutating_it() {
    let fixture = Fixture::new(json!({}));
    let mut base = fixture_backend(&fixture);
    base.invocation.as_mut().unwrap().ngram_cache_bytes = Some(1024 * 1024 * 1024);
    let options: crate::openai::ChatRuntimeOptions =
        serde_json::from_value(json!({"omlx":{"ngram_cache_mb":"auto"}})).unwrap();
    let configured = base.configured_for_chat(&options).unwrap();
    assert_eq!(configured.invocation().unwrap().ngram_cache_bytes, None);
    assert_eq!(
        base.invocation().unwrap().ngram_cache_bytes,
        Some(1024 * 1024 * 1024)
    );
    assert_ne!(configured.cache_identity(), base.cache_identity());
    assert_eq!(
        serde_json::to_value(options).unwrap()["omlx"]["ngram_cache_mb"],
        "auto"
    );
}

#[test]
fn chat_options_inherit_defaults_without_mutation_and_bound_expert_budget() {
    let fixture = Fixture::new(json!({}));
    let mut base = fixture_backend(&fixture);
    base.invocation.as_mut().unwrap().thinking = Some(true);
    base.invocation.as_mut().unwrap().expert_cache_bytes = Some(8 * 1024 * 1024 * 1024);
    let identity = base.cache_identity();
    let configured = base
        .configured_for_chat(&chat_options(Some(false), None))
        .unwrap();
    assert_eq!(configured.cache_identity(), identity);
    assert_eq!(configured.request_thinking, Some(false));
    assert_eq!(configured.invocation().unwrap().thinking, Some(true));
    assert!(Arc::ptr_eq(&base.servers, &configured.servers));
    let disabled = base
        .configured_for_chat(&chat_options(None, Some(0)))
        .unwrap();
    assert_eq!(disabled.invocation().unwrap().expert_cache_bytes, None);
    assert_eq!(disabled.request_thinking, None);
    assert_ne!(disabled.cache_identity(), identity);
    let maximum = base
        .configured_for_chat(&chat_options(None, Some(1_048_576)))
        .unwrap();
    assert_eq!(
        maximum.invocation().unwrap().expert_cache_bytes,
        Some(1_099_511_627_776)
    );
    assert!(
        base.configured_for_chat(&chat_options(None, Some(1_048_577)))
            .is_err()
    );
    assert_eq!(base.cache_identity(), identity);
    assert_eq!(base.request_thinking, None);
}

#[test]
fn thinking_chat_options_reuse_one_worker_and_do_not_leak_into_defaults() {
    let fixture = Fixture::new(json!({}));
    let mut base = fixture_backend(&fixture);
    base.invocation.as_mut().unwrap().thinking = None;
    base.invocation.as_mut().unwrap().expert_cache_bytes = None;
    let manifest = fixture_model_for_backend(&fixture);
    let request_path = resolve_model_dir(&fixture.store, &manifest)
        .unwrap()
        .join("request.json");
    let mut instance = None;
    for thinking in [Some(false), Some(true), None] {
        let configured: Arc<dyn GenerationBackend> = if let Some(thinking) = thinking {
            base.with_chat_options(&manifest, &chat_options(Some(thinking), None))
                .unwrap()
        } else {
            Arc::new(base.clone())
        };
        assert_eq!(
            configured.generate(&manifest, request()).unwrap().text,
            "answer"
        );
        let sent: Value = serde_json::from_slice(&fs::read(&request_path).unwrap()).unwrap();
        assert_eq!(
            sent.get("chat_template_kwargs").cloned(),
            thinking.map(|enabled| json!({"enable_thinking": enabled}))
        );
        let selected = configured
            .runtime_control_adapter_for(&manifest)
            .unwrap()
            .descriptor()
            .instance_id;
        if let Some(expected) = &instance {
            assert_eq!(&selected, expected);
        } else {
            instance = Some(selected);
        }
        assert_eq!(base.servers.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn thinking_chat_options_are_preserved_for_streaming() {
    use tokio_stream::StreamExt;
    let fixture = Fixture::new(json!({}));
    let mut base = fixture_backend(&fixture);
    base.invocation.as_mut().unwrap().thinking = None;
    base.invocation.as_mut().unwrap().expert_cache_bytes = None;
    let manifest = fixture_model_for_backend(&fixture);
    let configured = base
        .with_chat_options(&manifest, &chat_options(Some(false), None))
        .unwrap();
    let mut stream = configured.generate_stream(manifest.clone(), request());
    let mut done = false;
    while let Some(event) = stream.next().await {
        if matches!(event.unwrap(), GenerateStreamEvent::Done { .. }) {
            done = true;
        }
    }
    assert!(done);
    let path = resolve_model_dir(&fixture.store, &manifest)
        .unwrap()
        .join("request.json");
    let sent: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(sent["stream"], true);
    assert_eq!(
        sent["chat_template_kwargs"],
        json!({"enable_thinking":false})
    );
}

#[test]
#[cfg(unix)]
fn chat_options_probe_observes_the_applied_expert_budget_before_loading() {
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new(json!({}));
    let mut base = fixture_backend(&fixture);
    base.test_probe = None;
    let manifest = fixture_model_for_backend(&fixture);
    let interpreter = fixture.root.join("fixture-probe-python");
    fs::write(&interpreter, format!(
        "#!{}\nimport json, pathlib, sys\npayload=json.load(sys.stdin)\npathlib.Path(payload['model_dir'], 'probe-payload.json').write_text(json.dumps(payload))\nprint(json.dumps({{'ok':payload['expert_cache_bytes']==7340032,'detail':'configured probe','runtime':{{'omlx_version':'0.6.4'}}}}))\n",
        fixture.invocation.python.display()
    )).unwrap();
    fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o700)).unwrap();
    base.invocation.as_mut().unwrap().python = interpreter;
    base.invocation.as_mut().unwrap().expert_cache_bytes = None;
    assert!(
        base.with_chat_options(&manifest, &chat_options(Some(false), Some(7)))
            .is_ok()
    );
    let path = resolve_model_dir(&fixture.store, &manifest)
        .unwrap()
        .join("probe-payload.json");
    let payload: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(payload["expert_cache_bytes"], 7 * 1024 * 1024);
    assert!(base.servers.lock().unwrap().is_empty());
    assert!(!fixture.root.join("backends").exists());
    assert!(
        base.with_chat_options(&manifest, &chat_options(None, Some(8)))
            .is_err()
    );
    assert_eq!(base.invocation().unwrap().expert_cache_bytes, None);
}

#[test]
fn configured_expert_budget_selects_its_exact_worker_and_registry_is_bounded() {
    let fixture = Fixture::new(json!({}));
    let mut base = fixture_backend(&fixture);
    base.invocation.as_mut().unwrap().thinking = None;
    base.invocation.as_mut().unwrap().expert_cache_bytes = None;
    let manifest = fixture_model_for_backend(&fixture);
    let (default_worker, _) = base.cached_server(&manifest).unwrap();
    let mut second = OmlxProcess::start(
        &fixture.store,
        &fixture.invocation,
        &default_worker.model_dir,
        base.test_probe.as_ref().unwrap().clone(),
    )
    .unwrap();
    second.model_identity = default_worker.model_identity.clone();
    second.logical_model_id = default_worker.logical_model_id.clone();
    second.thinking = None;
    second.expert_cache_bytes = Some(8 * 1024 * 1024);
    let second_id = second.instance_id.clone();
    base.servers
        .lock()
        .unwrap()
        .insert("different-budget-fixture".into(), Arc::new(second));
    let configured = base
        .configured_for_chat(&chat_options(Some(false), Some(8)))
        .unwrap();
    assert_eq!(
        configured
            .runtime_control_adapter_for(&manifest)
            .unwrap()
            .descriptor()
            .instance_id,
        second_id
    );
    assert_eq!(
        base.runtime_control_adapter_for(&manifest)
            .unwrap()
            .descriptor()
            .instance_id,
        default_worker.instance_id
    );
    {
        let mut registry = base.servers.lock().unwrap();
        while registry.len() < MAX_CACHED_WORKERS {
            let key = format!("retained-fixture-{}", registry.len());
            registry.insert(key, default_worker.clone());
        }
    }
    assert!(
        base.cached_server(&manifest).is_ok(),
        "an existing worker remains reusable at capacity"
    );
    let error = configured.cached_server(&manifest).err().unwrap();
    assert!(error.to_string().contains("limit of 16 retained workers"));
    assert!(default_worker.is_running());
    assert_eq!(base.servers.lock().unwrap().len(), MAX_CACHED_WORKERS);
}

#[test]
fn persistent_cache_namespace_survives_restart_but_separates_model_and_runtime_settings() {
    let fixture = Fixture::new(json!({}));
    let cache = fixture.root.join("chat-cache");
    fs::create_dir(&cache).unwrap();
    let manifest = fixture_manifest();
    let report = ProbeReport {
        detail: "fixture".into(),
        version: "0.6.4".into(),
        tools: false,
        runtime: json!({"omlx_version":"0.6.4","mlx_version":"0.32.2"}),
        cache_paths: vec![],
    };
    let first =
        persistent_cache_directory(&cache, &manifest, &fixture.invocation, &report).unwrap();
    assert!(first.starts_with(cache.canonicalize().unwrap()));
    assert_eq!(
        first,
        persistent_cache_directory(&cache, &manifest, &fixture.invocation, &report).unwrap()
    );
    let mut restarted = fixture.invocation.clone();
    restarted
        .environment
        .push(("WERK_TEST_UNRELATED_SHELL_VALUE".into(), "changed".into()));
    assert_eq!(
        first,
        persistent_cache_directory(&cache, &manifest, &restarted, &report).unwrap()
    );
    restarted.thinking = Some(!fixture.invocation.thinking.unwrap_or(true));
    assert_ne!(
        first,
        persistent_cache_directory(&cache, &manifest, &restarted, &report).unwrap()
    );
    let mut changed_model = manifest.clone();
    changed_model.created_unix += 1;
    assert_ne!(
        first,
        persistent_cache_directory(&cache, &changed_model, &fixture.invocation, &report).unwrap()
    );
    let mut changed_runtime = report.clone();
    changed_runtime.runtime["mlx_version"] = json!("0.33.0");
    assert_ne!(
        first,
        persistent_cache_directory(&cache, &manifest, &fixture.invocation, &changed_runtime)
            .unwrap()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&first).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }
}

#[test]
fn persistent_cache_status_requires_verified_format_and_selected_model() {
    let status = json!({"installed":true,"active":true,"format":"omlx-exact-prefix-v1","model_id":"physical"});
    assert!(verified_persistent_cache_status(&status, "physical").unwrap());
    assert!(verified_persistent_cache_status(&status, "other").is_err());
    let mut inactive = status.clone();
    inactive["active"] = json!(false);
    assert!(!verified_persistent_cache_status(&inactive, "physical").unwrap());
    for invalid in [
        json!({}),
        json!({"installed":false,"active":true,"format":"omlx-exact-prefix-v1","model_id":"physical"}),
        json!({"installed":true,"active":true,"format":"unknown","model_id":"physical"}),
        json!({"installed":true,"active":"yes","format":"omlx-exact-prefix-v1","model_id":"physical"}),
    ] {
        assert!(verified_persistent_cache_status(&invalid, "physical").is_err());
    }
}

#[test]
fn unverified_native_cache_version_returns_normal_fallback_without_starting_worker() {
    let fixture = Fixture::new(json!({}));
    let backend = fixture_backend(&fixture);
    let manifest = fixture_manifest();
    let actual = fixture.store.model_dir(&manifest.id).join("files");
    if actual != fixture.model {
        fs::create_dir_all(actual.parent().unwrap()).unwrap();
        fs::rename(&fixture.model, &actual).unwrap();
    }
    let cache = fixture.root.join("absent-persistent-cache");
    assert!(
        backend
            .start_persistent_chat_session(&manifest, None, &cache)
            .unwrap()
            .is_none()
    );
    assert!(!cache.exists());
    assert!(backend.servers.lock().unwrap().is_empty());
    assert!(!fixture.root.join("backends").exists());
}

#[test]
fn generic_native_persistence_default_does_not_load_backend_or_create_files() {
    struct NoNativePersistence;
    impl GenerationBackend for NoNativePersistence {
        fn prepare(&self, _manifest: &ModelManifest) -> Result<()> {
            panic!("native persistence default must not prepare a backend");
        }
        fn generate(
            &self,
            _manifest: &ModelManifest,
            _request: GenerateRequest,
        ) -> Result<GenerateResponse> {
            panic!("native persistence default must not generate");
        }
        fn generate_stream(
            &self,
            _manifest: ModelManifest,
            _request: GenerateRequest,
        ) -> GenerateStream {
            panic!("native persistence default must not generate");
        }
    }
    let cache = env::temp_dir().join(format!("werk-absent-chat-cache-{}", random_id().unwrap()));
    assert!(
        NoNativePersistence
            .start_persistent_chat_session(&fixture_manifest(), None, &cache)
            .unwrap()
            .is_none()
    );
    assert!(!cache.exists());
}

#[test]
#[cfg(unix)]
fn persistent_cache_rejects_symlink_directory_without_touching_target() {
    let fixture = Fixture::new(json!({}));
    let cache = fixture.root.join("real-cache");
    fs::create_dir(&cache).unwrap();
    let link = fixture.root.join("linked-cache");
    std::os::unix::fs::symlink(&cache, &link).unwrap();
    let report = fixture_backend(&fixture).test_probe.unwrap();
    assert!(
        persistent_cache_directory(&link, &fixture_manifest(), &fixture.invocation, &report)
            .is_err()
    );
    assert_eq!(fs::read_dir(&cache).unwrap().count(), 0);
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
fn model_controls_require_unique_worker_when_chat_cache_namespaces_differ() {
    let fixture = Fixture::new(json!({}));
    let backend = fixture_backend(&fixture);
    let manifest = fixture_manifest();
    let actual = fixture.store.model_dir(&manifest.id).join("files");
    if actual != fixture.model {
        fs::create_dir_all(actual.parent().unwrap()).unwrap();
        fs::rename(&fixture.model, &actual).unwrap();
    }
    let actual = resolve_model_dir(&fixture.store, &manifest).unwrap();
    let (first, _) = backend.cached_server(&manifest).unwrap();
    let mut second = OmlxProcess::start(
        &fixture.store,
        &fixture.invocation,
        &actual,
        backend.test_probe.as_ref().unwrap().clone(),
    )
    .unwrap();
    second.model_identity = Some(ModelRuntimeIdentity::from_manifest(&manifest).unwrap());
    second.logical_model_id = Some(manifest.id.clone());
    let second = Arc::new(second);
    backend
        .servers
        .lock()
        .unwrap()
        .insert("another-chat-cache".into(), second.clone());
    let descriptor = backend
        .runtime_control_adapter_for(&manifest)
        .unwrap()
        .descriptor();
    assert_eq!(
        descriptor
            .capabilities
            .iter()
            .find(|cap| cap.id == crate::runtime_control::MODEL_RESIDENCY_CAPABILITY)
            .unwrap()
            .status,
        crate::werk_protocol::CapabilityStatus::Unavailable
    );
    first.child.lock().unwrap().kill().unwrap();
    first.child.lock().unwrap().wait().unwrap();
    assert_eq!(
        backend
            .runtime_control_adapter_for(&manifest)
            .unwrap()
            .descriptor()
            .instance_id,
        second.instance_id
    );
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

fn generic_tool_request() -> GenerateRequest {
    let mut req = request();
    req.tool_config = Some(super::super::ToolCallingConfig {
        tools: Some(
            serde_json::from_value(json!([{"type":"function","function":{
                "name":"lookup","parameters":{"type":"object","properties":{"a":{"type":"integer"}}}
            }}]))
            .unwrap(),
        ),
        tool_choice: Some(serde_json::from_value(json!("required")).unwrap()),
        parallel_tool_calls: Some(false),
    });
    req
}

#[tokio::test]
async fn generic_tools_cover_missing_native_parser_options_sessions_and_token_counting() {
    use tokio_stream::StreamExt;
    let fixture = Fixture::new(
        json!({"response":{"choices":[{"index":0,"message":{"role":"assistant","content":
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"a\":1}}</tool_call>"},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":10,"completion_tokens":7}}}),
    );
    let mut backend = fixture_backend(&fixture);
    backend.test_probe.as_mut().unwrap().tools = false;
    let manifest = fixture_model_for_backend(&fixture);
    let model_dir = resolve_model_dir(&fixture.store, &manifest).unwrap();
    let request = generic_tool_request();
    assert!(backend.supports_tool_calling(&manifest, false));
    assert_eq!(
        backend.count_tokens(&manifest, request.clone()).unwrap(),
        37
    );
    let response = backend.generate(&manifest, request.clone()).unwrap();
    assert_eq!(response.finish_reason, "tool_calls");
    assert_eq!(
        response.assistant_message.unwrap().tool_calls.unwrap()[0]
            .function
            .name,
        "lookup"
    );
    let count_body: Value =
        serde_json::from_slice(&fs::read(model_dir.join("tokenize.json")).unwrap()).unwrap();
    let chat_body: Value =
        serde_json::from_slice(&fs::read(model_dir.join("request.json")).unwrap()).unwrap();
    assert_eq!(count_body["messages"], chat_body["messages"]);
    assert!(chat_body.get("tools").is_none());
    assert!(
        chat_body["messages"]
            .to_string()
            .contains("werk-tool-call-v1")
    );
    let session = backend
        .start_chat_session(&manifest, None)
        .unwrap()
        .unwrap();
    assert_eq!(
        session.generate(request.clone()).unwrap().finish_reason,
        "tool_calls"
    );
    for mut stream in [
        backend.generate_stream(manifest, request.clone()),
        session.generate_stream(request),
    ] {
        let mut calls = 0;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                GenerateStreamEvent::ToolCallDelta(delta) => calls += delta.len(),
                GenerateStreamEvent::Done { finish_reason, .. } => {
                    assert_eq!(finish_reason, "tool_calls")
                }
                other => panic!("unexpected fallback event: {other:?}"),
            }
        }
        assert_eq!(calls, 1);
    }
}

#[test]
fn generic_tools_preserve_extended_options_and_native_auto_requests() {
    let fixture = Fixture::new(
        json!({"response":{"choices":[{"index":0,"message":{"role":"assistant","content":
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"a\":1}}</tool_call>"},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":10,"completion_tokens":7}}}),
    );
    let backend = fixture_backend(&fixture);
    let manifest = fixture_model_for_backend(&fixture);
    let model_dir = resolve_model_dir(&fixture.store, &manifest).unwrap();
    let mut options = std::collections::BTreeMap::new();
    options.insert("top_k".into(), json!(8));
    let value = backend
        .generate_api(&manifest, generic_tool_request(), options, None)
        .unwrap();
    assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(value["usage"]["prompt_tokens"], 10);
    let body: Value =
        serde_json::from_slice(&fs::read(model_dir.join("request.json")).unwrap()).unwrap();
    assert_eq!(body["top_k"], 8);
    assert!(body.get("tools").is_none());
    let mut native = generic_tool_request();
    native.tool_config.as_mut().unwrap().tool_choice = None;
    native.tool_config.as_mut().unwrap().parallel_tool_calls = None;
    let (prepared, policy) = backend.prepare_tool_request(&manifest, native).unwrap();
    assert!(policy.is_none());
    assert!(prepared.tool_config.is_some());
    backend.count_tokens(&manifest, prepared).unwrap();
    let native_count: Value =
        serde_json::from_slice(&fs::read(model_dir.join("tokenize.json")).unwrap()).unwrap();
    assert_eq!(native_count["tools"][0]["function"]["name"], "lookup");

    let (tx, mut rx) = mpsc::channel(2);
    let returned = backend
        .generate_api(
            &manifest,
            generic_tool_request(),
            Default::default(),
            Some(tx),
        )
        .unwrap();
    assert!(returned.is_null());
    let chunk = rx.blocking_recv().unwrap().unwrap();
    assert_eq!(chunk["object"], "chat.completion.chunk");
    assert!(chunk["choices"][0].get("message").is_none());
    assert_eq!(chunk["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
    assert_eq!(
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(chunk["choices"][0]["finish_reason"], "tool_calls");
    assert!(rx.blocking_recv().is_none());
}

#[test]
fn generic_tools_preserve_runtime_errors_and_reject_malformed_native_envelopes() {
    let fixture = Fixture::new(json!({"chat_http_error":true}));
    let backend = fixture_backend(&fixture);
    let manifest = fixture_model_for_backend(&fixture);
    let error = backend
        .generate(&manifest, generic_tool_request())
        .unwrap_err();
    assert!(format!("{error:#}").contains("quantization kernel failed"));
    let (_, policy) = backend
        .prepare_tool_request(&manifest, generic_tool_request())
        .unwrap();
    for mut value in [
        json!({"choices":[null]}),
        json!({"choices":[{"message":null}]}),
    ] {
        assert!(apply_generic_api_tools(&mut value, policy.as_ref().unwrap()).is_err());
    }
}

#[test]
fn generic_api_never_promotes_interrupted_tool_calls_including_streaming() {
    for reason in ["length", "content_filter"] {
        let fixture = Fixture::new(json!({"response":{"choices":[{"index":0,
            "message":{"role":"assistant","content":
                "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"a\":1}}</tool_call>"},
            "finish_reason":reason}],"usage":{"prompt_tokens":10,"completion_tokens":7}}}));
        let backend = fixture_backend(&fixture);
        let manifest = fixture_model_for_backend(&fixture);
        let result =
            backend.generate_api(&manifest, generic_tool_request(), Default::default(), None);
        assert!(
            result.is_err(),
            "{reason} must not become an executable tool call"
        );
        let (tx, mut rx) = mpsc::channel(2);
        let result = backend.generate_api(
            &manifest,
            generic_tool_request(),
            Default::default(),
            Some(tx),
        );
        assert!(
            result.is_err(),
            "{reason} must not become an executable streamed tool call"
        );
        assert!(
            rx.blocking_recv().is_none(),
            "no executable delta may escape validation"
        );
    }
}

#[test]
fn invalid_tool_options_fail_before_any_worker_or_chat_starts() {
    let fixture = Fixture::new(json!({}));
    let backend = fixture_backend(&fixture);
    let manifest = fixture_model_for_backend(&fixture);
    for invalid in ["required-without-tools", "unknown-name", "strict"] {
        let mut request = generic_tool_request();
        let config = request.tool_config.as_mut().unwrap();
        match invalid {
            "required-without-tools" => config.tools = None,
            "unknown-name" => {
                config.tool_choice = Some(
                    serde_json::from_value(
                        json!({"type":"function","function":{"name":"missing"}}),
                    )
                    .unwrap(),
                )
            }
            _ => config.tools.as_mut().unwrap()[0].function.strict = Some(true),
        }
        assert!(backend.generate(&manifest, request).is_err());
        assert!(backend.servers.lock().unwrap().is_empty());
        assert!(!fixture.model.join("chat_started").exists());
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

#[tokio::test]
#[cfg(unix)]
async fn persistent_cli_and_server_send_identical_multiturn_generation_requests() {
    use tokio_stream::StreamExt;
    let (fixture, mut backend, manifest) = server_cache_fixture(json!({"version":"0.6.4"}));
    let invocation = backend.invocation.as_mut().unwrap();
    invocation.expert_cache_bytes = Some(8 * 1024_u64.pow(3));
    invocation.thinking = Some(false);
    let directory = resolve_model_dir(&fixture.store, &manifest).unwrap();
    let cache = fixture.root.join("parity-chat-cache");
    fs::create_dir(&cache).unwrap();
    let cli = backend
        .start_persistent_chat_session(&manifest, Some(42), &cache)
        .unwrap()
        .unwrap();
    let mut req = request();
    req.messages = serde_json::from_value(json!([
        {"role":"user","content":"Ein Satz über Rust."},
        {"role":"assistant","content":"Rust bietet Speichersicherheit."},
        {"role":"user","content":"Und über Python?"}
    ]))
    .unwrap();
    req.temperature = Some(0.0);
    req.seed = Some(42);
    let mut stream = cli.generate_stream(req.clone());
    let mut done = false;
    while let Some(event) = stream.next().await {
        done |= matches!(event.unwrap(), GenerateStreamEvent::Done { .. });
    }
    assert!(done);
    drop(stream);
    let cli_body: Value =
        serde_json::from_slice(&fs::read(directory.join("request.json")).unwrap()).unwrap();
    let mut stream = backend.generate_stream(manifest, req);
    let mut done = false;
    while let Some(event) = stream.next().await {
        done |= matches!(event.unwrap(), GenerateStreamEvent::Done { .. });
    }
    assert!(done);
    let server_body: Value =
        serde_json::from_slice(&fs::read(directory.join("request.json")).unwrap()).unwrap();
    assert_eq!(cli_body, server_body);
    assert_eq!(cli_body["messages"].as_array().unwrap().len(), 3);
    assert_eq!(cli_body["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(cli_body["seed"], 42);
}

#[test]
#[cfg(unix)]
fn auto_expert_cache_activates_only_for_probe_verified_models() {
    for supported in [false, true] {
        let (fixture, mut backend, manifest) = server_cache_fixture(json!({"version":"0.6.4"}));
        backend.invocation.as_mut().unwrap().expert_cache_bytes = Some(0);
        if supported {
            backend.test_probe.as_mut().unwrap().runtime["expert_offload"] =
                json!({"cache_budget_mode":"auto"});
        }
        backend.prepare(&manifest).unwrap();
        let servers = backend.servers.lock().unwrap();
        assert_eq!(servers.len(), 1);
        let server = servers.values().next().unwrap();
        assert_eq!(server.expert_offload, supported);
        assert_eq!(server.expert_cache_bytes, Some(0));
        drop(servers);
        drop(backend);
        drop(fixture);
    }
}

#[test]
#[cfg(unix)]
fn auto_native_experts_require_verified_text_adapter_and_preserve_explicit_limits() {
    for (verified, budget, succeeds) in [(true, 0, true), (false, 0, false), (true, 1024, false)] {
        let (fixture, mut backend, manifest) =
            server_cache_fixture(json!({"version":"0.6.4", "native_experts":true}));
        backend.invocation.as_mut().unwrap().expert_cache_bytes = Some(budget);
        backend.test_probe.as_mut().unwrap().runtime["expert_offload"] = if verified {
            json!({"loader":"installed_native_text_port"})
        } else {
            json!({"cache_budget_mode":"auto"})
        };
        let result = backend.prepare(&manifest);
        assert_eq!(result.is_ok(), succeeds, "{result:?}");
        if succeeds {
            let servers = backend.servers.lock().unwrap();
            assert!(!servers.values().next().unwrap().expert_offload);
        }
        drop(backend);
        drop(fixture);
    }
}

#[test]
fn api_options_validate_without_starting_an_omlx_worker() {
    let fixture = Fixture::new(json!({}));
    let backend = fixture_backend(&fixture);
    let manifest = fixture_model_for_backend(&fixture);
    let mut options = std::collections::BTreeMap::from([("reasoning_effort".into(), json!("low"))]);
    backend
        .validate_api_options(&manifest, &request(), &options)
        .unwrap();
    options.insert("logprobs".into(), json!(true));
    let error = backend
        .validate_api_options(&manifest, &request(), &options)
        .unwrap_err();
    assert!(error.to_string().contains("does not support logprobs"));
    assert!(backend.servers.lock().unwrap().is_empty());
    assert!(backend.model_probes.lock().unwrap().is_empty());
}

#[test]
fn api_reasoning_effort_toggles_each_request_without_changing_worker_defaults() {
    let fixture = Fixture::new(json!({"response": {"choices": [{"index": 0,
        "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 4}}}));
    let mut backend = fixture_backend(&fixture);
    let invocation = backend.invocation.as_mut().unwrap();
    invocation.reasoning_effort = Some(OmlxReasoningEffort::High);
    invocation.thinking = Some(false);
    let manifest = fixture_model_for_backend(&fixture);
    let model_dir = resolve_model_dir(&fixture.store, &manifest).unwrap();
    let mut first_worker = None;
    for (effort, thinking) in [
        (Some(json!("low")), true),
        (Some(json!("none")), false),
        (Some(json!(0.35)), false),
        (Some(json!("adaptive")), false),
        (Some(json!("max")), true),
        (None, false),
    ] {
        let options: std::collections::BTreeMap<String, Value> = effort
            .clone()
            .map(|value| ("reasoning_effort".into(), value))
            .into_iter()
            .collect();
        assert_eq!(
            backend
                .count_api_tokens(&manifest, request(), options.clone())
                .unwrap(),
            37
        );
        let counted: Value =
            serde_json::from_slice(&fs::read(model_dir.join("tokenize.json")).unwrap()).unwrap();
        backend
            .generate_api(&manifest, request(), options, None)
            .unwrap();
        let body: Value =
            serde_json::from_slice(&fs::read(model_dir.join("request.json")).unwrap()).unwrap();
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], thinking);
        assert_eq!(
            counted["chat_template_kwargs"],
            body["chat_template_kwargs"]
        );
        assert_eq!(counted["reasoning_effort"], body["reasoning_effort"]);
        assert_eq!(counted["messages"], body["messages"]);
        match effort {
            Some(value) if value == json!("none") => {
                assert!(body.get("reasoning_effort").is_none());
                assert!(
                    body["chat_template_kwargs"]
                        .get("reasoning_effort")
                        .is_none()
                );
            }
            Some(value) => {
                assert_eq!(body["reasoning_effort"], value);
                assert_eq!(body["chat_template_kwargs"]["reasoning_effort"], value);
            }
            None => {
                assert_eq!(body["reasoning_effort"], "high");
                assert_eq!(body["chat_template_kwargs"]["reasoning_effort"], "high");
            }
        }
        let servers = backend.servers.lock().unwrap();
        assert_eq!(servers.len(), 1);
        let worker = servers.values().next().unwrap();
        assert_eq!(worker.reasoning_effort, Some(OmlxReasoningEffort::High));
        assert_eq!(worker.thinking, Some(false));
        if let Some(first) = &first_worker {
            assert!(Arc::ptr_eq(first, worker));
        } else {
            first_worker = Some(worker.clone());
        }
    }
}

#[test]
fn reasoning_effort_controls_native_payload_and_preserves_worker_reuse() {
    assert_eq!(reasoning_effort_enabled(None).unwrap(), None);
    for (name, effort) in [
        ("low", OmlxReasoningEffort::Low),
        ("high", OmlxReasoningEffort::High),
        ("max", OmlxReasoningEffort::Max),
    ] {
        assert_eq!(
            reasoning_effort_enabled(Some(name.into())).unwrap(),
            Some(effort)
        );
        let options: crate::openai::ChatRuntimeOptions =
            serde_json::from_value(json!({"omlx":{"reasoning_effort":name}})).unwrap();
        assert!(!options.omlx.as_ref().unwrap().is_empty());
        let fixture = Fixture::new(json!({}));
        let mut base = fixture_backend(&fixture);
        base.invocation.as_mut().unwrap().reasoning_effort = Some(OmlxReasoningEffort::Max);
        let configured = base.configured_for_chat(&options).unwrap();
        assert_eq!(configured.request_reasoning_effort, Some(effort));
        assert_eq!(base.request_reasoning_effort, None);
        assert_eq!(configured.cache_identity(), base.cache_identity());
        assert!(Arc::ptr_eq(&base.servers, &configured.servers));
        for thinking in [None, Some(false), Some(true)] {
            for stream in [false, true] {
                let body =
                    omlx_chat_completion_body("model", &request(), stream, thinking, Some(effort));
                assert_eq!(body["reasoning_effort"], name);
                assert_eq!(body["chat_template_kwargs"]["reasoning_effort"], name);
                assert_eq!(
                    body["chat_template_kwargs"].get("enable_thinking"),
                    thinking.map(|value| json!(value)).as_ref()
                );
            }
        }
    }
    for invalid in ["", "medium", "none", "0", " low", "LOW"] {
        assert!(reasoning_effort_enabled(Some(invalid.into())).is_err());
        assert!(
            serde_json::from_value::<crate::openai::OmlxChatOptions>(
                json!({"reasoning_effort":invalid})
            )
            .is_err()
        );
    }
}

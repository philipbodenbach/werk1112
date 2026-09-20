//! `run` frontend for an existing `werk serve`; owns no runtime or worker.
use super::{openai_transport::*, *};
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::{sync::Arc, time::Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
pub(crate) struct WerkServerClient {
    url: String,
    api_key: Option<String>,
    options: Option<crate::openai::ChatRuntimeOptions>,
}

impl WerkServerClient {
    pub(crate) fn new(url: &str, api_key: Option<String>) -> Result<Self> {
        // Same explicit host:port contract as the existing control-plane client.
        crate::werk_protocol::WerkProtocolClient::new(url, api_key.clone())?;
        Ok(Self {
            url: url.trim_end_matches('/').into(),
            api_key,
            options: None,
        })
    }

    fn complete(
        &self,
        model: &ModelManifest,
        request: GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse> {
        let started = Instant::now();
        let mut body = chat_completion_body(&model.id, &request, tx.is_some());
        if let Some(options) = &self.options {
            body["werk"] = serde_json::to_value(options)?;
        }
        let mut response = request_with_bearer_cancellable(
            &self.url,
            "/v1/chat/completions",
            "POST",
            Some(&body),
            None,
            self.api_key.as_deref(),
            tx.clone(),
        )?;
        let mut completion = OpenAiCompletion::default();
        let mut timings = GenerationTimings::default();
        let mut diagnostics = vec![
            "worker ownership: existing werk serve; live prefix cache is server-managed".into(),
        ];
        let mut finished = false;
        let mut consume = |value: &Value| -> Result<()> {
            if let Some(error) = value.get("error") {
                bail!("Werk server: {error}");
            }
            update_completion_from_event(&mut completion, value);
            if let Some(native) = value.pointer("/werk/timings") {
                timings = serde_json::from_value(native.clone())
                    .context("invalid Werk timing metadata")?;
            }
            if let Some(native) = value.pointer("/werk/backend_diagnostics") {
                diagnostics.extend(serde_json::from_value::<Vec<String>>(native.clone())?);
            }
            if tx.is_some() {
                if let Some(chunk) = delta_content(value) {
                    if !chunk.is_empty() {
                        if completion.first_token_seconds == 0.0 {
                            completion.first_token_seconds = started.elapsed().as_secs_f64();
                        }
                        completion.text.push_str(&chunk);
                        append_assistant_content(&mut completion.assistant_content, &chunk);
                        send_text_chunk(&tx, chunk)?;
                    }
                }
                if let Some(calls) = delta_tool_calls(value)? {
                    completion.saw_tool_call_delta = true;
                    if completion.first_token_seconds == 0.0 {
                        completion.first_token_seconds = started.elapsed().as_secs_f64();
                    }
                    send_tool_call_delta(&tx, calls)?;
                }
            } else {
                update_completion_from_message(&mut completion, value)?;
                completion.first_token_seconds = started.elapsed().as_secs_f64();
            }
            Ok(())
        };
        if tx.is_some() {
            let mut sse = SseAccumulator::default();
            stream_body(&mut response, |bytes| {
                sse.push(bytes, |event| {
                    if event == "[DONE]" {
                        finished = true;
                        return Ok(());
                    }
                    consume(&serde_json::from_str::<Value>(event)?)
                })
            })?;
            ensure!(
                finished && !completion.finish_reason.is_empty(),
                "Werk server stream ended without a completed response"
            );
        } else {
            let mut bytes = Vec::new();
            stream_body(&mut response, |chunk| {
                ensure!(
                    bytes.len() + chunk.len() <= 128 * 1024 * 1024,
                    "Werk response exceeds 128 MiB"
                );
                bytes.extend_from_slice(chunk);
                Ok(())
            })?;
            consume(&serde_json::from_slice::<Value>(&bytes)?)?;
        }
        ensure_visible_completion(&completion)?;
        // Client-observed latency includes HTTP and server queue/startup. Native
        // prefill/decode figures stay native; never estimate rates from total time.
        timings.first_token_seconds = completion.first_token_seconds;
        timings.total_seconds = started.elapsed().as_secs_f64();
        let assistant_message = Some(completion.assistant_message());
        Ok(GenerateResponse {
            text: completion.text,
            assistant_message,
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            finish_reason: completion.finish_reason,
            timings,
            backend_diagnostics: diagnostics,
        })
    }
}

impl GenerationBackend for WerkServerClient {
    fn supports_tool_calling(&self, _: &ModelManifest, _: bool) -> bool {
        true
    }
    fn with_chat_options(
        &self,
        _: &ModelManifest,
        options: &crate::openai::ChatRuntimeOptions,
    ) -> Result<Arc<dyn GenerationBackend>> {
        let mut client = self.clone();
        client.options = Some(options.clone());
        Ok(Arc::new(client))
    }
    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        self.complete(manifest, request, None)
    }
    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let client = self.clone();
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let result = client.complete(&manifest, request, Some(tx.clone()));
            send_stream_result(tx, result);
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{
        io::{BufRead, BufReader, Read, Write},
        net::TcpListener,
        thread,
    };
    use tokio_stream::StreamExt;

    fn manifest() -> ModelManifest {
        serde_json::from_value(json!({"id":"test", "source":{"kind":"local_path","path":"test"}, "format":"gguf", "backend":"test", "created_unix":1, "files":[]})).unwrap()
    }
    fn request() -> GenerateRequest {
        GenerateRequest {
            prompt: "unused".into(),
            messages: serde_json::from_value(json!([
            {"role":"user", "content":"Remember 37"}, {"role":"assistant","content":"OK"},
            {"role":"user","content":"Which number?"}]))
            .unwrap(),
            image_urls: vec![],
            max_tokens: 12,
            temperature: Some(0.0),
            top_p: None,
            stop: vec![],
            seed: Some(17),
            stream_granularity: StreamGranularity::Token,
            verbose: false,
            debug: false,
            tool_config: None,
        }
    }
    fn fixture(events: String, status: u16) -> (String, thread::JoinHandle<Value>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let join = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(socket);
            let mut length = 0;
            let mut authorization = false;
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            assert_eq!(line.trim(), "POST /v1/chat/completions HTTP/1.1");
            loop {
                line.clear();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(v) = line.strip_prefix("Content-Length:") {
                    length = v.trim().parse().unwrap();
                }
                authorization |= line.trim() == "Authorization: Bearer fixture-key";
            }
            assert!(authorization);
            let mut bytes = vec![0; length];
            reader.read_exact(&mut bytes).unwrap();
            let mut socket = reader.into_inner();
            write!(
                socket,
                "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{events}",
                events.len()
            )
            .unwrap();
            serde_json::from_slice(&bytes).unwrap()
        });
        (url, join)
    }
    #[tokio::test]
    async fn streams_from_existing_server_with_history_auth_and_native_timings() {
        let events = format!(
            "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"choices":[{"delta":{"content":"37"}}]}),
            json!({"choices":[{"finish_reason":"stop","delta":{}}],"werk":{"timings":{"cached_prompt_tokens":24,"prompt_seconds":0.25,"decode_seconds":0.5},"backend_diagnostics":["existing worker"]}}),
            json!({"choices":[],"usage":{"prompt_tokens":53,"completion_tokens":1}})
        );
        let (url, join) = fixture(events, 200);
        let client = WerkServerClient::new(&url, Some("fixture-key".into())).unwrap();
        let mut stream = client.generate_stream(manifest(), request());
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), GenerateStreamEvent::TextChunk(ref s) if s == "37")
        );
        match stream.next().await.unwrap().unwrap() {
            GenerateStreamEvent::Done {
                prompt_tokens,
                timings,
                backend_diagnostics,
                ..
            } => {
                assert_eq!(prompt_tokens, 53);
                assert_eq!(timings.cached_prompt_tokens, Some(24));
                assert_eq!(timings.prompt_seconds, 0.25);
                assert_eq!(timings.decode_seconds, 0.5);
                assert!(timings.first_token_seconds > 0.0);
                assert!(backend_diagnostics.contains(&"existing worker".into()));
            }
            _ => panic!("missing completion"),
        }
        let body = join.join().unwrap();
        assert_eq!(body["messages"].as_array().unwrap().len(), 3);
        assert_eq!(body["messages"][2]["content"], "Which number?");
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["seed"], 17);
    }
    #[tokio::test]
    async fn rejects_truncated_stream_and_http_errors() {
        for (events, status) in [
            (
                "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
                200,
            ),
            ("{\"error\":\"unauthorized\"}", 401),
        ] {
            let (url, join) = fixture(events.into(), status);
            let client = WerkServerClient::new(&url, Some("fixture-key".into())).unwrap();
            let events: Vec<_> = client
                .generate_stream(manifest(), request())
                .collect()
                .await;
            assert!(events.iter().any(Result::is_err));
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, Ok(GenerateStreamEvent::Done { .. })))
            );
            join.join().unwrap();
        }
    }
    #[test]
    fn rejects_invalid_server_url() {
        for url in [
            "https://example.com",
            "http://host:0",
            "http://user:secret@host:80",
            "http://host:80/v1",
        ] {
            assert!(WerkServerClient::new(url, None).is_err());
        }
    }
}

//! Shared OpenAI wire transport. Runtime discovery and lifecycle belong to each backend.
use super::{GenerateRequest, GenerateResponse, GenerateStreamEvent, GeneratedAssistantMessage};
use crate::openai::{ChatCompletionToolCall, ChatCompletionToolCallDelta};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream, ToSocketAddrs},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

pub(super) fn apply_api_options(
    body: &mut Value,
    options: std::collections::BTreeMap<String, Value>,
    allowed: &[&str],
) -> Result<()> {
    validate_api_options(&options, allowed)?;
    for (key, value) in options {
        body[key] = value;
    }
    Ok(())
}
pub(super) fn validate_api_options(
    options: &std::collections::BTreeMap<String, Value>,
    allowed: &[&str],
) -> Result<()> {
    for key in options.keys() {
        if !allowed.contains(&key.as_str()) {
            let name = if key == "__werk_matched_stop" {
                "stop_sequences with matched-stop metadata"
            } else {
                key.as_str()
            };
            bail!("the selected runtime does not support {name}");
        }
    }
    Ok(())
}

pub(super) fn generate_api(
    url: &str,
    bearer: Option<&str>,
    mut body: Value,
    tx: Option<mpsc::Sender<Result<Value, String>>>,
) -> Result<Value> {
    let matched_stops = body.as_object_mut().unwrap().remove("__werk_matched_stop");
    let mut watcher = None;
    let cancel = if let Some(raw) = tx.clone() {
        let (cancel, receiver) = mpsc::channel(1);
        watcher = Some(SocketCancellation(
            tokio::runtime::Handle::try_current()?.spawn(async move {
                raw.closed().await;
                drop(receiver);
            }),
        ));
        Some(cancel)
    } else {
        None
    };
    let mut response = request_with_bearer_cancellable(
        url,
        "/v1/chat/completions",
        "POST",
        Some(&body),
        None,
        bearer,
        cancel,
    )?;
    if response.status != 200 {
        bail!("native chat API returned HTTP {}", response.status);
    }
    if body.get("response_format").is_some()
        && response
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("warning"))
    {
        bail!("native runtime could not enforce the requested structured output");
    }
    let result = if let Some(tx) = tx {
        let mut accumulator = SseAccumulator::default();
        let mut done = false;
        stream_body(&mut response, |chunk| {
            accumulator.push(chunk, |event| {
                if event.trim() == "[DONE]" {
                    done = true;
                    return Ok(());
                }
                if done {
                    bail!("native API emitted data after DONE");
                }
                let mut value: Value = serde_json::from_str(event)
                    .context("native API returned invalid stream JSON")?;
                if value.get("error").is_some() {
                    bail!("native API stream error: {}", value["error"]);
                }
                validate_api_response(&value, &body, matched_stops.as_ref(), true)?;
                normalize_matched_stop(&mut value, matched_stops.as_ref());
                tx.blocking_send(Ok(value))
                    .map_err(|_| anyhow!("stream receiver closed"))
            })
        })?;
        if !done {
            bail!("native API stream ended without DONE");
        }
        Value::Null
    } else {
        let mut bytes = Vec::new();
        stream_body(&mut response, |chunk| {
            if bytes.len() + chunk.len() > 16 * 1024 * 1024 {
                bail!("native API response exceeds 16 MiB");
            }
            bytes.extend_from_slice(chunk);
            Ok(())
        })?;
        let mut value: Value =
            serde_json::from_slice(&bytes).context("native API returned invalid JSON")?;
        if value.get("error").is_some() {
            bail!("native API error: {}", value["error"]);
        }
        validate_api_response(&value, &body, matched_stops.as_ref(), false)?;
        normalize_matched_stop(&mut value, matched_stops.as_ref());
        value
    };
    drop(watcher);
    Ok(result)
}

fn validate_api_response(
    value: &Value,
    body: &Value,
    matched: Option<&Value>,
    stream: bool,
) -> Result<()> {
    let choices = value
        .get("choices")
        .and_then(Value::as_array)
        .context("native response is missing choices")?;
    let n = body.get("n").and_then(Value::as_u64).unwrap_or(1);
    if !stream {
        let indices: std::collections::HashSet<_> =
            choices.iter().filter_map(|c| c["index"].as_u64()).collect();
        if choices.len() != n as usize || indices.len() != n as usize {
            bail!("native API returned incomplete or duplicate choices");
        }
    }
    for choice in choices {
        ensure_choice_index(choice, n)?;
        if !stream && (!choice["finish_reason"].is_string() || !choice["message"].is_object()) {
            bail!("native API returned an incomplete completion");
        }
        if body.get("logprobs") == Some(&json!(true)) {
            let content = if stream {
                &choice["delta"]["content"]
            } else {
                &choice["message"]["content"]
            };
            if content.as_str().is_some_and(|s| !s.is_empty())
                && !choice["logprobs"]["content"].is_array()
            {
                bail!("native runtime did not provide requested logprobs");
            }
        }
        if let Some(stops) = matched
            && choice["finish_reason"].is_string()
        {
            let actual = choice
                .get("stop_reason")
                .context("native runtime cannot report matched stop sequences")?;
            if !actual.is_null() && !actual.is_string() && actual.as_u64().is_none() {
                bail!("native runtime reported invalid stop metadata");
            }
            if let Some(stop) = actual.as_str() {
                if choice["finish_reason"] != "stop" {
                    bail!("native stop metadata disagrees with finish reason");
                }
                let requested = stops
                    .as_array()
                    .is_some_and(|s| s.iter().any(|v| v.as_str() == Some(stop)));
                let template = body["stop"]
                    .as_array()
                    .is_some_and(|s| s.iter().any(|v| v.as_str() == Some(stop)));
                if !requested && !template {
                    bail!("native runtime reported an unexpected stop sequence");
                }
            }
        }
    }
    Ok(())
}
fn normalize_matched_stop(value: &mut Value, matched: Option<&Value>) {
    let Some(stops) = matched.and_then(Value::as_array) else {
        return;
    };
    if let Some(choices) = value["choices"].as_array_mut() {
        for choice in choices {
            if choice["stop_reason"].is_string() && !stops.contains(&choice["stop_reason"]) {
                // Model-template terminators are end_turn, not client stop_sequences.
                choice["stop_reason"] = Value::Null;
            }
        }
    }
}
fn ensure_choice_index(choice: &Value, n: u64) -> Result<()> {
    if !choice
        .get("index")
        .and_then(Value::as_u64)
        .is_some_and(|i| i < n)
    {
        bail!("native response has an invalid choice index");
    }
    Ok(())
}

pub(super) fn tokenization_json(base_url: &str, path: &str, body: &Value) -> Result<Value> {
    let mut response = request(
        base_url,
        path,
        "POST",
        Some(body),
        Some(Duration::from_secs(60)),
    )?;
    if response.status != 200 {
        bail!("native tokenizer returned HTTP {}", response.status);
    }
    let mut bytes = Vec::new();
    stream_body(&mut response, |chunk| {
        if bytes.len().saturating_add(chunk.len()) > 16 * 1024 * 1024 {
            bail!("native tokenizer response exceeds 16 MiB");
        }
        bytes.extend_from_slice(chunk);
        Ok(())
    })?;
    serde_json::from_slice(&bytes).context("native tokenizer returned invalid JSON")
}

#[derive(Default)]
pub(super) struct OpenAiCompletion {
    pub(super) text: String,
    pub(super) assistant_content: Option<Option<String>>,
    pub(super) tool_calls: Option<Vec<ChatCompletionToolCall>>,
    pub(super) saw_tool_call_delta: bool,
    pub(super) saw_reasoning_content: bool,
    pub(super) prompt_tokens: usize,
    pub(super) completion_tokens: usize,
    pub(super) prompt_seconds: f64,
    pub(super) decode_seconds: f64,
    pub(super) first_token_seconds: f64,
    pub(super) finish_reason: String,
}

impl OpenAiCompletion {
    pub(super) fn assistant_message(&self) -> GeneratedAssistantMessage {
        GeneratedAssistantMessage {
            content: self
                .assistant_content
                .clone()
                .unwrap_or_else(|| Some(self.text.clone())),
            tool_calls: self.tool_calls.clone(),
        }
    }
}

pub(super) struct HttpResponse {
    pub(super) status: u16,
    pub(super) headers: Vec<(String, String)>,
    pub(super) reader: BufReader<TcpStream>,
    pub(super) deadline: Option<HttpDeadline>,
    _cancellation: Option<SocketCancellation>,
}

struct SocketCancellation(tokio::task::JoinHandle<()>);

impl Drop for SocketCancellation {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct HttpDeadline {
    pub(super) started: Instant,
    pub(super) timeout: Duration,
}

impl HttpDeadline {
    pub(super) fn new(timeout: Duration) -> Self {
        Self {
            started: Instant::now(),
            timeout,
        }
    }

    pub(super) fn remaining(self) -> Result<Duration> {
        self.timeout
            .checked_sub(self.started.elapsed())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| anyhow!("OpenAI backend HTTP probe timed out"))
    }

    pub(super) fn remaining_capped(self, cap: Duration) -> Result<Duration> {
        Ok(self.remaining()?.min(cap))
    }
}

#[derive(Default)]
pub(super) struct SseAccumulator {
    pub(super) pending: Vec<u8>,
}

impl SseAccumulator {
    pub(super) fn push<F>(&mut self, bytes: &[u8], mut on_event: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        self.pending.extend_from_slice(bytes);
        if self.pending.len() > 4 * 1024 * 1024 {
            bail!("OpenAI SSE event exceeds 4 MiB");
        }
        while let Some(index) = find_sse_boundary(&self.pending) {
            let event = self.pending.drain(..index).collect::<Vec<_>>();
            while matches!(self.pending.first(), Some(b'\r' | b'\n')) {
                self.pending.remove(0);
            }
            let event = std::str::from_utf8(&event).context("OpenAI SSE event is not UTF-8")?;
            let mut data = Vec::new();
            let mut is_error = false;
            for line in event.lines() {
                if line
                    .strip_prefix("event:")
                    .is_some_and(|name| name.trim() == "error")
                {
                    is_error = true;
                }
            }
            for line in event.lines() {
                if let Some(value) = line.strip_prefix("data:") {
                    data.push(value.strip_prefix(' ').unwrap_or(value));
                }
            }
            if is_error {
                bail!("OpenAI upstream SSE error: {}", data.join("\n"));
            }
            if !data.is_empty() {
                on_event(&data.join("\n"))?;
            }
        }
        Ok(())
    }
}

pub(super) fn chat_completion_body(
    model_name: &str,
    request: &GenerateRequest,
    stream: bool,
) -> Value {
    let messages = if request.messages.is_empty() {
        json!([{
            "role": "user",
            "content": request.prompt,
        }])
    } else {
        chat_messages(&request.messages)
    };
    let mut body = json!({
        "model": model_name,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "stream": stream,
    });
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = json!(top_p);
    }
    if !request.stop.is_empty() {
        body["stop"] = json!(request.stop);
    }
    if let Some(seed) = request.seed {
        body["seed"] = json!(seed);
    }
    if let Some(tool_config) = &request.tool_config {
        if let Some(tools) = &tool_config.tools {
            body["tools"] = json!(tools);
        }
        if let Some(tool_choice) = &tool_config.tool_choice {
            body["tool_choice"] = json!(tool_choice);
        }
        if let Some(parallel_tool_calls) = tool_config.parallel_tool_calls {
            body["parallel_tool_calls"] = json!(parallel_tool_calls);
        }
    }
    body
}

pub(super) fn chat_messages(messages: &[crate::openai::ChatMessage]) -> Value {
    let mut messages =
        serde_json::to_value(messages).expect("ChatMessage serialization cannot fail");
    if let Some(messages) = messages.as_array_mut() {
        for message in messages {
            let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("input_image") {
                    part["type"] = json!("image_url");
                }
                if matches!(part.get("type").and_then(Value::as_str), Some("image_url"))
                    && let Some(url) = part.get("image_url").and_then(Value::as_str)
                {
                    part["image_url"] = json!({"url": url});
                }
            }
        }
    }
    messages
}

pub(super) fn update_completion_from_event(completion: &mut OpenAiCompletion, value: &Value) {
    if delta_has_reasoning_content(value) {
        completion.saw_reasoning_content = true;
    }
    if let Some(choice) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        && let Some(reason) = choice.get("finish_reason").and_then(Value::as_str)
        && !reason.is_empty()
    {
        completion.finish_reason = reason.to_string();
    }
    if let Some(usage) = value.get("usage") {
        if let Some(tokens) = usage.get("prompt_tokens").and_then(Value::as_u64) {
            completion.prompt_tokens = tokens as usize;
        }
        if let Some(tokens) = usage.get("completion_tokens").and_then(Value::as_u64) {
            completion.completion_tokens = tokens as usize;
        }
    }
}

pub(super) fn update_completion_from_message(
    completion: &mut OpenAiCompletion,
    value: &Value,
) -> Result<()> {
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .context("OpenAI backend non-streaming response has no completion choice")?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .context("OpenAI backend non-streaming response has no assistant message")?;

    completion.saw_reasoning_content |= ["reasoning", "reasoning_content"].iter().any(|field| {
        message
            .get(*field)
            .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
    });

    completion.assistant_content = match message.get("content") {
        Some(Value::Null) => Some(None),
        Some(Value::String(content)) => {
            completion.text.clone_from(content);
            Some(Some(content.clone()))
        }
        Some(_) => bail!("OpenAI backend assistant message content must be a string or null"),
        None => None,
    };
    completion.tool_calls = match message.get("tool_calls") {
        Some(Value::Null) | None => None,
        Some(tool_calls) => Some(
            serde_json::from_value(tool_calls.clone())
                .context("OpenAI backend returned invalid assistant message.tool_calls")?,
        ),
    };
    if completion.assistant_content.is_none()
        && completion
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty())
    {
        completion.assistant_content = Some(None);
    }
    Ok(())
}

pub(super) fn append_assistant_content(content: &mut Option<Option<String>>, chunk: &str) {
    match content {
        Some(Some(current)) => current.push_str(chunk),
        _ => *content = Some(Some(chunk.to_string())),
    }
}

pub(super) fn delta_tool_calls(value: &Value) -> Result<Option<Vec<ChatCompletionToolCallDelta>>> {
    let Some(tool_calls) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("tool_calls"))
    else {
        return Ok(None);
    };
    if tool_calls.is_null() {
        return Ok(None);
    }
    serde_json::from_value(tool_calls.clone())
        .map(Some)
        .context("OpenAI backend returned invalid delta.tool_calls")
}

pub(super) fn delta_has_reasoning_content(value: &Value) -> bool {
    let Some(delta) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
    else {
        return false;
    };
    ["reasoning", "reasoning_content"].into_iter().any(|key| {
        delta.get(key).is_some_and(|value| match value {
            Value::String(text) => !text.is_empty(),
            Value::Null => false,
            _ => true,
        })
    })
}

pub(super) fn ensure_visible_completion(completion: &OpenAiCompletion) -> Result<()> {
    let has_tool_calls = completion.saw_tool_call_delta
        || completion
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty());
    if completion.text.trim().is_empty() && completion.saw_reasoning_content && !has_tool_calls {
        bail!(
            "OpenAI backend generated hidden reasoning but no visible answer content; increase max tokens so the model can finish its answer, or disable the configured reasoning parser/mode when hidden reasoning is not wanted"
        );
    }
    Ok(())
}

pub(super) fn delta_content(value: &Value) -> Option<String> {
    value
        .get("choices")?
        .as_array()?
        .first()?
        .get("delta")?
        .get("content")
        .and_then(Value::as_str)
        .map(str::to_string)
}

pub(super) fn finalize_completion_stats(
    completion: &mut OpenAiCompletion,
    request: &GenerateRequest,
    elapsed_seconds: f64,
) {
    if completion.prompt_tokens == 0 && !request.prompt.trim().is_empty() {
        completion.prompt_tokens = estimate_tokens(&request.prompt);
    }
    if completion.prompt_seconds <= 0.0 && completion.first_token_seconds > 0.0 {
        completion.prompt_seconds = completion.first_token_seconds;
    }
    if completion.decode_seconds <= 0.0 {
        completion.decode_seconds = if completion.first_token_seconds > 0.0
            && elapsed_seconds > completion.first_token_seconds
        {
            elapsed_seconds - completion.first_token_seconds
        } else {
            elapsed_seconds
        };
    }
    if completion.completion_tokens == 0 {
        completion.completion_tokens = estimate_tokens(&completion.text);
    }
}

pub(super) fn get_with_timeout(
    base_url: &str,
    path: &str,
    timeout: Duration,
) -> Result<HttpResponse> {
    request(base_url, path, "GET", None, Some(timeout))
}

pub(super) fn post_json(base_url: &str, path: &str, body: &Value) -> Result<HttpResponse> {
    request(base_url, path, "POST", Some(body), None)
}

pub(super) fn request(
    base_url: &str,
    path: &str,
    method: &str,
    body: Option<&Value>,
    timeout: Option<Duration>,
) -> Result<HttpResponse> {
    request_with_bearer(base_url, path, method, body, timeout, None)
}

pub(super) fn request_with_bearer(
    base_url: &str,
    path: &str,
    method: &str,
    body: Option<&Value>,
    timeout: Option<Duration>,
    bearer: Option<&str>,
) -> Result<HttpResponse> {
    request_with_bearer_cancellable(base_url, path, method, body, timeout, bearer, None)
}

pub(super) fn request_with_bearer_cancellable(
    base_url: &str,
    path: &str,
    method: &str,
    body: Option<&Value>,
    timeout: Option<Duration>,
    bearer: Option<&str>,
    tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
) -> Result<HttpResponse> {
    if tx.as_ref().is_some_and(mpsc::Sender::is_closed) {
        bail!("stream receiver closed");
    }
    let (_, host, port) = parse_http_url(base_url)?;
    let deadline = timeout.map(HttpDeadline::new);
    let mut stream = match deadline {
        Some(deadline) => connect_http_with_deadline(&host, port, deadline),
        None => connect_http(&host, port),
    }
    .with_context(|| format!("failed to connect to OpenAI backend server at {base_url}"))?;
    let cancellation = if let Some(tx) = tx {
        let socket = stream.try_clone()?;
        Some(SocketCancellation(
            tokio::runtime::Handle::try_current()?.spawn(async move {
                tx.closed().await;
                let _ = socket.shutdown(std::net::Shutdown::Both);
            }),
        ))
    } else {
        None
    };
    stream.set_nodelay(true).ok();
    let body_text = body.map(serde_json::to_string).transpose()?;
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: text/event-stream\r\n"
    );
    if let Some(bearer) = bearer {
        if bearer.chars().any(char::is_control) {
            bail!("invalid OpenAI authorization token");
        }
        request.push_str(&format!("Authorization: Bearer {bearer}\r\n"));
    }
    if let Some(body_text) = &body_text {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body_text.len()));
    }
    request.push_str("\r\n");
    write_http_bytes(&mut stream, request.as_bytes(), deadline)?;
    if let Some(body_text) = body_text {
        write_http_bytes(&mut stream, body_text.as_bytes(), deadline)?;
    }
    if let Some(deadline) = deadline {
        stream.set_write_timeout(Some(deadline.remaining()?))?;
    }
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let status_line = read_http_line(&mut reader, deadline)?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| {
            anyhow!("invalid HTTP response from OpenAI backend server: {status_line:?}")
        })?;
    let mut headers = Vec::new();
    loop {
        let line = read_http_line(&mut reader, deadline)?;
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    if status >= 400 {
        // Keep upstream error bodies useful, but bounded by size and deadline.
        let mut response = HttpResponse {
            status,
            headers,
            reader,
            deadline,
            _cancellation: cancellation,
        };
        let mut text = Vec::new();
        let _ = stream_body(&mut response, |bytes| {
            let keep = bytes.len().min(65536_usize.saturating_sub(text.len()));
            text.extend_from_slice(&bytes[..keep]);
            if keep < bytes.len() {
                bail!("upstream error body limit reached");
            }
            Ok(())
        });
        bail!(
            "OpenAI backend HTTP {status}: {}",
            String::from_utf8_lossy(&text).trim()
        );
    }
    Ok(HttpResponse {
        status,
        headers,
        reader,
        deadline,
        _cancellation: cancellation,
    })
}

pub(super) fn write_http_bytes(
    stream: &mut TcpStream,
    mut bytes: &[u8],
    deadline: Option<HttpDeadline>,
) -> Result<()> {
    if deadline.is_none() {
        stream.write_all(bytes)?;
        return Ok(());
    }
    while !bytes.is_empty() {
        let remaining = deadline
            .context("missing OpenAI backend HTTP deadline")?
            .remaining()?;
        stream.set_write_timeout(Some(remaining))?;
        let written = stream.write(bytes)?;
        if written == 0 {
            bail!("OpenAI backend HTTP connection closed while writing request");
        }
        bytes = &bytes[written..];
    }
    Ok(())
}

pub(super) fn read_http_line(
    reader: &mut BufReader<TcpStream>,
    deadline: Option<HttpDeadline>,
) -> Result<String> {
    let Some(deadline) = deadline else {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        return Ok(line);
    };

    let mut bytes = Vec::new();
    loop {
        reader
            .get_mut()
            .set_read_timeout(Some(deadline.remaining()?))?;
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }
        let count = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|index| index + 1)
            .unwrap_or(available.len());
        let found_newline = available[..count].ends_with(b"\n");
        bytes.extend_from_slice(&available[..count]);
        reader.consume(count);
        if found_newline {
            break;
        }
    }
    String::from_utf8(bytes).context("OpenAI backend returned a non-UTF-8 HTTP header")
}

pub(super) fn connect_http(host: &str, port: u16) -> Result<TcpStream> {
    let addresses = resolve_http_addresses(host, port)?;
    connect_http_addresses(host, port, &addresses, |_| Ok(HTTP_CONNECT_TIMEOUT))
}

pub(super) fn connect_http_with_deadline(
    host: &str,
    port: u16,
    deadline: HttpDeadline,
) -> Result<TcpStream> {
    let addresses = resolve_http_addresses_with_deadline(host, port, deadline)?;
    connect_http_addresses(host, port, &addresses, |_| {
        deadline.remaining_capped(HTTP_CONNECT_TIMEOUT)
    })
}

pub(super) fn resolve_http_addresses(host: &str, port: u16) -> Result<Vec<SocketAddr>> {
    let addresses = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("could not resolve OpenAI backend host {host}"))?
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        bail!("OpenAI backend host {host} resolved to no socket addresses");
    }
    Ok(addresses)
}

pub(super) fn resolve_http_addresses_with_deadline(
    host: &str,
    port: u16,
    deadline: HttpDeadline,
) -> Result<Vec<SocketAddr>> {
    let host_owned = host.to_string();
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = (host_owned.as_str(), port)
            .to_socket_addrs()
            .map(|addresses| addresses.collect::<Vec<_>>())
            .map_err(|error| error.to_string());
        let _ = sender.send(result);
    });
    let addresses = receiver
        .recv_timeout(deadline.remaining()?)
        .map_err(|error| anyhow!("timed out resolving OpenAI backend host {host}: {error}"))?
        .map_err(|error| anyhow!("could not resolve OpenAI backend host {host}: {error}"))?;
    if addresses.is_empty() {
        bail!("OpenAI backend host {host} resolved to no socket addresses");
    }
    Ok(addresses)
}

pub(super) fn connect_http_addresses<F>(
    host: &str,
    port: u16,
    addresses: &[SocketAddr],
    mut timeout_for: F,
) -> Result<TcpStream>
where
    F: FnMut(&SocketAddr) -> Result<Duration>,
{
    let mut errors = Vec::new();
    for address in addresses {
        let timeout = timeout_for(address)?;
        match TcpStream::connect_timeout(address, timeout) {
            Ok(stream) => return Ok(stream),
            Err(error) => errors.push(format!("{address}: {error}")),
        }
    }
    Err(anyhow!(
        "could not connect to {host}:{port}: {}",
        errors.join("; ")
    ))
}

pub(super) fn stream_body<F>(response: &mut HttpResponse, mut on_bytes: F) -> Result<()>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    if header_contains(&response.headers, "transfer-encoding", "chunked") {
        loop {
            let size_line = read_http_line(&mut response.reader, response.deadline)?;
            let size_text = size_line
                .trim()
                .split_once(';')
                .map(|(size, _)| size)
                .unwrap_or_else(|| size_line.trim());
            let size = usize::from_str_radix(size_text, 16)
                .with_context(|| format!("invalid chunk size from OpenAI backend: {size_text}"))?;
            if size == 0 {
                break;
            }
            let mut remaining = size;
            let mut chunk = [0u8; 8192];
            while remaining > 0 {
                let count = remaining.min(chunk.len());
                read_http_exact(&mut response.reader, &mut chunk[..count], response.deadline)?;
                on_bytes(&chunk[..count])?;
                remaining -= count;
            }
            let mut crlf = [0u8; 2];
            read_http_exact(&mut response.reader, &mut crlf, response.deadline)?;
            if crlf != *b"\r\n" {
                bail!("invalid OpenAI HTTP chunk delimiter");
            }
        }
    } else if let Some(length) = response
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.parse::<usize>().ok())
    {
        let mut remaining = length;
        let mut buffer = [0u8; 8192];
        while remaining > 0 {
            let requested = remaining.min(buffer.len());
            let count = read_http_bytes(
                &mut response.reader,
                &mut buffer[..requested],
                response.deadline,
            )?;
            if count == 0 {
                bail!(
                    "OpenAI backend HTTP response ended before Content-Length bytes were received"
                );
            }
            on_bytes(&buffer[..count])?;
            remaining -= count;
        }
    } else {
        let mut buffer = [0u8; 8192];
        loop {
            let n = read_http_bytes(&mut response.reader, &mut buffer, response.deadline)?;
            if n == 0 {
                break;
            }
            on_bytes(&buffer[..n])?;
        }
    }
    Ok(())
}

pub(super) fn read_http_bytes(
    reader: &mut BufReader<TcpStream>,
    bytes: &mut [u8],
    deadline: Option<HttpDeadline>,
) -> Result<usize> {
    if let Some(deadline) = deadline {
        reader
            .get_mut()
            .set_read_timeout(Some(deadline.remaining()?))?;
    }
    Ok(reader.read(bytes)?)
}

pub(super) fn read_http_exact(
    reader: &mut BufReader<TcpStream>,
    mut bytes: &mut [u8],
    deadline: Option<HttpDeadline>,
) -> Result<()> {
    while !bytes.is_empty() {
        let count = read_http_bytes(reader, bytes, deadline)?;
        if count == 0 {
            bail!("OpenAI backend HTTP response ended unexpectedly");
        }
        bytes = &mut bytes[count..];
    }
    Ok(())
}

pub(super) fn parse_http_url(url: &str) -> Result<(String, String, u16)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("only http OpenAI backend URLs are supported: {url}"))?;
    let (host_port, _) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = host_port
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("OpenAI backend URL has no port: {url}"))?;
    Ok(("http".to_string(), host.to_string(), port.parse()?))
}

pub(super) fn header_contains(headers: &[(String, String)], name: &str, needle: &str) -> bool {
    headers.iter().any(|(header, value)| {
        header.eq_ignore_ascii_case(name) && value.to_ascii_lowercase().contains(needle)
    })
}

pub(super) fn find_sse_boundary(bytes: &[u8]) -> Option<usize> {
    let lf = bytes.windows(2).position(|window| window == b"\n\n");
    let crlf = bytes.windows(4).position(|window| window == b"\r\n\r\n");
    lf.into_iter().chain(crlf).min()
}

pub(super) fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

pub(super) fn send_stream_result(
    tx: mpsc::Sender<Result<GenerateStreamEvent, String>>,
    result: Result<GenerateResponse>,
) {
    match result {
        Ok(response) => {
            let _ = tx.blocking_send(Ok(GenerateStreamEvent::Done {
                finish_reason: response.finish_reason,
                prompt_tokens: response.prompt_tokens,
                completion_tokens: response.completion_tokens,
                timings: response.timings,
                backend_diagnostics: response.backend_diagnostics,
            }));
        }
        Err(err) => {
            let _ = tx.blocking_send(Err(format!("{err:#}")));
        }
    }
}

pub(super) fn send_text_chunk(
    tx: &Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    chunk: String,
) -> Result<()> {
    if let Some(tx) = tx {
        tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(chunk)))
            .map_err(|err| anyhow!("stream receiver closed: {err}"))?;
    }
    Ok(())
}

pub(super) fn send_tool_call_delta(
    tx: &Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    tool_calls: Vec<ChatCompletionToolCallDelta>,
) -> Result<()> {
    if let Some(tx) = tx {
        tx.blocking_send(Ok(GenerateStreamEvent::ToolCallDelta(tool_calls)))
            .map_err(|err| anyhow!("stream receiver closed: {err}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_stops_require_native_metadata_and_distinguish_template_terminators() {
        let request = json!({"stop":["END","<|im_end|>"]});
        let matched = json!(["END"]);
        for (stop, valid, expected) in [
            (json!("END"), true, json!("END")),
            (json!("<|im_end|>"), true, Value::Null),
            (Value::Null, true, Value::Null),
            (json!(42), true, json!(42)),
            (json!("unexpected"), false, Value::Null),
            (json!({}), false, Value::Null),
        ] {
            let mut response = json!({"choices":[{"index":0,"message":{"content":"ok"},"finish_reason":"stop","stop_reason":stop}]});
            assert_eq!(
                validate_api_response(&response, &request, Some(&matched), false).is_ok(),
                valid
            );
            if valid {
                normalize_matched_stop(&mut response, Some(&matched));
                assert_eq!(response["choices"][0]["stop_reason"], expected);
            }
        }
        let response =
            json!({"choices":[{"index":0,"message":{"content":"ok"},"finish_reason":"stop"}]});
        assert!(validate_api_response(&response, &request, Some(&matched), false).is_err());
    }

    #[test]
    fn mixed_line_delimiters_keep_earliest_sse_boundary() {
        let mut parser = SseAccumulator::default();
        let mut events = Vec::new();
        parser
            .push(b"data: first\r\n\r\ndata: second\n\n", |data| {
                events.push(data.to_string());
                Ok(())
            })
            .unwrap();
        assert_eq!(events, vec!["first", "second"]);
        assert!(parser.pending.is_empty());
    }

    #[test]
    fn multiline_sse_data_is_one_event_and_named_errors_are_rejected() {
        let mut parser = SseAccumulator::default();
        let mut events = Vec::new();
        parser
            .push(b"data: {\n", |data| {
                events.push(data.to_string());
                Ok(())
            })
            .unwrap();
        parser
            .push(b"data: \"choices\": []}\n\n", |data| {
                events.push(data.to_string());
                Ok(())
            })
            .unwrap();
        assert_eq!(events, vec!["{\n\"choices\": []}"]);
        let error = parser
            .push(b"event: error\ndata: {\"message\":\"failed\"}\n\n", |_| {
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("failed"));
    }
}

//! Minimal HTTP/JSON client for the Werk control plane.
//!
//! The client is deliberately loopback-oriented and dependency-free. It does
//! not follow redirects and never includes authentication material in errors.

use super::{
    CapabilitiesResponse, DecodeRequest, DecodeResponse, ExpertActionRequest, ExpertActionResponse,
    ExpertListFilter, ExpertListResponse, MemoryStatusResponse, PROTOCOL_VERSION_HEADER,
    PrefillRequest, PrefillResponse, ProtocolEnvelope, ProtocolErrorBody, ProtocolVersion,
    PruneStatesRequest, PruneStatesResponse, RuntimeInfo, StateActionRequest, StateActionResponse,
    StateListFilter, StateListResponse, StateTier,
};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fmt,
    io::{ErrorKind, Read, Write},
    net::{Ipv6Addr, Shutdown, TcpStream, ToSocketAddrs},
    time::{Duration, Instant},
};

const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_API_KEY_BYTES: usize = 16 * 1024;
const MAX_REQUEST_TARGET_BYTES: usize = 8 * 1024;
const MAX_RESOLVED_ADDRESSES: usize = 16;

#[derive(Clone)]
pub struct WerkProtocolClient {
    host: String,
    port: u16,
    api_key: Option<String>,
    timeout: Duration,
}

impl fmt::Debug for WerkProtocolClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WerkProtocolClient")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[derive(Debug)]
pub enum ClientError {
    InvalidBaseUrl(String),
    InvalidTimeout(String),
    Transport(String),
    Http {
        status: u16,
        error: Option<ProtocolErrorBody>,
        request_id: Option<String>,
    },
    InvalidResponse(String),
    IncompatibleProtocol(ProtocolVersion),
}

impl fmt::Display for ClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBaseUrl(message)
            | Self::InvalidTimeout(message)
            | Self::Transport(message)
            | Self::InvalidResponse(message) => formatter.write_str(message),
            Self::Http {
                status,
                error,
                request_id,
            } => {
                let request = request_id
                    .as_deref()
                    .map(|request_id| format!(" [request {request_id}]"))
                    .unwrap_or_default();
                if let Some(error) = error {
                    write!(
                        formatter,
                        "Werk Protocol HTTP {status}: {}{request}",
                        error.message
                    )
                } else {
                    write!(formatter, "Werk Protocol HTTP {status}{request}")
                }
            }
            Self::IncompatibleProtocol(version) => write!(
                formatter,
                "incompatible Werk Protocol version {}.{}",
                version.major, version.minor
            ),
        }
    }
}

impl std::error::Error for ClientError {}

impl WerkProtocolClient {
    pub fn new(base_url: &str, api_key: Option<String>) -> Result<Self, ClientError> {
        let without_scheme = base_url.strip_prefix("http://").ok_or_else(|| {
            ClientError::InvalidBaseUrl("Werk Protocol client requires an http:// URL".to_string())
        })?;
        let authority = without_scheme.trim_end_matches('/');
        if authority.contains('/') || authority.contains('@') {
            return Err(ClientError::InvalidBaseUrl(
                "Werk Protocol base URL must contain only host and port".to_string(),
            ));
        }
        let (host, port) = parse_authority(authority)?;
        let port = port.parse::<u16>().map_err(|_| {
            ClientError::InvalidBaseUrl("Werk Protocol base URL has an invalid port".to_string())
        })?;
        if port == 0 {
            return Err(ClientError::InvalidBaseUrl(
                "Werk Protocol base URL has an invalid port".to_string(),
            ));
        }
        Ok(Self {
            host,
            port,
            api_key,
            timeout: Duration::from_secs(30),
        })
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn info(&self) -> Result<RuntimeInfo, ClientError> {
        self.get("/werk/v1/info")
    }

    pub fn capabilities(&self) -> Result<CapabilitiesResponse, ClientError> {
        self.get("/werk/v1/capabilities")
    }

    pub fn memory_status(&self) -> Result<MemoryStatusResponse, ClientError> {
        self.get("/werk/v1/memory")
    }

    pub fn list_states(&self, filter: &StateListFilter) -> Result<StateListResponse, ClientError> {
        self.get(&state_list_path(filter))
    }

    pub fn state_action(
        &self,
        state_id: &str,
        request: &StateActionRequest,
    ) -> Result<StateActionResponse, ClientError> {
        self.post(
            &format!("/werk/v1/states/{}/actions", percent_encode(state_id)),
            request,
        )
    }

    pub fn prune_states(
        &self,
        request: &PruneStatesRequest,
    ) -> Result<PruneStatesResponse, ClientError> {
        self.post("/werk/v1/states/prune", request)
    }

    pub fn list_experts(
        &self,
        filter: &ExpertListFilter,
    ) -> Result<ExpertListResponse, ClientError> {
        self.get(&expert_list_path(filter))
    }

    pub fn expert_action(
        &self,
        request: &ExpertActionRequest,
    ) -> Result<ExpertActionResponse, ClientError> {
        self.post("/werk/v1/experts/actions", request)
    }

    pub fn prefill(&self, request: &PrefillRequest) -> Result<PrefillResponse, ClientError> {
        let secrets = match &request.input {
            super::PrefillInput::Text { text } => vec![text.as_str()],
            super::PrefillInput::Messages { messages } => messages
                .iter()
                .take(256)
                .map(|message| message.content.as_str())
                .collect(),
        };
        self.request("POST", "/werk/v1/prefill", Some(request), &secrets)
    }

    pub fn decode(&self, request: &DecodeRequest) -> Result<DecodeResponse, ClientError> {
        let mut secrets = Vec::with_capacity(1 + request.stop.len().min(16));
        secrets.push(request.handoff.as_str());
        secrets.extend(request.stop.iter().take(16).map(String::as_str));
        self.request("POST", "/werk/v1/decode", Some(request), &secrets)
    }

    pub fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, ClientError> {
        self.request::<(), T>("GET", path, None, &[])
    }

    pub fn post<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, ClientError> {
        self.request("POST", path, Some(body), &[])
    }

    fn request<B: Serialize, T: DeserializeOwned>(
        &self,
        method: &str,
        path: &str,
        body: Option<&B>,
        secrets: &[&str],
    ) -> Result<T, ClientError> {
        let result = self.request_unredacted(method, path, body);
        let mut redactions = Vec::with_capacity(secrets.len() + 1);
        redactions.extend_from_slice(secrets);
        if let Some(api_key) = self.api_key.as_deref() {
            redactions.push(api_key);
        }
        result.map_err(|error| redact_client_error(error, &redactions))
    }

    fn request_unredacted<B: Serialize, T: DeserializeOwned>(
        &self,
        method: &str,
        path: &str,
        body: Option<&B>,
    ) -> Result<T, ClientError> {
        if !valid_path(path) || path.len() > MAX_REQUEST_TARGET_BYTES {
            return Err(ClientError::InvalidBaseUrl(
                "Werk Protocol request path is invalid".to_string(),
            ));
        }
        let deadline = request_deadline(self.timeout)?;
        let body = body
            .map(serde_json::to_vec)
            .transpose()
            .map_err(|error| ClientError::InvalidResponse(error.to_string()))?
            .unwrap_or_default();
        if body.len() > MAX_REQUEST_BYTES {
            return Err(ClientError::InvalidResponse(
                "Werk Protocol request exceeds the client limit".to_string(),
            ));
        }
        let mut request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nAccept: application/json\r\n{PROTOCOL_VERSION_HEADER}: {}\r\nConnection: close\r\nContent-Length: {}\r\n",
            authority_header(&self.host, self.port),
            ProtocolVersion::V1,
            body.len()
        );
        if !body.is_empty() {
            request.push_str("Content-Type: application/json\r\n");
        }
        if let Some(api_key) = self.api_key.as_deref() {
            if api_key.is_empty()
                || api_key.len() > MAX_API_KEY_BYTES
                || api_key.contains(['\r', '\n'])
            {
                return Err(ClientError::InvalidBaseUrl(
                    "Werk Protocol API key contains invalid characters".to_string(),
                ));
            }
            request.push_str("Authorization: Bearer ");
            request.push_str(api_key);
            request.push_str("\r\n");
        }
        request.push_str("\r\n");

        let mut stream = connect_with_deadline(&self.host, self.port, deadline)?;
        write_all_with_deadline(&mut stream, request.as_bytes(), deadline)?;
        write_all_with_deadline(&mut stream, &body, deadline)?;
        let _ = stream.shutdown(Shutdown::Write);
        let response = read_response_with_deadline(&mut stream, deadline)?;
        if response.len() > MAX_RESPONSE_BYTES {
            return Err(ClientError::InvalidResponse(
                "Werk Protocol response exceeds the client limit".to_string(),
            ));
        }
        parse_response(&response)
    }
}

fn parse_authority(authority: &str) -> Result<(String, &str), ClientError> {
    if let Some(ipv6) = authority.strip_prefix('[') {
        let (host, port) = ipv6.split_once("]:").ok_or_else(|| {
            ClientError::InvalidBaseUrl(
                "Werk Protocol base URL has an invalid IPv6 authority".to_string(),
            )
        })?;
        if host.parse::<Ipv6Addr>().is_err() {
            return Err(ClientError::InvalidBaseUrl(
                "Werk Protocol base URL has an invalid IPv6 host".to_string(),
            ));
        }
        return Ok((host.to_string(), port));
    }

    let (host, port) = authority.rsplit_once(':').ok_or_else(|| {
        ClientError::InvalidBaseUrl("Werk Protocol base URL must include a port".to_string())
    })?;
    if host.is_empty()
        || !host
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '-'))
    {
        return Err(ClientError::InvalidBaseUrl(
            "Werk Protocol base URL has an invalid host".to_string(),
        ));
    }
    Ok((host.to_string(), port))
}

fn authority_header(host: &str, port: u16) -> String {
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

fn request_deadline(timeout: Duration) -> Result<Instant, ClientError> {
    if timeout.is_zero() {
        return Err(ClientError::InvalidTimeout(
            "Werk Protocol timeout must be greater than zero".to_string(),
        ));
    }
    Instant::now().checked_add(timeout).ok_or_else(|| {
        ClientError::InvalidTimeout("Werk Protocol timeout is too large".to_string())
    })
}

fn remaining_timeout(deadline: Instant) -> Result<Duration, ClientError> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(request_timed_out)
}

fn request_timed_out() -> ClientError {
    ClientError::Transport("Werk Protocol request timed out".to_string())
}

fn connect_with_deadline(
    host: &str,
    port: u16,
    deadline: Instant,
) -> Result<TcpStream, ClientError> {
    let addresses = (host, port)
        .to_socket_addrs()
        .map_err(|error| ClientError::Transport(error.to_string()))?;
    let mut attempted = 0usize;
    let mut last_error = None;
    for address in addresses.take(MAX_RESOLVED_ADDRESSES) {
        attempted += 1;
        match TcpStream::connect_timeout(&address, remaining_timeout(deadline)?) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = Some(error),
        }
    }
    if attempted == 0 {
        return Err(ClientError::Transport(
            "Werk Protocol host did not resolve to an address".to_string(),
        ));
    }
    Err(ClientError::Transport(format!(
        "Werk Protocol connection failed: {}",
        last_error.expect("at least one connection was attempted")
    )))
}

fn write_all_with_deadline(
    stream: &mut TcpStream,
    mut bytes: &[u8],
    deadline: Instant,
) -> Result<(), ClientError> {
    while !bytes.is_empty() {
        stream
            .set_write_timeout(Some(remaining_timeout(deadline)?))
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        match stream.write(bytes) {
            Ok(0) => {
                return Err(ClientError::Transport(
                    "Werk Protocol connection closed while writing".to_string(),
                ));
            }
            Ok(written) => bytes = &bytes[written..],
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) if matches!(error.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) => {
                return Err(request_timed_out());
            }
            Err(error) => return Err(ClientError::Transport(error.to_string())),
        }
    }
    Ok(())
}

fn read_response_with_deadline(
    stream: &mut TcpStream,
    deadline: Instant,
) -> Result<Vec<u8>, ClientError> {
    let mut response = Vec::new();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        stream
            .set_read_timeout(Some(remaining_timeout(deadline)?))
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        match stream.read(&mut buffer) {
            Ok(0) => return Ok(response),
            Ok(read) => {
                if response.len().saturating_add(read) > MAX_RESPONSE_BYTES {
                    return Err(ClientError::InvalidResponse(
                        "Werk Protocol response exceeds the client limit".to_string(),
                    ));
                }
                response.extend_from_slice(&buffer[..read]);
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) if matches!(error.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) => {
                return Err(request_timed_out());
            }
            Err(error) => return Err(ClientError::Transport(error.to_string())),
        }
    }
}

fn redact_client_error(mut error: ClientError, secrets: &[&str]) -> ClientError {
    match &mut error {
        ClientError::InvalidBaseUrl(message)
        | ClientError::InvalidTimeout(message)
        | ClientError::Transport(message)
        | ClientError::InvalidResponse(message) => {
            *message = redact_text(message, secrets);
        }
        ClientError::Http {
            error: Some(body),
            request_id,
            ..
        } => {
            body.message = redact_text(&body.message, secrets);
            if let Some(details) = &mut body.details {
                redact_json_strings(details, secrets);
            }
            redact_request_id(request_id, secrets);
        }
        ClientError::Http {
            error: None,
            request_id,
            ..
        } => redact_request_id(request_id, secrets),
        ClientError::IncompatibleProtocol(_) => {}
    }
    error
}

fn redact_request_id(request_id: &mut Option<String>, secrets: &[&str]) {
    let Some(value) = request_id else {
        return;
    };
    let redacted = redact_text(value, secrets);
    if redacted.contains("[redacted]") || !valid_request_id(&redacted) {
        *request_id = None;
    } else {
        *value = redacted;
    }
}

fn valid_request_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn redact_text(value: &str, secrets: &[&str]) -> String {
    let mut redacted = value.to_string();
    for secret in secrets.iter().copied().filter(|secret| !secret.is_empty()) {
        redacted = redacted.replace(secret, "[redacted]");
    }
    redacted.chars().take(2_048).collect()
}

fn redact_json_strings(value: &mut serde_json::Value, secrets: &[&str]) {
    match value {
        serde_json::Value::String(text) => *text = redact_text(text, secrets),
        serde_json::Value::Array(values) => {
            for value in values {
                redact_json_strings(value, secrets);
            }
        }
        serde_json::Value::Object(values) => {
            let prior = std::mem::take(values);
            for (key, mut value) in prior {
                redact_json_strings(&mut value, secrets);
                values.insert(redact_text(&key, secrets), value);
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

fn state_list_path(filter: &StateListFilter) -> String {
    let mut query = Vec::new();
    if let Some(model_id) = &filter.model_id {
        query.push(format!("model_id={}", percent_encode(model_id)));
    }
    if let Some(tier) = filter.tier {
        query.push(format!("tier={}", state_tier_name(tier)));
    }
    if let Some(limit) = filter.limit {
        query.push(format!("limit={limit}"));
    }
    if let Some(cursor) = &filter.cursor {
        query.push(format!("cursor={}", percent_encode(cursor)));
    }
    query_path("/werk/v1/states", query)
}

fn expert_list_path(filter: &ExpertListFilter) -> String {
    let mut query = Vec::new();
    if let Some(model_id) = &filter.model_id {
        query.push(format!("model_id={}", percent_encode(model_id)));
    }
    if let Some(tier) = filter.tier {
        let tier = match tier {
            super::ExpertTier::Vram => "vram",
            super::ExpertTier::Ram => "ram",
            super::ExpertTier::External => "external",
        };
        query.push(format!("tier={tier}"));
    }
    if let Some(limit) = filter.limit {
        query.push(format!("limit={limit}"));
    }
    if let Some(cursor) = &filter.cursor {
        query.push(format!("cursor={}", percent_encode(cursor)));
    }
    if filter.allow_experimental {
        query.push("allow_experimental=true".to_string());
    }
    query_path("/werk/v1/experts", query)
}

fn query_path(base: &str, query: Vec<String>) -> String {
    if query.is_empty() {
        base.to_string()
    } else {
        format!("{base}?{}", query.join("&"))
    }
}

fn state_tier_name(tier: StateTier) -> &'static str {
    match tier {
        StateTier::Vram => "vram",
        StateTier::Ram => "ram",
        StateTier::Disk => "disk",
        StateTier::External => "external",
    }
}

fn percent_encode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

fn valid_path(path: &str) -> bool {
    path.starts_with("/werk/v1/")
        && !path.contains(['\r', '\n', ' ', '#'])
        && !path
            .split('?')
            .next()
            .unwrap_or(path)
            .split('/')
            .any(|part| part == "..")
}

fn parse_response<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ClientError> {
    let split = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| ClientError::InvalidResponse("invalid HTTP response".to_string()))?;
    let head = std::str::from_utf8(&bytes[..split])
        .map_err(|_| ClientError::InvalidResponse("invalid HTTP headers".to_string()))?;
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| ClientError::InvalidResponse("invalid HTTP status".to_string()))?;
    let body = response_body(head, &bytes[split + 4..])?;
    let declared_protocol = response_protocol_version(head)?;
    if let Some(version) = declared_protocol
        && !ProtocolVersion::V1.accepts(version)
    {
        return Err(ClientError::IncompatibleProtocol(version));
    }
    #[derive(serde::Deserialize)]
    struct VersionEnvelope {
        protocol: ProtocolVersion,
    }
    let body_protocol = serde_json::from_slice::<VersionEnvelope>(&body)
        .ok()
        .map(|envelope| envelope.protocol);
    if let Some(version) = body_protocol
        && !ProtocolVersion::V1.accepts(version)
    {
        return Err(ClientError::IncompatibleProtocol(version));
    }
    if let (Some(declared), Some(body)) = (declared_protocol, body_protocol)
        && declared != body
    {
        return Err(ClientError::InvalidResponse(
            "Werk Protocol version header does not match its envelope".to_string(),
        ));
    }
    if !(200..300).contains(&status) {
        #[derive(serde::Deserialize)]
        struct ErrorEnvelope {
            protocol: ProtocolVersion,
            request_id: String,
            error: ProtocolErrorBody,
        }
        // A reverse proxy may return an unrelated JSON error. Preserve its
        // HTTP status, but accept typed Werk error fields only when the body
        // carries the compatible protocol envelope already checked above.
        let envelope = body_protocol.and_then(|version| {
            serde_json::from_slice::<ErrorEnvelope>(&body)
                .ok()
                .filter(|envelope| {
                    envelope.protocol == version && valid_request_id(&envelope.request_id)
                })
        });
        let (error, request_id) = envelope
            .map(|envelope| (Some(envelope.error), Some(envelope.request_id)))
            .unwrap_or((None, None));
        return Err(ClientError::Http {
            status,
            error,
            request_id,
        });
    }
    let body_protocol = body_protocol.ok_or_else(|| {
        ClientError::InvalidResponse("Werk Protocol response has no protocol version".to_string())
    })?;
    let envelope = serde_json::from_slice::<ProtocolEnvelope<T>>(&body)
        .map_err(|error| ClientError::InvalidResponse(error.to_string()))?;
    if envelope.protocol != body_protocol {
        return Err(ClientError::InvalidResponse(
            "Werk Protocol version changed while decoding its envelope".to_string(),
        ));
    }
    if !valid_request_id(&envelope.request_id) {
        return Err(ClientError::InvalidResponse(
            "Werk Protocol response has an invalid request ID".to_string(),
        ));
    }
    Ok(envelope.data)
}

fn response_protocol_version(head: &str) -> Result<Option<ProtocolVersion>, ClientError> {
    let mut protocol = None;
    for line in head.lines().skip(1) {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if !name.eq_ignore_ascii_case(PROTOCOL_VERSION_HEADER) {
            continue;
        }
        let version = value.trim().parse::<ProtocolVersion>().map_err(|_| {
            ClientError::InvalidResponse("invalid Werk Protocol version header".to_string())
        })?;
        if protocol.replace(version).is_some() {
            return Err(ClientError::InvalidResponse(
                "duplicate Werk Protocol version header".to_string(),
            ));
        }
    }
    Ok(protocol)
}

fn response_body(head: &str, received: &[u8]) -> Result<Vec<u8>, ClientError> {
    let mut content_length = None;
    let mut chunked = false;
    for line in head.lines().skip(1) {
        let (name, value) = line.split_once(':').ok_or_else(|| {
            ClientError::InvalidResponse("invalid HTTP response header".to_string())
        })?;
        if name.trim() != name || name.is_empty() || !name.bytes().all(valid_header_name_byte) {
            return Err(ClientError::InvalidResponse(
                "invalid HTTP response header".to_string(),
            ));
        }
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            let parsed = value.parse::<usize>().map_err(|_| {
                ClientError::InvalidResponse("invalid HTTP content length".to_string())
            })?;
            if content_length
                .replace(parsed)
                .is_some_and(|prior| prior != parsed)
            {
                return Err(ClientError::InvalidResponse(
                    "conflicting HTTP content lengths".to_string(),
                ));
            }
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            let encodings = value.split(',').map(str::trim).collect::<Vec<_>>();
            if encodings
                .last()
                .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
                && encodings
                    .iter()
                    .all(|value| value.eq_ignore_ascii_case("chunked"))
            {
                chunked = true;
            } else {
                return Err(ClientError::InvalidResponse(
                    "unsupported HTTP transfer encoding".to_string(),
                ));
            }
        }
    }
    if chunked && content_length.is_some() {
        return Err(ClientError::InvalidResponse(
            "HTTP response contains both chunked encoding and content length".to_string(),
        ));
    }
    if chunked {
        return decode_chunked(received);
    }
    if let Some(length) = content_length {
        if length > MAX_RESPONSE_BYTES || received.len() != length {
            return Err(ClientError::InvalidResponse(
                "truncated or oversized HTTP response body".to_string(),
            ));
        }
    }
    Ok(received.to_vec())
}

fn valid_header_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

fn decode_chunked(bytes: &[u8]) -> Result<Vec<u8>, ClientError> {
    let mut position = 0usize;
    let mut decoded = Vec::new();
    loop {
        let line_end = bytes[position..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .map(|offset| position + offset)
            .ok_or_else(|| {
                ClientError::InvalidResponse("truncated HTTP chunk header".to_string())
            })?;
        let size_text = std::str::from_utf8(&bytes[position..line_end])
            .map_err(|_| ClientError::InvalidResponse("invalid HTTP chunk size".to_string()))?;
        let size_text = size_text.split(';').next().unwrap_or_default().trim();
        if size_text.is_empty() || size_text.len() > 16 {
            return Err(ClientError::InvalidResponse(
                "invalid HTTP chunk size".to_string(),
            ));
        }
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| ClientError::InvalidResponse("invalid HTTP chunk size".to_string()))?;
        position = line_end + 2;
        if size == 0 {
            if bytes.get(position..position + 2) == Some(b"\r\n") {
                position += 2;
            } else {
                let trailers_end = bytes[position..]
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|offset| position + offset + 4)
                    .ok_or_else(|| {
                        ClientError::InvalidResponse("truncated HTTP chunk trailers".to_string())
                    })?;
                position = trailers_end;
            }
            if position != bytes.len() {
                return Err(ClientError::InvalidResponse(
                    "unexpected bytes after HTTP chunks".to_string(),
                ));
            }
            return Ok(decoded);
        }
        if decoded.len().saturating_add(size) > MAX_RESPONSE_BYTES {
            return Err(ClientError::InvalidResponse(
                "Werk Protocol response exceeds the client limit".to_string(),
            ));
        }
        let chunk_end = position
            .checked_add(size)
            .ok_or_else(|| ClientError::InvalidResponse("invalid HTTP chunk size".to_string()))?;
        if bytes.get(chunk_end..chunk_end + 2) != Some(b"\r\n") {
            return Err(ClientError::InvalidResponse(
                "truncated HTTP chunk data".to_string(),
            ));
        }
        decoded.extend_from_slice(&bytes[position..chunk_end]);
        position = chunk_end + 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::werk_protocol::RuntimeInfo;
    use std::{net::TcpListener, thread};

    #[test]
    fn debug_redacts_api_key() {
        let client =
            WerkProtocolClient::new("http://127.0.0.1:11434", Some("werk-secret".to_string()))
                .unwrap();
        let debug = format!("{client:?}");
        assert!(!debug.contains("werk-secret"));
        assert!(debug.contains("[redacted]"));
    }

    #[test]
    fn base_url_parser_accepts_bracketed_ipv6_and_rejects_url_authority_features() {
        let client = WerkProtocolClient::new("http://[::1]:11434/", None).unwrap();
        assert_eq!(client.host, "::1");
        assert_eq!(client.port, 11434);
        assert_eq!(authority_header(&client.host, client.port), "[::1]:11434");

        for invalid in [
            "https://127.0.0.1:11434",
            "http://user@127.0.0.1:11434",
            "http://127.0.0.1:11434/werk/v1",
            "http://::1:11434",
            "http://[not-ipv6]:11434",
        ] {
            assert!(
                matches!(
                    WerkProtocolClient::new(invalid, None),
                    Err(ClientError::InvalidBaseUrl(_))
                ),
                "unexpectedly accepted {invalid}"
            );
        }
    }

    #[test]
    fn displayed_and_debugged_http_errors_redact_credentials_and_handoffs() {
        let error = ClientError::Http {
            status: 409,
            error: Some(ProtocolErrorBody {
                code: super::super::ProtocolErrorCode::IncompatibleState,
                message: "credential-secret failed for handoff-secret".to_string(),
                retryable: false,
                details: Some(serde_json::json!({
                    "credential-secret": "handoff-secret and credential-secret"
                })),
            }),
            request_id: Some("credential-secret".to_string()),
        };

        let error = redact_client_error(error, &["credential-secret", "handoff-secret"]);
        let rendered = format!("{error} {error:?}");
        assert!(!rendered.contains("credential-secret"));
        assert!(!rendered.contains("handoff-secret"));
        assert!(rendered.contains("[redacted]"));
    }

    #[test]
    fn client_parses_envelope_and_sends_bearer_without_leaking_it() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let read = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..read]);
            assert!(request.starts_with("GET /werk/v1/info HTTP/1.1"));
            assert!(request.contains("Authorization: Bearer secret"));
            assert!(request.contains("Accept: application/json"));
            assert!(request.contains(&format!("{PROTOCOL_VERSION_HEADER}: 1.0")));
            let body = serde_json::json!({
                "protocol": {"major": 1, "minor": 0},
                "request_id": "req_test",
                "data": {
                    "service": "werk1112",
                    "service_version": "1.5.1",
                    "protocol": {"major": 1, "minor": 0},
                    "active_backend": "test",
                    "limits": {
                        "max_page_size": 100,
                        "max_state_ids_per_operation": 100,
                        "max_expert_ids_per_operation": 256,
                        "max_request_bytes": 1048576,
                        "max_handoff_bytes": 4096,
                        "max_ttl_seconds": 2592000
                    }
                }
            })
            .to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\n{PROTOCOL_VERSION_HEADER}: 1.0\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });
        let client = WerkProtocolClient::new(
            &format!("http://127.0.0.1:{port}"),
            Some("secret".to_string()),
        )
        .unwrap();
        let info: RuntimeInfo = client.get("/werk/v1/info").unwrap();
        assert_eq!(info.active_backend, "test");
        server.join().unwrap();
    }

    #[test]
    fn client_rejects_cross_major_responses() {
        let body = br#"{"protocol":{"major":2,"minor":0},"request_id":"r","data":{"ok":true}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );
        assert!(matches!(
            parse_response::<serde_json::Value>(response.as_bytes()),
            Err(ClientError::IncompatibleProtocol(ProtocolVersion {
                major: 2,
                minor: 0
            }))
        ));
    }

    #[test]
    fn client_checks_the_envelope_version_before_versioned_payload_fields() {
        let success = br#"{"protocol":{"major":2,"minor":0},"request_id":"r","data":{"capabilities":[{"id":"new","status":"future_status","detail":"new"}]}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
            success.len(),
            String::from_utf8_lossy(success)
        );
        assert!(matches!(
            parse_response::<CapabilitiesResponse>(response.as_bytes()),
            Err(ClientError::IncompatibleProtocol(ProtocolVersion {
                major: 2,
                minor: 0
            }))
        ));

        let error = br#"{"protocol":{"major":2,"minor":0},"request_id":"r","error":{"code":"future_error","message":"new","retryable":false}}"#;
        let response = format!(
            "HTTP/1.1 500 Error\r\nContent-Length: {}\r\n\r\n{}",
            error.len(),
            String::from_utf8_lossy(error)
        );
        assert!(matches!(
            parse_response::<serde_json::Value>(response.as_bytes()),
            Err(ClientError::IncompatibleProtocol(ProtocolVersion {
                major: 2,
                minor: 0
            }))
        ));
    }

    #[test]
    fn client_never_treats_an_unversioned_json_error_as_a_werk_error() {
        let body = br#"{"error":{"code":"unauthorized","message":"foreign","retryable":false}}"#;
        let response = format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );

        assert!(matches!(
            parse_response::<serde_json::Value>(response.as_bytes()),
            Err(ClientError::Http {
                status: 401,
                error: None,
                request_id: None,
            })
        ));
    }

    #[test]
    fn client_requires_safe_request_ids_in_versioned_envelopes() {
        let success =
            br#"{"protocol":{"major":1,"minor":0},"request_id":"../bad","data":{"ok":true}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
            success.len(),
            String::from_utf8_lossy(success)
        );
        assert!(matches!(
            parse_response::<serde_json::Value>(response.as_bytes()),
            Err(ClientError::InvalidResponse(_))
        ));

        let error = br#"{"protocol":{"major":1,"minor":0},"request_id":"../bad","error":{"code":"unauthorized","message":"foreign","retryable":false}}"#;
        let response = format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Length: {}\r\n\r\n{}",
            error.len(),
            String::from_utf8_lossy(error)
        );
        assert!(matches!(
            parse_response::<serde_json::Value>(response.as_bytes()),
            Err(ClientError::Http {
                status: 401,
                error: None,
                request_id: None,
            })
        ));
    }

    #[test]
    fn client_rejects_mismatched_or_duplicate_version_headers() {
        let body = br#"{"protocol":{"major":1,"minor":0},"request_id":"r","data":{"ok":true}}"#;
        let mismatched = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n{PROTOCOL_VERSION_HEADER}: 1.1\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );
        assert!(matches!(
            parse_response::<serde_json::Value>(mismatched.as_bytes()),
            Err(ClientError::IncompatibleProtocol(ProtocolVersion {
                major: 1,
                minor: 1
            }))
        ));

        let duplicate = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n{PROTOCOL_VERSION_HEADER}: 1.0\r\n{PROTOCOL_VERSION_HEADER}: 1.0\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );
        assert!(matches!(
            parse_response::<serde_json::Value>(duplicate.as_bytes()),
            Err(ClientError::InvalidResponse(_))
        ));
    }

    #[test]
    fn zero_timeout_and_oversized_request_targets_fail_before_network_io() {
        let zero_timeout = WerkProtocolClient::new("http://127.0.0.1:9", None)
            .unwrap()
            .with_timeout(Duration::ZERO);
        assert!(matches!(
            zero_timeout.info(),
            Err(ClientError::InvalidTimeout(_))
        ));

        let client = WerkProtocolClient::new("http://127.0.0.1:9", None).unwrap();
        let path = format!(
            "/werk/v1/states?cursor={}",
            "a".repeat(MAX_REQUEST_TARGET_BYTES)
        );
        assert!(matches!(
            client.get::<serde_json::Value>(&path),
            Err(ClientError::InvalidBaseUrl(_))
        ));
    }

    #[test]
    fn client_decodes_bounded_chunked_envelopes() {
        let json = br#"{"protocol":{"major":1,"minor":0},"request_id":"r","data":{"ok":true}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
            json.len(),
            String::from_utf8_lossy(json)
        );
        let value: serde_json::Value = parse_response(response.as_bytes()).unwrap();
        assert_eq!(value, serde_json::json!({"ok": true}));
    }

    #[test]
    fn client_rejects_truncated_content_length_and_conflicting_framing() {
        let truncated = b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n{}";
        assert!(matches!(
            parse_response::<serde_json::Value>(truncated),
            Err(ClientError::InvalidResponse(_))
        ));
        let ambiguous =
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n";
        assert!(matches!(
            parse_response::<serde_json::Value>(ambiguous),
            Err(ClientError::InvalidResponse(_))
        ));
    }
}

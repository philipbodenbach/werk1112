use axum::{
    Json, Router,
    body::Body,
    extract::{
        DefaultBodyLimit, Path, Query, State,
        rejection::{JsonRejection, PathRejection, QueryRejection},
    },
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::Response,
    routing::{get, post},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::{Deserialize, Deserializer, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::werk_protocol::{
    ControlContext, DecodeRequest, ExpertActionRequest, ExpertListFilter, PROTOCOL_VERSION_HEADER,
    PersistencePolicy, PrefillInput, PrefillRequest, ProtocolEnvelope, ProtocolError,
    ProtocolErrorBody, ProtocolErrorCode, ProtocolVersion, PruneStatesRequest, StateActionRequest,
    StateListFilter,
};

use super::state::ApiState;

const MAX_WERK_BODY_BYTES: usize = 1024 * 1024;
const MAX_WERK_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
static FALLBACK_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn routes() -> Router<ApiState> {
    let reads = Router::new()
        .route("/werk/v1/info", get(info_handler))
        .route("/werk/v1/capabilities", get(capabilities_handler))
        .route("/werk/v1/chat/reasoning", get(reasoning_handler))
        .route("/werk/v1/chat/limits", get(generation_limits_handler))
        .route("/werk/v1/memory", get(memory_handler))
        .route("/werk/v1/states", get(list_states_handler))
        .route("/werk/v1/experts", get(list_experts_handler));

    let writes = Router::new()
        .route(
            "/werk/v1/chat/count_tokens",
            post(chat_count_tokens_handler),
        )
        .route("/werk/v1/states/prune", post(prune_states_handler))
        .route("/werk/v1/states/{id}/actions", post(state_action_handler))
        .route("/werk/v1/experts/actions", post(expert_action_handler))
        .route("/werk/v1/prefill", post(prefill_handler))
        .route("/werk/v1/decode", post(decode_handler))
        .layer(DefaultBodyLimit::max(MAX_WERK_BODY_BYTES));

    reads.merge(writes)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReasoningQuery {
    model: Option<String>,
    max_output_tokens: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationLimitsQuery {
    model: Option<String>,
}

async fn generation_limits_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    query: Result<Query<GenerationLimitsQuery>, QueryRejection>,
) -> Response {
    let (request_id, _) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Query(query) = match query {
        Ok(query) => query,
        Err(error) => {
            return protocol_error(
                request_id,
                ProtocolError::new(ProtocolErrorCode::InvalidRequest, error.to_string()),
            );
        }
    };
    let model = query
        .model
        .or_else(|| state.default_model.clone())
        .unwrap_or_default();
    if model.is_empty() || model.len() > 512 {
        return protocol_error(
            request_id,
            ProtocolError::new(
                ProtocolErrorCode::InvalidRequest,
                "A valid model is required",
            ),
        );
    }
    let targets = if let Some(registry) = &state.deployments {
        match registry.preload_targets(&model) {
            Ok(targets) => targets,
            Err(error) => {
                return protocol_error(
                    request_id,
                    ProtocolError::new(ProtocolErrorCode::InvalidRequest, error.to_string()),
                );
            }
        }
    } else {
        vec![(model.clone(), state, model.clone())]
    };
    let result = crate::logging::spawn_blocking(
        move || -> anyhow::Result<crate::generation_limits::ChatGenerationLimits> {
            let mut profiles = Vec::new();
            for (_, state, physical_model) in targets {
                let manifest = state.store.get(&physical_model)?;
                let mut profile = crate::generation_limits::profile(
                    &manifest,
                    super::generation::generation_context_size(&state, &manifest),
                );
                profile.max_concurrent_requests = state.backend.inference_capacity(&manifest);
                profiles.push(profile);
            }
            anyhow::ensure!(!profiles.is_empty(), "No matching deployment");
            // An alias must be safe on every replica. An unknown replica prevents
            // advertising a verified ceiling; known differing limits use the minimum.
            let minimum = |values: Vec<Option<usize>>| -> Option<usize> {
                values
                    .into_iter()
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .min()
            };
            Ok(crate::generation_limits::ChatGenerationLimits {
                schema_version: 1,
                model,
                context_tokens: minimum(profiles.iter().map(|p| p.context_tokens).collect()),
                max_output_tokens: minimum(profiles.iter().map(|p| p.max_output_tokens).collect()),
                max_concurrent_requests: minimum(
                    profiles.iter().map(|p| p.max_concurrent_requests).collect(),
                ),
            })
        },
    )
    .await;
    match result {
        Ok(Ok(profile)) => success(request_id, profile),
        Ok(Err(error)) => protocol_error(
            request_id,
            ProtocolError::new(ProtocolErrorCode::InvalidRequest, error.to_string()),
        ),
        Err(_) => protocol_error(
            request_id,
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "Generation limits lookup failed",
            ),
        ),
    }
}

async fn reasoning_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    query: Result<Query<ReasoningQuery>, QueryRejection>,
) -> Response {
    let (request_id, _) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Query(query) = match query {
        Ok(query) => query,
        Err(error) => {
            return protocol_error(
                request_id,
                ProtocolError::new(ProtocolErrorCode::InvalidRequest, error.to_string()),
            );
        }
    };
    let model = query
        .model
        .or_else(|| state.default_model.clone())
        .unwrap_or_default();
    let output = query.max_output_tokens.unwrap_or(256);
    if model.is_empty() || model.len() > 512 || output == 0 || output > 2_000_000 {
        return protocol_error(
            request_id,
            ProtocolError::new(
                ProtocolErrorCode::InvalidRequest,
                "model and max_output_tokens (1..2000000) are required",
            ),
        );
    }
    // Expand aliases read-only: do not take inference permits or move routing cursors.
    let targets = if let Some(registry) = &state.deployments {
        match registry.preload_targets(&model) {
            Ok(targets) => targets,
            Err(error) => {
                return protocol_error(
                    request_id,
                    ProtocolError::new(ProtocolErrorCode::InvalidRequest, error.to_string()),
                );
            }
        }
    } else {
        vec![(model.clone(), state, model.clone())]
    };
    let result = crate::logging::spawn_blocking(
        move || -> anyhow::Result<crate::reasoning::ReasoningProfile> {
            let mut profiles = Vec::new();
            for (_, state, physical_model) in targets {
                let manifest = state.store.get(&physical_model)?;
                let mut profile = crate::reasoning::profile(
                    &manifest,
                    state.backend.reasoning_config(&manifest).as_ref(),
                    super::generation::generation_context_size(&state, &manifest),
                    output,
                );
                profile.model = model.clone();
                profiles.push(profile);
            }
            let first = profiles
                .first()
                .ok_or_else(|| anyhow::anyhow!("No matching deployment"))?;
            anyhow::ensure!(
                profiles.iter().all(|p| p == first),
                "Deployment replicas report different reasoning profiles; select a deployment ID"
            );
            Ok(profiles.remove(0))
        },
    )
    .await;
    match result {
        Ok(Ok(profile)) => success(request_id, profile),
        Ok(Err(error)) => protocol_error(
            request_id,
            ProtocolError::new(ProtocolErrorCode::InvalidRequest, error.to_string()),
        ),
        Err(_) => protocol_error(
            request_id,
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "Reasoning profile lookup failed",
            ),
        ),
    }
}

async fn chat_count_tokens_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Result<Json<crate::openai::ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let (request_id, _) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Json(mut request) = match body {
        Ok(body) => body,
        Err(error) => return json_rejection(request_id, error),
    };
    // This route never generates, regardless of an inherited stream setting.
    request.stream = Some(false);
    let prepared = match super::generation::prepare(
        state,
        request,
        super::generation::ContextPolicy::Count,
        "/werk/v1/chat/count_tokens",
        &headers,
    )
    .await
    {
        Ok(value) => value,
        Err(error) => {
            return protocol_error(
                request_id,
                ProtocolError::new(
                    if error.status == StatusCode::TOO_MANY_REQUESTS {
                        ProtocolErrorCode::ResourceExhausted
                    } else if error.status.is_server_error() {
                        ProtocolErrorCode::Unavailable
                    } else {
                        ProtocolErrorCode::InvalidRequest
                    },
                    error.message,
                )
                .retryable(
                    error.status == StatusCode::TOO_MANY_REQUESTS || error.status.is_server_error(),
                ),
            );
        }
    };
    let model = prepared
        .state
        .requested_alias
        .clone()
        .unwrap_or_else(|| prepared.manifest.id.clone());
    match crate::logging::spawn_blocking(move || {
        let _permit = prepared.state.deployment_permit;
        prepared.state.backend.count_api_tokens(
            &prepared.manifest,
            prepared.request,
            prepared.api_options,
        )
    })
    .await
    {
        Ok(Ok(count)) => success(
            request_id,
            crate::werk_protocol::ChatTokenCountResponse {
                model,
                input_tokens: count as u64,
            },
        ),
        Ok(Err(error)) => {
            let unsupported = error.is::<crate::backend::UnsupportedTokenCounting>();
            protocol_error(
                request_id,
                ProtocolError::new(
                    if unsupported {
                        ProtocolErrorCode::Unsupported
                    } else {
                        ProtocolErrorCode::Unavailable
                    },
                    format!("Native chat token count unavailable: {error}"),
                )
                .retryable(!unsupported),
            )
        }
        Err(_) => protocol_error(
            request_id,
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "Native chat token count task failed",
            ),
        ),
    }
}

async fn info_handler(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    match state.werk_control.info(context).await {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

async fn capabilities_handler(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    match state.werk_control.capabilities(context).await {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

async fn memory_handler(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    match state.werk_control.memory_status(context).await {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

async fn list_states_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    query: Result<Query<StateListFilter>, QueryRejection>,
) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Query(filter) = match query {
        Ok(query) => query,
        Err(_) => return invalid_request(request_id, "invalid state-list query"),
    };
    match state.werk_control.list_states(context, filter).await {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

async fn state_action_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    path: Result<Path<String>, PathRejection>,
    body: Result<Json<StateActionRequest>, JsonRejection>,
) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Path(state_id) = match path {
        Ok(path) => path,
        Err(_) => return invalid_request(request_id, "invalid runtime-state path"),
    };
    let Json(request) = match body {
        Ok(body) => body,
        Err(rejection) => return json_rejection(request_id, rejection),
    };
    match state
        .werk_control
        .state_action(context, state_id, request)
        .await
    {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

async fn prune_states_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Result<Json<PruneStatesRequest>, JsonRejection>,
) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Json(request) = match body {
        Ok(body) => body,
        Err(rejection) => return json_rejection(request_id, rejection),
    };
    match state.werk_control.prune_states(context, request).await {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

async fn list_experts_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    query: Result<Query<ExpertListFilter>, QueryRejection>,
) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Query(filter) = match query {
        Ok(query) => query,
        Err(_) => return invalid_request(request_id, "invalid expert-list query"),
    };
    match state.werk_control.list_experts(context, filter).await {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

async fn expert_action_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Result<Json<ExpertActionRequest>, JsonRejection>,
) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Json(request) = match body {
        Ok(body) => body,
        Err(rejection) => return json_rejection(request_id, rejection),
    };
    match state.werk_control.expert_action(context, request).await {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

async fn prefill_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Result<Json<WirePrefillRequest>, JsonRejection>,
) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Json(request) = match body {
        Ok(body) => body,
        Err(rejection) => return json_rejection(request_id, rejection),
    };
    let PrefillRequestWithPresence {
        mut request,
        policy_was_supplied,
        experimental_decision_was_supplied,
    } = request.into_request();
    state.server_persistence.apply_prefill_defaults(
        &mut request,
        policy_was_supplied,
        experimental_decision_was_supplied,
    );
    match state.werk_control.prefill(context, request).await {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePrefillRequest {
    model_id: String,
    input: PrefillInput,
    #[serde(default)]
    policy: Supplied<PersistencePolicy>,
    #[serde(default)]
    allow_experimental: Supplied<bool>,
}

impl WirePrefillRequest {
    fn into_request(self) -> PrefillRequestWithPresence {
        let (policy, policy_was_supplied) = self.policy.into_value_or_default();
        let (allow_experimental, experimental_decision_was_supplied) =
            self.allow_experimental.into_value_or_default();
        PrefillRequestWithPresence {
            request: PrefillRequest {
                model_id: self.model_id,
                input: self.input,
                policy,
                allow_experimental,
            },
            policy_was_supplied,
            experimental_decision_was_supplied,
        }
    }
}

struct PrefillRequestWithPresence {
    request: PrefillRequest,
    policy_was_supplied: bool,
    experimental_decision_was_supplied: bool,
}

enum Supplied<T> {
    Missing,
    Value(T),
}

impl<T> Default for Supplied<T> {
    fn default() -> Self {
        Self::Missing
    }
}

impl<'de, T> Deserialize<'de> for Supplied<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Self::Value)
    }
}

impl<T: Default> Supplied<T> {
    fn into_value_or_default(self) -> (T, bool) {
        match self {
            Self::Missing => (T::default(), false),
            Self::Value(value) => (value, true),
        }
    }
}

async fn decode_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Result<Json<DecodeRequest>, JsonRejection>,
) -> Response {
    let state = match super::deployments::control_state(state, &headers) {
        Ok(state) => state,
        Err(response) => return response,
    };
    let (request_id, context) = match request_context(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Json(request) = match body {
        Ok(body) => body,
        Err(rejection) => return json_rejection(request_id, rejection),
    };
    match state.werk_control.decode(context, request).await {
        Ok(data) => success(request_id, data),
        Err(error) => protocol_error(request_id, error),
    }
}

pub(super) async fn request_context(
    state: &ApiState,
    headers: &HeaderMap,
) -> Result<(String, ControlContext), Response> {
    let request_id = crate::logging::capture()
        .request_id
        .unwrap_or_else(new_request_id);
    negotiate_protocol(headers).map_err(|error| protocol_error(request_id.clone(), error))?;
    let principal = state
        .werk_principal(headers)
        .await
        .map_err(|error| protocol_error(request_id.clone(), error))?;
    Ok((
        request_id.clone(),
        ControlContext::new(principal, request_id),
    ))
}

fn negotiate_protocol(headers: &HeaderMap) -> Result<(), ProtocolError> {
    for value in headers.get_all(PROTOCOL_VERSION_HEADER) {
        let requested = value
            .to_str()
            .ok()
            .and_then(|value| value.parse::<ProtocolVersion>().ok())
            .ok_or_else(incompatible_protocol)?;
        if !requested.accepts(ProtocolVersion::V1) {
            return Err(incompatible_protocol());
        }
    }
    if !accepts_protocol_response(headers) {
        return Err(incompatible_protocol());
    }
    Ok(())
}

fn accepts_protocol_response(headers: &HeaderMap) -> bool {
    let mut supplied = false;
    for value in headers.get_all(header::ACCEPT) {
        supplied = true;
        let Ok(value) = value.to_str() else {
            continue;
        };
        if value.split(',').any(acceptable_media_range) {
            return true;
        }
    }
    !supplied
}

fn acceptable_media_range(value: &str) -> bool {
    let mut parts = value.split(';');
    let media_type = parts.next().unwrap_or_default().trim();
    let mut quality = 1.0_f32;
    let mut saw_quality = false;
    for parameter in parts {
        let Some((name, value)) = parameter.trim().split_once('=') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("q") {
            if saw_quality {
                return false;
            }
            saw_quality = true;
            let Ok(parsed) = value.trim().parse::<f32>() else {
                return false;
            };
            if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
                return false;
            }
            quality = parsed;
        }
    }
    quality > 0.0
        && (media_type.eq_ignore_ascii_case("application/json")
            || media_type.eq_ignore_ascii_case("application/*")
            || media_type == "*/*")
}

fn incompatible_protocol() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::IncompatibleProtocol,
        "the requested Werk Protocol representation is not supported",
    )
}

fn new_request_id() -> String {
    let mut bytes = [0u8; 16];
    if getrandom::getrandom(&mut bytes).is_err() {
        let sequence = FALLBACK_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        bytes[..8].copy_from_slice(&sequence.rotate_left(17).to_be_bytes());
        bytes[8..].copy_from_slice(&(sequence ^ 0xa6d3_9b27_51c8_e40f).to_be_bytes());
    }
    format!("req_{}", URL_SAFE_NO_PAD.encode(bytes))
}

pub(super) fn success<T: Serialize>(request_id: String, data: T) -> Response {
    json_response(
        StatusCode::OK,
        request_id.clone(),
        ProtocolEnvelope::v1(request_id, data),
    )
}

#[derive(Serialize)]
struct ProtocolErrorEnvelope {
    protocol: ProtocolVersion,
    request_id: String,
    error: ProtocolErrorBody,
}

fn protocol_error(request_id: String, error: ProtocolError) -> Response {
    let status = error_status(error.code);
    let details = safe_error_details(&error);
    let message = match error.code {
        ProtocolErrorCode::Internal => "internal runtime-control error".to_string(),
        ProtocolErrorCode::CorruptState => "runtime state is corrupt".to_string(),
        ProtocolErrorCode::Unauthorized => {
            "authentication credentials are missing or invalid".to_string()
        }
        _ => error.message,
    };
    let envelope = ProtocolErrorEnvelope {
        protocol: ProtocolVersion::V1,
        request_id: request_id.clone(),
        error: ProtocolErrorBody {
            code: error.code,
            message,
            retryable: error.retryable,
            // Never forward arbitrary backend JSON. Compatibility mismatch
            // fields are the sole typed, allow-listed diagnostic exposed by
            // this transport because clients need them to explain a refused
            // restore without receiving paths, prompts, or credentials.
            details,
        },
    };
    let mut response = json_response(status, request_id, envelope);
    if error.code == ProtocolErrorCode::Unauthorized {
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    }
    response
}

fn safe_error_details(error: &ProtocolError) -> Option<serde_json::Value> {
    if error.code != ProtocolErrorCode::IncompatibleState {
        return None;
    }
    let fields = error
        .details
        .as_ref()?
        .as_object()?
        .get("mismatch_fields")?
        .as_array()?;
    const ALLOWED_FIELDS: &[&str] = &[
        "model_fingerprint",
        "tokenizer_fingerprint",
        "prompt_fingerprint",
        "chat_template_fingerprint",
        "backend",
        "backend_version",
        "runtime_adapter_version",
        "accelerator_family",
        "tensor_dtype",
        "kv_dtype",
        "quantization",
        "cache_layout",
        "block_size",
        "context",
        "multimodal_processor_fingerprints",
        "producer_protocol",
    ];
    if fields.len() > ALLOWED_FIELDS.len() {
        return None;
    }
    let mut safe = Vec::with_capacity(fields.len());
    for field in fields {
        let field = field.as_str()?;
        if !ALLOWED_FIELDS.contains(&field) || safe.iter().any(|seen| *seen == field) {
            return None;
        }
        safe.push(field);
    }
    Some(serde_json::json!({ "mismatch_fields": safe }))
}

fn invalid_request(request_id: String, message: &'static str) -> Response {
    protocol_error(
        request_id,
        ProtocolError::new(ProtocolErrorCode::InvalidRequest, message),
    )
}

fn json_rejection(request_id: String, rejection: JsonRejection) -> Response {
    if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        let envelope = ProtocolErrorEnvelope {
            protocol: ProtocolVersion::V1,
            request_id: request_id.clone(),
            error: ProtocolErrorBody {
                code: ProtocolErrorCode::ResourceExhausted,
                message: "request body exceeds the Werk Protocol limit".to_string(),
                retryable: false,
                details: None,
            },
        };
        return json_response(StatusCode::PAYLOAD_TOO_LARGE, request_id, envelope);
    }
    invalid_request(request_id, "invalid JSON request body")
}

fn error_status(code: ProtocolErrorCode) -> StatusCode {
    match code {
        ProtocolErrorCode::InvalidRequest => StatusCode::BAD_REQUEST,
        ProtocolErrorCode::IncompatibleProtocol => StatusCode::NOT_ACCEPTABLE,
        ProtocolErrorCode::Unauthorized => StatusCode::UNAUTHORIZED,
        ProtocolErrorCode::Forbidden => StatusCode::FORBIDDEN,
        ProtocolErrorCode::NotFound => StatusCode::NOT_FOUND,
        ProtocolErrorCode::Conflict | ProtocolErrorCode::IncompatibleState => StatusCode::CONFLICT,
        ProtocolErrorCode::ExpiredHandoff => StatusCode::GONE,
        ProtocolErrorCode::ExperimentalOptInRequired => StatusCode::PRECONDITION_REQUIRED,
        ProtocolErrorCode::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
        ProtocolErrorCode::Unsupported => StatusCode::NOT_IMPLEMENTED,
        ProtocolErrorCode::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        ProtocolErrorCode::CorruptState | ProtocolErrorCode::Internal => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

fn json_response<T: Serialize>(status: StatusCode, request_id: String, body: T) -> Response {
    let (status, payload) = match serde_json::to_vec(&body) {
        Ok(payload) if payload.len() <= MAX_WERK_RESPONSE_BYTES => (status, payload),
        Ok(_) => {
            let fallback = ProtocolErrorEnvelope {
                protocol: ProtocolVersion::V1,
                request_id: request_id.clone(),
                error: ProtocolErrorBody {
                    code: ProtocolErrorCode::ResourceExhausted,
                    message: "runtime-control response exceeds the protocol limit".to_string(),
                    retryable: false,
                    details: None,
                },
            };
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::to_vec(&fallback)
                    .expect("the static Werk Protocol error envelope is serializable"),
            )
        }
        Err(_) => {
            let fallback = ProtocolErrorEnvelope {
                protocol: ProtocolVersion::V1,
                request_id: request_id.clone(),
                error: ProtocolErrorBody {
                    code: ProtocolErrorCode::Internal,
                    message: "internal runtime-control error".to_string(),
                    retryable: false,
                    details: None,
                },
            };
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::to_vec(&fallback)
                    .expect("the static Werk Protocol error envelope is serializable"),
            )
        }
    };
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CACHE_CONTROL, "no-store")
        .header(header::VARY, header::ACCEPT.as_str())
        .header(PROTOCOL_VERSION_HEADER, ProtocolVersion::V1.to_string())
        .header("x-content-type-options", "nosniff");
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        builder = builder.header("x-werk-request-id", value);
    }
    builder
        .body(Body::from(payload))
        .expect("Werk Protocol response headers are valid")
}

#[cfg(test)]
mod tests;

use super::*;
use crate::{
    backend::{
        GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent, GenerationBackend,
        GenerationTimings,
    },
    model_store::{ModelManifest, ModelStore},
    werk_protocol::{
        BoxControlFuture, CapabilitiesResponse, DecodeResponse, ExpertActionResponse,
        ExpertListResponse, MemoryStatusResponse, MemoryTierStatus, PrefillResponse, PressureLevel,
        ProtocolLimits, ProtocolResult, PruneStatesResponse, RuntimeInfo, StateActionResponse,
        StateListResponse, WerkControl,
    },
};
use axum::{
    body::{self, Body},
    http::{Method, Request},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tower::ServiceExt;

#[derive(Clone)]
struct TestGenerationBackend;

impl GenerationBackend for TestGenerationBackend {
    fn generate(
        &self,
        _manifest: &ModelManifest,
        _request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        Ok(GenerateResponse {
            text: String::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: "stop".to_string(),
            timings: GenerationTimings {
                load_seconds: 0.0,
                warmup_seconds: 0.0,
                first_token_seconds: 0.0,
                prompt_seconds: 0.0,
                decode_seconds: 0.0,
                total_seconds: 0.0,
            },
            backend_diagnostics: Vec::new(),
        })
    }

    fn generate_stream(
        &self,
        _manifest: ModelManifest,
        _request: GenerateRequest,
    ) -> GenerateStream {
        Box::pin(tokio_stream::empty::<Result<GenerateStreamEvent, String>>())
    }
}

#[derive(Clone, Default)]
struct FakeControl {
    principals: Arc<Mutex<Vec<String>>>,
    info_error: Option<ProtocolError>,
}

impl WerkControl for FakeControl {
    fn info(&self, context: ControlContext) -> BoxControlFuture<'_, RuntimeInfo> {
        let principals = self.principals.clone();
        let error = self.info_error.clone();
        Box::pin(async move {
            if let Some(error) = error {
                return Err(error);
            }
            principals
                .lock()
                .expect("principal recorder")
                .push(context.principal_id().to_string());
            Ok(RuntimeInfo {
                service: "test".to_string(),
                service_version: "1".to_string(),
                protocol: ProtocolVersion::V1,
                active_backend: "test".to_string(),
                limits: ProtocolLimits::default(),
            })
        })
    }

    fn capabilities(&self, _context: ControlContext) -> BoxControlFuture<'_, CapabilitiesResponse> {
        ready(Ok(CapabilitiesResponse {
            capabilities: Vec::new(),
        }))
    }

    fn list_states(
        &self,
        _context: ControlContext,
        _filter: StateListFilter,
    ) -> BoxControlFuture<'_, StateListResponse> {
        ready(Ok(StateListResponse {
            states: Vec::new(),
            next_cursor: None,
        }))
    }

    fn state_action(
        &self,
        _context: ControlContext,
        _state_id: String,
        _request: StateActionRequest,
    ) -> BoxControlFuture<'_, StateActionResponse> {
        ready(Err(ProtocolError::new(
            ProtocolErrorCode::NotFound,
            "runtime state was not found",
        )))
    }

    fn prune_states(
        &self,
        _context: ControlContext,
        request: PruneStatesRequest,
    ) -> BoxControlFuture<'_, PruneStatesResponse> {
        ready(Ok(PruneStatesResponse {
            matched: 0,
            removed: 0,
            bytes: Some(0),
            dry_run: request.dry_run,
        }))
    }

    fn memory_status(
        &self,
        _context: ControlContext,
    ) -> BoxControlFuture<'_, MemoryStatusResponse> {
        let tier = MemoryTierStatus {
            capacity_bytes: None,
            available_bytes: None,
            managed_bytes: 0,
            reserved_bytes: 0,
            pressure: PressureLevel::Unknown,
        };
        ready(Ok(MemoryStatusResponse {
            observed_at_unix_ms: 1,
            overall_pressure: PressureLevel::Unknown,
            topology: "unknown".to_string(),
            host: tier.clone(),
            accelerator: tier,
            last_action_unix_ms: None,
            counters: BTreeMap::new(),
        }))
    }

    fn list_experts(
        &self,
        _context: ControlContext,
        _filter: ExpertListFilter,
    ) -> BoxControlFuture<'_, ExpertListResponse> {
        ready(Ok(ExpertListResponse {
            experts: Vec::new(),
            next_cursor: None,
        }))
    }

    fn expert_action(
        &self,
        _context: ControlContext,
        request: ExpertActionRequest,
    ) -> BoxControlFuture<'_, ExpertActionResponse> {
        ready(Ok(ExpertActionResponse {
            experts: Vec::new(),
            changed: 0,
            dry_run: request.dry_run,
        }))
    }

    fn prefill(
        &self,
        _context: ControlContext,
        _request: PrefillRequest,
    ) -> BoxControlFuture<'_, PrefillResponse> {
        ready(Err(ProtocolError::new(
            ProtocolErrorCode::Unsupported,
            "prefill is unsupported",
        )))
    }

    fn decode(
        &self,
        _context: ControlContext,
        _request: DecodeRequest,
    ) -> BoxControlFuture<'_, DecodeResponse> {
        ready(Err(ProtocolError::new(
            ProtocolErrorCode::Unsupported,
            "decode is unsupported",
        )))
    }
}

fn ready<T: Send + 'static>(result: ProtocolResult<T>) -> BoxControlFuture<'static, T> {
    Box::pin(async move { result })
}

fn app(control: FakeControl, api_keys: Vec<String>) -> Router {
    super::super::router::router(
        ApiState::new(test_store(), Arc::new(TestGenerationBackend))
            .with_werk_control(Arc::new(control))
            .with_api_keys(api_keys),
    )
}

fn test_store() -> ModelStore {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    ModelStore::resolve(Some(std::env::temp_dir().join(format!(
        "werk-api-control-test-{}-{nonce}",
        std::process::id()
    ))))
    .expect("test store")
}

async fn request(
    app: &Router,
    method: Method,
    uri: &str,
    body: Option<String>,
    credential: Option<(&str, &str)>,
) -> Response {
    let mut builder = Request::builder().method(method).uri(uri);
    if body.is_some() {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
    }
    if let Some((name, value)) = credential {
        builder = builder.header(name, value);
    }
    app.clone()
        .oneshot(
            builder
                .body(Body::from(body.unwrap_or_default()))
                .expect("request"),
        )
        .await
        .expect("response")
}

async fn json_body(response: Response) -> Value {
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("JSON response")
}

#[tokio::test]
async fn werk_routes_are_additive_versioned_and_no_store() {
    let app = app(FakeControl::default(), Vec::new());

    let response = request(&app, Method::GET, "/werk/v1/info", None, None).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
    assert!(response.headers().contains_key("x-werk-request-id"));
    assert_eq!(
        response.headers()[PROTOCOL_VERSION_HEADER],
        ProtocolVersion::V1.to_string()
    );
    assert_eq!(response.headers()[header::VARY], "accept");
    let body = json_body(response).await;
    assert_eq!(body["protocol"], json!({"major": 1, "minor": 0}));
    assert_eq!(body["data"]["service"], "test");
    assert!(body["request_id"].as_str().unwrap().starts_with("req_"));

    let legacy = request(&app, Method::GET, "/v1/models", None, None).await;
    assert_eq!(legacy.status(), StatusCode::OK);
}

#[tokio::test]
async fn protocol_content_negotiation_is_explicit_and_backward_compatible() {
    let app = app(FakeControl::default(), Vec::new());

    for accept in ["application/json", "application/*", "*/*"] {
        let response = request(
            &app,
            Method::GET,
            "/werk/v1/info",
            None,
            Some(("accept", accept)),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK, "{accept}");
    }

    let response = request(
        &app,
        Method::GET,
        "/werk/v1/info",
        None,
        Some((
            "accept",
            "application/vnd.werk.v2+json, application/json;q=0",
        )),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_ACCEPTABLE);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "incompatible_protocol");

    let response = request(
        &app,
        Method::GET,
        "/werk/v1/info",
        None,
        Some((PROTOCOL_VERSION_HEADER, "2.0")),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_ACCEPTABLE);

    let future_v1_client = request(
        &app,
        Method::GET,
        "/werk/v1/info",
        None,
        Some((PROTOCOL_VERSION_HEADER, "1.1")),
    )
    .await;
    assert_eq!(future_v1_client.status(), StatusCode::OK);
}

#[tokio::test]
async fn all_control_routes_dispatch_and_use_typed_errors() {
    let app = app(FakeControl::default(), Vec::new());
    let requests = [
        (Method::GET, "/werk/v1/capabilities", None, StatusCode::OK),
        (Method::GET, "/werk/v1/memory", None, StatusCode::OK),
        (Method::GET, "/werk/v1/states", None, StatusCode::OK),
        (Method::GET, "/werk/v1/experts", None, StatusCode::OK),
        (
            Method::POST,
            "/werk/v1/states/state_1/actions",
            Some(json!({"action":"pin"}).to_string()),
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/werk/v1/states/prune",
            Some(json!({"selector":{"kind":"ids","ids":["state_1"]}}).to_string()),
            StatusCode::OK,
        ),
        (
            Method::POST,
            "/werk/v1/experts/actions",
            Some(
                json!({
                    "model_id":"model",
                    "expert_ids":["expert_1"],
                    "action":"pin",
                    "target_tier":null
                })
                .to_string(),
            ),
            StatusCode::OK,
        ),
        (
            Method::POST,
            "/werk/v1/prefill",
            Some(json!({"model_id":"model","input":{"type":"text","text":"x"}}).to_string()),
            StatusCode::NOT_IMPLEMENTED,
        ),
        (
            Method::POST,
            "/werk/v1/decode",
            Some(
                json!({
                    "handoff":"opaque",
                    "max_tokens":1,
                    "temperature":null,
                    "top_p":null,
                    "seed":null
                })
                .to_string(),
            ),
            StatusCode::NOT_IMPLEMENTED,
        ),
    ];

    for (method, uri, body, expected) in requests {
        let response = request(&app, method, uri, body, None).await;
        assert_eq!(response.status(), expected, "{uri}");
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let body = json_body(response).await;
        assert_eq!(body["protocol"], json!({"major": 1, "minor": 0}));
        assert!(body["request_id"].is_string());
    }
}

#[tokio::test]
async fn authentication_derives_distinct_opaque_principals() {
    let control = FakeControl::default();
    let principals = control.principals.clone();
    let app = app(
        control,
        vec!["alpha-secret".to_string(), "beta-secret".to_string()],
    );

    let unauthorized = request(&app, Method::GET, "/werk/v1/info", None, None).await;
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(unauthorized.headers()[header::WWW_AUTHENTICATE], "Bearer");
    let unauthorized = json_body(unauthorized).await;
    assert_eq!(unauthorized["error"]["code"], "unauthorized");
    assert!(!unauthorized.to_string().contains("alpha-secret"));

    let bearer = request(
        &app,
        Method::GET,
        "/werk/v1/info",
        None,
        Some(("authorization", "Bearer alpha-secret")),
    )
    .await;
    assert_eq!(bearer.status(), StatusCode::OK);
    let api_key = request(
        &app,
        Method::GET,
        "/werk/v1/info",
        None,
        Some(("x-api-key", "beta-secret")),
    )
    .await;
    assert_eq!(api_key.status(), StatusCode::OK);

    let principals = principals.lock().expect("principal recorder");
    assert_eq!(principals.len(), 2);
    assert_ne!(principals[0], principals[1]);
    for principal in principals.iter() {
        assert!(principal.starts_with("p_"));
        assert!(!principal.contains("secret"));
    }
}

#[tokio::test]
async fn oversized_and_internal_errors_are_safe_json_envelopes() {
    let oversized_app = app(FakeControl::default(), Vec::new());
    let oversized = request(
        &oversized_app,
        Method::POST,
        "/werk/v1/prefill",
        Some(format!("\"{}\"", "x".repeat(MAX_WERK_BODY_BYTES + 1))),
        None,
    )
    .await;
    assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(oversized.headers()[header::CACHE_CONTROL], "no-store");
    let oversized = json_body(oversized).await;
    assert_eq!(oversized["error"]["code"], "resource_exhausted");

    let app = app(
        FakeControl {
            principals: Arc::default(),
            info_error: Some(
                ProtocolError::new(
                    ProtocolErrorCode::Internal,
                    "failed at /home/person/private with alpha-secret",
                )
                .with_details(json!({"path":"/home/person/private"})),
            ),
        },
        Vec::new(),
    );
    let response = request(&app, Method::GET, "/werk/v1/info", None, None).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = json_body(response).await;
    let serialized = body.to_string();
    assert_eq!(body["error"]["code"], "internal");
    assert_eq!(body["error"]["message"], "internal runtime-control error");
    assert!(!serialized.contains("/home/person/private"));
    assert!(!serialized.contains("alpha-secret"));
}

#[tokio::test]
async fn incompatible_state_exposes_only_allow_listed_mismatch_fields() {
    let valid_app = app(
        FakeControl {
            principals: Arc::default(),
            info_error: Some(
                ProtocolError::new(
                    ProtocolErrorCode::IncompatibleState,
                    "runtime state is incompatible",
                )
                .with_details(json!({
                    "mismatch_fields": ["backend", "kv_dtype"]
                })),
            ),
        },
        Vec::new(),
    );
    let response = request(&valid_app, Method::GET, "/werk/v1/info", None, None).await;
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let body = json_body(response).await;
    assert_eq!(
        body["error"]["details"]["mismatch_fields"],
        json!(["backend", "kv_dtype"])
    );

    let invalid_app = app(
        FakeControl {
            principals: Arc::default(),
            info_error: Some(
                ProtocolError::new(
                    ProtocolErrorCode::IncompatibleState,
                    "runtime state is incompatible",
                )
                .with_details(json!({
                    "mismatch_fields": ["backend", "/private/path"]
                })),
            ),
        },
        Vec::new(),
    );
    let response = request(&invalid_app, Method::GET, "/werk/v1/info", None, None).await;
    let body = json_body(response).await;
    assert!(body["error"].get("details").is_none());
    assert!(!body.to_string().contains("/private/path"));
}

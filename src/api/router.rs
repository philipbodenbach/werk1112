use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::{HeaderName, HeaderValue, Method, header},
    routing::{get, post},
};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, CorsLayer};

use super::{
    automatic1111::{
        get_options_handler, progress_handler, sd_models_handler, set_options_handler,
        txt2img_handler,
    },
    chat::{chat_completions_handler, model_handler, models_handler},
    media::{
        audio_generations_handler, audio_speech_handler, audio_transcriptions_handler,
        audio_translations_handler, cancel_job_handler, capabilities_handler,
        comfy_image_edits_unsupported_handler, create_job_handler, get_job_handler,
        image_edits_handler, image_generations_handler, output_handler, parameters_handler,
        video_generations_handler,
    },
    state::ApiState,
    werk,
};
use crate::werk_protocol::PROTOCOL_VERSION_HEADER;

const DEFAULT_API_BODY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const MAX_API_BODY_LIMIT_BYTES: usize = 512 * 1024 * 1024;

pub fn router(state: ApiState) -> Router {
    router_with_body_limit(state, configured_api_body_limit_bytes())
}

pub(in crate::api) fn router_with_body_limit(state: ApiState, body_limit_bytes: usize) -> Router {
    let cors_origins = state
        .cors_origins()
        .iter()
        .map(|origin| origin.header_value())
        .collect::<Vec<_>>();
    let router = Router::new()
        .route("/metrics", get(super::observability::metrics))
        .route(
            "/werk/v1/observability",
            get(super::observability::snapshot),
        )
        .route(
            "/v1/files",
            get(super::files::list)
                .post(super::files::upload)
                .layer::<_, std::convert::Infallible>(axum::middleware::from_fn(
                    super::files::response_headers,
                ))
                .layer(DefaultBodyLimit::max(
                    (crate::file_store::MAX_FILE_BYTES + 1024 * 1024).min(body_limit_bytes),
                )),
        )
        .route(
            "/v1/files/{id}",
            get(super::files::retrieve)
                .delete(super::files::delete)
                .layer(axum::middleware::from_fn(super::files::response_headers)),
        )
        .route(
            "/v1/files/{id}/content",
            get(super::files::content)
                .layer(axum::middleware::from_fn(super::files::response_headers)),
        )
        .route(
            "/v1/messages/count_tokens",
            post(super::anthropic::count_tokens_handler)
                .layer(DefaultBodyLimit::max(body_limit_bytes)),
        )
        .route(
            "/v1/messages",
            post(super::anthropic::messages_handler).layer(DefaultBodyLimit::max(body_limit_bytes)),
        )
        .route("/v1/models", get(models_handler))
        .route(
            "/v1/embeddings",
            post(super::text_analysis::embeddings).layer(DefaultBodyLimit::max(1024 * 1024)),
        )
        .route(
            "/v1/rerank",
            post(super::text_analysis::rerank).layer(DefaultBodyLimit::max(1024 * 1024)),
        )
        .route(
            "/rerank",
            post(super::text_analysis::rerank).layer(DefaultBodyLimit::max(1024 * 1024)),
        )
        .route(
            "/v1/classifications",
            post(super::text_analysis::classify).layer(DefaultBodyLimit::max(1024 * 1024)),
        )
        .route(
            "/v1/systemone",
            post(super::text_analysis::classify).layer(DefaultBodyLimit::max(1024 * 1024)),
        )
        .route("/werk/v1/deployments", get(super::deployments::diagnostics))
        .route("/v1/models/{id}", get(model_handler))
        .route(
            "/v1/chat/completions",
            post(chat_completions_handler).layer(DefaultBodyLimit::max(body_limit_bytes)),
        )
        .route("/v1/images/generations", post(image_generations_handler))
        .route("/v1/images/edits", post(image_edits_handler))
        .route(
            "/proxy/openai/images/generations",
            post(image_generations_handler),
        )
        .route(
            "/proxy/openai/images/edits",
            post(comfy_image_edits_unsupported_handler),
        )
        .route("/v1/videos/generations", post(video_generations_handler))
        .route("/v1/audio/generations", post(audio_generations_handler))
        .route("/v1/audio/speech", post(audio_speech_handler))
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcriptions_handler).layer(DefaultBodyLimit::max(body_limit_bytes)),
        )
        .route(
            "/v1/audio/translations",
            post(audio_translations_handler).layer(DefaultBodyLimit::max(body_limit_bytes)),
        )
        .route("/v1/capabilities", get(capabilities_handler))
        .route("/v1/tools", get(super::tools::list))
        .route(
            "/v1/tools/call",
            post(super::tools::call).layer(DefaultBodyLimit::max(body_limit_bytes)),
        )
        .route("/v1/parameters", get(parameters_handler))
        .route("/v1/outputs/{id}", get(output_handler))
        .route(
            "/v1/jobs",
            post(create_job_handler).layer(DefaultBodyLimit::max(body_limit_bytes)),
        )
        .route(
            "/v1/jobs/{id}",
            get(get_job_handler).delete(cancel_job_handler),
        )
        .route("/sdapi/v1/txt2img", post(txt2img_handler))
        .route("/sdapi/v1/sd-models", get(sd_models_handler))
        .route(
            "/sdapi/v1/options",
            get(get_options_handler).post(set_options_handler),
        )
        .route("/sdapi/v1/progress", get(progress_handler))
        .merge(werk::routes())
        .with_state(state);

    if cors_origins.is_empty() {
        router
    } else {
        router.layer(browser_cors_layer(cors_origins))
    }
}

fn configured_api_body_limit_bytes() -> usize {
    std::env::var("WERK_API_BODY_LIMIT_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=MAX_API_BODY_LIMIT_BYTES).contains(value))
        .unwrap_or(DEFAULT_API_BODY_LIMIT_BYTES)
}

fn browser_cors_layer(origins: Vec<HeaderValue>) -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            header::ACCEPT,
            HeaderName::from_static(PROTOCOL_VERSION_HEADER),
            HeaderName::from_static("x-api-key"),
            HeaderName::from_static("x-werk-session-id"),
            HeaderName::from_static("x-werk-deployment"),
            HeaderName::from_static("anthropic-version"),
            HeaderName::from_static("anthropic-beta"),
            HeaderName::from_static("openai-organization"),
            HeaderName::from_static("openai-project"),
            HeaderName::from_static("x-stainless-lang"),
            HeaderName::from_static("x-stainless-package-version"),
            HeaderName::from_static("x-stainless-os"),
            HeaderName::from_static("x-stainless-arch"),
            HeaderName::from_static("x-stainless-runtime"),
            HeaderName::from_static("x-stainless-runtime-version"),
            HeaderName::from_static("x-stainless-retry-count"),
            HeaderName::from_static("x-stainless-timeout"),
            HeaderName::from_static("x-stainless-read-timeout"),
            HeaderName::from_static("x-stainless-helper-method"),
            HeaderName::from_static("x-stainless-async"),
        ])
        .expose_headers([
            HeaderName::from_static("request-id"),
            HeaderName::from_static("x-werk-output-id"),
            HeaderName::from_static("x-werk-request-id"),
            HeaderName::from_static(PROTOCOL_VERSION_HEADER),
        ])
}

pub async fn serve(addr: SocketAddr, state: ApiState) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    serve_with_listener(listener, state).await
}

/// Serve on a listener reserved before potentially expensive model preparation.
pub async fn serve_with_listener(listener: TcpListener, state: ApiState) -> anyhow::Result<()> {
    let addr = listener.local_addr()?;
    let console = crate::terminal::interactive(crate::terminal::Stream::Out);
    if console {
        crate::terminal::panel(
            crate::terminal::Stream::Out,
            "Server ready",
            &format!(
                "LISTENING\nhttp://{addr}\n\nDefault model: {}\nAuthentication: {}\n\nAPI\nOpenAI: /v1 · Anthropic: /v1/messages · Werk: /werk/v1\n\nMONITOR\nwerk top --url http://{addr}\n\nCtrl+C stops the server. Requests appear below.",
                state
                    .default_model
                    .as_deref()
                    .unwrap_or("select a model per request"),
                if state.api_key_auth_enabled() {
                    "enabled · Bearer / X-API-Key"
                } else {
                    "disabled"
                }
            ),
        );
        crate::terminal::heading(crate::terminal::Stream::Out, "Requests");
    } else {
        crate::ui_println!("Server running at http://{addr}");
    }
    if !console && state.api_key_auth_enabled() {
        crate::ui_println!(
            "API key auth enabled; use Authorization: Bearer <key> or X-API-Key: <key> (A1111 clients may use Basic werk:<key>)"
        );
    }
    let app = router(state);
    let app = if console {
        app.layer(axum::middleware::from_fn(console_request))
    } else {
        app
    };
    axum::serve(listener, app).await?;
    Ok(())
}

async fn console_request(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = request.method().clone();
    let path = request
        .extensions()
        .get::<axum::extract::MatchedPath>()
        .map(|path| path.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());
    let started = std::time::Instant::now();
    let response = next.run(request).await;
    // Headers can precede a streaming body. Do not call this inference duration.
    if !matches!(
        path.as_str(),
        "/health" | "/werk/v1/observability" | "/metrics"
    ) {
        crate::ui_println!(
            "{}  {} {}  {} · headers {:.1} ms",
            chrono::DateTime::<chrono::Utc>::from(std::time::SystemTime::now())
                .format("%H:%M:%S UTC"),
            method,
            path,
            response.status().as_u16(),
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    response
}

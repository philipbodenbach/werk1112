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
};

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
        .route("/v1/models", get(models_handler))
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
            HeaderName::from_static("x-api-key"),
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
        .expose_headers([HeaderName::from_static("x-werk-output-id")])
}

pub async fn serve(addr: SocketAddr, state: ApiState) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    println!("Server running at http://{addr}");
    if state.api_key_auth_enabled() {
        println!(
            "API key auth enabled; use Authorization: Bearer <key> or X-API-Key: <key> (A1111 clients may use Basic werk:<key>)"
        );
    }
    axum::serve(listener, router(state)).await?;
    Ok(())
}

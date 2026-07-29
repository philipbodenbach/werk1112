use axum::{
    Json,
    extract::{
        Query, State,
        rejection::{JsonRejection, QueryRejection},
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc};

use crate::capabilities::InferenceTask;

use super::{
    super::state::ApiState,
    auth::authorize_automatic1111,
    compatibility::harmless_default,
    request::{ProgressQuery, Txt2ImgRequest},
    response::{ProgressResponse, SdModelItem, automatic1111_error, execute_txt2img},
};

pub(in crate::api) async fn txt2img_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    payload: Result<Json<Txt2ImgRequest>, JsonRejection>,
) -> Response {
    if let Err(response) = authorize_automatic1111(&state, &headers) {
        return response;
    }
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(rejection) => return json_rejection_response(rejection),
    };
    if let Err(error) = request.validate() {
        return automatic1111_error(StatusCode::UNPROCESSABLE_ENTITY, error);
    }

    let model = match resolve_checkpoint(&state, request.checkpoint_override()) {
        Ok(model) => model,
        Err(error) => return automatic1111_error(StatusCode::BAD_REQUEST, error),
    };
    let seed = match concrete_seed(request.seed()) {
        Ok(seed) => seed,
        Err(error) => return automatic1111_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let parameters = request.normalized_parameters();
    let warnings = request.compatibility_warnings();
    let send_images = request.send_images();
    let steps = request.steps();
    let batch_size = request.batch_size();
    let image_count = request.image_count();
    let inference = request.into_inference(model, seed);

    let compatibility = Arc::clone(&state.automatic1111);
    let generation_lock = Arc::clone(&compatibility.generation_lock)
        .lock_owned()
        .await;
    let service = Arc::clone(&state.inference_service);
    match tokio::task::spawn_blocking(move || {
        // Keep both guards in the worker so a disconnected HTTP client does
        // not make an ongoing generation look idle or bypass serialization.
        let _generation_lock = generation_lock;
        let _active = compatibility.begin(steps);
        execute_txt2img(
            service,
            inference,
            parameters,
            send_images,
            warnings,
            seed,
            batch_size,
            image_count,
        )
    })
    .await
    {
        Ok(Ok(response)) => Json(response).into_response(),
        Ok(Err(error)) => automatic1111_error(StatusCode::BAD_REQUEST, error),
        Err(error) => automatic1111_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("AUTOMATIC1111 generation task failed: {error}"),
        ),
    }
}

pub(in crate::api) async fn sd_models_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = authorize_automatic1111(&state, &headers) {
        return response;
    }
    match state.store.list() {
        Ok(models) => Json(
            models
                .into_iter()
                .filter(|model| model.supports_task(InferenceTask::ImageGeneration))
                .map(|model| SdModelItem {
                    title: model.id.clone(),
                    model_name: model.id.clone(),
                    hash: None,
                    sha256: None,
                    // A1111 requires a string here. An identifier avoids leaking a
                    // server-side absolute model path to remote clients.
                    filename: model.id,
                    config: None,
                })
                .collect::<Vec<_>>(),
        )
        .into_response(),
        Err(error) => automatic1111_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
    }
}

pub(in crate::api) async fn get_options_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = authorize_automatic1111(&state, &headers) {
        return response;
    }
    let checkpoint = if let Some(checkpoint) = selected_or_default_checkpoint(&state) {
        checkpoint
    } else {
        match image_checkpoint_ids(&state) {
            Ok(checkpoints) => match checkpoints.as_slice() {
                [checkpoint] => checkpoint.clone(),
                _ => String::new(),
            },
            Err(error) => {
                return automatic1111_error(StatusCode::INTERNAL_SERVER_ERROR, error);
            }
        }
    };
    Json(json!({
        "sd_model_checkpoint": checkpoint,
        "samples_format": "png"
    }))
    .into_response()
}

pub(in crate::api) async fn set_options_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    payload: Result<Json<BTreeMap<String, Value>>, JsonRejection>,
) -> Response {
    if let Err(response) = authorize_automatic1111(&state, &headers) {
        return response;
    }
    let Json(options) = match payload {
        Ok(payload) => payload,
        Err(rejection) => return json_rejection_response(rejection),
    };
    for (name, value) in &options {
        match name.as_str() {
            "sd_model_checkpoint" => {
                let Some(checkpoint) = value.as_str() else {
                    return automatic1111_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "sd_model_checkpoint must be a string".to_string(),
                    );
                };
                if let Err(error) = validate_image_checkpoint(&state, checkpoint) {
                    return automatic1111_error(StatusCode::BAD_REQUEST, error);
                }
            }
            "samples_format" if value.as_str() == Some("png") => {}
            _ if harmless_default(value) => {}
            _ => {
                return automatic1111_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("AUTOMATIC1111 option '{name}' is not supported by Werk"),
                );
            }
        }
    }
    if let Some(checkpoint) = options.get("sd_model_checkpoint").and_then(Value::as_str) {
        state
            .automatic1111
            .set_selected_checkpoint(Some(checkpoint.to_string()));
    }
    Json(Value::Null).into_response()
}

pub(in crate::api) async fn progress_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    query: Result<Query<ProgressQuery>, QueryRejection>,
) -> Response {
    if let Err(response) = authorize_automatic1111(&state, &headers) {
        return response;
    }
    let Query(query) = match query {
        Ok(query) => query,
        Err(rejection) => return query_rejection_response(rejection),
    };
    let snapshot = state.automatic1111.progress();
    let _skip_current_image = query.skip_current_image;
    let (progress, state_value, textinfo) = if snapshot.active {
        (
            0.01,
            progress_state(
                "scripts_txt2img",
                1,
                snapshot.started_unix.to_string(),
                snapshot.steps,
            ),
            Some(
                "Generating via Werk inference router; step-level progress is unavailable"
                    .to_string(),
            ),
        )
    } else {
        (0.0, progress_state("", 0, "0".to_string(), 0), None)
    };
    Json(ProgressResponse {
        progress,
        eta_relative: 0.0,
        state: state_value,
        current_image: None,
        textinfo,
    })
    .into_response()
}

fn progress_state(job: &str, job_count: u32, job_timestamp: String, steps: u32) -> Value {
    json!({
        "skipped": false,
        "interrupted": false,
        "stopping_generation": false,
        "job": job,
        "job_count": job_count,
        "job_timestamp": job_timestamp,
        "job_no": 0,
        "sampling_step": 0,
        "sampling_steps": steps
    })
}

fn resolve_checkpoint(state: &ApiState, requested: Option<&str>) -> Result<String, String> {
    if let Some(checkpoint) = requested {
        validate_image_checkpoint(state, checkpoint)?;
        return Ok(checkpoint.to_string());
    }
    if let Some(checkpoint) = selected_or_default_checkpoint(state) {
        validate_image_checkpoint(state, &checkpoint)?;
        return Ok(checkpoint);
    }
    let candidates = image_checkpoint_ids(state)?;
    match candidates.as_slice() {
        [checkpoint] => Ok(checkpoint.clone()),
        [] => Err("no installed model declares image-generation".to_string()),
        _ => Err(
            "multiple image-generation models are installed; configure --image-model or POST sd_model_checkpoint to /sdapi/v1/options"
                .to_string(),
        ),
    }
}

fn selected_or_default_checkpoint(state: &ApiState) -> Option<String> {
    state
        .automatic1111
        .selected_checkpoint()
        .or_else(|| state.default_image_model.clone())
        .or_else(|| {
            state.default_model.as_ref().and_then(|model| {
                state
                    .store
                    .get(model)
                    .ok()
                    .filter(|manifest| manifest.supports_task(InferenceTask::ImageGeneration))
                    .map(|manifest| manifest.id)
            })
        })
}

fn validate_image_checkpoint(state: &ApiState, checkpoint: &str) -> Result<(), String> {
    let manifest = state
        .store
        .get(checkpoint)
        .map_err(|_| format!("model '{checkpoint}' was not found"))?;
    if !manifest.supports_task(InferenceTask::ImageGeneration) {
        return Err(format!(
            "model '{}' does not declare image-generation",
            manifest.id
        ));
    }
    Ok(())
}

fn image_checkpoint_ids(state: &ApiState) -> Result<Vec<String>, String> {
    state
        .store
        .list()
        .map_err(|error| error.to_string())
        .map(|models| {
            models
                .into_iter()
                .filter(|model| model.supports_task(InferenceTask::ImageGeneration))
                .map(|model| model.id)
                .collect()
        })
}

fn concrete_seed(seed: i64) -> Result<u64, String> {
    if seed >= 0 {
        return Ok(seed as u64);
    }
    let mut bytes = [0_u8; 8];
    getrandom::getrandom(&mut bytes)
        .map_err(|error| format!("failed to generate random seed: {error}"))?;
    Ok(u64::from_le_bytes(bytes) & i64::MAX as u64)
}

fn json_rejection_response(rejection: JsonRejection) -> Response {
    automatic1111_error(rejection.status(), rejection.body_text())
}

fn query_rejection_response(rejection: QueryRejection) -> Response {
    automatic1111_error(rejection.status(), rejection.body_text())
}

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{fs, sync::Arc};

use crate::{
    inference::InferenceRequest,
    inference_service::{InferenceResult, InferenceService},
};

use super::super::media::output::encode_base64;

#[derive(Debug, Serialize)]
pub(super) struct Txt2ImgResponse {
    images: Vec<String>,
    parameters: Value,
    info: String,
}

#[derive(Debug, Serialize)]
pub(super) struct SdModelItem {
    pub(super) title: String,
    pub(super) model_name: String,
    pub(super) hash: Option<String>,
    pub(super) sha256: Option<String>,
    pub(super) filename: String,
    pub(super) config: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct ProgressResponse {
    pub(super) progress: f64,
    pub(super) eta_relative: f64,
    pub(super) state: Value,
    pub(super) current_image: Option<String>,
    pub(super) textinfo: Option<String>,
}

#[derive(Debug, Serialize)]
struct Automatic1111Error {
    error: &'static str,
    detail: String,
    body: String,
    errors: String,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_txt2img(
    service: Arc<InferenceService>,
    request: InferenceRequest,
    parameters: Value,
    send_images: bool,
    mut compatibility_warnings: Vec<String>,
    seed: u64,
    batch_size: u32,
    image_count: u32,
) -> Result<Txt2ImgResponse, String> {
    let result = service
        .execute(request)
        .map_err(|error| error.to_string())?;
    let result_id = result.id.clone();
    let prepared = prepare_txt2img_response(
        &result,
        parameters,
        send_images,
        &mut compatibility_warnings,
        seed,
        batch_size,
        image_count,
    );
    let cleanup = service.output_store().remove_result(&result_id);
    match (prepared, cleanup) {
        (Ok(mut response), Ok(())) => {
            append_info_warning(&mut response.info, compatibility_warnings);
            Ok(response)
        }
        (Ok(mut response), Err(error)) => {
            compatibility_warnings.push(format!(
                "temporary output cleanup failed for result '{result_id}': {error:#}"
            ));
            append_info_warning(&mut response.info, compatibility_warnings);
            Ok(response)
        }
        (Err(error), _) => Err(error),
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_txt2img_response(
    result: &InferenceResult,
    parameters: Value,
    send_images: bool,
    compatibility_warnings: &mut Vec<String>,
    seed: u64,
    batch_size: u32,
    image_count: u32,
) -> Result<Txt2ImgResponse, String> {
    if result.outputs.is_empty() {
        return Err("image backend did not produce an output".to_string());
    }
    let images = if send_images {
        result
            .outputs
            .iter()
            .map(|output| {
                fs::read(&output.path)
                    .map(|bytes| encode_base64(&bytes))
                    .map_err(|error| format!("failed to read generated image: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };

    compatibility_warnings.extend(result.warnings.iter().cloned());
    let actual_output_count = result.outputs.len();
    if actual_output_count != image_count as usize {
        compatibility_warnings.push(format!(
            "requested {image_count} image(s), but the backend returned {actual_output_count}"
        ));
    }
    if actual_output_count > 1 && result.outputs.iter().any(|output| output.seed.is_none()) {
        compatibility_warnings.push(
            "the backend did not report every per-image seed; missing values use the effective request seed"
                .to_string(),
        );
    }
    let effective = &result.effective_request;
    let width = effective.u64_parameter("image.width").unwrap_or(0);
    let height = effective.u64_parameter("image.height").unwrap_or(0);
    let steps = effective.u64_parameter("image.steps").unwrap_or(0);
    let cfg_scale = effective.f64_parameter("image.guidance").unwrap_or(0.0);
    let actual_seed = result
        .outputs
        .first()
        .and_then(|output| output.seed)
        .unwrap_or(seed);
    let all_seeds = result
        .outputs
        .iter()
        .map(|output| output.seed.unwrap_or(seed))
        .collect::<Vec<_>>();
    let prompt = effective.prompt.clone().unwrap_or_default();
    let negative_prompt = effective.negative_prompt.clone().unwrap_or_default();
    let info = json!({
        "prompt": prompt,
        "all_prompts": vec![effective.prompt.clone().unwrap_or_default(); actual_output_count],
        "negative_prompt": negative_prompt,
        "all_negative_prompts": vec![effective.negative_prompt.clone().unwrap_or_default(); actual_output_count],
        "seed": actual_seed,
        "all_seeds": all_seeds,
        "subseed": -1,
        "all_subseeds": vec![-1; actual_output_count],
        "subseed_strength": 0,
        "width": width,
        "height": height,
        "sampler_name": "Werk automatic",
        "cfg_scale": cfg_scale,
        "steps": steps,
        "batch_size": batch_size,
        "restore_faces": false,
        "sd_model_name": result.model,
        "sd_model_hash": null,
        "index_of_first_image": 0,
        "infotexts": [],
        "styles": [],
        "job_timestamp": result.created_unix.to_string(),
        "version": env!("CARGO_PKG_VERSION")
    });
    Ok(Txt2ImgResponse {
        images,
        parameters,
        info: info.to_string(),
    })
}

fn append_info_warning(info: &mut String, warnings: Vec<String>) {
    if warnings.is_empty() {
        return;
    }
    let Ok(mut value) = serde_json::from_str::<Value>(info) else {
        return;
    };
    if let Some(object) = value.as_object_mut() {
        object.insert("werk_warnings".to_string(), json!(warnings));
        *info = value.to_string();
    }
}

pub(super) fn automatic1111_error(status: StatusCode, detail: String) -> Response {
    (
        status,
        Json(Automatic1111Error {
            error: "HTTPException",
            detail,
            body: String::new(),
            errors: String::new(),
        }),
    )
        .into_response()
}

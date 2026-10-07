use super::{response::api_error_with_code, state::ApiState};
use crate::{
    backend::text_analysis::{Options, endpoint, task_for},
    capabilities::InferenceTask,
};
use anyhow::{Result, ensure};
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

pub(super) async fn embeddings(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle(state, headers, body, InferenceTask::TextEmbedding).await
}
pub(super) async fn rerank(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle(state, headers, body, InferenceTask::TextReranking).await
}
pub(super) async fn classify(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle(state, headers, body, InferenceTask::TextClassification).await
}
fn error(status: StatusCode, message: impl Into<String>, code: &str) -> Response {
    api_error_with_code(status, message.into(), None, Some(code.into()))
}

async fn handle(
    state: ApiState,
    headers: HeaderMap,
    mut body: Value,
    task: InferenceTask,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    if state.deployments.is_some() {
        return error(
            StatusCode::BAD_REQUEST,
            "Text-analysis endpoints do not use chat deployment profiles. Start werk serve without --deployments and select an installed model.",
            "unsupported_deployment",
        );
    }
    if let Err(err) = validate(&mut body, task) {
        return error(StatusCode::BAD_REQUEST, err.to_string(), "invalid_input");
    }
    let options: Options =
        match serde_json::from_value(body.get("werk").cloned().unwrap_or(json!({}))) {
            Ok(options) => options,
            Err(err) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!("Invalid werk options: {err}"),
                    "invalid_options",
                );
            }
        };
    let options = match state.text_policy.apply(options) {
        Ok(options) => options,
        Err(err) => {
            return error(
                StatusCode::BAD_REQUEST,
                err.to_string(),
                "unsupported_runtime",
            );
        }
    };
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .or(state.default_model.as_deref())
        .unwrap_or("")
        .to_string();
    if model.is_empty() {
        return error(
            StatusCode::BAD_REQUEST,
            "Specify model or start werk serve --model MODEL",
            "missing_model",
        );
    }
    // Hold the permit inside spawn_blocking even if the client disconnects.
    let permit = match state.text_gate.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "Text-analysis workers are busy. Retry after the current request completes.",
                "runtime_busy",
            );
        }
    };
    let mut observed = state.telemetry.begin(&model);
    observed.analysis_begin(&task.to_string());
    let request_id = observed.id();
    let started = std::time::Instant::now();
    state.log_verbose(format!(
        "[werk serve] request #{request_id} model={model} task={task} accepted"
    ));
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let outcome = (|| {
        let manifest = state.store.get(&model).map_err(|err| (StatusCode::NOT_FOUND, err.to_string()))?;
        if task_for(&manifest) != Some(task) {
            return Err((StatusCode::BAD_REQUEST, format!("Model '{}' cannot execute {}. Declared tasks: {:?}. Use {} with the appropriate model; these models do not generate chat responses.", manifest.id, task, manifest.metadata.tasks,
                manifest.metadata.tasks.first().copied().map(endpoint).unwrap_or("/v1/models"))));
        }
        if task == InferenceTask::TextClassification {
            let is_laya = manifest.architecture.as_deref() == Some("laya");
            if is_laya && body.get("input").is_some() || !is_laya && body.get("input").is_none() {
                return Err((StatusCode::BAD_REQUEST, "Laya requires state and questions; sequence classifiers require input texts. Use the request schema matching the architecture.".into()));
            }
        }
        if options.backend == "candle" && manifest.architecture.as_deref() != Some("xlm-roberta") {
            return Err((StatusCode::BAD_REQUEST, "Candle supports XLM-RoBERTa sequence classification/reranking, but has no loader for this architecture. Use backend=auto or transformers; run werk backend install text-analysis.".into()));
        }
        if options.backend == "vllm" && task == InferenceTask::TextClassification {
            return Err((StatusCode::BAD_REQUEST, "This classification task requires its PyTorch or native Candle adapter. Use backend=auto or transformers.".into()));
        }
        if options.checkpoint != "auto" && manifest.architecture.as_deref() != Some("laya") {
            return Err((StatusCode::BAD_REQUEST, "werk.checkpoint applies only to Laya; omit it for this architecture.".into()));
        }
        let mut result = state.text_backend.execute_observed(&manifest, task, body, &options, |phase, runtime| {
            observed.analysis_phase(phase, runtime);
            state.log_verbose(format!("[werk serve] request #{request_id} {phase}{}",
                runtime.map(|(backend, device)| format!(" runtime={backend} device={device}")).unwrap_or_default()));
        })
            .map_err(|err| (if err.is::<crate::backend::text_analysis::InvalidInput>() { StatusCode::BAD_REQUEST } else { StatusCode::SERVICE_UNAVAILABLE }, format!("{err:#}")))?;
        result.as_object_mut().unwrap().remove("ok");
        result.as_object_mut().unwrap().remove("warnings");
        result["model"] = json!(manifest.id);
        result["werk"]["request_id"] = json!(request_id);
        result["werk"]["request_seconds"] = json!(started.elapsed().as_secs_f64());
        Ok(result)
        })();
        match &outcome {
            Ok(response) => {
                observed.analysis_complete(response);
                if state.verbose { log_stats(request_id, task, response); }
            }
            Err((status, message)) => {
                observed.error();
                state.log_verbose(format!("[werk serve] request #{request_id} failed status={status} elapsed={:.3}s: {message}", started.elapsed().as_secs_f64()));
            }
        }
        outcome
    }).await;
    match result {
        Ok(Ok(response)) => Json(response).into_response(),
        Ok(Err((status, message))) => error(status, message, "text_analysis_failed"),
        Err(err) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Text-analysis worker failed: {err}; retry the request"),
            "worker_failed",
        ),
    }
}

fn log_stats(id: u64, task: InferenceTask, response: &Value) {
    let meta = &response["werk"];
    let text = |key: &str| meta[key].as_str().unwrap_or("n/a");
    let seconds = |key: &str| {
        meta[key]
            .as_f64()
            .map(|s| format!("{s:.3} s"))
            .unwrap_or_else(|| "n/a".into())
    };
    let tokens = response["usage"]["input_tokens"]
        .as_u64()
        .or_else(|| response["usage"]["prompt_tokens"].as_u64())
        .or_else(|| response["usage"]["total_tokens"].as_u64())
        .map(|n| n.to_string())
        .unwrap_or_else(|| "n/a".into());
    let results = response["answers"]
        .as_object()
        .map(|a| a.len())
        .or_else(|| response["results"].as_array().map(|a| a.len()))
        .or_else(|| response["data"].as_array().map(|a| a.len()))
        .unwrap_or(0);
    crate::terminal::panel(
        crate::terminal::Stream::Err,
        &format!("Request #{id} complete · {task}"),
        &format!(
            "Model: {}\nRuntime: {}\nDevice: {}\nPrecision: {}\nModel cache: {}\n\nRequest total: {}\nWorker total: {}\nLoad: {}\nInference: {}\nInput tokens: {tokens}\nResults: {results}",
            response["model"].as_str().unwrap_or("n/a"),
            text("runtime"),
            text("device"),
            text("dtype"),
            match meta["model_cache_hit"].as_bool() {
                Some(true) => "hit",
                Some(false) => "miss",
                None => "n/a",
            },
            seconds("request_seconds"),
            seconds("total_seconds"),
            seconds("load_seconds"),
            seconds("inference_seconds")
        ),
    );
}

fn strings(value: &Value, name: &str) -> Result<()> {
    let values: Vec<&Value> = if value.is_string() {
        vec![value]
    } else {
        value
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("{name} must be a string or array of strings"))?
            .iter()
            .collect()
    };
    ensure!(
        !values.is_empty() && values.len() <= 128,
        "{name} must contain 1..128 texts; split larger requests into batches"
    );
    for value in values {
        ensure!(
            value
                .as_str()
                .is_some_and(|text| !text.trim().is_empty() && text.len() <= 131072),
            "{name} must contain nonempty strings of at most 128 KiB"
        );
    }
    Ok(())
}
pub(super) fn validate(body: &mut Value, task: InferenceTask) -> Result<()> {
    ensure!(body.is_object(), "request must be a JSON object");
    let allowed: &[&str] = match task {
        InferenceTask::TextEmbedding => &[
            "model",
            "input",
            "input_type",
            "dimensions",
            "encoding_format",
            "werk",
            "user",
        ],
        InferenceTask::TextReranking => &[
            "model",
            "query",
            "documents",
            "texts",
            "top_n",
            "return_documents",
            "werk",
        ],
        InferenceTask::TextClassification => &["model", "input", "state", "questions", "werk"],
        _ => &[],
    };
    for name in body.as_object().unwrap().keys() {
        ensure!(
            allowed.contains(&name.as_str()),
            "Unsupported field '{name}' for {task}; remove it or use the matching endpoint"
        );
    }
    ensure!(
        body.get("model").is_none_or(Value::is_string),
        "model must be a string"
    );
    match task {
        InferenceTask::TextEmbedding => {
            strings(&body["input"], "input")?;
            ensure!(
                body.get("encoding_format").is_none_or(|v| v == "float"),
                "Only encoding_format=float is supported; omit encoding_format or select float"
            );
            ensure!(
                body.get("input_type")
                    .is_none_or(|v| v == "query" || v == "document"),
                "input_type must be query or document"
            );
            ensure!(
                body.get("dimensions").is_none_or(|v| v
                    .as_u64()
                    .is_some_and(|n| [128, 256, 512, 768].contains(&n))),
                "dimensions must be 128, 256, 512 or 768"
            );
        }
        InferenceTask::TextReranking => {
            ensure!(
                body.get("documents").is_none() || body.get("texts").is_none(),
                "Use documents or texts, not both"
            );
            ensure!(body["query"].is_string(), "query must be a string");
            strings(&body["query"], "query")?;
            if body.get("documents").is_none() {
                if let Some(texts) = body.get("texts").cloned() {
                    body["documents"] = texts;
                }
            }
            let docs = body
                .get_mut("documents")
                .and_then(Value::as_array_mut)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "documents must be an array of strings or objects containing text"
                    )
                })?;
            for doc in docs.iter_mut() {
                if doc.is_object() {
                    *doc = doc.get("text").cloned().unwrap_or(Value::Null);
                }
            }
            let count = docs.len();
            strings(&body["documents"], "documents")?;
            ensure!(
                body.get("top_n")
                    .is_none_or(|v| v.as_u64().is_some_and(|n| n > 0 && n <= count as u64)),
                "top_n must be 1..documents.len()"
            );
            ensure!(
                body.get("return_documents").is_none_or(Value::is_boolean),
                "return_documents must be a boolean"
            );
        }
        InferenceTask::TextClassification => {
            if body.get("input").is_some() {
                ensure!(
                    body.get("state").is_none() && body.get("questions").is_none(),
                    "Use classifier input or Laya state/questions, not both"
                );
                strings(&body["input"], "input")?;
                return Ok(());
            }
            ensure!(
                body.get("state")
                    .is_some_and(|v| v.is_string() || v.is_object()),
                "state must be a string or object"
            );
            let questions = body["questions"].as_object().ok_or_else(|| {
                anyhow::anyhow!("questions must be an object of typed Laya questions")
            })?;
            ensure!(
                !questions.is_empty() && questions.len() <= 32,
                "questions must contain 1..32 entries"
            );
            for question in questions.values() {
                ensure!(
                    question["instructions"]
                        .as_str()
                        .is_some_and(|s| !s.trim().is_empty()),
                    "each question requires nonempty instructions"
                );
                match question["type"].as_str() {
                    Some("noul") => {}
                    Some("choice") => ensure!(
                        question["criteria"]
                            .as_object()
                            .is_some_and(|o| !o.is_empty()
                                && o.len() <= 64
                                && o.values().all(Value::is_string)),
                        "choice criteria must contain 1..64 string descriptions"
                    ),
                    Some("score") => ensure!(
                        question["criteria"].as_array().is_some_and(|a| a.len() >= 2
                            && a.len() <= 32
                            && a.iter().all(Value::is_string)),
                        "score criteria must contain 2..32 strings"
                    ),
                    _ => anyhow::bail!("question type must be noul, choice or score"),
                }
            }
        }
        _ => anyhow::bail!("unsupported text-analysis task"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_shapes_and_limits_are_checked_before_model_loading() {
        let mut input = json!({"query":"hello","documents":[{"text":"world"}],"top_n":1});
        validate(&mut input, InferenceTask::TextReranking).unwrap();
        assert_eq!(input["documents"], json!(["world"]));
        for mut input in [
            json!({"input":[]}),
            json!({"input":[1,2]}),
            json!({"input":"hello","dimensions":64}),
            json!({"input":"hello","encoding_format":"base64"}),
        ] {
            assert!(validate(&mut input, InferenceTask::TextEmbedding).is_err());
        }
        assert!(
            validate(
                &mut json!({"input":"hello", "input_type":"query", "dimensions":128}),
                InferenceTask::TextEmbedding
            )
            .is_ok()
        );
        assert!(validate(&mut json!({"state":"ticket", "questions":{"topic":{"type":"choice","instructions":"route","criteria":["bad shape"]}}}), InferenceTask::TextClassification).is_err());
    }
}

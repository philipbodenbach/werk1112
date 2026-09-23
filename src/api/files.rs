//! The colliding /v1/files paths select their wire contract by Anthropic headers.
use super::state::ApiState;
use crate::file_store::{MAX_FILE_BYTES, StoredFile};
use axum::{
    Json,
    extract::{Multipart, Path, Query, State, multipart::MultipartRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(super) struct Wire {
    anthropic: bool,
    beta: bool,
}
impl Wire {
    fn detect(headers: &HeaderMap) -> Self {
        Self {
            anthropic: headers.contains_key("anthropic-version")
                || headers.contains_key("anthropic-beta"),
            beta: headers
                .get("anthropic-beta")
                .is_some_and(|h| h == "files-api-2025-04-14"),
        }
    }
    fn object(self, f: &StoredFile) -> Value {
        if self.anthropic {
            let mut v = json!({"id":f.id,"type":"file","filename":f.filename,"mime_type":f.mime_type,
                "size_bytes":f.bytes,"created_at":date(f.created_at),"downloadable":false});
            if !self.beta {
                v["expires_at"] = f.expires_at.map(date).into();
            }
            v
        } else {
            json!({"id":f.id,"object":"file","bytes":f.bytes,"filename":f.filename,
                "created_at":f.created_at,"expires_at":f.expires_at,"purpose":f.purpose,"status":"processed"})
        }
    }
    fn error(self, status: StatusCode, message: impl Into<String>) -> Response {
        if self.anthropic {
            let kind = match status {
                StatusCode::UNAUTHORIZED => "authentication_error",
                StatusCode::NOT_FOUND => "not_found_error",
                StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
                StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
                _ if status.is_server_error() => "api_error",
                _ => "invalid_request_error",
            };
            let id = request_id();
            let mut response = (status, Json(json!({"type":"error","error":{"type":kind,"message":message.into()},"request_id":id}))).into_response();
            response
                .headers_mut()
                .insert("request-id", id.parse().unwrap());
            response
        } else {
            super::response::api_error(status, message.into(), None)
        }
    }
}
fn request_id() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    format!(
        "req_werk_files_{}_{}",
        crate::model_store::unix_ts(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}
pub(super) async fn response_headers(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .entry("request-id")
        .or_insert_with(|| request_id().parse().unwrap());
    response
        .headers_mut()
        .insert("cache-control", "no-store".parse().unwrap());
    response
}
fn date(seconds: u64) -> String {
    chrono::DateTime::from_timestamp(seconds as i64, 0)
        .unwrap_or_default()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
async fn access(state: &ApiState, headers: &HeaderMap) -> Result<(Wire, String), Response> {
    let wire = Wire::detect(headers);
    if state.authorize(headers).is_err() {
        return Err(wire.error(StatusCode::UNAUTHORIZED, "invalid or missing API key"));
    }
    if wire.anthropic
        && headers
            .get("anthropic-version")
            .is_none_or(|v| v != "2023-06-01")
    {
        return Err(wire.error(
            StatusCode::BAD_REQUEST,
            "anthropic-version must be 2023-06-01",
        ));
    }
    if headers.contains_key("anthropic-beta") && !wire.beta {
        return Err(wire.error(
            StatusCode::BAD_REQUEST,
            "unsupported anthropic-beta feature",
        ));
    }
    let principal = state
        .werk_principal(headers)
        .await
        .map_err(|_| wire.error(StatusCode::UNAUTHORIZED, "file access denied"))?;
    Ok((wire, principal))
}

pub(super) async fn upload(
    State(state): State<ApiState>,
    headers: HeaderMap,
    input: Result<Multipart, MultipartRejection>,
) -> Response {
    let (wire, principal) = match access(&state, &headers).await {
        Ok(v) => v,
        Err(e) => return e,
    };
    let _permit = match state.upload_gate.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return wire.error(StatusCode::TOO_MANY_REQUESTS, "too many concurrent uploads"),
    };
    let mut form = match input {
        Ok(f) => f,
        Err(e) => return wire.error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    let mut file = None;
    let mut fields = BTreeMap::new();
    loop {
        let field = match form.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return wire.error(e.status(), "invalid or oversized multipart upload"),
        };
        let name = field.name().unwrap_or_default().to_owned();
        if name == "file" {
            if file.is_some() {
                return wire.error(StatusCode::BAD_REQUEST, "duplicate file field");
            }
            let filename = field.file_name().unwrap_or("document").to_owned();
            let mime = field.content_type().map(str::to_owned);
            let mut field = field;
            let mut bytes = Vec::new();
            loop {
                match field.chunk().await {
                    Ok(Some(chunk)) if bytes.len() + chunk.len() <= MAX_FILE_BYTES => {
                        bytes.extend_from_slice(&chunk)
                    }
                    Ok(Some(_)) => {
                        return wire.error(StatusCode::PAYLOAD_TOO_LARGE, "file exceeds 50 MiB");
                    }
                    Ok(None) => break,
                    Err(e) => return wire.error(e.status(), "invalid or oversized upload"),
                }
            }
            let mime = mime.unwrap_or_else(|| {
                if bytes.starts_with(b"%PDF-") {
                    "application/pdf".into()
                } else if !bytes.contains(&0) && std::str::from_utf8(&bytes).is_ok() {
                    "text/plain".into()
                } else {
                    "application/octet-stream".into()
                }
            });
            file = Some((filename, mime, bytes));
        } else {
            if !matches!(
                name.as_str(),
                "purpose" | "expires_after[anchor]" | "expires_after[seconds]"
            ) || fields.contains_key(&name)
            {
                return wire.error(StatusCode::BAD_REQUEST, "unknown or duplicate upload field");
            }
            let mut field = field;
            let mut bytes = Vec::new();
            while let Some(chunk) = match field.chunk().await {
                Ok(c) => c,
                Err(_) => return wire.error(StatusCode::BAD_REQUEST, "invalid upload field"),
            } {
                if bytes.len() + chunk.len() > 256 {
                    return wire.error(StatusCode::BAD_REQUEST, "upload field too long");
                }
                bytes.extend_from_slice(&chunk);
            }
            let text = match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => return wire.error(StatusCode::BAD_REQUEST, "upload field must be UTF-8"),
            };
            fields.insert(name, text);
        }
    }
    let Some((filename, mime, bytes)) = file else {
        return wire.error(StatusCode::BAD_REQUEST, "file is required");
    };
    let purpose = match fields.remove("purpose") {
        Some(p)
            if !wire.anthropic
                && matches!(
                    p.as_str(),
                    "user_data" | "assistants" | "batch" | "fine-tune" | "vision" | "evals"
                ) =>
        {
            p
        }
        None if wire.anthropic => "user_data".into(),
        _ => {
            return wire.error(
                StatusCode::BAD_REQUEST,
                "valid OpenAI purpose required; Anthropic uploads omit purpose",
            );
        }
    };
    let expires = match (
        fields.remove("expires_after[anchor]"),
        fields.remove("expires_after[seconds]"),
    ) {
        (None, None) => {
            if purpose == "batch" {
                Some(crate::model_store::unix_ts() + 2_592_000)
            } else {
                None
            }
        }
        (Some(anchor), Some(seconds)) if !wire.anthropic && anchor == "created_at" => {
            match seconds.parse::<u64>() {
                Ok(n) if (3600..=2_592_000).contains(&n) => Some(crate::model_store::unix_ts() + n),
                _ => {
                    return wire.error(
                        StatusCode::BAD_REQUEST,
                        "expires_after.seconds must be 3600..2592000",
                    );
                }
            }
        }
        _ => return wire.error(StatusCode::BAD_REQUEST, "invalid expires_after"),
    };
    match tokio::task::spawn_blocking(move || {
        let _permit = _permit;
        state
            .files
            .put(&principal, filename, mime, purpose, expires, &bytes)
    })
    .await
    {
        Ok(Ok(file)) => Json(wire.object(&file)).into_response(),
        Ok(Err(e)) => wire.error(StatusCode::BAD_REQUEST, e.to_string()),
        Err(_) => wire.error(StatusCode::INTERNAL_SERVER_ERROR, "file upload failed"),
    }
}

pub(super) async fn list(
    State(state): State<ApiState>,
    headers: HeaderMap,
    query: Result<Query<BTreeMap<String, String>>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let (wire, principal) = match access(&state, &headers).await {
        Ok(v) => v,
        Err(e) => return e,
    };
    let query = match query {
        Ok(Query(q)) => q,
        Err(_) => return wire.error(StatusCode::BAD_REQUEST, "invalid file query"),
    };
    let allowed: &[&str] = if !wire.anthropic {
        &["purpose", "limit", "order", "after"]
    } else if wire.beta {
        &["limit", "after_id", "before_id"]
    } else {
        &["limit", "page"]
    };
    if query.keys().any(|k| !allowed.contains(&k.as_str())) {
        return wire.error(StatusCode::BAD_REQUEST, "unsupported file query parameter");
    }
    let limit = match query.get("limit").map(|s| s.parse::<usize>()).transpose() {
        Ok(n) => n.unwrap_or(if wire.anthropic { 20 } else { 10000 }),
        Err(_) => 0,
    };
    if limit == 0 || limit > if wire.anthropic { 1000 } else { 10000 } {
        return wire.error(StatusCode::BAD_REQUEST, "invalid file page limit");
    }
    if query
        .get("order")
        .is_some_and(|v| v != "asc" && v != "desc")
        || (query.contains_key("after_id") && query.contains_key("before_id"))
    {
        return wire.error(StatusCode::BAD_REQUEST, "invalid file pagination");
    }
    let mut files = match tokio::task::spawn_blocking(move || state.files.list(&principal)).await {
        Ok(Ok(f)) => f,
        _ => {
            return wire.error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "file inventory unavailable",
            );
        }
    };
    if let Some(purpose) = query.get("purpose") {
        files.retain(|f| &f.purpose == purpose);
    }
    files.sort_by(|a, b| (b.created_at, &b.id).cmp(&(a.created_at, &a.id)));
    if query.get("order").is_some_and(|v| v == "asc") {
        files.reverse();
    }
    let cursor = query
        .get("after")
        .or(query.get("after_id"))
        .or(query.get("before_id"))
        .or(query.get("page"));
    if let Some(id) = cursor {
        let Some(index) = files.iter().position(|f| &f.id == id) else {
            return wire.error(StatusCode::BAD_REQUEST, "invalid file cursor");
        };
        if query.contains_key("before_id") {
            files.truncate(index);
        } else {
            files.drain(..=index);
        }
    }
    let more = files.len() > limit;
    files.truncate(limit);
    let first = files.first().map(|f| f.id.clone());
    let last = files.last().map(|f| f.id.clone());
    let data: Vec<_> = files.iter().map(|f| wire.object(f)).collect();
    Json(if wire.anthropic && !wire.beta {
        json!({"data":data,"next_page":if more {last}else{None}})
    } else {
        let mut v = json!({"data":data,"has_more":more,"first_id":first,"last_id":last});
        if !wire.anthropic {
            v["object"] = "list".into();
        }
        v
    })
    .into_response()
}

pub(super) async fn retrieve(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let (wire, principal) = match access(&state, &headers).await {
        Ok(v) => v,
        Err(e) => return e,
    };
    match tokio::task::spawn_blocking(move || state.files.metadata(&principal, &id)).await {
        Ok(Ok(file)) => Json(wire.object(&file)).into_response(),
        _ => wire.error(StatusCode::NOT_FOUND, "file not found"),
    }
}
pub(super) async fn content(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let (wire, principal) = match access(&state, &headers).await {
        Ok(v) => v,
        Err(e) => return e,
    };
    if wire.anthropic {
        return wire.error(
            StatusCode::BAD_REQUEST,
            "uploaded files are not downloadable through the Anthropic contract",
        );
    }
    let permit = match state.upload_gate.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return wire.error(
                StatusCode::TOO_MANY_REQUESTS,
                "too many concurrent file transfers",
            );
        }
    };
    match tokio::task::spawn_blocking(move || (state.files.get(&principal, &id), permit)).await {
        Ok((Ok((_, bytes)), permit)) => {
            // Keep admission until the body is consumed or disconnected, including
            // slow clients. At most four upload/download buffers can coexist.
            let mut bytes = axum::body::Bytes::from(bytes);
            let stream = tokio_stream::iter(std::iter::from_fn(move || {
                let _keep_admission = &permit;
                if bytes.is_empty() {
                    None
                } else {
                    let size = bytes.len().min(65536);
                    Some(Ok::<_, std::convert::Infallible>(bytes.split_to(size)))
                }
            }));
            (
                [
                    ("content-type", "application/octet-stream"),
                    ("content-disposition", "attachment"),
                ],
                axum::body::Body::from_stream(stream),
            )
                .into_response()
        }
        _ => wire.error(StatusCode::NOT_FOUND, "file not found"),
    }
}
pub(super) async fn delete(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let (wire, principal) = match access(&state, &headers).await {
        Ok(v) => v,
        Err(e) => return e,
    };
    let result_id = id.clone();
    match tokio::task::spawn_blocking(move || state.files.delete(&principal, &id)).await {
        Ok(Ok(())) => Json(if wire.anthropic {
            json!({"id":result_id,"type":"file_deleted"})
        } else {
            json!({"id":result_id,"object":"file","deleted":true})
        })
        .into_response(),
        _ => wire.error(StatusCode::NOT_FOUND, "file not found"),
    }
}

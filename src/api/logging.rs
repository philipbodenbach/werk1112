//! Access logging wraps response bodies without buffering or consuming SSE data.
use crate::logging::{self, Level};
use axum::{
    body::{Body, Bytes, HttpBody},
    extract::{MatchedPath, Request},
    http::HeaderValue,
    middleware::Next,
    response::Response,
};
use serde_json::json;
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Instant,
};

struct Access {
    context: logging::Context,
    method: String,
    route: String,
    started: Instant,
    headers_seconds: Option<f64>,
    status: Option<u16>,
    bytes: u64,
    finished: bool,
}
impl Access {
    fn finish(&mut self, outcome: &str) {
        if self.finished {
            return;
        }
        self.finished = true;
        let status = self.status.unwrap_or(0);
        let level = if outcome == "error" || status >= 500 {
            Level::Error
        } else if outcome == "cancelled" || status >= 400 {
            Level::Warn
        } else if matches!(
            self.route.as_str(),
            "/health" | "/metrics" | "/werk/v1/observability"
        ) {
            Level::Trace
        } else {
            Level::Info
        };
        logging::emit_in(
            &self.context,
            level,
            match outcome {
                "error" => "http.request.failed",
                "cancelled" => "http.request.cancelled",
                _ => "http.request.completed",
            },
            &format!(
                "{} {} {} · {outcome} · {:.2} s",
                self.method,
                self.route,
                self.status
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "pending".into()),
                self.started.elapsed().as_secs_f64()
            ),
            json!({
                "method":self.method,"route":self.route,"status":self.status,"outcome":outcome,
                "headers_seconds":self.headers_seconds,"duration_seconds":self.started.elapsed().as_secs_f64(),"response_bytes":self.bytes
            }),
        );
    }
}
impl Drop for Access {
    fn drop(&mut self) {
        self.finish(if std::thread::panicking() {
            "error"
        } else {
            "cancelled"
        });
    }
}
pub(super) async fn access(request: Request, next: Next) -> Response {
    let context = logging::request_context();
    let request_id = context.request_id.clone().unwrap();
    // Matched templates cannot expose file IDs, query strings or arbitrary URL paths.
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str())
        .unwrap_or("<unmatched>")
        .to_string();
    let method = request.method().to_string();
    let mut access = Access {
        context: context.clone(),
        method: method.clone(),
        route: route.clone(),
        started: Instant::now(),
        headers_seconds: None,
        status: None,
        bytes: 0,
        finished: false,
    };
    let monitoring = matches!(
        route.as_str(),
        "/health" | "/metrics" | "/werk/v1/observability"
    );
    logging::emit_in(
        &context,
        if monitoring {
            Level::Trace
        } else {
            Level::Debug
        },
        "http.request.started",
        "HTTP request started",
        json!({"method":method,"route":route}),
    );
    let mut response = logging::scope(context, next.run(request)).await;
    access.status = Some(response.status().as_u16());
    access.headers_seconds = Some(access.started.elapsed().as_secs_f64());
    response
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());
    let (parts, body) = response.into_parts();
    if body.is_end_stream() {
        access.finish("completed");
        return Response::from_parts(parts, body);
    }
    Response::from_parts(
        parts,
        Body::new(LoggedBody {
            inner: Box::pin(body),
            access,
        }),
    )
}
struct LoggedBody {
    inner: Pin<Box<Body>>,
    access: Access,
}
impl HttpBody for LoggedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Bytes>, Self::Error>>> {
        let this = self.get_mut();
        // Restore correlation while polling a streaming response. Guards retain their own context.
        let result = logging::with_context(this.access.context.clone(), || {
            this.inner.as_mut().poll_frame(cx)
        });
        match &result {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.access.bytes += data.len() as u64;
                }
                if this.inner.is_end_stream() {
                    this.access.finish("completed");
                }
            }
            Poll::Ready(Some(Err(_))) => this.access.finish("error"),
            Poll::Ready(None) => this.access.finish("completed"),
            Poll::Pending => {}
        }
        result
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::tests::{fixture, records};
    use axum::{Router, routing::get};
    use tower::ServiceExt;

    #[tokio::test]
    async fn streaming_completion_waits_for_body_and_preserves_frames_and_correlation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let root = fixture(path.clone(), Level::Debug);
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let receiver = std::sync::Arc::new(std::sync::Mutex::new(Some(rx)));
        let router = Router::new()
            .route(
                "/stream",
                get(move || {
                    let rx = receiver.lock().unwrap().take().unwrap();
                    async move {
                        logging::spawn_blocking(|| {
                            logging::emit(Level::Debug, "worker", "worker", json!({}))
                        })
                        .await
                        .unwrap();
                        Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx))
                    }
                }),
            )
            .layer(axum::middleware::from_fn(access));
        let response = logging::scope(
            root.clone(),
            router.oneshot(
                Request::builder()
                    .uri("/stream?prompt=private-query")
                    .header("authorization", "private-header")
                    .header("x-request-id", "untrusted-client-id")
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .await
        .unwrap();
        let id = response.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        assert!(id.starts_with("req_werk_"));
        assert_eq!(records(&root, &path).len(), 2);
        tx.send(Ok(Bytes::from_static(b"data: private-content\n\n")))
            .await
            .unwrap();
        drop(tx);
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(body, "data: private-content\n\n");
        let events = records(&root, &path);
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(|e| e["request_id"] == id));
        let event = events.last().unwrap();
        assert_eq!(event["event"], "http.request.completed");
        assert_eq!(event["fields"]["response_bytes"], body.len());
        assert!(
            event["fields"]["duration_seconds"].as_f64().unwrap()
                >= event["fields"]["headers_seconds"].as_f64().unwrap()
        );
        let raw = std::fs::read_to_string(path).unwrap();
        for private in [
            "private-query",
            "private-header",
            "private-content",
            "untrusted-client-id",
        ] {
            assert!(!raw.contains(private));
        }
    }

    #[tokio::test]
    async fn failed_and_abandoned_streams_emit_once_and_unknown_paths_are_not_logged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let root = fixture(path.clone(), Level::Info);
        let router = Router::new()
            .route(
                "/cancel",
                get(|| async {
                    Body::from_stream(tokio_stream::pending::<Result<Bytes, std::io::Error>>())
                }),
            )
            .route(
                "/error",
                get(|| async {
                    Body::from_stream(tokio_stream::iter([Err::<Bytes, _>(
                        std::io::Error::other("private-body-error"),
                    )]))
                }),
            )
            .route("/metrics", get(|| async { "metrics" }))
            .route(
                "/denied",
                get(|| async { axum::http::StatusCode::UNAUTHORIZED }),
            )
            .layer(axum::middleware::from_fn(access));
        for uri in [
            "/cancel",
            "/error",
            "/metrics",
            "/denied",
            "/private-path?secret=private-query",
        ] {
            let response = logging::scope(
                root.clone(),
                router
                    .clone()
                    .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap()),
            )
            .await
            .unwrap();
            if uri == "/cancel" {
                drop(response);
            } else {
                let _ = axum::body::to_bytes(response.into_body(), 1024).await;
            }
        }
        let events = records(&root, &path);
        assert_eq!(events.len(), 4); // metrics success is TRACE
        assert_eq!(events[0]["event"], "http.request.cancelled");
        assert_eq!(events[0]["level"], "warn");
        assert_eq!(events[1]["event"], "http.request.failed");
        assert_eq!(events[1]["level"], "error");
        assert_eq!(events[2]["fields"]["status"], 401);
        assert_eq!(events[2]["level"], "warn");
        assert_eq!(events[3]["fields"]["route"], "<unmatched>");
        assert_eq!(events[3]["fields"]["status"], 404);
        assert!(!std::fs::read_to_string(path).unwrap().contains("private-"));
    }
}

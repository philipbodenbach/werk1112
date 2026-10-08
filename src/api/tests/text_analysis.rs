use super::support::*;

#[tokio::test]
async fn text_analysis_routes_require_authentication_and_validate_before_loading() {
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
    let app =
        router(ApiState::new(store, Arc::new(MockBackend)).with_api_keys(vec!["secret".into()]));
    for path in [
        "/v1/embeddings",
        "/v1/rerank",
        "/rerank",
        "/v1/classifications",
        "/v1/systemone",
    ] {
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        assert_eq!(
            app.clone().oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .header("authorization", "Bearer secret")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: Value =
            serde_json::from_slice(&body::to_bytes(response.into_body(), 10000).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["code"], "invalid_input");
    }
}

#[tokio::test]
async fn wrong_task_recommends_the_matching_endpoint_without_loading_weights() {
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
    let source = dir.path().join("source");
    fs::create_dir_all(&source).unwrap();
    fs::write(source.join("model.safetensors"), "fixture").unwrap();
    fs::write(
        source.join("config.json"),
        r#"{"model_type":"laya","architectures":["LayaTypedDecisions"]}"#,
    )
    .unwrap();
    store
        .import_path(&source, "custom/decision-finetune")
        .unwrap();
    let mut state = ApiState::new(store, Arc::new(MockBackend));
    state.text_policy = crate::backend::text_analysis::Policy {
        backend: Some("candle".into()),
        device: None,
    };
    let app = router(state);
    let request = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model":"custom/decision-finetune","input":"hello"}).to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&body::to_bytes(response.into_body(), 10000).await.unwrap())
            .unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("/v1/classifications")
    );
    let request = Request::builder()
        .method("POST")
        .uri("/v1/classifications")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model":"custom/decision-finetune",
                "state":{"text":"hello"},
                "questions":{"greeting":{"type":"noul","instructions":"Is this a greeting?"}}
            })
            .to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&body::to_bytes(response.into_body(), 10000).await.unwrap())
            .unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Candle supports XLM-RoBERTa")
    );

    let request = Request::builder()
        .uri("/v1/capabilities")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&body::to_bytes(response.into_body(), 100000).await.unwrap())
            .unwrap();
    let serialized = body.to_string();
    assert!(serialized.contains("no compatible loader"));
    assert!(!serialized.contains("\"available_tasks\":[\"text-classification\"]"));
}

#[tokio::test]
async fn observability_analysis_routes_report_active_and_completed_requests_without_content() {
    use crate::{backend::text_analysis::TextAnalysisBackend, media_companion::CompanionClient};
    let python = if cfg!(windows) { "python" } else { "python3" };
    const WORKER: &str = r#"
import json, sys, time
for line in sys.stdin:
    frame = json.loads(line)
    response = {"ok": True}
    if frame["operation"] == "execute":
        time.sleep(0.1)
        request = frame["payload"]
        result_key = {"text-classification":"answers", "text-reranking":"results", "text-embedding":"data"}[request["task"]]
        response.update({result_key: {"private-question": "private-answer"} if result_key == "answers" else [{}],
            "usage":{"input_tokens":47}, "werk":{"runtime":"fixture", "device":"cpu", "dtype":"float32", "model_cache_hit":False, "load_seconds":0.1, "inference_seconds":0.2, "total_seconds":0.3}})
    print(json.dumps({"transport_version":1,"request_id":frame["request_id"],"response":response}), flush=True)
"#;
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
    for (model, config) in [
        (
            "decision",
            json!({"model_type":"laya","architectures":["LayaTypedDecisions"]}),
        ),
        (
            "ranker",
            json!({"model_type":"xlm-roberta","architectures":["XLMRobertaForSequenceClassification"],"num_labels":1,"id2label":{"0":"LABEL_0"}}),
        ),
        (
            "embedder",
            json!({"model_type":"embedding_gemma2","architectures":["EmbeddingGemma2Model"]}),
        ),
    ] {
        let source = dir.path().join(model);
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), config.to_string()).unwrap();
        fs::write(source.join("model.safetensors"), "fixture").unwrap();
        store.import_path(&source, model).unwrap();
    }
    let mut state = ApiState::new(store.clone(), Arc::new(MockBackend));
    state.text_backend = TextAnalysisBackend::with_test_client(
        store,
        CompanionClient::from_embedded_python(python, WORKER, "observability fixture"),
        json!({"cuda":false,"architectures":{"laya":true,"xlm-roberta":true,"embedding_gemma2":true}}),
    );
    let telemetry = state.telemetry.clone();
    let app = router(state);
    for (path, payload) in [
        (
            "/v1/classifications",
            json!({"model":"decision","state":{"text":"private-input"},"questions":{"private-question":{"type":"noul","instructions":"private-instructions"}}}),
        ),
        (
            "/v1/rerank",
            json!({"model":"ranker","query":"private-query","documents":["private-document"]}),
        ),
        (
            "/v1/embeddings",
            json!({"model":"embedder","input":"private-input"}),
        ),
    ] {
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();
        let pending = tokio::spawn(app.clone().oneshot(request));
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let snapshot = telemetry.snapshot();
                if snapshot.totals.active == 1
                    && snapshot.requests[0].state == "loading / inference"
                {
                    break;
                }
                assert!(
                    !pending.is_finished(),
                    "request finished without an observable active phase"
                );
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let response = pending.await.unwrap().unwrap();
        let status = response.status();
        let response: Value =
            serde_json::from_slice(&body::to_bytes(response.into_body(), 100000).await.unwrap())
                .unwrap();
        assert_eq!(status, StatusCode::OK, "{response}");
        let snapshot = telemetry.snapshot();
        let record = &snapshot.requests[0];
        assert_eq!(response["werk"]["request_id"], record.id);
        assert_eq!(record.prompt_tokens, Some(47));
        assert_eq!(
            record.analysis.as_ref().unwrap().inference_seconds,
            Some(0.2)
        );
        assert_eq!(record.analysis.as_ref().unwrap().results, Some(1));
        assert_eq!(snapshot.totals.active, 0);
        assert!(record.decode_tokens_per_second.is_none());
        assert!(
            !serde_json::to_string(&snapshot)
                .unwrap()
                .contains("private-")
        );
    }
    assert_eq!(telemetry.snapshot().totals.completed, 3);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let metrics = String::from_utf8(
        body::to_bytes(response.into_body(), 100000)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(metrics.contains("werk_requests_completed_total 3"));
    assert!(metrics.contains("werk_last_request_inference_seconds{model=\"decision\"} 0.2"));
}

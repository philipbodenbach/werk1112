use super::support::*;
use base64::Engine;

fn app(format: ModelFormat) -> (tempfile::TempDir, ApiState, Router) {
    let root = tempfile::tempdir().unwrap();
    let store = ModelStore::resolve(Some(root.path().to_owned())).unwrap();
    store.ensure().unwrap();
    let manifest = ModelManifest {
        id: "document-model".into(),
        source: ModelSource::LocalPath {
            path: "test".into(),
        },
        storage: Default::default(),
        format,
        architecture: Some("qwen2".into()),
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock".into(),
        created_unix: 1,
        files: vec![],
        artifacts: vec![],
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir(&manifest.id)).unwrap();
    fs::write(
        store
            .model_dir(&manifest.id)
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let state = ApiState::new(store, Arc::new(PromptEchoBackend))
        .with_api_keys(vec!["alice".into(), "bob".into()]);
    let router = router(state.clone());
    (root, state, router)
}
async fn send(
    app: &Router,
    method: &str,
    path: &str,
    key: &str,
    anthropic: bool,
    body: Value,
) -> Response {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json");
    if anthropic {
        request = request.header("anthropic-version", "2023-06-01");
    }
    app.clone()
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap()
}
async fn value(response: Response) -> Value {
    serde_json::from_slice(
        &body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap(),
    )
    .unwrap()
}
async fn upload(app: &Router, anthropic: bool) -> Value {
    let mut data="--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"notes.txt\"\r\nContent-Type: text/plain\r\n\r\nDocument secret 42\r\n".to_owned();
    if !anthropic {
        data.push_str(
            "--boundary\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nuser_data\r\n",
        );
    }
    data.push_str("--boundary--\r\n");
    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/files")
        .header("authorization", "Bearer alice")
        .header("content-type", "multipart/form-data; boundary=boundary");
    if anthropic {
        request = request.header("anthropic-version", "2023-06-01");
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::from(data)).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    value(response).await
}

#[tokio::test]
async fn uploaded_files_cross_api_formats_survive_restart_and_are_isolated() {
    let (_root, state, router) = app(ModelFormat::SafeTensors);
    let uploaded = upload(&router, false).await;
    assert_eq!(uploaded["object"], "file");
    assert_eq!(uploaded["purpose"], "user_data");
    let id = uploaded["id"].as_str().unwrap();
    let restarted = super::support::router(
        ApiState::new(state.store.as_ref().clone(), Arc::new(PromptEchoBackend))
            .with_api_keys(vec!["alice".into(), "bob".into()]),
    );
    let metadata = value(
        send(
            &restarted,
            "GET",
            &format!("/v1/files/{id}"),
            "alice",
            true,
            json!({}),
        )
        .await,
    )
    .await;
    assert_eq!(metadata["type"], "file");
    assert_eq!(metadata["size_bytes"], 18);
    assert_eq!(metadata["downloadable"], false);
    let request = json!({"model":"document-model","max_tokens":32,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"file","file_id":id}}]}]});
    let response = send(
        &restarted,
        "POST",
        "/v1/messages",
        "alice",
        true,
        request.clone(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        value(response).await["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Document secret 42")
    );
    assert_eq!(
        send(&restarted, "POST", "/v1/messages", "bob", true, request)
            .await
            .status(),
        StatusCode::BAD_REQUEST
    );
    for endpoint in [format!("/v1/files/{id}"), format!("/v1/files/{id}/content")] {
        assert_eq!(
            send(&restarted, "GET", &endpoint, "bob", false, json!({}))
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
    }
    assert_eq!(
        send(
            &restarted,
            "DELETE",
            &format!("/v1/files/{id}"),
            "bob",
            false,
            json!({})
        )
        .await
        .status(),
        StatusCode::NOT_FOUND
    );
    let listed = value(send(&restarted, "GET", "/v1/files", "bob", false, json!({})).await).await;
    assert_eq!(listed["data"], json!([]));
    let deleted = value(
        send(
            &restarted,
            "DELETE",
            &format!("/v1/files/{id}"),
            "alice",
            true,
            json!({}),
        )
        .await,
    )
    .await;
    assert_eq!(deleted["type"], "file_deleted");
    assert_eq!(
        send(
            &restarted,
            "GET",
            &format!("/v1/files/{id}"),
            "alice",
            false,
            json!({})
        )
        .await
        .status(),
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn inline_documents_use_the_same_pipeline_for_gguf_cuda_and_mlx_models() {
    for format in [
        ModelFormat::Gguf,
        ModelFormat::SafeTensors,
        ModelFormat::Mlx,
    ] {
        let (_root, _state, app) = app(format);
        let encoded =
            base64::engine::general_purpose::STANDARD.encode("Shared document for any backend");
        let openai = json!({"model":"document-model","messages":[{"role":"user","content":[{"type":"text","text":"Before"},{"type":"file","file":{"filename":"notes.txt","file_data":encoded}},{"type":"text","text":"After"}]}]});
        let anthropic = json!({"model":"document-model","max_tokens":256,"messages":[{"role":"user","content":[{"type":"text","text":"Before"},{"type":"document","title":"notes.txt","source":{"type":"text","media_type":"text/plain","data":"Shared document for any backend"}},{"type":"text","text":"After"}]}]});
        let o = send(&app, "POST", "/v1/chat/completions", "alice", false, openai).await;
        assert_eq!(o.status(), StatusCode::OK);
        let a = send(&app, "POST", "/v1/messages", "alice", true, anthropic).await;
        assert_eq!(a.status(), StatusCode::OK);
        let o = value(o).await;
        let a = value(a).await;
        assert_eq!(
            o["choices"][0]["message"]["content"],
            a["content"][0]["text"]
        );
        let text = a["content"][0]["text"].as_str().unwrap();
        assert!(text.find("Before").unwrap() < text.find("Shared document").unwrap());
        assert!(text.find("Shared document").unwrap() < text.find("After").unwrap());
    }
}

#[tokio::test]
async fn files_pagination_and_download_use_the_selected_contract() {
    let (_root, _state, app) = app(ModelFormat::Gguf);
    let first = upload(&app, true).await;
    let _second = upload(&app, false).await;
    let page = value(send(&app, "GET", "/v1/files?limit=1", "alice", true, json!({})).await).await;
    assert_eq!(page["data"].as_array().unwrap().len(), 1);
    let next = page["next_page"].as_str().unwrap();
    let second = value(
        send(
            &app,
            "GET",
            &format!("/v1/files?limit=1&page={next}"),
            "alice",
            true,
            json!({}),
        )
        .await,
    )
    .await;
    assert_eq!(second["data"].as_array().unwrap().len(), 1);
    assert!(second["next_page"].is_null());
    let id = first["id"].as_str().unwrap();
    let response = send(
        &app,
        "GET",
        &format!("/v1/files/{id}/content"),
        "alice",
        false,
        json!({}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body::to_bytes(response.into_body(), 1024).await.unwrap(),
        "Document secret 42"
    );
    assert_eq!(
        send(
            &app,
            "GET",
            &format!("/v1/files/{id}/content"),
            "alice",
            true,
            json!({})
        )
        .await
        .status(),
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn unsupported_openai_fields_are_errors_instead_of_silent_ignores() {
    let (_root, _state, app) = app(ModelFormat::Gguf);
    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        "alice",
        false,
        json!({"model":"document-model","messages":[],"not_a_parameter":true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        value(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not_a_parameter")
    );
}

#[tokio::test]
async fn file_transfer_admission_is_released_when_clients_disconnect() {
    let (_root, state, app) = app(ModelFormat::Gguf);
    let uploaded = upload(&app, false).await;
    let path = format!("/v1/files/{}/content", uploaded["id"].as_str().unwrap());
    let mut held = Vec::new();
    for _ in 0..4 {
        let response = send(&app, "GET", &path, "alice", false, json!({})).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["cache-control"], "no-store");
        assert!(response.headers().contains_key("request-id"));
        held.push(response);
    }
    assert_eq!(state.upload_gate.available_permits(), 0);
    assert_eq!(
        send(&app, "GET", &path, "alice", false, json!({}))
            .await
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    drop(held);
    assert_eq!(state.upload_gate.available_permits(), 4);
    assert_eq!(
        send(&app, "GET", &path, "alice", false, json!({}))
            .await
            .status(),
        StatusCode::OK
    );
}

#[tokio::test]
#[ignore = "requires PDF dependencies from docs/document-requirements.txt"]
async fn real_pdf_is_expanded_through_both_apis_for_every_model_layout() {
    let encoded = base64::engine::general_purpose::STANDARD.encode(include_bytes!(
        "../../../tests/fixtures/documents/sample.pdf"
    ));
    for format in [
        ModelFormat::Gguf,
        ModelFormat::SafeTensors,
        ModelFormat::Mlx,
    ] {
        let (_root, _state, app) = app(format);
        let openai = json!({"model":"document-model","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"sample.pdf","file_data":encoded}}]}]});
        let anthropic = json!({"model":"document-model","max_tokens":256,"messages":[{"role":"user","content":[{"type":"document","title":"sample.pdf","source":{"type":"base64","media_type":"application/pdf","data":encoded}}]}]});
        let o = send(&app, "POST", "/v1/chat/completions", "alice", false, openai).await;
        let a = send(&app, "POST", "/v1/messages", "alice", true, anthropic).await;
        assert_eq!(o.status(), StatusCode::OK);
        assert_eq!(a.status(), StatusCode::OK);
        let o = value(o).await;
        let a = value(a).await;
        assert_eq!(
            o["choices"][0]["message"]["content"],
            a["content"][0]["text"]
        );
        assert!(
            a["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("Werk document 42")
        );
    }
}

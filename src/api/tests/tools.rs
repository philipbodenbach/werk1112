use super::support::*;
fn call(name: &str, arguments: Value) -> Value {
    json!({"id":"call_fixture","type":"function","function":{"name":name,"arguments":arguments.to_string()}})
}

#[tokio::test]
async fn catalog_covers_all_modalities_without_installed_models_and_requires_auth() {
    let app = router(
        ApiState::new(test_store(), Arc::new(MockBackend)).with_api_keys(vec!["secret".into()]),
    );
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/tools")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/tools")
                .header(header::AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    let tools = body["tools"].as_array().unwrap();
    for name in [
        "text_generation",
        "image_understanding",
        "image_generation",
        "image_editing",
        "video_generation",
        "image_to_video",
        "music_generation",
        "song_continuation",
        "speech_to_text",
        "audio_understanding",
        "text_to_speech",
        "voice_conversion",
        "get_job",
        "cancel_job",
    ] {
        let tool = tools
            .iter()
            .find(|t| t["function"]["name"] == name)
            .expect(name);
        assert_eq!(tool["function"]["parameters"]["type"], "object");
        assert!(
            !tool["function"]["parameters"]["required"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
    let response = post_json(
        &app,
        "/v1/tools/call",
        call("image_generation", json!({"model":"media","prompt":"test"})),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn media_tools_execute_jobs_across_image_video_music_speech_and_audio() {
    let app = media_app(vec![]);
    for (name, expected_mime) in [
        ("image_generation", "image/png"),
        ("video_generation", "video/mp4"),
        ("audio_generation", "audio/wav"),
        ("music_generation", "audio/wav"),
        ("text_to_speech", "audio/wav"),
        ("speech_to_text", "text/plain"),
    ] {
        let mut arguments = json!({"model":"media","prompt":"fixture", "parameters":{"routing.backend":"mock-media"}});
        if name == "speech_to_text" {
            arguments.as_object_mut().unwrap().remove("prompt");
            arguments["inputs"] = json!([{"modality":"audio","role":"input_audio","source":{"kind":"base64","data":"AAEC"},"mime_type":"audio/wav"}]);
        }
        let response = post_json(&app, "/v1/tools/call", call(name, arguments), None).await;
        let status = response.status();
        let body = response_json(response).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{name}: {body}");
        assert_eq!(body["message"]["role"], "tool");
        assert_eq!(body["message"]["tool_call_id"], "call_fixture");
        let id = body["result"]["id"].as_str().unwrap();
        let mut completed = false;
        for _ in 0..200 {
            let response = post_json(
                &app,
                "/v1/tools/call",
                call("get_job", json!({"id":id})),
                None,
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let body = response_json(response).await;
            assert_ne!(body["result"]["status"], "failed", "{name}: {body}");
            if body["result"]["status"] == "completed" {
                assert_eq!(body["result"]["result"]["runtime"], "mock-media-cpu");
                assert_eq!(body["result"]["outputs"][0]["mime_type"], expected_mime);
                let url = body["result"]["outputs"][0]["url"].as_str().unwrap();
                let output = app
                    .clone()
                    .oneshot(Request::builder().uri(url).body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                assert_eq!(output.status(), StatusCode::OK);
                let content: Value =
                    serde_json::from_str(body["message"]["content"].as_str().unwrap()).unwrap();
                assert_eq!(content["id"], id);
                completed = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(completed, "{name} job did not complete");
    }
}

#[tokio::test]
async fn tool_dispatch_rejects_invalid_arguments_and_task_overrides() {
    let app = media_app(vec![]);
    for payload in [
        call("unknown", json!({"model":"media"})),
        call(
            "image_generation",
            json!({"model":"media","task":"music_generation","prompt":"test"}),
        ),
        call(
            "image_generation",
            json!({"model":"media","prompt":"test","parameters":{"image.width":-1}}),
        ),
        call("get_job", json!({"id":"../outside"})),
        json!({"id":"call_bad","type":"function","function":{"name":"image_generation","arguments":"[1,2]"}}),
    ] {
        let response = post_json(&app, "/v1/tools/call", payload, None).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn cancellation_is_an_explicit_tool_and_preserves_correlation() {
    let store = test_store();
    install_media_model(&store);
    let state = ApiState::new(store, Arc::new(MockBackend));
    let record = state
        .job_manager
        .store()
        .create(crate::inference::InferenceRequest::new(
            "media",
            InferenceTask::VideoGeneration,
        ))
        .unwrap();
    let app = router(state);
    let response = post_json(
        &app,
        "/v1/tools/call",
        call("cancel_job", json!({"id":record.id})),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["result"]["status"], "cancelled");
    assert_eq!(body["tool_call_id"], "call_fixture");
}

#[tokio::test]
async fn catalog_can_select_one_task_without_loading_a_model() {
    let app = media_app(vec![]);
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/tools?task=music-generation")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    let names = body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["music_generation", "get_job", "cancel_job"]);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/tools?task=unknown")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_and_vision_functions_use_chat_adapter_and_return_correlated_results() {
    let store = test_store();
    install_media_model(&store);
    let mut manifest = store.get("media").unwrap();
    manifest.metadata.tasks = vec![
        InferenceTask::TextGeneration,
        InferenceTask::ImageUnderstanding,
    ];
    fs::write(
        store
            .model_dir("media")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let app = router(ApiState::new(store, Arc::new(MockBackend)));
    for (name, content) in [
        ("text_generation", json!("hello")),
        (
            "image_understanding",
            json!([{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAEC","detail":"low"}}]),
        ),
    ] {
        let response = post_json(&app,"/v1/tools/call",call(name,json!({"model":"media","max_tokens":64,"messages":[{"role":"user","content":content}]})),None).await;
        let status = response.status();
        let body = response_json(response).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["result"]["choices"][0]["message"]["content"], "hello");
        assert_eq!(body["message"]["tool_call_id"], "call_fixture");
    }
}

#[tokio::test]
async fn vision_tool_requires_an_actual_image_before_dispatch() {
    let app = media_app(vec![]);
    for content in [
        json!("There is no image here"),
        json!([{"type":"text","text":"image_url: this is only text"}]),
        json!([{"type":"image_url","image_url":{"url":" "}}]),
    ] {
        let response = post_json(
            &app,
            "/v1/tools/call",
            call(
                "image_understanding",
                json!({
                    "model":"media", "messages":[{"role":"user","content":content}]
                }),
            ),
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("requires an image_url"),
            "{body}"
        );
    }
}

fn assert_compact_tool_job(body: &Value, input_data: &str) {
    assert!(!body.to_string().contains(input_data));
    let job = &body["result"];
    assert!(job.get("request").is_none());
    assert!(job["result"].get("effective_request").is_none());
    assert!(job["result"].get("backend_metadata").is_none());
    let content: Value =
        serde_json::from_str(body["message"]["content"].as_str().unwrap()).unwrap();
    assert_eq!(content, *job);
    assert!(body["message"]["content"].as_str().unwrap().len() < 4096);
}

#[tokio::test]
async fn media_tool_results_do_not_echo_request_bytes_at_submission_or_completion() {
    let app = media_app(vec![]);
    let input_data = "AAEC".repeat(8192);
    let response = post_json(&app, "/v1/tools/call", call("speech_to_text", json!({
        "model":"media", "inputs":[{"modality":"audio","role":"input_audio","source":{"kind":"base64","data":input_data},"mime_type":"audio/wav"}],
        "parameters":{"routing.backend":"mock-media"}
    })), None).await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let body = response_json(response).await;
    assert_compact_tool_job(&body, &input_data);
    let id = body["result"]["id"].as_str().unwrap().to_string();
    for _ in 0..200 {
        let response = post_json(
            &app,
            "/v1/tools/call",
            call("get_job", json!({"id":id})),
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert_compact_tool_job(&body, &input_data);
        assert_ne!(body["result"]["status"], "failed", "{body}");
        if body["result"]["status"] == "completed" {
            assert_eq!(body["result"]["result"]["runtime"], "mock-media-cpu");
            assert_eq!(body["result"]["outputs"][0]["mime_type"], "text/plain");
            assert!(
                body["result"]["outputs"][0]["url"]
                    .as_str()
                    .unwrap()
                    .starts_with("/v1/outputs/")
            );
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("media job did not complete");
}

#[tokio::test]
async fn cancellation_tool_does_not_echo_queued_media_inputs() {
    let store = test_store();
    install_media_model(&store);
    let state = ApiState::new(store, Arc::new(MockBackend));
    let input_data = "AAEC".repeat(8192);
    let request = crate::inference_service::tool_inference_request(InferenceTask::SpeechToText, json!({
        "model":"media", "inputs":[{"modality":"audio","role":"input_audio","source":{"kind":"base64","data":input_data}}]
    })).unwrap();
    let record = state.job_manager.store().create(request).unwrap();
    let app = router(state);
    let response = post_json(
        &app,
        "/v1/tools/call",
        call("cancel_job", json!({"id":record.id})),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_compact_tool_job(&body, &input_data);
    assert_eq!(body["result"]["status"], "cancelled");
}

#[tokio::test]
async fn inpainting_catalog_and_execution_require_distinct_source_and_mask_roles() {
    let store = test_store();
    install_media_model(&store);
    let mut manifest = store.get("media").unwrap();
    manifest.metadata.tasks.extend([
        InferenceTask::ImageInpainting,
        InferenceTask::ImageOutpainting,
        InferenceTask::VideoInpainting,
    ]);
    manifest
        .metadata
        .input_modalities
        .push(InputModality::Video);
    fs::write(
        store
            .model_dir("media")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let service = InferenceService::with_backend(store.clone(), Arc::new(MockMediaBackend));
    let app = router(ApiState::new(store, Arc::new(MockBackend)).with_inference_service(service));
    for (name, modality, source, mask, expected_mime) in [
        ("image_inpainting", "image", "image", "mask", "image/png"),
        (
            "image_outpainting",
            "image",
            "initial_image",
            "mask_image",
            "image/png",
        ),
        (
            "video_inpainting",
            "video",
            "source_video",
            "mask_video",
            "video/mp4",
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/tools?task={name}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        let inputs = &body["tools"][0]["function"]["parameters"]["properties"]["inputs"];
        assert_eq!(inputs["minItems"], 2);
        let groups = inputs["allOf"].as_array().unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups[0]["contains"]["properties"]["modality"]["const"],
            modality
        );
        assert!(
            groups[0]["contains"]["properties"]["role"]["enum"]
                .as_array()
                .unwrap()
                .contains(&json!(source))
        );
        assert!(
            groups[1]["contains"]["properties"]["role"]["enum"]
                .as_array()
                .unwrap()
                .contains(&json!(mask))
        );
        let source_input =
            json!({"modality":modality,"role":source,"source":{"kind":"base64","data":"AAEC"}});
        let mut arguments = json!({"model":"media","prompt":"repair", "inputs":[source_input], "parameters":{"routing.backend":"mock-media"}});
        let response = post_json(&app, "/v1/tools/call", call(name, arguments.clone()), None).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert!(
            body["error"]["message"].as_str().unwrap().contains("mask"),
            "{body}"
        );
        arguments["inputs"].as_array_mut().unwrap().push(
            json!({"modality":modality,"role":mask,"source":{"kind":"base64","data":"AAEC"}}),
        );
        let response = post_json(&app, "/v1/tools/call", call(name, arguments), None).await;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = response_json(response).await;
        let id = body["result"]["id"].as_str().unwrap().to_string();
        let mut completed = false;
        for _ in 0..200 {
            let response = post_json(
                &app,
                "/v1/tools/call",
                call("get_job", json!({"id":id})),
                None,
            )
            .await;
            let body = response_json(response).await;
            assert_ne!(body["result"]["status"], "failed", "{body}");
            if body["result"]["status"] == "completed" {
                assert_eq!(body["result"]["outputs"][0]["mime_type"], expected_mime);
                completed = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(completed, "{name} job did not complete");
    }
}

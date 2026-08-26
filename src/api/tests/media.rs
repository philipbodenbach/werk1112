use super::support::*;

#[tokio::test]
async fn generic_audio_jobs_accept_large_bounded_json_bodies() {
    let store = test_store();
    let body_limit = 3 * 1024 * 1024;
    let app = crate::api::router::router_with_body_limit(
        ApiState::new(store, Arc::new(MockBackend)),
        body_limit,
    );
    let payload = |encoded_size| {
        json!({
            "model": "missing-audio-model",
            "task": "speech-to-text",
            "inputs": [{
                "modality": "audio",
                "role": "input_audio",
                "source": {"kind": "base64", "data": "A".repeat(encoded_size)}
            }]
        })
    };

    let above_default_size = 2 * 1024 * 1024 + 64 * 1024;
    let above_axum_default = post_json(&app, "/v1/jobs", payload(above_default_size), None).await;
    assert_eq!(above_axum_default.status(), StatusCode::BAD_REQUEST);

    for endpoint in ["/v1/audio/transcriptions", "/v1/audio/translations"] {
        let response = post_json(
            &app,
            endpoint,
            json!({
                "model": "missing-audio-model",
                "file": {
                    "base64": "A".repeat(above_default_size),
                    "mime_type": "audio/wav"
                }
            }),
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{endpoint}");
    }

    let unrelated_large_body = post_json(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "missing-chat-model",
            "messages": [{"role": "user", "content": "A".repeat(above_default_size)}]
        }),
        None,
    )
    .await;
    assert_eq!(unrelated_large_body.status(), StatusCode::PAYLOAD_TOO_LARGE);

    let above_configured_limit =
        post_json(&app, "/v1/jobs", payload(body_limit + 1024), None).await;
    assert_eq!(
        above_configured_limit.status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
}

#[test]
fn media_request_shapes_normalize_openai_and_werk_fields() {
    let parsed: ImageGenerationApiRequest = serde_json::from_value(json!({
        "model": "media",
        "prompt": "a small station",
        "negative_prompt": "crowded",
        "n": 2,
        "size": "640x480",
        "response_format": "b64_json",
        "output_format": "webp",
        "steps": 12,
        "parameters": {"guidance": 4.5},
        "backend": "mock-media",
        "quality": "low",
        "stream": false,
        "background": "auto",
        "moderation": "auto",
        "output_compression": 80,
        "partial_images": 0,
        "style": "natural",
        "allow_cpu_offload": true
    }))
    .unwrap();
    let (request, response_format) = parsed.into_inference("media".to_string()).unwrap();
    assert_eq!(request.task, InferenceTask::ImageGeneration);
    assert!(matches!(response_format, DirectResponseFormat::Base64));
    assert_eq!(
        request
            .parameters
            .get("image.width")
            .and_then(ParameterValue::as_u64),
        Some(640)
    );
    assert_eq!(
        request
            .parameters
            .get("image.height")
            .and_then(ParameterValue::as_u64),
        Some(480)
    );
    assert_eq!(
        request
            .parameters
            .get("image.num_images")
            .and_then(ParameterValue::as_u64),
        Some(2)
    );
    assert_eq!(
        request
            .parameters
            .get("image.steps")
            .and_then(ParameterValue::as_u64),
        Some(12)
    );
    assert_eq!(
        request
            .parameters
            .get("image.guidance")
            .and_then(ParameterValue::as_f64),
        Some(4.5)
    );
    assert_eq!(request.routing.backend.as_deref(), Some("mock-media"));
    assert_eq!(request.routing.quality.as_deref(), Some("draft"));
    assert_eq!(request.routing.allow_cpu_offload, OverrideBool::Enabled);
    assert_eq!(
        request
            .parameters
            .get("image.output_format")
            .and_then(ParameterValue::as_str),
        Some("webp")
    );
    assert!(
        request
            .prompt
            .as_deref()
            .is_some_and(|prompt| prompt.ends_with("Visual style: natural and realistic."))
    );

    let parsed: ImageEditApiRequest = serde_json::from_value(json!({
        "model": "media",
        "prompt": "replace the sky",
        "image": "data:image/png;base64,AAEC"
    }))
    .unwrap();
    let (request, _) = parsed.into_inference().unwrap();
    assert!(matches!(
        request.inputs[0].source,
        InferenceInputSource::Base64 { ref data } if data == "AAEC"
    ));
    assert_eq!(request.inputs[0].mime_type.as_deref(), Some("image/png"));

    let serialized = serde_json::to_value(ApiMediaInput::Object(ApiMediaInputObject {
        path: None,
        url: Some("https://example.test/input.wav".to_string()),
        base64: None,
        mime_type: Some("audio/wav".to_string()),
    }))
    .unwrap();
    assert_eq!(
        serialized["url"],
        Value::String("https://example.test/input.wav".to_string())
    );
}

#[test]
fn openai_image_defaults_are_accepted_without_becoming_backend_parameters() {
    let parsed: ImageGenerationApiRequest = serde_json::from_value(json!({
        "prompt": "a small station",
        "size": "auto",
        "quality": "auto",
        "stream": false,
        "background": "auto",
        "moderation": "auto",
        "output_compression": 100,
        "partial_images": 0
    }))
    .unwrap();
    assert_eq!(parsed.requested_model(), None);
    let (request, response_format) = parsed.into_inference("media".to_string()).unwrap();
    assert!(matches!(response_format, DirectResponseFormat::Base64));
    assert_eq!(request.model, "media");
    assert_eq!(request.routing.quality, None);
    assert!(!request.parameters.contains_key("image.width"));
    assert!(!request.parameters.contains_key("image.height"));
    for compatibility_field in [
        "image.background",
        "image.moderation",
        "image.output_compression",
        "image.partial_images",
        "image.stream",
    ] {
        assert!(!request.parameters.contains_key(compatibility_field));
    }

    let parsed: ImageGenerationApiRequest = serde_json::from_value(json!({
        "model": "media",
        "prompt": "a small station",
        "stream": true
    }))
    .unwrap();
    let error = parsed.into_inference("media".to_string()).unwrap_err();
    assert!(error.contains("streaming image generation is not supported"));
}

#[tokio::test]
async fn direct_media_routes_return_openai_data_and_werk_metadata() {
    let app = media_app(Vec::new());

    let response = post_json(
        &app,
        "/v1/images/generations",
        json!({
            "prompt": "an orbital greenhouse",
            "size": "512x512",
            "response_format": "b64_json"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["task"], "image_generation");
    assert_eq!(value["werk"]["model"], "media");
    assert_eq!(value["data"][0]["mime_type"], "image/png");
    assert_eq!(value["data"][0]["b64_json"], encode_base64(b"mock image"));
    assert_eq!(
        value["werk"]["effective_request"]["parameters"]["image.width"]["value"],
        512
    );
    let embedded_output_id = value["data"][0]["id"].as_str().unwrap();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/outputs/{embedded_output_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let response = post_json(
        &app,
        "/v1/images/generations",
        json!({
            "model": "gpt-image-1",
            "prompt": "an orbital greenhouse"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["model"], "media");
    assert!(value["data"][0]["b64_json"].is_string());

    let response = post_json(
        &app,
        "/v1/images/edits",
        json!({
            "model": "media",
            "prompt": "make it blue",
            "image": {"base64": "AAEC", "mime_type": "image/png"},
            "response_format": "url"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["task"], "image_editing");
    let output_url = value["data"][0]["url"].as_str().unwrap().to_string();
    assert!(output_url.starts_with("/v1/outputs/"));
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(output_url)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "image/png"
    );
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&bytes[..], b"mock image");

    let response = post_json(
        &app,
        "/v1/audio/speech",
        json!({
            "model": "media",
            "input": "hello",
            "voice": "test",
            "response_format": "wav"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "audio/wav"
    );
    let speech_output_id = response
        .headers()
        .get("x-werk-output-id")
        .and_then(|value| value.to_str().ok())
        .unwrap()
        .to_string();
    assert!(speech_output_id.starts_with("out-"));
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&bytes[..], b"mock audio");
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/outputs/{speech_output_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let response = post_json(
        &app,
        "/v1/audio/transcriptions",
        json!({
            "model": "media",
            "file": {"base64": "AAEC", "mime_type": "audio/wav"},
            "prompt": "Names and technical terms",
            "response_format": "text"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["task"], "speech_to_text");
    assert_eq!(value["data"][0]["text"], "mock transcript");
    assert_eq!(
        value["werk"]["effective_request"]["parameters"]["stt.initial_prompt"]["value"],
        "Names and technical terms"
    );
    assert!(value["werk"]["effective_request"]["prompt"].is_null());
    let transcript_output_id = value["data"][0]["id"].as_str().unwrap();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/outputs/{transcript_output_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let response = post_json(
        &app,
        "/v1/audio/translations",
        json!({
            "model": "media",
            "file": {"base64": "AAEC", "mime_type": "audio/wav"},
            "response_format": "text"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["task"], "speech_translation");
    assert_eq!(value["data"][0]["text"], "mock transcript");
    assert!(value["data"][0]["url"].is_null());
    assert_eq!(
        value["werk"]["effective_request"]["parameters"]["stt.operation"]["value"],
        "translate"
    );
    let translation_output_id = value["data"][0]["id"].as_str().unwrap();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/outputs/{translation_output_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn open_webui_openai_image_request_returns_embedded_image() {
    let app = media_app(vec!["sk-media".to_string()]);
    let response = post_json(
        &app,
        "/v1/images/generations?api-version=2025-04-01-preview",
        json!({
            "model": "gpt-image-1",
            "prompt": "an orbital greenhouse",
            "n": 1,
            "size": "512x512"
        }),
        Some("sk-media"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["model"], "media");
    assert_eq!(value["data"][0]["b64_json"], encode_base64(b"mock image"));

    let output_id = value["data"][0]["id"].as_str().unwrap();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/outputs/{output_id}"))
                .header(header::AUTHORIZATION, "Bearer sk-media")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn comfyui_hosted_openai_proxy_accepts_x_api_key() {
    let app = media_app(vec!["sk-media".to_string()]);
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/proxy/openai/images/generations")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-api-key", "sk-media")
                .body(Body::from(
                    json!({
                        "model": "gpt-image-1",
                        "prompt": "an orbital greenhouse",
                        "quality": "low",
                        "background": "opaque",
                        "moderation": "low",
                        "n": 1,
                        "seed": 7,
                        "size": "512x512"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["model"], "media");
    assert_eq!(value["data"][0]["b64_json"], encode_base64(b"mock image"));
    assert_eq!(
        value["werk"]["effective_request"]["parameters"]["image.seed"]["value"],
        7
    );
    assert!(value["werk"]["effective_request"]["parameters"]["image.moderation"].is_null());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/proxy/openai/images/edits")
                .header("x-api-key", "sk-media")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    assert!(
        response_json(response).await["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("multipart"))
    );
}

#[tokio::test]
async fn image_default_resolves_omissions_and_openai_aliases_with_multiple_models() {
    let store = test_store();
    install_media_model(&store);
    let mut other = store.get("media").unwrap();
    other.id = "other-image".to_string();
    fs::create_dir_all(store.model_dir(&other.id)).unwrap();
    fs::write(
        store
            .model_dir(&other.id)
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&other).unwrap(),
    )
    .unwrap();

    let inference_service =
        InferenceService::with_backend(store.clone(), Arc::new(MockMediaBackend));
    let app = router(
        ApiState::new(store, Arc::new(MockBackend))
            .with_default_image_model(Some("media".to_string()))
            .with_inference_service(inference_service),
    );

    for model in [None, Some("gpt-image-1")] {
        let mut payload = json!({"prompt": "an orbital greenhouse"});
        if let Some(model) = model {
            payload["model"] = json!(model);
        }
        let response = post_json(&app, "/v1/images/generations", payload, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response_json(response).await["werk"]["model"], "media");
    }

    let response = post_json(
        &app,
        "/v1/images/generations",
        json!({
            "prompt": "an orbital greenhouse",
            "stream": true
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        response_json(response).await["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("streaming image generation is not supported"))
    );
}

#[tokio::test]
async fn long_media_and_generic_job_routes_return_job_records() {
    let app = media_app(Vec::new());

    let response = post_json(
        &app,
        "/v1/videos/generations",
        json!({
            "model": "media",
            "prompt": "slow camera orbit",
            "size": "832x480",
            "response_format": "mp4"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let video_job = response_json(response).await;
    assert_eq!(video_job["request"]["task"], "video_generation");
    let video_id = video_job["id"].as_str().unwrap().to_string();

    let response = post_json(
        &app,
        "/v1/audio/generations",
        json!({
            "model": "media",
            "prompt": "quiet analogue ambient",
            "task": "music-generation",
            "n": 2,
            "response_format": "wav"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let audio_job = response_json(response).await;
    assert_eq!(audio_job["request"]["task"], "music_generation");

    let response = post_json(
        &app,
        "/v1/audio/speech",
        json!({
            "model": "media",
            "input": "background speech",
            "async": true
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);

    let response = post_json(
        &app,
        "/v1/jobs",
        json!({
            "model": "media",
            "task": "image-generation",
            "prompt": "generic job request",
            "parameters": {"width": 512, "height": 512}
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let generic_job = response_json(response).await;
    assert_eq!(generic_job["request"]["task"], "image_generation");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/jobs/{video_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let fetched = response_json(response).await;
    assert_eq!(fetched["id"], video_id);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/jobs/{video_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let cancelled = response_json(response).await;
    assert!(matches!(
        cancelled["status"].as_str(),
        Some("cancelled" | "completed" | "failed")
    ));
}

#[tokio::test]
async fn capability_and_parameter_routes_are_authenticated() {
    let app = media_app(vec!["sk-media".to_string()]);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = post_json(
        &app,
        "/v1/images/generations",
        json!({"model": "media", "prompt": "unauthorized"}),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .header(header::AUTHORIZATION, "Bearer sk-media")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let capabilities = response_json(response).await;
    assert_eq!(capabilities["object"], "werk.capabilities");
    assert!(
        capabilities["models"][0]["available_tasks"]
            .as_array()
            .is_some_and(|tasks| tasks.contains(&json!("image_generation")))
    );
    assert_eq!(
        capabilities["models"][0]["task_statuses"]["image_generation"]["status"],
        "available"
    );
    assert_eq!(
        capabilities["models"][0]["task_statuses"]["image_generation"]["adapter"],
        "mock-media"
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/parameters?task=image-generation&model=media&backend=mock-media")
                .header(header::AUTHORIZATION, "Bearer sk-media")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let parameters = response_json(response).await;
    assert_eq!(parameters["object"], "werk.parameter_schema");
    assert_eq!(parameters["task"], "image_generation");
    assert_eq!(parameters["parameter_support"]["image.width"], "native");
    assert_eq!(parameters["runtime_candidates"][0]["id"], "mock-media-cpu");
    assert_eq!(parameters["task_readiness"]["status"], "available");
    assert_eq!(parameters["task_readiness"]["adapter"], "mock-media");
    assert!(
        parameters["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .any(|parameter| parameter["path"] == "image.width")
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/parameters?task=image-generation&backend=cuda")
                .header(header::AUTHORIZATION, "Bearer sk-media")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let parameters = response_json(response).await;
    assert_eq!(parameters["backend"], "cuda");
    assert!(
        parameters["runtime_candidates"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

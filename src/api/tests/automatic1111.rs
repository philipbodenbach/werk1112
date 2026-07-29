use super::support::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use std::sync::{Condvar, Mutex};

#[tokio::test]
async fn txt2img_returns_exact_a1111_shape_and_cleans_embedded_outputs() {
    let store = test_store();
    install_media_model(&store);
    let output_root = store.home().join("outputs");
    let inference_service =
        InferenceService::with_backend(store.clone(), Arc::new(MockMediaBackend));
    let app = router(
        ApiState::new(store, Arc::new(MockBackend))
            .with_default_image_model(Some("media".to_string()))
            .with_inference_service(inference_service),
    );

    let response = post_json(
        &app,
        "/sdapi/v1/txt2img",
        json!({
            "prompt": "a small red robot",
            "negative_prompt": "blurry",
            "seed": -1,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "seed_enable_extras": true,
            "batch_size": 1,
            "n_iter": 1,
            "steps": 4,
            "cfg_scale": 1.5,
            "width": 512,
            "height": 512,
            "sampler_name": "Euler",
            "scheduler": "Automatic",
            "ddim_discretize": "uniform",
            "s_noise": 1,
            "denoising_strength": null,
            "override_settings": null,
            "enable_hr": false,
            "restore_faces": false,
            "tiling": false,
            "styles": [],
            "refiner_checkpoint": "None",
            "refiner_switch_at": 0.8,
            "script_args": null,
            "alwayson_scripts": null,
            "save_images": false,
            "send_images": true,
            "unknown_false_default": false
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    let keys = value
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(keys, vec!["images", "info", "parameters"]);
    assert_eq!(value["images"][0], encode_base64(b"mock image"));
    assert_eq!(value["parameters"]["seed"], -1);
    assert_eq!(value["parameters"]["sampler_index"], "Euler");
    assert_eq!(value["parameters"]["seed_resize_from_h"], -1);
    assert_eq!(value["parameters"]["seed_resize_from_w"], -1);
    assert_eq!(value["parameters"]["seed_enable_extras"], true);
    assert_eq!(value["parameters"]["s_noise"], 1.0);
    assert_eq!(value["parameters"]["denoising_strength"], Value::Null);
    assert_eq!(value["parameters"]["override_settings"], Value::Null);
    assert_eq!(value["parameters"]["unknown_false_default"], false);

    let info: Value = serde_json::from_str(value["info"].as_str().unwrap()).unwrap();
    assert_eq!(info["prompt"], "a small red robot");
    assert_eq!(info["negative_prompt"], "blurry");
    assert_eq!(info["width"], 512);
    assert_eq!(info["height"], 512);
    assert_eq!(info["steps"], 4);
    assert_eq!(info["cfg_scale"], 1.5);
    assert_eq!(info["batch_size"], 1);
    assert_eq!(info["sampler_name"], "Werk automatic");
    assert_eq!(info["sd_model_name"], "media");
    assert!(info["seed"].as_u64().is_some());
    assert!(
        info["werk_warnings"][0]
            .as_str()
            .is_some_and(|warning| warning.contains("sampler_name"))
    );

    assert!(output_root.is_dir());
    assert_eq!(fs::read_dir(output_root).unwrap().count(), 0);
}

#[tokio::test]
async fn unique_image_model_is_reported_and_result_arrays_follow_actual_outputs() {
    let app = media_app(Vec::new());
    let options = response_json(get(&app, "/sdapi/v1/options", None).await).await;
    assert_eq!(options["sd_model_checkpoint"], "media");

    let response = post_json(
        &app,
        "/sdapi/v1/txt2img",
        json!({
            "prompt": "two requested variations",
            "batch_size": 2,
            "n_iter": 1,
            "steps": 2,
            "seed": 3
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    let info: Value = serde_json::from_str(value["info"].as_str().unwrap()).unwrap();
    assert_eq!(value["images"].as_array().unwrap().len(), 1);
    assert_eq!(info["all_prompts"].as_array().unwrap().len(), 1);
    assert_eq!(info["all_negative_prompts"].as_array().unwrap().len(), 1);
    assert_eq!(info["all_seeds"].as_array().unwrap().len(), 1);
    assert!(
        info["werk_warnings"]
            .as_array()
            .is_some_and(|warnings| warnings.iter().any(|warning| warning
                .as_str()
                .is_some_and(|warning| warning.contains("backend returned 1"))))
    );
}

#[tokio::test]
async fn models_and_options_select_image_checkpoints_without_mutating_request_overrides() {
    let store = test_store();
    install_media_model(&store);
    install_model_clone(&store, "media", "other-image");
    install_non_image_model(&store, "chat-only");
    let inference_service =
        InferenceService::with_backend(store.clone(), Arc::new(MockMediaBackend));
    let app = router(
        ApiState::new(store, Arc::new(MockBackend))
            .with_default_image_model(Some("media".to_string()))
            .with_inference_service(inference_service),
    );

    let models = response_json(get(&app, "/sdapi/v1/sd-models", None).await).await;
    assert_eq!(models.as_array().unwrap().len(), 2);
    assert!(
        models
            .as_array()
            .unwrap()
            .iter()
            .all(|model| model["model_name"] != "chat-only")
    );
    let media = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["model_name"] == "media")
        .unwrap();
    assert_eq!(media["filename"], "media");
    assert_eq!(media["hash"], Value::Null);

    let options = response_json(get(&app, "/sdapi/v1/options", None).await).await;
    assert_eq!(options["sd_model_checkpoint"], "media");
    assert_eq!(options["samples_format"], "png");

    let response = post_json(
        &app,
        "/sdapi/v1/options",
        json!({"sd_model_checkpoint": "other-image"}),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response_json(response).await, Value::Null);
    let options = response_json(get(&app, "/sdapi/v1/options", None).await).await;
    assert_eq!(options["sd_model_checkpoint"], "other-image");

    let response = post_json(
        &app,
        "/sdapi/v1/txt2img",
        json!({"prompt": "selected model", "steps": 2}),
        None,
    )
    .await;
    let value = response_json(response).await;
    let info: Value = serde_json::from_str(value["info"].as_str().unwrap()).unwrap();
    assert_eq!(info["sd_model_name"], "other-image");

    let response = post_json(
        &app,
        "/sdapi/v1/txt2img",
        json!({
            "prompt": "temporary override",
            "steps": 2,
            "override_settings": {"sd_model_checkpoint": "media"}
        }),
        None,
    )
    .await;
    let value = response_json(response).await;
    let info: Value = serde_json::from_str(value["info"].as_str().unwrap()).unwrap();
    assert_eq!(info["sd_model_name"], "media");
    let options = response_json(get(&app, "/sdapi/v1/options", None).await).await;
    assert_eq!(options["sd_model_checkpoint"], "other-image");
}

#[tokio::test]
async fn unsupported_features_return_a1111_error_shape_and_x_api_key_is_accepted() {
    let store = test_store();
    install_media_model(&store);
    let inference_service =
        InferenceService::with_backend(store.clone(), Arc::new(MockMediaBackend));
    let app = router(
        ApiState::new(store, Arc::new(MockBackend))
            .with_default_image_model(Some("media".to_string()))
            .with_inference_service(inference_service)
            .with_api_keys(vec!["sk-media".to_string()]),
    );

    let response = get_with_header(&app, "/sdapi/v1/options", "x-api-key", "sk-media").await;
    assert_eq!(response.status(), StatusCode::OK);

    let basic = format!("Basic {}", STANDARD.encode("werk:sk-media"));
    let response = get_with_header(&app, "/sdapi/v1/options", "authorization", &basic).await;
    assert_eq!(response.status(), StatusCode::OK);

    let response = get(&app, "/sdapi/v1/options", None).await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response.headers().get(header::WWW_AUTHENTICATE).unwrap(),
        "Basic realm=\"Werk1112\""
    );
    assert_eq!(response_json(response).await["error"], "HTTPException");

    let response = post_json(
        &app,
        "/sdapi/v1/txt2img",
        json!({"prompt": "robot", "enable_hr": true}),
        Some("sk-media"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let value = response_json(response).await;
    let keys = value
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(keys, vec!["body", "detail", "error", "errors"]);
    assert_eq!(value["error"], "HTTPException");
    assert!(
        value["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("enable_hr"))
    );

    let response = post_json(
        &app,
        "/sdapi/v1/txt2img",
        json!({
            "prompt": "robot",
            "seed_resize_from_h": 512,
            "seed_resize_from_w": 512
        }),
        Some("sk-media"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert!(
        response_json(response).await["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("seed_resize_from_h"))
    );

    let response = post_json(
        &app,
        "/sdapi/v1/txt2img",
        json!({"prompt": "robot", "s_noise": 0.5}),
        Some("sk-media"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert!(
        response_json(response).await["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("s_noise"))
    );
}

#[tokio::test]
async fn extractor_rejections_use_a1111_errors_and_auth_takes_precedence() {
    let app = media_app(vec!["sk-media".to_string()]);

    let response = post_raw_json(&app, "/sdapi/v1/txt2img", "{", None).await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response.headers().get(header::WWW_AUTHENTICATE).unwrap(),
        "Basic realm=\"Werk1112\""
    );
    assert_a1111_error(response).await;

    let response = post_raw_json(&app, "/sdapi/v1/txt2img", "{", Some("sk-media")).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_a1111_error(response).await;

    let response = post_raw_json(&app, "/sdapi/v1/options", "{", Some("sk-media")).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_a1111_error(response).await;

    let response = get(
        &app,
        "/sdapi/v1/progress?skip_current_image=not-a-boolean",
        Some("sk-media"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_a1111_error(response).await;
}

#[tokio::test]
async fn progress_reports_only_honest_coarse_active_and_idle_state() {
    let store = test_store();
    install_media_model(&store);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let backend = BlockingMediaBackend {
        inner: MockMediaBackend,
        started: Arc::new(Mutex::new(Some(started_tx))),
        gate: Arc::clone(&gate),
    };
    let inference_service = InferenceService::with_backend(store.clone(), Arc::new(backend));
    let app = router(
        ApiState::new(store, Arc::new(MockBackend))
            .with_default_image_model(Some("media".to_string()))
            .with_inference_service(inference_service),
    );

    let generation_app = app.clone();
    let generation = tokio::spawn(async move {
        post_json(
            &generation_app,
            "/sdapi/v1/txt2img",
            json!({"prompt": "blocked generation", "steps": 7}),
            None,
        )
        .await
    });
    started_rx.await.unwrap();

    let active =
        response_json(get(&app, "/sdapi/v1/progress?skip_current_image=true", None).await).await;
    assert_eq!(active["progress"], 0.01);
    assert_eq!(active["eta_relative"], 0.0);
    assert_eq!(active["state"]["job"], "scripts_txt2img");
    assert_eq!(active["state"]["job_count"], 1);
    assert_eq!(active["state"]["sampling_step"], 0);
    assert_eq!(active["state"]["sampling_steps"], 7);
    assert_eq!(active["current_image"], Value::Null);
    assert!(
        active["textinfo"]
            .as_str()
            .is_some_and(|text| text.contains("step-level progress is unavailable"))
    );

    let (released, condvar) = &*gate;
    *released.lock().unwrap() = true;
    condvar.notify_all();
    assert_eq!(generation.await.unwrap().status(), StatusCode::OK);

    let idle = response_json(get(&app, "/sdapi/v1/progress", None).await).await;
    assert_eq!(idle["progress"], 0.0);
    assert_eq!(idle["state"]["job"], "");
    assert_eq!(idle["state"]["job_count"], 0);
    assert_eq!(idle["state"]["sampling_steps"], 0);
    assert_eq!(idle["textinfo"], Value::Null);
}

#[derive(Clone)]
struct BlockingMediaBackend {
    inner: MockMediaBackend,
    started: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    gate: Arc<(Mutex<bool>, Condvar)>,
}

impl MediaInferenceBackend for BlockingMediaBackend {
    fn probe(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
        task: InferenceTask,
        schema_paths: &[String],
    ) -> BackendProbe {
        self.inner.probe(store, manifest, task, schema_paths)
    }

    fn execute(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
        request: &EffectiveInferenceRequest,
        output_dir: &Path,
        runtime: &str,
    ) -> anyhow::Result<BackendExecution> {
        if let Some(started) = self.started.lock().unwrap().take() {
            let _ = started.send(());
        }
        let (released, condvar) = &*self.gate;
        let mut released = released.lock().unwrap();
        while !*released {
            released = condvar.wait(released).unwrap();
        }
        self.inner
            .execute(store, manifest, request, output_dir, runtime)
    }
}

async fn get(app: &Router, uri: &str, bearer_token: Option<&str>) -> Response {
    let mut builder = Request::builder().uri(uri);
    if let Some(token) = bearer_token {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    app.clone()
        .oneshot(builder.body(Body::empty()).unwrap())
        .await
        .unwrap()
}

async fn get_with_header(app: &Router, uri: &str, name: &str, value: &str) -> Response {
    app.clone()
        .oneshot(
            Request::builder()
                .uri(uri)
                .header(name, value)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn post_raw_json(
    app: &Router,
    uri: &str,
    payload: &str,
    bearer_token: Option<&str>,
) -> Response {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(token) = bearer_token {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    app.clone()
        .oneshot(builder.body(Body::from(payload.to_string())).unwrap())
        .await
        .unwrap()
}

async fn assert_a1111_error(response: Response) {
    let value = response_json(response).await;
    assert_eq!(value["error"], "HTTPException");
    assert!(
        value["detail"]
            .as_str()
            .is_some_and(|detail| !detail.is_empty())
    );
    assert_eq!(value["body"], "");
    assert_eq!(value["errors"], "");
}

fn install_model_clone(store: &ModelStore, source: &str, id: &str) {
    let mut manifest = store.get(source).unwrap();
    manifest.id = id.to_string();
    fs::create_dir_all(store.model_dir(id)).unwrap();
    fs::write(
        store.model_dir(id).join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

fn install_non_image_model(store: &ModelStore, id: &str) {
    let manifest = ModelManifest {
        id: id.to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: ModelMetadata::default(),
    };
    fs::create_dir_all(store.model_dir(id)).unwrap();
    fs::write(
        store.model_dir(id).join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

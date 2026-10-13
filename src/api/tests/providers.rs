use super::support::*;

#[tokio::test]
async fn provider_extensions_require_existing_server_auth_and_reject_invalid_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::resolve(Some(dir.path().into())).unwrap();
    let state =
        ApiState::new(store, Arc::new(MockBackend)).with_api_keys(vec!["server-secret".into()]);
    let app = router(state);
    for (method, path, payload) in [
        ("GET", "/werk/v1/model-support", json!(null)),
        (
            "POST",
            "/werk/v1/images/flux-3",
            json!({"prompt":"x","seed":1}),
        ),
        ("POST", "/werk/v1/audio/dialogue", json!({"inputs":[]})),
        (
            "POST",
            "/werk/v1/actions/predictions",
            json!({"model":"droid","task":"x","state":[0,0,0,0,0,0,0,0],"images":{}}),
        ),
    ] {
        let request = |auth: bool| {
            let mut b = Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json");
            if auth {
                b = b.header("authorization", "Bearer server-secret");
            }
            b.body(Body::from(payload.to_string())).unwrap()
        };
        assert_eq!(
            app.clone().oneshot(request(false)).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
        let response = app.clone().oneshot(request(true)).await.unwrap();
        assert_eq!(
            response.status(),
            if method == "GET" {
                StatusCode::OK
            } else {
                StatusCode::BAD_REQUEST
            }
        );
    }
}

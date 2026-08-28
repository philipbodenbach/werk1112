use super::support::*;
use crate::api::CorsOrigin;
use axum::http::{Method, header};

fn cors_app(origins: &[&str], api_keys: Vec<String>) -> Router {
    let origins = origins
        .iter()
        .map(|origin| origin.parse::<CorsOrigin>().unwrap())
        .collect();
    router(
        ApiState::new(test_store(), Arc::new(MockBackend))
            .with_api_keys(api_keys)
            .with_cors_origins(origins),
    )
}

fn comma_separated_header(response: &Response, name: header::HeaderName) -> Vec<String> {
    response
        .headers()
        .get(name)
        .unwrap()
        .to_str()
        .unwrap()
        .split(',')
        .map(|value| value.trim().to_ascii_lowercase())
        .collect()
}

#[tokio::test]
async fn cors_is_disabled_without_an_explicit_origin() {
    let app = cors_app(&[], Vec::new());
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header(header::ORIGIN, "https://app.example.test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none()
    );

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::OPTIONS)
                .uri("/v1/chat/completions")
                .header(header::ORIGIN, "https://app.example.test")
                .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none()
    );
}

#[tokio::test]
async fn allowed_origin_preflight_supports_werk_and_openai_sdk_headers() {
    let app = cors_app(&["https://app.example.test"], vec!["sk-test".to_string()]);
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::OPTIONS)
                .uri("/v1/chat/completions")
                .header(header::ORIGIN, "https://app.example.test")
                .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                .header(
                    header::ACCESS_CONTROL_REQUEST_HEADERS,
                    "authorization,x-api-key,content-type,accept,openai-organization,openai-project,x-stainless-lang,x-stainless-package-version,x-stainless-os,x-stainless-arch,x-stainless-runtime,x-stainless-runtime-version,x-stainless-retry-count,x-stainless-timeout,x-stainless-helper-method",
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .unwrap(),
        "https://app.example.test"
    );
    assert!(response.headers().get(header::WWW_AUTHENTICATE).is_none());
    assert!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_CREDENTIALS)
            .is_none()
    );

    let methods = comma_separated_header(&response, header::ACCESS_CONTROL_ALLOW_METHODS);
    assert_eq!(methods, vec!["get", "post", "delete"]);
    let headers = comma_separated_header(&response, header::ACCESS_CONTROL_ALLOW_HEADERS);
    for expected in [
        "authorization",
        "x-api-key",
        "content-type",
        "accept",
        "openai-organization",
        "openai-project",
        "x-stainless-lang",
        "x-stainless-package-version",
        "x-stainless-os",
        "x-stainless-arch",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
        "x-stainless-retry-count",
        "x-stainless-timeout",
        "x-stainless-read-timeout",
        "x-stainless-helper-method",
        "x-stainless-async",
    ] {
        assert!(
            headers.iter().any(|header| header == expected),
            "{expected}"
        );
    }
    let vary = comma_separated_header(&response, header::VARY);
    assert!(vary.iter().any(|value| value == "origin"));
    assert!(
        vary.iter()
            .any(|value| value == "access-control-request-method")
    );
    assert!(
        vary.iter()
            .any(|value| value == "access-control-request-headers")
    );
}

#[tokio::test]
async fn allowed_origin_decorates_authenticated_and_auth_error_responses() {
    let app = cors_app(
        &["https://app.example.test", "tauri://localhost"],
        vec!["sk-test".to_string()],
    );

    let unauthorized = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header(header::ORIGIN, "https://app.example.test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        unauthorized
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .unwrap(),
        "https://app.example.test"
    );
    assert_eq!(
        unauthorized
            .headers()
            .get(header::WWW_AUTHENTICATE)
            .unwrap(),
        "Bearer"
    );

    let authorized = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header(header::ORIGIN, "tauri://localhost")
                .header(header::AUTHORIZATION, "Bearer sk-test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(authorized.status(), StatusCode::OK);
    assert_eq!(
        authorized
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .unwrap(),
        "tauri://localhost"
    );
    assert!(
        comma_separated_header(&authorized, header::ACCESS_CONTROL_EXPOSE_HEADERS)
            .iter()
            .any(|header| header == "x-werk-output-id")
    );
}

#[tokio::test]
async fn configured_origin_is_matched_exactly() {
    let app = cors_app(&["https://app.example.test"], vec!["sk-test".to_string()]);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header(header::ORIGIN, "https://app.example.test.evil")
                .header(header::AUTHORIZATION, "Bearer sk-test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none()
    );
}

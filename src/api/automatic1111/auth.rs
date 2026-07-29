use axum::{
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::Response,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};

use super::super::state::ApiState;
use super::response::automatic1111_error;

#[allow(clippy::result_large_err)]
pub(super) fn authorize_automatic1111(
    state: &ApiState,
    headers: &HeaderMap,
) -> Result<(), Response> {
    if !state.api_key_auth_enabled() {
        return Ok(());
    }

    if let Some(value) = headers.get("x-api-key")
        && let Ok(value) = value.to_str()
        && state.api_key_matches(value.trim())
    {
        return Ok(());
    }

    if let Some(value) = headers.get(header::AUTHORIZATION)
        && let Ok(value) = value.to_str()
        && let Some((scheme, credential)) = value.split_once(' ')
    {
        let credential = credential.trim();
        if scheme.eq_ignore_ascii_case("bearer") && state.api_key_matches(credential) {
            return Ok(());
        }
        if scheme.eq_ignore_ascii_case("basic")
            && let Ok(decoded) = STANDARD.decode(credential)
            && let Ok(decoded) = String::from_utf8(decoded)
            && let Some((username, password)) = decoded.split_once(':')
            && username == "werk"
            && state.api_key_matches(password)
        {
            return Ok(());
        }
    }

    let mut response = automatic1111_error(
        StatusCode::UNAUTHORIZED,
        "missing or invalid API credentials; use Bearer, X-API-Key, or Basic werk:<Werk API key>"
            .to_string(),
    );
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Basic realm=\"Werk1112\""),
    );
    Err(response)
}

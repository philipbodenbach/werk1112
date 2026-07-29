use axum::http::{HeaderValue, Uri};
use std::{fmt, str::FromStr};

/// One exact browser origin allowed to access the HTTP API.
#[derive(Clone, PartialEq, Eq)]
pub struct CorsOrigin(HeaderValue);

impl CorsOrigin {
    pub(super) fn header_value(&self) -> HeaderValue {
        self.0.clone()
    }

    pub fn as_str(&self) -> &str {
        self.0
            .to_str()
            .expect("validated CORS origins contain visible ASCII only")
    }
}

impl fmt::Debug for CorsOrigin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CorsOrigin")
            .field(&self.as_str())
            .finish()
    }
}

impl FromStr for CorsOrigin {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.is_empty() {
            return Err("CORS origin must not be empty".to_string());
        }
        if value.eq_ignore_ascii_case("null") {
            return Err("CORS origin 'null' is not allowed".to_string());
        }
        if value.contains('*') {
            return Err("wildcard CORS origins are not allowed".to_string());
        }

        let uri = value
            .parse::<Uri>()
            .map_err(|error| format!("invalid CORS origin '{value}': {error}"))?;
        let scheme = uri
            .scheme_str()
            .ok_or_else(|| "CORS origin must include a scheme".to_string())?;
        if scheme.eq_ignore_ascii_case("file") {
            return Err("file:// has the opaque 'null' origin and is not allowed".to_string());
        }
        let authority = uri
            .authority()
            .ok_or_else(|| "CORS origin must include a host".to_string())?;
        if authority.as_str().contains('@') {
            return Err("CORS origin must not contain user information".to_string());
        }
        let authority_start = value
            .find("://")
            .map(|index| index + 3)
            .ok_or_else(|| "CORS origin must include a scheme separator".to_string())?;
        if &value[authority_start..] != authority.as_str() {
            return Err(
                "CORS origin must contain only scheme, host, and optional port".to_string(),
            );
        }

        let value = value
            .parse::<HeaderValue>()
            .map_err(|error| format!("invalid CORS origin header: {error}"))?;
        Ok(Self(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_exact_network_and_desktop_origins() {
        for origin in [
            "http://127.0.0.1:3000",
            "https://app.example.test",
            "tauri://localhost",
            "http://[::1]:5173",
        ] {
            assert_eq!(origin.parse::<CorsOrigin>().unwrap().as_str(), origin);
        }
    }

    #[test]
    fn rejects_wildcard_null_and_non_origin_values() {
        for origin in [
            "*",
            "null",
            "file:///tmp/app.html",
            "https://*.example.test",
            "https://app.example.test/path",
            "https://user@app.example.test",
            "app.example.test",
        ] {
            assert!(
                origin.parse::<CorsOrigin>().is_err(),
                "unexpectedly accepted {origin}"
            );
        }
    }
}

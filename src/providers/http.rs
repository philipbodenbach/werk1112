use crate::{
    backend::{
        PreloadHandle,
        runtime_cache::{self, RuntimeCache},
    },
    model_support::Provider,
};
use anyhow::{Result, anyhow, ensure};
use reqwest::{
    blocking::{Client, Response},
    header::HeaderValue,
};
use serde_json::Value;
use std::{
    io::Read,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct ProviderError {
    pub code: &'static str,
    pub status: Option<u16>,
    pub message: String,
}
impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for ProviderError {}
pub fn error(code: &'static str, message: impl Into<String>) -> anyhow::Error {
    ProviderError {
        code,
        status: None,
        message: message.into(),
    }
    .into()
}

pub struct ProviderClient {
    pub client: Client,
    pub origin: String,
    provider: Provider,
    pub(crate) request_timeout: Duration,
    #[cfg(test)]
    test_key: bool,
}
// Keys are read per operation, never stored in Debug/serialized configuration or cache keys.
fn credential(provider: Provider) -> Result<HeaderValue> {
    let key = std::env::var(provider.key_env())
        .ok()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| error("missing_credentials", format!("set {}", provider.key_env())))?;
    let text = if provider == Provider::Mistral {
        format!("Bearer {key}")
    } else {
        key
    };
    let mut value = HeaderValue::from_str(&text).map_err(|_| {
        error(
            "invalid_credentials",
            format!("invalid {} header value", provider.key_env()),
        )
    })?;
    value.set_sensitive(true);
    Ok(value)
}
fn clients() -> &'static RuntimeCache<String, ProviderClient> {
    static CACHE: OnceLock<RuntimeCache<String, ProviderClient>> = OnceLock::new();
    CACHE.get_or_init(|| RuntimeCache::bounded(3))
}
pub fn client(provider: Provider) -> Result<Arc<ProviderClient>> {
    let (client, _, _) = clients().get_or_try_init(
        provider.origin().into(),
        |_| true,
        || ProviderClient::new(provider, provider.origin()),
    )?;
    Ok(client)
}
pub fn preload(provider: Provider) -> Result<PreloadHandle> {
    let _ = credential(provider)?;
    runtime_cache::preload(|| {
        client(provider)?;
        Ok(())
    })
}
impl ProviderClient {
    fn new(provider: Provider, origin: &str) -> Result<Self> {
        Ok(Self {
            provider,
            request_timeout: Duration::from_secs(300),
            origin: origin.into(),
            #[cfg(test)]
            test_key: false,
            client: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(300))
                .build()
                .map_err(|_| error("provider_configuration", "cannot initialize HTTPS client"))?,
        })
    }
    #[cfg(test)]
    pub(crate) fn for_test(provider: Provider, origin: &str) -> Result<Self> {
        let mut client = Self::new(provider, origin)?;
        client.test_key = true;
        Ok(client)
    }
    fn credential(&self) -> Result<HeaderValue> {
        #[cfg(test)]
        if self.test_key {
            let mut value = HeaderValue::from_static("test-secret");
            value.set_sensitive(true);
            return Ok(value);
        }
        credential(self.provider)
    }
    pub fn post(&self, path: &str, body: &Value) -> Result<Response> {
        self.send(&format!("{}{path}", self.origin), Some(body), true)
    }
    pub fn get(&self, url: &str, authenticated: bool) -> Result<Response> {
        self.send(url, None, authenticated)
    }
    fn send(&self, url: &str, body: Option<&Value>, authenticated: bool) -> Result<Response> {
        // Validate before attaching a secret, including regional BFL polling hosts.
        if authenticated {
            self.validate_poll_url(url)?;
        }
        let started = Instant::now();
        for attempt in 0..3 {
            let remaining = self
                .request_timeout
                .checked_sub(started.elapsed())
                .ok_or_else(|| error("upstream_timeout", "provider request timed out"))?;
            let mut request = if body.is_some() {
                self.client.post(url)
            } else {
                self.client.get(url)
            }
            .timeout(remaining);
            if authenticated {
                request = request.header(self.provider.header(), self.credential()?);
            }
            if let Some(body) = body {
                request = request
                    .header("content-type", "application/json")
                    .body(serde_json::to_vec(body)?);
            }
            // Never replay an ambiguous POST/network failure: generation may already be billed.
            let response = request.send().map_err(|e| {
                error(
                    if e.is_timeout() {
                        "upstream_timeout"
                    } else {
                        "upstream_transport"
                    },
                    "provider request failed (URL and credentials redacted)",
                )
            })?;
            let status = response.status().as_u16();
            let retry = status == 429 || (body.is_none() && matches!(status, 502 | 504));
            if retry && attempt < 2 {
                let delay = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(1 << attempt);
                if delay <= 10 {
                    std::thread::sleep(Duration::from_secs(delay));
                    continue;
                }
            }
            if status == 503 && body.is_none() && authenticated && self.provider == Provider::Bfl {
                // BFL terminal task failures can use 503 with a status JSON body.
                return Ok(response);
            }
            if !response.status().is_success() {
                let code = match status {
                    401 | 403 => "provider_access_denied",
                    402 => "provider_payment_required",
                    429 => "provider_rate_limited",
                    400 | 422 => "upstream_invalid_request",
                    _ => "upstream_error",
                };
                return Err(ProviderError { code, status: Some(status), message: format!("provider returned HTTP {status}; check account access and documented request parameters") }.into());
            }
            return Ok(response);
        }
        Err(anyhow!("unreachable provider retry state"))
    }
    pub fn validate_poll_url(&self, value: &str) -> Result<()> {
        let url = reqwest::Url::parse(value)
            .map_err(|_| error("upstream_schema", "invalid provider URL"))?;
        let origin = reqwest::Url::parse(&self.origin)?;
        let same_origin = url.origin() == origin.origin();
        let bfl_region = self.provider == Provider::Bfl
            && url.scheme() == "https"
            && url.port_or_known_default() == Some(443)
            && url
                .host_str()
                .is_some_and(|s| s == "api.bfl.ai" || s.ends_with(".bfl.ai"));
        ensure!(
            (same_origin || bfl_region) && url.username().is_empty() && url.password().is_none(),
            "upstream_schema: refused credential forwarding to unexpected origin"
        );
        Ok(())
    }
}
pub fn read_bytes(mut response: Response, max: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    response
        .by_ref()
        .take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| error("upstream_transport", "provider response read failed"))?;
    ensure!(
        bytes.len() as u64 <= max,
        "upstream_schema: provider response exceeds size limit"
    );
    Ok(bytes)
}
pub fn read_json(response: Response) -> Result<Value> {
    serde_json::from_slice(&read_bytes(response, 8 * 1024 * 1024)?)
        .map_err(|_| error("upstream_schema", "provider returned invalid JSON"))
}

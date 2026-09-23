//! Model-independent input documents. All decoders receive ordinary text/image parts.
use crate::{
    file_store::{FileStore, MAX_FILE_BYTES},
    openai::{ContentPart, ImageUrlSpec},
};
use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::{
    io::Read,
    net::{IpAddr, ToSocketAddrs},
    time::Duration,
};

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DocumentOptions {
    #[serde(default)]
    pub mode: DocumentMode,
    #[serde(default)]
    pub ocr: bool,
}
#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentMode {
    #[default]
    Auto,
    Text,
    Vision,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DocumentSource {
    Base64 { media_type: String, data: String },
    Text { media_type: String, data: String },
    Url { url: String },
    File { file_id: String },
    Content { content: DocumentContent },
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum DocumentContent {
    Text(String),
    Blocks(Vec<DocumentTextBlock>),
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DocumentTextBlock {
    Text { text: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct FilePart {
    pub file_id: Option<String>,
    pub file_data: Option<String>,
    pub filename: Option<String>,
    #[serde(skip)]
    pub source: Option<DocumentSource>,
    #[serde(skip)]
    pub context: Option<String>,
}
impl FilePart {
    pub fn resolve(self, store: &FileStore, principal: &str) -> Result<InputDocument> {
        ensure!(
            self.filename.as_ref().is_none_or(|s| s.len() <= 512),
            "document title/filename exceeds 512 bytes"
        );
        ensure!(
            self.context.as_ref().is_none_or(|s| s.len() <= 65536),
            "document context exceeds 64 KiB"
        );
        if let Some(source) = self.source {
            return resolve_source(source, self.filename, self.context, store, principal);
        }
        match (self.file_id, self.file_data) {
            (Some(id), None) => resolve_source(
                DocumentSource::File { file_id: id },
                self.filename,
                self.context,
                store,
                principal,
            ),
            (None, Some(data)) => {
                let (mime, raw) = decode_data(&data)?;
                Ok(InputDocument {
                    filename: self.filename.unwrap_or_else(|| "document".into()),
                    mime,
                    data: raw,
                    context: self.context,
                })
            }
            _ => bail!("file requires exactly one of file_id or file_data"),
        }
    }
}
pub struct InputDocument {
    pub filename: String,
    pub mime: String,
    pub data: Vec<u8>,
    pub context: Option<String>,
}
fn resolve_source(
    source: DocumentSource,
    title: Option<String>,
    context: Option<String>,
    store: &FileStore,
    principal: &str,
) -> Result<InputDocument> {
    let (name, mime, data) = match source {
        DocumentSource::File { file_id } => {
            let (f, b) = store
                .get(principal, &file_id)
                .map_err(|_| anyhow::anyhow!("document file not found"))?;
            (f.filename, f.mime_type, b)
        }
        DocumentSource::Base64 { media_type, data } => {
            ensure!(
                data.len() <= MAX_FILE_BYTES * 4 / 3 + 4,
                "document exceeds 50 MiB"
            );
            (
                "document".into(),
                media_type,
                base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .context("invalid document base64")?,
            )
        }
        DocumentSource::Text { media_type, data } => {
            ensure!(
                media_type == "text/plain",
                "text document source requires text/plain"
            );
            ("document.txt".into(), media_type, data.into_bytes())
        }
        DocumentSource::Content { content } => {
            let text = match content {
                DocumentContent::Text(t) => t,
                DocumentContent::Blocks(b) => b
                    .into_iter()
                    .map(|DocumentTextBlock::Text { text }| text)
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            (
                "document.txt".into(),
                "text/plain".into(),
                text.into_bytes(),
            )
        }
        DocumentSource::Url { url } => fetch_url(&url)?,
    };
    ensure!(
        !data.is_empty() && data.len() <= MAX_FILE_BYTES,
        "document must contain 1..50 MiB"
    );
    Ok(InputDocument {
        filename: title.unwrap_or(name),
        mime,
        data,
        context,
    })
}
fn decode_data(data: &str) -> Result<(String, Vec<u8>)> {
    ensure!(
        data.len() <= MAX_FILE_BYTES * 4 / 3 + 512,
        "document exceeds 50 MiB"
    );
    let (mime, base64) = if let Some(data) = data.strip_prefix("data:") {
        let (head, body) = data.split_once(',').context("invalid file data URL")?;
        let mime = head
            .strip_suffix(";base64")
            .context("file data URL must use base64")?;
        (mime.to_owned(), body)
    } else {
        ("application/octet-stream".into(), data)
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64)
        .context("invalid file base64")?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_FILE_BYTES,
        "document must contain 1..50 MiB"
    );
    Ok((mime, bytes))
}
fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_unspecified()
                || ip.is_multicast()
                || a == 0
                || a >= 240
                || (a == 100 && (64..128).contains(&b))
                || (a == 198 && (b == 18 || b == 19))
                || (a == 192 && b == 0 && c == 0)
                || (a == 192 && b == 88 && c == 99))
        }
        IpAddr::V6(ip) => ip
            .to_ipv4_mapped()
            .map(|v| public_ip(IpAddr::V4(v)))
            .unwrap_or_else(|| {
                let s = ip.segments();
                s[0] & 0xe000 == 0x2000
                    && !(s[0] == 0x2001 && s[1] == 0x0db8)
                    && s[0] != 0x2002
                    && !(s[0] == 0x2001 && s[1] < 0x200)
            }),
    }
}
fn fetch_url(raw: &str) -> Result<(String, String, Vec<u8>)> {
    let mut url = reqwest::Url::parse(raw).context("invalid document URL")?;
    let started = std::time::Instant::now();
    for _ in 0..5 {
        ensure!(
            matches!(url.scheme(), "https" | "http")
                && url.username().is_empty()
                && url.password().is_none(),
            "document URL must use HTTP(S) without credentials"
        );
        let host = url.host_str().context("document URL has no host")?;
        let port = url
            .port_or_known_default()
            .context("invalid document URL port")?;
        let addresses: Vec<_> = (host, port)
            .to_socket_addrs()
            .context("document host lookup failed")?
            .collect();
        ensure!(
            !addresses.is_empty() && addresses.iter().all(|a| public_ip(a.ip())),
            "document URLs must resolve to public addresses; upload local files instead"
        );
        let remaining = Duration::from_secs(30)
            .checked_sub(started.elapsed())
            .context("document download timed out")?;
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .timeout(remaining)
            .resolve_to_addrs(host, &addresses)
            .build()?;
        let response = client
            .get(url.clone())
            .send()
            .context("document download failed")?;
        if response.status().is_redirection() {
            let location = response
                .headers()
                .get("location")
                .context("document redirect has no location")?
                .to_str()?;
            url = url.join(location)?;
            continue;
        }
        ensure!(
            response.status().is_success(),
            "document download returned HTTP {}",
            response.status()
        );
        ensure!(
            response
                .content_length()
                .is_none_or(|n| n <= MAX_FILE_BYTES as u64),
            "document exceeds 50 MiB"
        );
        let mime = response
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("application/octet-stream")
            .split(';')
            .next()
            .unwrap()
            .to_owned();
        let name = url
            .path_segments()
            .and_then(|mut s| s.next_back())
            .filter(|s| !s.is_empty())
            .unwrap_or("document")
            .to_owned();
        let mut bytes = Vec::new();
        response
            .take(MAX_FILE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= MAX_FILE_BYTES, "document exceeds 50 MiB");
        return Ok((name, mime, bytes));
    }
    bail!("too many document redirects")
}

pub fn text_part(text: String) -> ContentPart {
    ContentPart {
        kind: "text".into(),
        text: Some(text),
        image_url: None,
        file: None,
    }
}
#[derive(Deserialize)]
struct Page {
    text: String,
    image: Option<String>,
}
#[derive(Deserialize)]
struct Extracted {
    pages: Vec<Page>,
    representation: String,
}

pub async fn expand(
    input: InputDocument,
    options: &DocumentOptions,
    has_vision: bool,
) -> Result<Vec<ContentPart>> {
    let visual = match options.mode {
        DocumentMode::Auto => has_vision,
        DocumentMode::Text => false,
        DocumentMode::Vision => {
            ensure!(
                has_vision,
                "visual document mode requires a vision-capable model"
            );
            true
        }
    };
    let pdf = input.data.starts_with(b"%PDF-");
    let office = input.data.starts_with(b"PK\x03\x04");
    let extracted = if !pdf && !office {
        ensure!(
            input.data.len() <= 8 * 1024 * 1024,
            "extracted document text exceeds 8 MiB"
        );
        let text=String::from_utf8(input.data).context("unsupported document encoding or format; expected UTF-8, PDF or supported office document")?;
        ensure!(
            !text.contains('\0'),
            "binary content is not a text document"
        );
        Extracted {
            pages: vec![Page { text, image: None }],
            representation: "text".into(),
        }
    } else {
        let config = serde_json::json!({"filename":input.filename,"mime":input.mime,"vision":visual && pdf,"ocr":options.ocr});
        let python = std::env::var_os("WERK_DOCUMENT_PYTHON").unwrap_or_else(|| {
            if cfg!(windows) {
                "python".into()
            } else {
                "python3".into()
            }
        });
        let mut command = tokio::process::Command::new(python);
        command
            .args(["-I", "-c", include_str!("document_worker.py")])
            .arg(config.to_string())
            .kill_on_drop(true)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        #[cfg(unix)]
        {
            command.process_group(0);
        }
        let mut child = command
            .spawn()
            .context("document worker unavailable; configure WERK_DOCUMENT_PYTHON")?;
        #[cfg(unix)]
        let mut group = DocumentProcessGroup(child.id());
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stdin = child.stdin.take().unwrap();
        let write_in = async move {
            stdin.write_all(&input.data).await?;
            drop(stdin);
            anyhow::Ok(())
        };
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let read_out = async move {
            let mut bytes = Vec::new();
            stdout
                .take(64 * 1024 * 1024 + 1)
                .read_to_end(&mut bytes)
                .await?;
            anyhow::Ok(bytes)
        };
        let read_err = async move {
            let mut bytes = Vec::new();
            stderr.take(16385).read_to_end(&mut bytes).await?;
            anyhow::Ok(bytes)
        };
        let result = tokio::time::timeout(Duration::from_secs(90), async {
            tokio::try_join!(
                read_out,
                read_err,
                async { Ok(child.wait().await?) },
                write_in
            )
        })
        .await;
        let (out, err, status, ()) = match result {
            Ok(r) => r?,
            Err(_) => {
                child.kill().await.ok();
                bail!("document processing timed out");
            }
        };
        ensure!(
            out.len() <= 64 * 1024 * 1024 && err.len() <= 16384,
            "document worker output exceeds limit"
        );
        ensure!(
            status.success(),
            "document processing failed: {}",
            String::from_utf8_lossy(&err).trim()
        );
        #[cfg(unix)]
        {
            group.0 = None;
        }
        serde_json::from_slice::<Extracted>(&out).context("invalid document worker result")?
    };
    ensure!(
        !extracted.pages.is_empty() && extracted.pages.len() <= 200,
        "document must contain 1..200 pages"
    );
    let mut parts = Vec::new();
    parts.push(text_part(format!(
        "[Document {}; representation: {}{}]",
        serde_json::to_string(&input.filename)?,
        extracted.representation,
        input
            .context
            .map(|c| format!("; context: {c}"))
            .unwrap_or_default()
    )));
    for (i, page) in extracted.pages.into_iter().enumerate() {
        parts.push(text_part(format!("[Page {}]\n{}", i + 1, page.text)));
        if let Some(image) = page.image {
            parts.push(ContentPart {
                kind: "image_url".into(),
                text: None,
                image_url: Some(ImageUrlSpec::Url(format!("data:image/png;base64,{image}"))),
                file: None,
            });
        }
    }
    parts.push(text_part("[End document]".into()));
    Ok(parts)
}

#[cfg(unix)]
struct DocumentProcessGroup(Option<u32>);
#[cfg(unix)]
impl Drop for DocumentProcessGroup {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn document_urls_reject_nonpublic_addresses_including_mapped_ipv6() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "198.18.0.1",
            "192.0.0.8",
            "192.88.99.1",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
            "fe80::1",
            "2001::1",
            "2001:db8::1",
            "2002:7f00:1::",
        ] {
            assert!(!public_ip(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"] {
            assert!(public_ip(ip.parse().unwrap()), "{ip}");
        }
        for url in [
            "file:///etc/passwd",
            "http://user:password@example.com/doc",
            "http://127.0.0.1/doc",
            "http://169.254.169.254/doc",
        ] {
            assert!(fetch_url(url).is_err(), "{url}");
        }
    }
}

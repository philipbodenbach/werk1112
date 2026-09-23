use super::state::ApiState;
use crate::{
    capabilities::InferenceTask,
    model_store::ModelManifest,
    openai::{ChatCompletionRequest, MessageContent},
};
use anyhow::{Result, bail, ensure};
use axum::http::HeaderMap;

pub(super) async fn expand(
    state: &ApiState,
    headers: &HeaderMap,
    manifest: &ModelManifest,
    request: &mut ChatCompletionRequest,
) -> Result<()> {
    let mut count = 0;
    for message in &request.messages {
        if let Some(MessageContent::Parts(parts)) = &message.content {
            for part in parts {
                match part.kind.as_str() {
                    "text" => ensure!(
                        part.text.is_some() && part.image_url.is_none() && part.file.is_none(),
                        "text content requires only text"
                    ),
                    "image_url" | "input_image" => ensure!(
                        part.image_url.is_some() && part.text.is_none() && part.file.is_none(),
                        "image content requires only image_url"
                    ),
                    "file" => {
                        ensure!(
                            matches!(message.role.as_str(), "user" | "tool"),
                            "files require user or tool role"
                        );
                        ensure!(
                            part.file.is_some() && part.text.is_none() && part.image_url.is_none(),
                            "file content requires only file"
                        );
                        count += 1;
                    }
                    _ => bail!("unsupported message content type: {}", part.kind),
                }
            }
        }
    }
    if count == 0 {
        return Ok(());
    }
    ensure!(count <= 16, "at most 16 documents per request");
    let mut permit = state
        .document_gate
        .clone()
        .try_acquire_owned()
        .map_err(|_| anyhow::anyhow!("too many concurrent document requests"))?;
    let principal = state
        .werk_principal(headers)
        .await
        .map_err(|_| anyhow::anyhow!("document file access denied"))?;
    let options = request
        .werk
        .as_ref()
        .and_then(|w| w.documents.clone())
        .unwrap_or_default();
    let vision = if manifest.supports_task(InferenceTask::ImageUnderstanding)
        && options.mode != crate::documents::DocumentMode::Text
    {
        let backend = state.backend.clone();
        let selected = manifest.clone();
        let (ready, returned) = tokio::task::spawn_blocking(move || {
            (
                backend.task_readiness(&selected, InferenceTask::ImageUnderstanding),
                permit,
            )
        })
        .await?;
        permit = returned;
        ready.is_some_and(|r| r.status == crate::inference::TaskReadinessStatus::Available)
    } else {
        false
    };
    let mut input_bytes = 0;
    let mut output_bytes = 0;
    for message in &mut request.messages {
        let Some(MessageContent::Parts(parts)) = &mut message.content else {
            continue;
        };
        if !parts.iter().any(|p| p.kind == "file") {
            continue;
        }
        let mut expanded = Vec::new();
        for mut part in std::mem::take(parts) {
            if part.kind != "file" {
                expanded.push(part);
                continue;
            }
            let file = part.file.take().unwrap();
            let store = state.files.clone();
            let owner = principal.clone();
            // A cancelled HTTP request cannot release the permit while its
            // bounded blocking download/read is still executing.
            let (resolved, returned) =
                tokio::task::spawn_blocking(move || (file.resolve(&store, &owner), permit)).await?;
            permit = returned;
            let input = resolved?;
            input_bytes += input.data.len();
            ensure!(
                input_bytes <= crate::file_store::MAX_FILE_BYTES,
                "combined documents exceed 50 MiB"
            );
            let result = crate::documents::expand(input, &options, vision).await?;
            output_bytes += result
                .iter()
                .map(|p| {
                    p.text.as_ref().map_or(0, String::len)
                        + p.image_url.as_ref().map_or(0, |u| u.url().len())
                })
                .sum::<usize>();
            ensure!(
                output_bytes <= 64 * 1024 * 1024,
                "expanded documents exceed 64 MiB"
            );
            expanded.extend(result);
        }
        *parts = expanded;
    }
    Ok(())
}

//! Ordered startup preloading. The CLI requires every target to be ready.
//! No HTTP request, user telemetry or
//! persistent conversation is created for the synthetic warmup.
use super::state::ApiState;
use crate::{
    backend::{GenerateRequest, PreloadHandle, StreamGranularity},
    capabilities::InferenceTask,
    logging::Level,
    openai::{ChatMessage, MessageContent, messages_to_prompt_for_model_with_template},
};
use anyhow::Result;
use serde_json::json;
use std::{collections::HashSet, time::Instant};

#[derive(Default, Debug)]
pub(crate) struct PreloadReport {
    pub attempted: usize,
    pub ready: usize,
}

fn event(level: Level, name: &str, message: String, fields: serde_json::Value) {
    if crate::logging::enabled() {
        crate::logging::emit(level, name, &message, fields);
    } else {
        crate::ui_eprintln!("{message}");
    }
}

impl ApiState {
    pub(crate) async fn preload_models(&self, models: &[String]) -> PreloadReport {
        let mut seen = HashSet::new();
        let mut report = PreloadReport::default();
        'targets: for name in models {
            let targets = match &self.deployments {
                Some(registry) => registry.preload_targets(name),
                None => Ok(vec![(name.clone(), self.clone(), name.clone())]),
            };
            let targets = match targets {
                Ok(targets) => targets,
                Err(error) => {
                    if seen.insert(name.clone()) {
                        report.attempted += 1;
                        event(
                            Level::Warn,
                            "serve.preload.failed",
                            format!("Preload '{name}' failed: {error:#}"),
                            json!({"model":name,"error":format!("{error:#}"),"duration_seconds":0.0}),
                        );
                    }
                    break;
                }
            };
            for (id, state, model) in targets {
                if !seen.insert(id.clone()) {
                    continue;
                }
                report.attempted += 1;
                event(
                    Level::Info,
                    "serve.preload.started",
                    format!("Preparing preload target '{id}'..."),
                    json!({"model":model,"target":id}),
                );
                let started = Instant::now();
                let activity = crate::terminal::Activity::start(format!("Preload {id}"));
                activity.phase("Preparing runtime and warming model");
                let result =
                    crate::logging::spawn_blocking(move || state.preload_one(&model)).await;
                let result = result
                    .unwrap_or_else(|error| Err(anyhow::anyhow!("preload task failed: {error}")));
                let seconds = started.elapsed().as_secs_f64();
                match result {
                    Ok(handle) => {
                        activity.finish();
                        self.preloaded
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(handle);
                        report.ready += 1;
                        event(
                            Level::Info,
                            "serve.preload.ready",
                            format!(
                                "Preloaded '{id}' in {seconds:.2}s; retained until server shutdown"
                            ),
                            json!({"target":id,"duration_seconds":seconds}),
                        );
                    }
                    Err(error) => {
                        event(
                            Level::Warn,
                            "serve.preload.failed",
                            format!("Preload '{id}' failed after {seconds:.2}s: {error:#}"),
                            json!({"target":id,"duration_seconds":seconds,"error":format!("{error:#}")}),
                        );
                        break 'targets;
                    }
                }
            }
        }
        event(
            if report.ready == report.attempted {
                Level::Info
            } else {
                Level::Warn
            },
            "serve.preload.finished",
            format!(
                "Preload complete: {} of {} ready; {}",
                report.ready,
                report.attempted,
                if report.ready == report.attempted {
                    "starting request service"
                } else {
                    "startup failed"
                }
            ),
            json!({"ready":report.ready,"attempted":report.attempted,"failed":report.attempted-report.ready}),
        );
        report
    }

    fn preload_one(&self, model: &str) -> Result<PreloadHandle> {
        let manifest = self.store.get(model)?;
        if let Some((provider, _)) = crate::model_support::remote(&manifest) {
            let handle = crate::providers::http::preload(provider)?;
            event(
                Level::Info,
                "serve.preload.remote_prepared",
                format!(
                    "Prepared remote provider for '{model}'; credentials validated locally, no inference or billing, no local weights"
                ),
                json!({"model":model,"mode":"remote_client_only"}),
            );
            return Ok(handle);
        }
        if crate::model_support::is_action(&manifest) {
            return self.actions.preload(&self.store, &manifest);
        }
        anyhow::ensure!(
            manifest.metadata.tasks.is_empty()
                || manifest.supports_task(InferenceTask::TextGeneration)
                || manifest.supports_task(InferenceTask::ImageUnderstanding),
            "preload currently supports text generation only"
        );
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text(
                "Count from one to thirty-two, writing each number in words.".into(),
            )),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let prompt = messages_to_prompt_for_model_with_template(
            &manifest,
            &messages,
            self.prompt_options(&manifest, false)?,
        );
        let handle = self.backend.preload(
            &manifest,
            GenerateRequest {
                prompt: prompt.prompt,
                messages,
                image_urls: Vec::new(),
                max_tokens: 32,
                temperature: Some(0.0),
                top_p: None,
                stop: prompt.stop,
                seed: Some(0),
                stream_granularity: StreamGranularity::Token,
                verbose: false,
                debug: false,
                tool_config: None,
            },
        )?;
        Ok(handle.with_owner(self.backend.clone()))
    }
}

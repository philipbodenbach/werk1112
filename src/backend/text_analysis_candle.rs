//! Native XLM-RoBERTa sequence classification (including one-logit rerankers).
use super::{
    runtime_cache::RuntimeCache,
    text_analysis::{InvalidInput, Options},
};
use crate::{
    capabilities::InferenceTask,
    model_store::{ModelManifest, ModelRuntimeIdentity, ModelStore},
};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaForSequenceClassification};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};
use tokenizers::{Tokenizer, TruncationParams};

pub(super) struct NativeModel {
    model: XLMRobertaForSequenceClassification,
    tokenizer: Tokenizer,
    config: Config,
    metadata: Value,
    device: Device,
    _files: Option<super::model_file_cache::CacheReleaseGuard>,
}
#[derive(Clone)]
pub(super) struct NativeBackend {
    models: Arc<RuntimeCache<(ModelRuntimeIdentity, String), Mutex<NativeModel>>>,
}
impl Default for NativeBackend {
    fn default() -> Self {
        Self {
            models: Arc::new(RuntimeCache::bounded(2)),
        }
    }
}
impl NativeBackend {
    pub fn execute(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
        task: InferenceTask,
        body: &Value,
        options: &Options,
        device_name: &str,
    ) -> Result<Value> {
        ensure!(
            manifest.architecture.as_deref() == Some("xlm-roberta"),
            "Candle has no loader for this architecture; use backend=auto or transformers"
        );
        let started = Instant::now();
        let key = (
            ModelRuntimeIdentity::from_manifest(manifest)?,
            device_name.into(),
        );
        let (cached, hit, load_seconds) = self.models.get_or_try_init(key.clone(), |_| true, || {
            let root = store.model_files_dir(manifest);
            let metadata: Value = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
            let config: Config = serde_json::from_value(metadata.clone())?;
            ensure!(config.position_embedding_type == "absolute", "Candle XLM-RoBERTa requires absolute position embeddings; use transformers for this variant");
            let labels = metadata["num_labels"].as_u64().map(|n| n as usize)
                .or_else(|| metadata["id2label"].as_object().map(|labels| labels.len())).unwrap_or(2);
            ensure!(task != InferenceTask::TextReranking || labels == 1, "reranking requires exactly one relevance logit");
            let device = if device_name == "cuda" {
                #[cfg(feature = "candle-cuda")]
                { Device::new_cuda(0).context("Candle CUDA initialization failed; check nvidia-smi or select device=cpu")? }
                #[cfg(not(feature = "candle-cuda"))]
                { anyhow::bail!("This Werk binary has no Candle CUDA support; build with --features cuda, or use transformers CUDA / Candle CPU") }
            } else { Device::Cpu };
            let files = manifest.files.iter().filter(|file| {
                let path = std::path::Path::new(&file.path);
                path.extension().is_some_and(|ext| ext == "safetensors") && path.parent() == Some(std::path::Path::new("files"))
            }).map(|file| store.absolute_model_file(manifest, &file.path)).collect::<Vec<_>>();
            ensure!(!files.is_empty(), "No root safetensors weights found; download the complete sequence-classification checkpoint");
            let guard = super::model_file_cache::CacheReleaseGuard::prepare_manifest(store, manifest);
            // Matching upstream F32 accumulation is the conservative native fallback.
            let weights = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device)? };
            let model = XLMRobertaForSequenceClassification::new(labels, &config, weights)?;
            let mut tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            tokenizer.with_padding(None);
            tokenizer.with_truncation(None).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            Ok(Mutex::new(NativeModel { model, tokenizer, config, metadata, device, _files: guard }))
        })?;
        let inference_started = Instant::now();
        let outcome = (|| -> Result<Value> {
            let cached = cached
                .lock()
                .map_err(|_| anyhow::anyhow!("Candle model lock poisoned; restart werk serve"))?;
            let max_length = options.max_length.min(
                cached
                    .config
                    .max_position_embeddings
                    .saturating_sub(cached.config.pad_token_id as usize + 1),
            );
            let input = if task == InferenceTask::TextReranking {
                &body["documents"]
            } else {
                &body["input"]
            };
            let texts = if let Some(text) = input.as_str() {
                vec![text]
            } else {
                input
                    .as_array()
                    .context("input must be an array")?
                    .iter()
                    .map(|text| text.as_str().context("input text must be a string"))
                    .collect::<Result<Vec<_>>>()?
            };
            let mut tokenizer = cached.tokenizer.clone();
            if options.truncate {
                tokenizer
                    .with_truncation(Some(TruncationParams {
                        max_length,
                        ..Default::default()
                    }))
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            }
            let encoded = texts.iter().map(|text| {
                let encoding = if task == InferenceTask::TextReranking {
                    tokenizer.encode((body["query"].as_str().context("query required")?, *text), true)
                } else { tokenizer.encode(*text, true) }.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                if encoding.len() > max_length { return Err(InvalidInput(format!("Input exceeds {max_length} tokens. Split the text or explicitly set werk.truncate=true")).into()); }
                Ok(encoding)
            }).collect::<Result<Vec<_>>>()?;
            let tokens: usize = encoded.iter().map(|e| e.len()).sum();
            let mut logits = Vec::new();
            for batch in encoded.chunks(options.batch_size) {
                let width = batch.iter().map(|e| e.len()).max().unwrap();
                let mut ids = Vec::with_capacity(width * batch.len());
                let mut mask = Vec::with_capacity(width * batch.len());
                let mut types = Vec::with_capacity(width * batch.len());
                for encoding in batch {
                    for i in 0..width {
                        ids.push(
                            encoding
                                .get_ids()
                                .get(i)
                                .copied()
                                .unwrap_or(cached.config.pad_token_id),
                        );
                        mask.push(u32::from(i < encoding.len()));
                        types.push(encoding.get_type_ids().get(i).copied().unwrap_or(0));
                    }
                }
                let shape = (batch.len(), width);
                let output = cached.model.forward(
                    &Tensor::from_vec(ids, shape, &cached.device)?,
                    &Tensor::from_vec(mask, shape, &cached.device)?,
                    &Tensor::from_vec(types, shape, &cached.device)?,
                )?;
                logits.extend(output.to_dtype(DType::F32)?.to_vec2::<f32>()?);
            }
            ensure!(
                logits.iter().flatten().all(|v| v.is_finite()),
                "Candle returned non-finite logits; check checkpoint integrity or use transformers"
            );
            let mut response = if task == InferenceTask::TextReranking {
                let mut results = logits.iter().enumerate().map(|(index, row)| {
                    let mut result = json!({"index":index,"relevance_score":1.0 / (1.0 + (-row[0] as f64).exp())});
                    if body["return_documents"].as_bool() == Some(true) { result["document"] = json!({"text":texts[index]}); }
                    result
                }).collect::<Vec<_>>();
                results.sort_by(|a, b| {
                    b["relevance_score"]
                        .as_f64()
                        .unwrap()
                        .total_cmp(&a["relevance_score"].as_f64().unwrap())
                });
                results.truncate(body["top_n"].as_u64().unwrap_or(results.len() as u64) as usize);
                json!({"results":results})
            } else {
                let multilabel = cached.metadata["problem_type"] == "multi_label_classification";
                let data = logits.iter().enumerate().map(|(index,row)| {
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denom: f64 = row.iter().map(|v| ((*v - max) as f64).exp()).sum();
                    json!({"index": index,"scores": row.iter().enumerate().map(|(i,value)| json!({
                        "label":cached.metadata["id2label"][i.to_string()].as_str().map(str::to_owned).unwrap_or_else(||i.to_string()),
                        "score": if multilabel { 1.0 / (1.0 + (-*value as f64).exp()) } else { ((*value - max) as f64).exp()/denom }
                    })).collect::<Vec<_>>()})
                }).collect::<Vec<_>>();
                json!({"data":data})
            };
            response["ok"] = json!(true);
            response["usage"] = json!({"total_tokens":tokens});
            response["werk"] = json!({"runtime":"candle", "device":device_name, "dtype":"float32", "model_cache_hit":hit, "load_seconds":load_seconds,"inference_seconds":inference_started.elapsed().as_secs_f64(),"total_seconds":started.elapsed().as_secs_f64(), "max_length":max_length});
            Ok(response)
        })();
        drop(cached);
        if outcome.is_err() {
            self.models.remove_if(|_| true);
        }
        outcome
    }
}

use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::{
    DType, Int, Shape, Tensor, TensorData, activation, backend::Backend, module,
    ops::AttentionModuleOptions, s,
};
use safetensors::{Dtype as SafeDType, SafeTensors};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::PathBuf, time::Instant};
use tokenizers::Tokenizer;

use crate::backend::{GenerateRequest, GenerateResponse, GenerationTimings};
use crate::{
    backend::BurnMode,
    model_store::{ModelManifest, ModelStore},
};

type TokenCallback = Box<dyn FnMut(String) + Send>;

pub enum BurnPhi3Runtime {
    #[cfg(feature = "burn-cuda")]
    Cuda(Box<Phi3Generator<burn::backend::Cuda>>),
    #[cfg(feature = "burn-cpu")]
    Cpu(Box<Phi3Generator<burn::backend::Flex>>),
}

impl BurnPhi3Runtime {
    pub fn load(
        store: &ModelStore,
        manifest: &ModelManifest,
        tokenizer: Tokenizer,
        mode: BurnMode,
    ) -> Result<Self> {
        match mode {
            BurnMode::Cuda => load_cuda(store, manifest, tokenizer),
            BurnMode::Cpu => load_cpu(store, manifest, tokenizer),
        }
    }

    pub fn generate(
        &mut self,
        request: GenerateRequest,
        on_token: Option<TokenCallback>,
    ) -> Result<GenerateResponse> {
        match self {
            #[cfg(feature = "burn-cuda")]
            Self::Cuda(model) => model.generate(request, on_token),
            #[cfg(feature = "burn-cpu")]
            Self::Cpu(model) => model.generate(request, on_token),
        }
    }
}

#[cfg(feature = "burn-cuda")]
fn load_cuda(
    store: &ModelStore,
    manifest: &ModelManifest,
    tokenizer: Tokenizer,
) -> Result<BurnPhi3Runtime> {
    let device = Default::default();
    let generator = Phi3Generator::<burn::backend::Cuda>::load(store, manifest, tokenizer, device)?;
    Ok(BurnPhi3Runtime::Cuda(Box::new(generator)))
}

#[cfg(not(feature = "burn-cuda"))]
fn load_cuda(
    _store: &ModelStore,
    _manifest: &ModelManifest,
    _tokenizer: Tokenizer,
) -> Result<BurnPhi3Runtime> {
    bail!(
        "This Werk binary was built without Burn CUDA support. Rebuild with: cargo install --path . --locked --force --features burn-cuda. Burn CUDA requires native CUDA and NCCL system libraries."
    )
}

#[cfg(feature = "burn-cpu")]
fn load_cpu(
    store: &ModelStore,
    manifest: &ModelManifest,
    tokenizer: Tokenizer,
) -> Result<BurnPhi3Runtime> {
    let device = Default::default();
    let generator = Phi3Generator::<burn::backend::Flex>::load(store, manifest, tokenizer, device)?;
    Ok(BurnPhi3Runtime::Cpu(Box::new(generator)))
}

#[cfg(not(feature = "burn-cpu"))]
fn load_cpu(
    _store: &ModelStore,
    _manifest: &ModelManifest,
    _tokenizer: Tokenizer,
) -> Result<BurnPhi3Runtime> {
    bail!(
        "This Werk binary was built without Burn CPU support. Rebuild with: cargo install --path . --locked --force --features burn-cpu"
    )
}

pub fn probe_phi3_files(store: &ModelStore, manifest: &ModelManifest) -> Result<String> {
    let config = read_phi3_config(store, manifest)?;
    ensure_phi3_config_supported(&config)?;
    let keys = safetensor_keys(store, manifest)?;
    required_weight_keys(&config)
        .into_iter()
        .find(|key| !keys.contains(key))
        .map_or_else(
            || Ok("Phi-3 SafeTensors keys validated".to_string()),
            |missing| bail!("Phi-3 Burn weights are missing tensor '{missing}'"),
        )
}

#[derive(Debug, Clone, Deserialize)]
struct Phi3Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    max_position_embeddings: usize,
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    hidden_act: Option<String>,
}

impl Phi3Config {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

pub(crate) struct Phi3Generator<B: Backend> {
    model: Phi3Model<B>,
    tokenizer: Tokenizer,
    eos_token: Option<u32>,
    device: B::Device,
}

impl<B: Backend> Phi3Generator<B> {
    fn load(
        store: &ModelStore,
        manifest: &ModelManifest,
        tokenizer: Tokenizer,
        device: B::Device,
    ) -> Result<Self> {
        let config = read_phi3_config(store, manifest)?;
        ensure_phi3_config_supported(&config)?;
        let tensors = load_tensors::<B>(store, manifest, &device)?;
        let model = Phi3Model::new(config.clone(), tensors, &device)?;
        let eos_token = config.eos_token_id.or_else(|| eos_token_id(&tokenizer));
        let mut generator = Self {
            model,
            tokenizer,
            eos_token,
            device,
        };
        generator.prepare_smoke()?;
        Ok(generator)
    }

    fn prepare_smoke(&mut self) -> Result<()> {
        let ids = self
            .tokenizer
            .encode("Say ok.", true)
            .map_err(|err| anyhow!("Burn Phi-3 tokenizer smoke failed: {err}"))?
            .get_ids()
            .to_vec();
        if ids.is_empty() {
            bail!("Burn Phi-3 tokenizer smoke produced zero tokens");
        }
        self.model.clear_kv_cache();
        let input = token_tensor::<B>(&ids, &self.device);
        let _ = self.model.forward(input, 0)?;
        self.model.clear_kv_cache();
        Ok(())
    }

    fn generate(
        &mut self,
        request: GenerateRequest,
        mut on_token: Option<TokenCallback>,
    ) -> Result<GenerateResponse> {
        if !request.image_urls.is_empty() {
            bail!("Burn Phi-3 backend is text-only for now");
        }

        let started = Instant::now();
        let mut tokens = self
            .tokenizer
            .encode(request.prompt.clone(), true)
            .map_err(|err| anyhow!("Burn Phi-3 tokenization failed: {err}"))?
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            bail!("Burn Phi-3 tokenizer produced no prompt tokens");
        }
        let prompt_tokens = tokens.len();

        let prompt_started = Instant::now();
        self.model.clear_kv_cache();
        let mut logits = self
            .model
            .forward(token_tensor::<B>(&tokens, &self.device), 0)?;
        let mut index_pos = tokens.len();
        let prompt_seconds = prompt_started.elapsed().as_secs_f64();

        let decode_started = Instant::now();
        let mut generated = Vec::new();
        let mut generated_text = String::new();
        let mut finish_reason = "length".to_string();
        for _ in 0..request.max_tokens {
            let next_token = greedy_token(logits)?;
            if Some(next_token) == self.eos_token {
                finish_reason = "stop".to_string();
                break;
            }

            tokens.push(next_token);
            generated.push(next_token);

            let decoded = self
                .tokenizer
                .decode(&generated, true)
                .unwrap_or_else(|_| String::new());
            let previous_text = generated_text.clone();
            generated_text = decoded;
            let reached_stop = stop_reached(&mut generated_text, &request.stop);
            let safe_piece = if generated_text.starts_with(&previous_text)
                && generated_text.len() > previous_text.len()
            {
                generated_text[previous_text.len()..].to_string()
            } else if previous_text != generated_text && !reached_stop {
                generated_text.clone()
            } else {
                String::new()
            };
            if let Some(callback) = on_token.as_mut()
                && !safe_piece.is_empty()
            {
                callback(safe_piece);
            }
            if reached_stop {
                finish_reason = "stop".to_string();
                break;
            }

            logits = self
                .model
                .forward(token_tensor::<B>(&[next_token], &self.device), index_pos)?;
            index_pos += 1;
        }
        let decode_seconds = decode_started.elapsed().as_secs_f64();

        Ok(GenerateResponse {
            text: generated_text,
            prompt_tokens,
            completion_tokens: generated.len(),
            finish_reason,
            timings: GenerationTimings {
                load_seconds: 0.0,
                warmup_seconds: 0.0,
                first_token_seconds: 0.0,
                prompt_seconds,
                decode_seconds,
                total_seconds: started.elapsed().as_secs_f64(),
            },
        })
    }
}

struct Phi3Model<B: Backend> {
    config: Phi3Config,
    embed_tokens: Tensor<B, 2>,
    layers: Vec<DecoderLayer<B>>,
    norm: Tensor<B, 1>,
    lm_head: Tensor<B, 2>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
}

impl<B: Backend> Phi3Model<B> {
    fn new(config: Phi3Config, mut tensors: TensorMaps<B>, device: &B::Device) -> Result<Self> {
        let embed_tokens = tensors.take2("model.embed_tokens.weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(index, &config, &mut tensors)?);
        }
        let norm = tensors.take1("model.norm.weight")?;
        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            tensors.take2("lm_head.weight")?
        };
        let (cos, sin) = rotary_cache::<B>(&config, device);
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos,
            sin,
        })
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.kv_cache = None;
        }
    }

    fn forward(
        &mut self,
        input_ids: Tensor<B, 2, Int>,
        seqlen_offset: usize,
    ) -> Result<Tensor<B, 3>> {
        let dims = input_ids.dims();
        let b_size = dims[0];
        let seq_len = dims[1];
        let mut xs = module::embedding(self.embed_tokens.clone(), input_ids);
        for layer in &mut self.layers {
            xs = layer.forward(xs, &self.cos, &self.sin, seqlen_offset)?;
        }
        let xs = xs
            .narrow(1, seq_len - 1, 1)
            .reshape([b_size, 1, self.config.hidden_size]);
        let xs = rms_norm(xs, self.norm.clone(), self.config.rms_norm_eps);
        Ok(module::linear(xs, self.lm_head.clone(), None))
    }
}

struct DecoderLayer<B: Backend> {
    self_attn: Attention<B>,
    mlp: Mlp<B>,
    input_layernorm: Tensor<B, 1>,
    post_attention_layernorm: Tensor<B, 1>,
}

impl<B: Backend> DecoderLayer<B> {
    fn new(index: usize, config: &Phi3Config, tensors: &mut TensorMaps<B>) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(index, config, tensors)?,
            mlp: Mlp::new(index, tensors)?,
            input_layernorm: tensors
                .take1(&format!("model.layers.{index}.input_layernorm.weight"))?,
            post_attention_layernorm: tensors.take1(&format!(
                "model.layers.{index}.post_attention_layernorm.weight"
            ))?,
        })
    }

    fn forward(
        &mut self,
        xs: Tensor<B, 3>,
        cos: &Tensor<B, 2>,
        sin: &Tensor<B, 2>,
        seqlen_offset: usize,
    ) -> Result<Tensor<B, 3>> {
        let residual = xs.clone();
        let normed = rms_norm(
            xs,
            self.input_layernorm.clone(),
            self.self_attn.rms_norm_eps,
        );
        let attended = self.self_attn.forward(normed, cos, sin, seqlen_offset)?;
        let xs = attended + residual;
        let residual = xs.clone();
        let normed = rms_norm(
            xs,
            self.post_attention_layernorm.clone(),
            self.self_attn.rms_norm_eps,
        );
        Ok(self.mlp.forward(normed) + residual)
    }
}

struct Attention<B: Backend> {
    qkv_proj: Tensor<B, 2>,
    o_proj: Tensor<B, 2>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    kv_cache: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
}

impl<B: Backend> Attention<B> {
    fn new(index: usize, config: &Phi3Config, tensors: &mut TensorMaps<B>) -> Result<Self> {
        Ok(Self {
            qkv_proj: tensors.take2(&format!("model.layers.{index}.self_attn.qkv_proj.weight"))?,
            o_proj: tensors.take2(&format!("model.layers.{index}.self_attn.o_proj.weight"))?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            num_kv_groups: config.num_attention_heads / config.num_key_value_heads,
            head_dim: config.head_dim(),
            rms_norm_eps: config.rms_norm_eps,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: Tensor<B, 3>,
        cos: &Tensor<B, 2>,
        sin: &Tensor<B, 2>,
        seqlen_offset: usize,
    ) -> Result<Tensor<B, 3>> {
        let [b_sz, q_len, _] = xs.dims();
        let qkv = module::linear(xs, self.qkv_proj.clone(), None);
        let query_pos = self.num_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        let query_states = qkv
            .clone()
            .narrow(2, 0, query_pos)
            .reshape([b_sz, q_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let key_states = qkv
            .clone()
            .narrow(2, query_pos, kv_width)
            .reshape([b_sz, q_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let value_states = qkv
            .narrow(2, query_pos + kv_width, kv_width)
            .reshape([b_sz, q_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let query_states = apply_rope(query_states, cos, sin, seqlen_offset);
        let key_states = apply_rope(key_states, cos, sin, seqlen_offset);

        let (key_states, value_states) = match self.kv_cache.take() {
            Some((prev_k, prev_v)) => (
                Tensor::cat(vec![prev_k, key_states], 2),
                Tensor::cat(vec![prev_v, value_states], 2),
            ),
            None => (key_states, value_states),
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = repeat_kv(key_states, self.num_kv_groups);
        let value_states = repeat_kv(value_states, self.num_kv_groups);
        let output = module::attention(
            query_states,
            key_states,
            value_states,
            None,
            None,
            AttentionModuleOptions {
                scale: Some(1.0 / (self.head_dim as f64).sqrt()),
                softcap: None,
                is_causal: q_len > 1 && seqlen_offset == 0,
            },
        );
        Ok(module::linear(
            output
                .swap_dims(1, 2)
                .reshape([b_sz, q_len, self.num_heads * self.head_dim]),
            self.o_proj.clone(),
            None,
        ))
    }
}

struct Mlp<B: Backend> {
    gate_up_proj: Tensor<B, 2>,
    down_proj: Tensor<B, 2>,
    intermediate_size: usize,
}

impl<B: Backend> Mlp<B> {
    fn new(index: usize, tensors: &mut TensorMaps<B>) -> Result<Self> {
        let gate_up_proj =
            tensors.take2(&format!("model.layers.{index}.mlp.gate_up_proj.weight"))?;
        let intermediate_size = gate_up_proj.dims()[0] / 2;
        Ok(Self {
            gate_up_proj,
            down_proj: tensors.take2(&format!("model.layers.{index}.mlp.down_proj.weight"))?,
            intermediate_size,
        })
    }

    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let up_states = module::linear(xs, self.gate_up_proj.clone(), None);
        let gate = up_states.clone().narrow(2, 0, self.intermediate_size);
        let up = up_states.narrow(2, self.intermediate_size, self.intermediate_size);
        module::linear(up * activation::silu(gate), self.down_proj.clone(), None)
    }
}

struct TensorMaps<B: Backend> {
    one_d: HashMap<String, Tensor<B, 1>>,
    two_d: HashMap<String, Tensor<B, 2>>,
}

impl<B: Backend> TensorMaps<B> {
    fn take1(&mut self, name: &str) -> Result<Tensor<B, 1>> {
        self.one_d
            .remove(name)
            .with_context(|| format!("missing Burn Phi-3 1D tensor '{name}'"))
    }

    fn take2(&mut self, name: &str) -> Result<Tensor<B, 2>> {
        self.two_d
            .remove(name)
            .with_context(|| format!("missing Burn Phi-3 2D tensor '{name}'"))
    }
}

fn rms_norm<B: Backend>(xs: Tensor<B, 3>, weight: Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let hidden = xs.dims()[2];
    let variance = xs.clone().powf_scalar(2.0).mean_dim(2);
    let xs = xs / (variance + eps).sqrt();
    xs * weight.reshape([1, 1, hidden])
}

fn apply_rope<B: Backend>(
    xs: Tensor<B, 4>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
    offset: usize,
) -> Tensor<B, 4> {
    let [b, h, seq, dim] = xs.dims();
    let half = dim / 2;
    let cos = cos
        .clone()
        .slice([offset..offset + seq, 0..half])
        .reshape([1, 1, seq, half]);
    let sin = sin
        .clone()
        .slice([offset..offset + seq, 0..half])
        .reshape([1, 1, seq, half]);
    let even = xs.clone().slice(s![.., .., .., 0..dim;2]);
    let odd = xs.slice(s![.., .., .., 1..dim;2]);
    let out_even = even.clone() * cos.clone() - odd.clone() * sin.clone();
    let out_odd = even * sin + odd * cos;
    Tensor::cat(
        vec![
            out_even.unsqueeze_dim::<5>(4),
            out_odd.unsqueeze_dim::<5>(4),
        ],
        4,
    )
    .reshape([b, h, seq, dim])
}

fn repeat_kv<B: Backend>(xs: Tensor<B, 4>, groups: usize) -> Tensor<B, 4> {
    if groups == 1 {
        return xs;
    }
    let [b, h, seq, dim] = xs.dims();
    xs.reshape([b, h, 1, seq, dim])
        .expand([b, h, groups, seq, dim])
        .reshape([b, h * groups, seq, dim])
}

fn rotary_cache<B: Backend>(
    config: &Phi3Config,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let partial = config
        .partial_rotary_factor
        .map(|factor| (factor * config.head_dim() as f64) as usize)
        .unwrap_or(config.head_dim());
    let half = partial / 2;
    let max_seq_len = config.max_position_embeddings;
    let mut cos = Vec::with_capacity(max_seq_len * half);
    let mut sin = Vec::with_capacity(max_seq_len * half);
    for position in 0..max_seq_len {
        for index in 0..half {
            let dim_index = 2 * index;
            let inv_freq =
                1.0_f32 / (config.rope_theta.powf(dim_index as f64 / partial as f64) as f32);
            let freq = position as f32 * inv_freq;
            cos.push(freq.cos());
            sin.push(freq.sin());
        }
    }
    (
        Tensor::from_data(
            TensorData::new(cos, Shape::new([max_seq_len, half])),
            device,
        ),
        Tensor::from_data(
            TensorData::new(sin, Shape::new([max_seq_len, half])),
            device,
        ),
    )
}

fn load_tensors<B: Backend>(
    store: &ModelStore,
    manifest: &ModelManifest,
    device: &B::Device,
) -> Result<TensorMaps<B>> {
    let mut one_d = HashMap::new();
    let mut two_d = HashMap::new();
    for path in safetensor_paths(store, manifest)? {
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        for (name, view) in tensors.tensors() {
            let dtype = burn_dtype(view.dtype())
                .with_context(|| format!("unsupported dtype for tensor '{name}'"))?;
            match view.shape().len() {
                1 => {
                    let shape = view.shape();
                    let data = TensorData::from_bytes_vec(
                        view.data().to_vec(),
                        Shape::new([shape[0]]),
                        dtype,
                    );
                    one_d.insert(name, Tensor::<B, 1>::from_data(data, (device, dtype)));
                }
                2 => {
                    let shape = view.shape();
                    let data = TensorData::from_bytes_vec(
                        view.data().to_vec(),
                        Shape::new([shape[0], shape[1]]),
                        dtype,
                    );
                    two_d.insert(name, Tensor::<B, 2>::from_data(data, (device, dtype)));
                }
                _ => {}
            }
        }
    }
    Ok(TensorMaps { one_d, two_d })
}

fn token_tensor<B: Backend>(tokens: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    let ids = tokens.iter().map(|id| *id as i32).collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(ids, Shape::new([1, tokens.len()])), device)
}

fn greedy_token<B: Backend>(logits: Tensor<B, 3>) -> Result<u32> {
    let vocab_size = logits.dims()[2];
    let token = logits
        .reshape([vocab_size])
        .argmax(0)
        .try_into_data()
        .context("failed to read Burn Phi-3 sampled token")?
        .to_vec::<i32>()
        .context("failed to convert Burn Phi-3 sampled token")?
        .first()
        .copied()
        .context("Burn Phi-3 sampling returned no token")?;
    u32::try_from(token).context("Burn Phi-3 sampled negative token id")
}

fn read_phi3_config(store: &ModelStore, manifest: &ModelManifest) -> Result<Phi3Config> {
    let config_path = manifest
        .config_path
        .as_deref()
        .context("safetensors model requires config.json")?;
    let path = store.absolute_model_file(manifest, config_path);
    let data =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("failed to parse {}", path.display()))
}

fn ensure_phi3_config_supported(config: &Phi3Config) -> Result<()> {
    if config.hidden_size % config.num_attention_heads != 0 {
        bail!("Phi-3 hidden_size must be divisible by num_attention_heads");
    }
    if config.vocab_size == 0
        || config.hidden_size == 0
        || config.intermediate_size == 0
        || config.num_hidden_layers == 0
    {
        bail!("Phi-3 config has invalid zero-sized dimensions");
    }
    let _ = config.bos_token_id;
    if config.num_attention_heads % config.num_key_value_heads != 0 {
        bail!("Phi-3 num_attention_heads must be divisible by num_key_value_heads");
    }
    if !matches!(
        config.hidden_act.as_deref().unwrap_or("silu"),
        "silu" | "swish"
    ) {
        bail!(
            "Burn Phi-3 supports silu/swish activations only, got {:?}",
            config.hidden_act
        );
    }
    if config.rope_scaling.is_some() || config.original_max_position_embeddings.is_some() {
        bail!("Burn Phi-3 currently supports the 4K RoPE configuration only");
    }
    Ok(())
}

fn safetensor_paths(store: &ModelStore, manifest: &ModelManifest) -> Result<Vec<PathBuf>> {
    let mut paths = manifest
        .files
        .iter()
        .filter(|file| file.path.ends_with(".safetensors"))
        .map(|file| store.absolute_model_file(manifest, &file.path))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        bail!("SafeTensors model has no .safetensors weight files");
    }
    Ok(paths)
}

fn safetensor_keys(store: &ModelStore, manifest: &ModelManifest) -> Result<Vec<String>> {
    let mut keys = Vec::new();
    for path in safetensor_paths(store, manifest)? {
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        keys.extend(tensors.names().into_iter().cloned());
    }
    Ok(keys)
}

fn required_weight_keys(config: &Phi3Config) -> Vec<String> {
    let mut keys = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
    ];
    if !config.tie_word_embeddings {
        keys.push("lm_head.weight".to_string());
    }
    for index in 0..config.num_hidden_layers {
        keys.extend([
            format!("model.layers.{index}.self_attn.qkv_proj.weight"),
            format!("model.layers.{index}.self_attn.o_proj.weight"),
            format!("model.layers.{index}.mlp.gate_up_proj.weight"),
            format!("model.layers.{index}.mlp.down_proj.weight"),
            format!("model.layers.{index}.input_layernorm.weight"),
            format!("model.layers.{index}.post_attention_layernorm.weight"),
        ]);
    }
    keys
}

fn burn_dtype(dtype: SafeDType) -> Result<DType> {
    match dtype {
        SafeDType::F16 => Ok(DType::F16),
        SafeDType::BF16 => Ok(DType::BF16),
        SafeDType::F32 => Ok(DType::F32),
        other => bail!("Burn Phi-3 only supports F16, BF16, and F32 weights, got {other:?}"),
    }
}

fn eos_token_id(tokenizer: &Tokenizer) -> Option<u32> {
    ["<eos>", "</s>", "<|end_of_text|>", "<|im_end|>"]
        .into_iter()
        .find_map(|token| tokenizer.token_to_id(token))
}

fn stop_reached(text: &mut String, stops: &[String]) -> bool {
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(index) = text.find(stop) {
            text.truncate(index);
            return true;
        }
    }
    false
}

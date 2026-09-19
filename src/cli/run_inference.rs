//! One-shot frontend for the existing terminal-chat and canonical media paths.
use super::*;
use crate::{
    backend::ToolCallingConfig,
    openai::{
        ChatCompletionFunctionCall, ChatCompletionRequest, ChatCompletionToolCall,
        ChatCompletionToolCallDelta,
    },
};

#[derive(Debug, Clone, Args, Default)]
pub struct RunOptions {
    #[arg(long, value_name = "PATH", conflicts_with_all = ["prompt", "images"], help = "Read an OpenAI chat request or canonical media InferenceRequest from JSON; - reads stdin")]
    pub request: Option<PathBuf>,
    #[arg(long, value_parser = parse_inference_task, help = "Inference task; inferred when the model has an unambiguous default")]
    pub task: Option<InferenceTask>,
    #[arg(
        long = "input",
        value_name = "MODALITY[:ROLE]=PATH_OR_URL",
        help = "Media input, e.g. audio=clip.wav or image:mask_image=mask.png; repeatable"
    )]
    pub inputs: Vec<String>,
    #[arg(
        long = "set",
        value_name = "PATH=VALUE",
        help = "Canonical media parameter, e.g. image.steps=20 or routing.precision=float16; repeatable"
    )]
    pub parameters: Vec<String>,
    #[arg(
        long,
        value_name = "PATH",
        help = "Publish media output to a file or directory"
    )]
    pub output: Option<PathBuf>,
    #[arg(
        long,
        help = "Print structured results; with text streaming, emit NDJSON events"
    )]
    pub json: bool,
    #[arg(
        long,
        help = "Stream generated text; with --json, also stream tool-call deltas"
    )]
    pub stream: bool,
    #[arg(
        long,
        value_enum,
        help = "Text stream granularity: token or chunk; implies --stream"
    )]
    pub stream_granularity: Option<StreamGranularityArg>,
    #[arg(
        long,
        alias = "single-turn",
        help = "Use only this invocation's messages; conflicts with conversation persistence"
    )]
    pub no_history: bool,
}

pub(super) struct RunTurn {
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    pub json: bool,
    pub stop: Vec<String>,
    pub tool_config: Option<ToolCallingConfig>,
    pub requires_tools: bool,
}

fn read_request(path: &Path) -> Result<Value> {
    const LIMIT: u64 = 128 * 1024 * 1024;
    let mut source: Box<dyn Read> = if path == Path::new("-") {
        Box::new(io::stdin())
    } else {
        Box::new(
            fs::File::open(path)
                .with_context(|| format!("cannot read request {}", path.display()))?,
        )
    };
    let mut bytes = Vec::new();
    source.by_ref().take(LIMIT + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() as u64 <= LIMIT, "request exceeds 128 MiB");
    let request: Value = serde_json::from_slice(&bytes).context("invalid request JSON")?;
    anyhow::ensure!(request.is_object(), "request must be a JSON object");
    Ok(request)
}

fn bind_request_model(value: &mut Value, model: &str) -> Result<()> {
    match value.get("model") {
        Some(Value::String(request_model)) if request_model != model => {
            bail!("request model '{request_model}' differs from positional model '{model}'")
        }
        Some(Value::String(_)) | Some(Value::Null) | None => {}
        _ => bail!("request model must be a string"),
    }
    value["model"] = Value::String(model.to_string());
    Ok(())
}

fn default_task(manifest: &ModelManifest) -> Result<InferenceTask> {
    let tasks = &manifest.metadata.tasks;
    if tasks.is_empty() || tasks.contains(&InferenceTask::TextGeneration) {
        return Ok(InferenceTask::TextGeneration);
    }
    if tasks.contains(&InferenceTask::ImageUnderstanding) {
        return Ok(InferenceTask::ImageUnderstanding);
    }
    if tasks.len() == 1 {
        return Ok(tasks[0]);
    }
    bail!(
        "model '{}' supports multiple media tasks; select --task or supply --request with a task",
        manifest.id
    )
}

fn parse_input(value: &str) -> Result<InferenceInput> {
    let (kind, source) = value
        .split_once('=')
        .context("--input requires MODALITY[:ROLE]=PATH_OR_URL")?;
    let (modality, role) = kind.split_once(':').unwrap_or((kind, kind));
    let modality = modality
        .parse::<InputModality>()
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        !role.trim().is_empty() && !source.trim().is_empty(),
        "input role and source must not be empty"
    );
    Ok(string_input(modality, role, source))
}

fn apply_media_parameters(request: &mut InferenceRequest, values: &[String]) -> Result<()> {
    let mut routing = serde_json::to_value(&request.routing)?;
    for (key, value) in parse_set_overrides(values).map_err(anyhow::Error::msg)? {
        if let Some(field) = key.strip_prefix("routing.") {
            anyhow::ensure!(
                routing.get(field).is_some(),
                "unknown routing parameter '{key}'"
            );
            routing[field] = value;
        } else {
            request
                .parameters
                .insert(key, ParameterValue::from_json(value)?);
        }
    }
    request.routing = serde_json::from_value(routing).context("invalid routing override")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn execute(
    model_home: Option<PathBuf>,
    backend_override: BackendArg,
    device_override: Option<DeviceArg>,
    llama_options: LlamaRuntimeOptions,
    selection_options: SelectionOptions,
    model: String,
    prompt: Vec<String>,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<u64>,
    chat_template: Option<ChatTemplateArg>,
    images: Vec<String>,
    verbose: bool,
    debug: bool,
    options: RunOptions,
    persistence_args: ChatPersistenceArgs,
) -> Result<()> {
    let store = ModelStore::resolve(model_home)?;
    let manifest = store.get(&model)?;
    let mut raw = options.request.as_deref().map(read_request).transpose()?;
    if let Some(raw) = raw.as_mut() {
        bind_request_model(raw, &manifest.id)?;
    }
    let request_task = raw
        .as_ref()
        .and_then(|raw| raw.get("task"))
        .map(|task| serde_json::from_value::<InferenceTask>(task.clone()))
        .transpose()?;
    if let (Some(explicit), Some(from_request)) = (options.task, request_task) {
        anyhow::ensure!(
            explicit == from_request,
            "--task disagrees with request task"
        );
    }
    let task = match options.task.or(request_task) {
        Some(task) => task,
        None if raw
            .as_ref()
            .is_some_and(|raw| raw.get("messages").is_some()) =>
        {
            InferenceTask::TextGeneration
        }
        None => default_task(&manifest)?,
    };
    if !matches!(
        task,
        InferenceTask::TextGeneration | InferenceTask::ImageUnderstanding
    ) {
        anyhow::ensure!(
            !persistence_args.is_enabled(),
            "conversation persistence applies to text/vision and tool conversations; media uses the existing output store"
        );
        anyhow::ensure!(
            !options.stream && options.stream_granularity.is_none(),
            "this media task returns completed outputs; text --stream is not supported"
        );
        anyhow::ensure!(
            temperature.is_none()
                && top_p.is_none()
                && chat_template.is_none()
                && max_tokens.is_none()
                && !options.no_history,
            "text sampling/history/template options do not apply to media; use --set with the task's parameter schema"
        );
        let mut request = if let Some(mut raw) = raw {
            raw["task"] = serde_json::to_value(task)?;
            serde_json::from_value::<InferenceRequest>(raw)
                .context("expected a canonical media InferenceRequest")?
        } else {
            let mut request = InferenceRequest::new(&manifest.id, task);
            request.prompt = (!prompt.is_empty()).then(|| prompt.join(" "));
            request
        };
        request.inputs.extend(
            options
                .inputs
                .iter()
                .map(|v| parse_input(v))
                .collect::<Result<Vec<_>>>()?,
        );
        request.inputs.extend(
            images
                .iter()
                .map(|source| string_input(InputModality::Image, "image", source)),
        );
        if let Some(seed) = seed {
            let prefix = match task.output_modality() {
                OutputModality::Image => "image",
                OutputModality::Video => "video",
                OutputModality::Audio => "audio",
                _ => {
                    bail!("--seed does not apply to this task; use its canonical parameter schema")
                }
            };
            request
                .parameters
                .insert(format!("{prefix}.seed"), seed.into());
        }
        apply_media_parameters(&mut request, &options.parameters)?;
        let selected = media_routing(&RoutingArgs::default(), backend_override, device_override)?;
        for (label, supplied, target) in [
            ("backend", selected.backend, &mut request.routing.backend),
            (
                "accelerator",
                selected.accelerator,
                &mut request.routing.accelerator,
            ),
            ("device", selected.device, &mut request.routing.device),
        ] {
            if let Some(supplied) = supplied {
                anyhow::ensure!(
                    target.as_ref().is_none_or(|current| current == &supplied),
                    "conflicting media {label} in request and CLI"
                );
                *target = Some(supplied);
            }
        }
        return execute_media_request(
            &store,
            request,
            options.output,
            verbose,
            debug,
            options.json,
        );
    }
    anyhow::ensure!(
        options.inputs.is_empty() && options.parameters.is_empty() && options.output.is_none(),
        "text/vision uses --image or OpenAI --request; --input, --set and --output are media options"
    );
    let images = normalize_cli_image_sources(&images)?;
    let request = if let Some(raw) = raw {
        serde_json::from_value::<ChatCompletionRequest>(raw)
            .context("expected an OpenAI chat request with messages")?
    } else {
        anyhow::ensure!(
            !prompt.is_empty(),
            "text/vision requires a prompt or --request"
        );
        serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": manifest.id, "messages": [vision_user_message(&prompt.join(" "), &images)]
        }))?
    };
    anyhow::ensure!(
        !request.messages.is_empty(),
        "chat messages must not be empty"
    );
    if let Some(runtime) = request.werk.as_ref() {
        runtime.validate()?;
    }
    let max_tokens = max_tokens
        .or(request.max_completion_tokens)
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    let temperature = temperature.or(request.temperature);
    let top_p = top_p.or(request.top_p);
    let seed = seed.or(request.seed);
    let stream =
        options.stream || options.stream_granularity.is_some() || request.stream.unwrap_or(false);
    let selection_options = conversation_selection_options(selection_options, &persistence_args);
    let persistence = persistence_args.open(&store, &manifest.id)?;
    let has_images = !image_urls_from_messages(&request.messages).is_empty()
        || persistence
            .as_ref()
            .is_some_and(|(_, messages)| !image_urls_from_messages(messages).is_empty());
    let requires_tools = request.requires_tool_calling()
        || persistence
            .as_ref()
            .is_some_and(|(_, messages)| messages.iter().any(ChatMessage::uses_tool_calling));
    anyhow::ensure!(
        !requires_tools || matches!(chat_template, None | Some(ChatTemplateArg::Model)),
        "tool conversations require the backend/model chat template; omit --chat-template or use model"
    );
    let runtime_options = request.werk.as_ref().filter(|options| !options.is_empty());
    anyhow::ensure!(
        !has_images || runtime_options.is_none(),
        "werk.omlx options currently support text chat only"
    );
    let backend_choice = resolve_backend(backend_override, device_override)?;
    let route_choice = if runtime_options.is_some() {
        anyhow::ensure!(
            matches!(backend_choice, BackendChoice::Auto | BackendChoice::Omlx),
            "werk.omlx controls require the auto or omlx backend"
        );
        BackendChoice::Omlx
    } else {
        backend_choice
    };
    let selected_route = routed_backend_for_request_with_tools(
        &store,
        route_choice,
        &manifest,
        has_images,
        requires_tools,
        selection_options,
    )?;
    let selected_backend = selected_route.choice;
    selected_route.report();
    print_routing_debug(
        &store,
        backend_override,
        &manifest,
        has_images,
        &selected_route,
        debug,
    );
    let mut backend = selected_route.build(store, llama_options.clone(), selection_options)?;
    if let Some(runtime) = runtime_options {
        backend = backend.with_chat_options(&manifest, runtime)?;
    }
    anyhow::ensure!(
        !requires_tools || backend.supports_tool_calling(&manifest, has_images),
        "selected backend does not support OpenAI tool calling for this model"
    );
    let stop = request.stop_strings();
    let tool_config = (request.tools.is_some()
        || request.tool_choice.is_some()
        || request.parallel_tool_calls.is_some())
    .then_some(ToolCallingConfig {
        tools: request.tools,
        tool_choice: request.tool_choice,
        parallel_tool_calls: request.parallel_tool_calls,
    });
    let context_size = chat_context_size(selected_backend, &manifest, llama_options.ctx_size);
    chat_loop(
        backend,
        manifest,
        selected_backend,
        context_size,
        max_tokens,
        temperature,
        top_p,
        seed,
        !options.no_history,
        chat_template,
        Vec::new(),
        options
            .stream_granularity
            .unwrap_or(StreamGranularityArg::Token)
            .into(),
        verbose,
        debug,
        terminal_spinner_enabled(debug) && !options.json,
        persistence,
        Some(RunTurn {
            messages: request.messages,
            stream,
            json: options.json,
            stop,
            tool_config,
            requires_tools,
        }),
    )
    .await
}

#[derive(Default)]
pub(super) struct ToolCallAccumulator(BTreeMap<usize, ChatCompletionToolCall>);

impl ToolCallAccumulator {
    pub fn extend(&mut self, deltas: Vec<ChatCompletionToolCallDelta>) -> Result<()> {
        for delta in deltas {
            anyhow::ensure!(delta.index < 1024, "tool-call index exceeds 1023");
            let call = self
                .0
                .entry(delta.index)
                .or_insert_with(|| ChatCompletionToolCall {
                    id: String::new(),
                    kind: "function".into(),
                    function: ChatCompletionFunctionCall {
                        name: String::new(),
                        arguments: String::new(),
                    },
                });
            if let Some(id) = delta.id {
                call.id.push_str(&id);
            }
            if let Some(kind) = delta.kind {
                call.kind = kind;
            }
            if let Some(function) = delta.function {
                if let Some(name) = function.name {
                    call.function.name.push_str(&name);
                }
                if let Some(arguments) = function.arguments {
                    call.function.arguments.push_str(&arguments);
                }
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<ChatCompletionToolCall>> {
        let calls: Vec<_> = self.0.into_values().collect();
        anyhow::ensure!(
            calls
                .iter()
                .all(|call| !call.id.is_empty() && !call.function.name.is_empty()),
            "backend returned incomplete tool-call metadata"
        );
        Ok(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{GenerateResponse, GenerateStream, GenerationTimings};
    use std::{
        sync::Mutex,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn fixture() -> (ModelStore, ModelManifest) {
        let home = std::env::temp_dir().join(format!(
            "werk-run-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = ModelStore::resolve(Some(home)).unwrap();
        let manifest = ModelManifest {
            storage: Default::default(),
            id: "test-model".into(),
            source: ModelSource::LocalPath {
                path: "test".into(),
            },
            format: ModelFormat::Gguf,
            architecture: Some("llama".into()),
            tokenizer_path: None,
            config_path: None,
            model_path: Some("files/model.gguf".into()),
            backend: "test".into(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        };
        (store, manifest)
    }

    fn text_message(role: &str, text: &str) -> ChatMessage {
        serde_json::from_value(json!({"role":role,"content":text})).unwrap()
    }

    fn persistence() -> ChatPersistenceArgs {
        ChatPersistenceArgs {
            session: Some("shared".into()),
            ..Default::default()
        }
    }

    fn turn(messages: Vec<ChatMessage>) -> RunTurn {
        RunTurn {
            messages,
            stream: false,
            json: true,
            stop: vec!["END".into()],
            tool_config: None,
            requires_tools: false,
        }
    }

    #[derive(Default)]
    struct Recording {
        requests: Mutex<Vec<GenerateRequest>>,
        calls: Mutex<Vec<&'static str>>,
        events: Mutex<Vec<Result<GenerateStreamEvent, String>>>,
    }

    struct MockBackend {
        state: Arc<Recording>,
        native: bool,
    }
    struct MockSession(Arc<Recording>);

    fn stream(state: &Recording, request: GenerateRequest) -> GenerateStream {
        state.requests.lock().unwrap().push(request);
        Box::pin(tokio_stream::iter(state.events.lock().unwrap().clone()))
    }

    impl GenerationBackend for MockBackend {
        fn prepare(&self, _: &ModelManifest) -> Result<()> {
            self.state.calls.lock().unwrap().push("prepare");
            Ok(())
        }
        fn start_persistent_chat_session(
            &self,
            _: &ModelManifest,
            _: Option<u64>,
            _: &Path,
        ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
            self.state.calls.lock().unwrap().push("persistent");
            Ok(self.native.then(|| {
                Box::new(MockSession(self.state.clone())) as Box<dyn ChatGenerationSession>
            }))
        }
        fn generate(&self, _: &ModelManifest, _: GenerateRequest) -> Result<GenerateResponse> {
            unreachable!()
        }
        fn generate_stream(&self, _: ModelManifest, request: GenerateRequest) -> GenerateStream {
            self.state.calls.lock().unwrap().push("backend_generate");
            stream(&self.state, request)
        }
    }
    impl ChatGenerationSession for MockSession {
        fn generate(&self, _: GenerateRequest) -> Result<GenerateResponse> {
            unreachable!()
        }
        fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
            self.0.calls.lock().unwrap().push("session_generate");
            stream(&self.0, request)
        }
    }
    fn done(reason: &str) -> Result<GenerateStreamEvent, String> {
        Ok(GenerateStreamEvent::Done {
            finish_reason: reason.into(),
            prompt_tokens: 10,
            completion_tokens: 2,
            timings: GenerationTimings::default(),
            backend_diagnostics: Vec::new(),
        })
    }
    async fn invoke(
        store: &ModelStore,
        manifest: &ModelManifest,
        state: Arc<Recording>,
        run: RunTurn,
        native: bool,
    ) -> Result<()> {
        chat_loop(
            Arc::new(MockBackend { state, native }),
            manifest.clone(),
            BackendChoice::LlamaServer(LlamaCppMode::Cuda),
            Some(4096),
            128,
            Some(0.4),
            Some(0.8),
            Some(17),
            true,
            None,
            Vec::new(),
            StreamGranularity::Chunk,
            false,
            false,
            false,
            persistence().open(store, &manifest.id)?,
            Some(run),
        )
        .await
    }

    #[tokio::test]
    async fn run_resumes_chat_archive_and_reuses_native_session_before_prepare() {
        let (store, manifest) = fixture();
        let state = Arc::new(Recording::default());
        *state.events.lock().unwrap() = vec![
            Ok(GenerateStreamEvent::TextChunk("answer".into())),
            done("stop"),
        ];
        invoke(
            &store,
            &manifest,
            state.clone(),
            turn(vec![text_message("user", "first")]),
            true,
        )
        .await
        .unwrap();
        let (storage, archive) = persistence().open(&store, &manifest.id).unwrap().unwrap();
        assert_eq!(archive.len(), 2);
        assert!(storage.resumed());
        drop(storage);
        invoke(
            &store,
            &manifest,
            state.clone(),
            turn(vec![text_message("user", "second")]),
            true,
        )
        .await
        .unwrap();
        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].messages.len(), 3);
        assert_eq!(
            requests[1].messages[0].content.as_ref().unwrap().as_text(),
            "first"
        );
        assert_eq!(requests[1].max_tokens, 128);
        assert_eq!(requests[1].temperature, Some(0.4));
        assert_eq!(requests[1].top_p, Some(0.8));
        assert_eq!(requests[1].seed, Some(17));
        assert_eq!(requests[1].stream_granularity, StreamGranularity::Chunk);
        assert!(requests[1].stop.contains(&"END".to_string()));
        assert_eq!(
            *state.calls.lock().unwrap(),
            [
                "persistent",
                "session_generate",
                "persistent",
                "session_generate"
            ]
        );
        assert_eq!(
            persistence()
                .open(&store, &manifest.id)
                .unwrap()
                .unwrap()
                .1
                .len(),
            4
        );
    }

    #[tokio::test]
    async fn run_failed_or_unfinished_stream_keeps_previous_archive_and_releases_lock() {
        let (store, manifest) = fixture();
        let state = Arc::new(Recording::default());
        *state.events.lock().unwrap() = vec![
            Ok(GenerateStreamEvent::TextChunk("saved".into())),
            done("stop"),
        ];
        invoke(
            &store,
            &manifest,
            state.clone(),
            turn(vec![text_message("user", "initial")]),
            false,
        )
        .await
        .unwrap();
        let (storage, _) = persistence().open(&store, &manifest.id).unwrap().unwrap();
        let path = storage.path().unwrap().to_path_buf();
        let before = fs::read(&path).unwrap();
        drop(storage);
        for failure in [
            vec![
                Ok(GenerateStreamEvent::TextChunk("partial".into())),
                Err("failed".into()),
            ],
            vec![Ok(GenerateStreamEvent::TextChunk("unfinished".into()))],
        ] {
            *state.events.lock().unwrap() = failure;
            assert!(
                invoke(
                    &store,
                    &manifest,
                    state.clone(),
                    turn(vec![text_message("user", "retry")]),
                    false
                )
                .await
                .is_err()
            );
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(
                persistence()
                    .open(&store, &manifest.id)
                    .unwrap()
                    .unwrap()
                    .1
                    .len(),
                2
            );
        }
    }

    fn delta(value: Value) -> ChatCompletionToolCallDelta {
        serde_json::from_value(value).unwrap()
    }

    #[tokio::test]
    async fn run_persists_tool_only_answers_and_resumes_tool_results() {
        let (store, manifest) = fixture();
        let state = Arc::new(Recording::default());
        *state.events.lock().unwrap() = vec![
            Ok(GenerateStreamEvent::ToolCallDelta(vec![delta(json!({
                "index":0,"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}
            }))])),
            done("tool_calls"),
        ];
        let mut run = turn(vec![text_message("user", "weather?")]);
        run.requires_tools = true;
        let tool_request: ChatCompletionRequest = serde_json::from_value(json!({"messages":[], "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}],"tool_choice":"auto","parallel_tool_calls":true})).unwrap();
        run.tool_config = Some(ToolCallingConfig {
            tools: tool_request.tools,
            tool_choice: tool_request.tool_choice,
            parallel_tool_calls: tool_request.parallel_tool_calls,
        });
        invoke(&store, &manifest, state.clone(), run, true)
            .await
            .unwrap();
        assert!(state.requests.lock().unwrap()[0].tool_config.is_some());
        let (storage, history) = persistence().open(&store, &manifest.id).unwrap().unwrap();
        assert!(history[1].content.is_none());
        assert_eq!(history[1].tool_calls.as_ref().unwrap()[0].id, "call_1");
        drop(storage);
        *state.events.lock().unwrap() = vec![
            Ok(GenerateStreamEvent::TextChunk("sunny".into())),
            done("stop"),
        ];
        let result = serde_json::from_value(
            json!({"role":"tool","tool_call_id":"call_1","content":"sunny"}),
        )
        .unwrap();
        invoke(&store, &manifest, state.clone(), turn(vec![result]), true)
            .await
            .unwrap();
        assert_eq!(state.requests.lock().unwrap()[1].messages[2].role, "tool");
        assert_eq!(
            persistence()
                .open(&store, &manifest.id)
                .unwrap()
                .unwrap()
                .1
                .len(),
            4
        );
    }

    #[tokio::test]
    async fn run_images_bypass_text_only_sessions_including_restored_images() {
        let (store, manifest) = fixture();
        let state = Arc::new(Recording::default());
        *state.events.lock().unwrap() = vec![
            Ok(GenerateStreamEvent::TextChunk("image".into())),
            done("stop"),
        ];
        invoke(
            &store,
            &manifest,
            state.clone(),
            turn(vec![vision_user_message(
                "describe",
                &["https://example.org/a.png".into()],
            )]),
            true,
        )
        .await
        .unwrap();
        invoke(
            &store,
            &manifest,
            state.clone(),
            turn(vec![text_message("user", "again")]),
            true,
        )
        .await
        .unwrap();
        assert_eq!(
            *state.calls.lock().unwrap(),
            ["backend_generate", "backend_generate"]
        );
        assert!(!state.requests.lock().unwrap()[1].image_urls.is_empty());
    }

    #[test]
    fn run_cli_exposes_persistence_streaming_and_media_without_conflicts() {
        for backend in ["auto", "cuda", "cpu", "mlx", "omlx", "vllm"] {
            let cli = Cli::try_parse_from([
                "werk",
                "--backend",
                backend,
                "run",
                "model",
                "hello",
                "--session",
                "shared",
                "--persistence-mode",
                "disk",
                "--persistence-ttl-seconds",
                "60",
                "--stream-granularity",
                "chunk",
                "--json",
            ])
            .unwrap();
            let command = cli.command.unwrap();
            assert!(!should_print_startup_banner_for(&command, true, true));
            let Commands::Run {
                persistence,
                options,
                ..
            } = command
            else {
                panic!()
            };
            assert!(persistence.is_enabled());
            assert_eq!(persistence.persistence_ttl_seconds, Some(60));
            assert_eq!(
                options.stream_granularity,
                Some(StreamGranularityArg::Chunk)
            );
        }
        for flag in ["--no-history", "--single-turn"] {
            assert!(
                Cli::try_parse_from(["werk", "run", "model", "hello", "--persistence", flag])
                    .is_err()
            );
        }
        assert!(
            Cli::try_parse_from([
                "werk",
                "run",
                "model",
                "hello",
                "--persistence-ttl-seconds",
                "0"
            ])
            .is_err()
        );
        assert!(Cli::try_parse_from(["werk", "run", "model", "--request", "-"]).is_ok());
        assert!(
            Cli::try_parse_from(["werk", "run", "model", "hello", "--request", "a.json"]).is_err()
        );
        assert!(
            Cli::try_parse_from([
                "werk",
                "run",
                "model",
                "--task",
                "speech-to-text",
                "--input",
                "audio=clip.wav"
            ])
            .is_ok()
        );
    }

    #[test]
    fn run_tool_delta_assembly_handles_parallel_calls_and_rejects_incomplete_metadata() {
        let mut calls = ToolCallAccumulator::default();
        calls
            .extend(vec![
                delta(json!({"index":1,"id":"b","function":{"name":"second","arguments":"{"}})),
                delta(json!({"index":0,"id":"a","function":{"name":"first","arguments":"{}"}})),
            ])
            .unwrap();
        calls
            .extend(vec![delta(json!({"index":1,"function":{"arguments":"}"}}))])
            .unwrap();
        let calls = calls.finish().unwrap();
        assert_eq!(calls[0].id, "a");
        assert_eq!(calls[1].function.arguments, "{}");
        let mut incomplete = ToolCallAccumulator::default();
        incomplete
            .extend(vec![delta(
                json!({"index":0,"function":{"arguments":"{}"}}),
            )])
            .unwrap();
        assert!(incomplete.finish().is_err());
    }

    #[test]
    fn run_media_inputs_parameters_routing_and_model_binding_use_canonical_schema() {
        let input = parse_input("image:mask_image=https://example.org/a.png?x=1").unwrap();
        assert_eq!(input.role, "mask_image");
        assert!(matches!(input.source, InferenceInputSource::Url { .. }));
        assert!(parse_input("audio=").is_err());
        let mut request = InferenceRequest::new("test-model", InferenceTask::ImageGeneration);
        apply_media_parameters(
            &mut request,
            &[
                "image.steps=20".into(),
                "routing.precision=float16".into(),
                "routing.accelerator=cuda".into(),
            ],
        )
        .unwrap();
        assert_eq!(request.routing.precision.as_deref(), Some("float16"));
        assert_eq!(
            serde_json::to_value(&request.parameters["image.steps"]).unwrap(),
            json!(20)
        );
        assert!(apply_media_parameters(&mut request, &["routing.typo=true".into()]).is_err());
        let mut raw = json!({"task":"image_generation","prompt":"mountains"});
        bind_request_model(&mut raw, "test-model").unwrap();
        let request: InferenceRequest = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(request.model, "test-model");
        assert!(bind_request_model(&mut raw, "other").is_err());
    }

    #[test]
    fn run_media_task_inference_rejects_ambiguity() {
        let (_, mut manifest) = fixture();
        manifest.metadata.tasks = vec![InferenceTask::ImageGeneration];
        assert_eq!(
            default_task(&manifest).unwrap(),
            InferenceTask::ImageGeneration
        );
        manifest.metadata.tasks.push(InferenceTask::ImageEditing);
        assert!(default_task(&manifest).is_err());
        manifest.metadata.tasks.push(InferenceTask::TextGeneration);
        assert_eq!(
            default_task(&manifest).unwrap(),
            InferenceTask::TextGeneration
        );
    }
    #[test]
    fn run_tool_context_trimming_removes_whole_turns_and_preserves_current_chain() {
        let (_, manifest) = fixture();
        let tool_call: ChatMessage = serde_json::from_value(json!({"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]})).unwrap();
        let tool_result: ChatMessage = serde_json::from_value(
            json!({"role":"tool","tool_call_id":"call_1","content":"x".repeat(5000)}),
        )
        .unwrap();
        let mut messages = vec![
            text_message("system", "instructions"),
            text_message("user", "old"),
            tool_call.clone(),
            tool_result.clone(),
            text_message("assistant", "old answer"),
            text_message("user", "new"),
        ];
        let removed = trim_chat_history_to_context(
            &manifest,
            BackendChoice::LlamaServer(LlamaCppMode::Cuda),
            None,
            &mut messages,
            Some(1024),
            128,
        )
        .unwrap();
        assert_eq!(removed, 4);
        assert_eq!(messages.len(), 2);
        let mut current = vec![text_message("user", "current"), tool_call, tool_result];
        assert!(
            trim_chat_history_to_context(
                &manifest,
                BackendChoice::LlamaServer(LlamaCppMode::Cuda),
                None,
                &mut current,
                Some(512),
                128
            )
            .is_err()
        );
        assert_eq!(current.len(), 3);
    }

    #[test]
    fn run_and_chat_persistence_modes_share_policy_and_cache_defaults() {
        let (store, manifest) = fixture();
        for command in ["run", "chat"] {
            let mut args = vec!["werk", command, "test-model"];
            if command == "run" {
                args.push("hello");
            }
            args.extend(["--session", "shared", "--persistence-mode", "memory"]);
            let command = Cli::try_parse_from(args).unwrap().command.unwrap();
            let persistence = match command {
                Commands::Run { persistence, .. } | Commands::Chat { persistence, .. } => {
                    persistence
                }
                _ => unreachable!(),
            };
            let (storage, _) = persistence.open(&store, &manifest.id).unwrap().unwrap();
            assert!(!storage.is_durable());
            let options = conversation_selection_options(SelectionOptions::default(), &persistence);
            assert_eq!(options.vllm_automatic_prefix_caching, Some(true));
            assert!(!options.omlx_server_prefix_cache);
        }
        let options = conversation_selection_options(SelectionOptions::default(), &persistence());
        assert!(options.omlx_server_prefix_cache);
        let fresh = ChatPersistenceArgs {
            persistence_reuse: Some(ServePersistenceReuseArg::Disabled),
            ..persistence()
        };
        let options = conversation_selection_options(SelectionOptions::default(), &fresh);
        assert_eq!(options.vllm_automatic_prefix_caching, Some(false));
        assert!(!options.omlx_server_prefix_cache);
    }
}

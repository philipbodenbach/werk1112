use axum::{
    http::{HeaderMap, header},
    response::Response,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};
use tokio::sync::Semaphore;

use crate::{
    backend::{ChatGenerationSession, GenerationBackend},
    inference_service::{InferenceService, JobManager},
    model_store::{ModelManifest, ModelRuntimeIdentity, ModelStore},
    openai::ChatTemplateOptions,
    runtime_control::{
        LocalWerkControl, PrincipalDeriver, RoutedRuntimeAdapter, RuntimeRoutedGenerationBackend,
        ServerPersistenceConfig,
    },
    werk_protocol::{ProtocolError, ProtocolErrorCode, WerkControl},
};

use super::{
    automatic1111::Automatic1111State,
    cors::CorsOrigin,
    response::{auth_error, constant_time_eq},
};

const MAX_PRINCIPAL_DERIVATIONS: usize = 8;
const MAX_CHAT_SESSIONS: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ChatSessionKey {
    model: ModelRuntimeIdentity,
    seed: Option<u64>,
}

#[derive(Default)]
struct ChatSessionCache {
    entries: HashMap<ChatSessionKey, Arc<dyn ChatGenerationSession>>,
    recency: VecDeque<ChatSessionKey>,
}

impl ChatSessionCache {
    fn get(&mut self, key: ChatSessionKey) -> Option<Arc<dyn ChatGenerationSession>> {
        let session = self.entries.get(&key).cloned()?;
        self.touch(key);
        Some(session)
    }

    fn insert(
        &mut self,
        key: ChatSessionKey,
        session: Arc<dyn ChatGenerationSession>,
    ) -> Arc<dyn ChatGenerationSession> {
        if let Some(existing) = self.entries.get(&key).cloned() {
            self.touch(key);
            return existing;
        }

        self.entries.insert(key, session.clone());
        self.recency.push_back(key);
        while self.entries.len() > MAX_CHAT_SESSIONS {
            let Some(expired) = self.recency.pop_front() else {
                break;
            };
            self.entries.remove(&expired);
        }
        session
    }

    fn touch(&mut self, key: ChatSessionKey) {
        if let Some(index) = self.recency.iter().position(|candidate| *candidate == key) {
            self.recency.remove(index);
        }
        self.recency.push_back(key);
    }
}

pub type PromptOptionsResolver = Arc<
    dyn Fn(&ModelStore, &ModelManifest, bool) -> anyhow::Result<ChatTemplateOptions<'static>>
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct ApiState {
    pub(super) store: Arc<ModelStore>,
    pub(super) backend: Arc<dyn GenerationBackend>,
    pub(super) werk_control: Arc<dyn WerkControl>,
    pub(super) server_persistence: ServerPersistenceConfig,
    pub(super) default_model: Option<String>,
    pub(super) default_image_model: Option<String>,
    pub(super) chat_context_size: usize,
    prompt_options_resolver: Option<PromptOptionsResolver>,
    chat_sessions: Arc<Mutex<ChatSessionCache>>,
    api_keys: Arc<Vec<String>>,
    principal_deriver: PrincipalDeriver,
    principal_derivation_gate: Arc<Semaphore>,
    cors_origins: Arc<Vec<CorsOrigin>>,
    pub(super) automatic1111: Arc<Automatic1111State>,
    pub(super) verbose: bool,
    pub(super) inference_service: Arc<InferenceService>,
    pub(super) job_manager: Arc<JobManager>,
}

impl ApiState {
    pub fn new(store: ModelStore, backend: Arc<dyn GenerationBackend>) -> Self {
        Self::new_with_default_model(store, backend, None)
    }

    pub fn new_with_default_model(
        store: ModelStore,
        backend: Arc<dyn GenerationBackend>,
        default_model: Option<String>,
    ) -> Self {
        Self::new_with_default_model_and_prompt_options(store, backend, default_model, None)
    }

    pub fn new_with_default_model_and_prompt_options(
        store: ModelStore,
        backend: Arc<dyn GenerationBackend>,
        default_model: Option<String>,
        prompt_options_resolver: Option<PromptOptionsResolver>,
    ) -> Self {
        Self::new_with_default_model_prompt_options_and_verbose(
            store,
            backend,
            default_model,
            prompt_options_resolver,
            false,
        )
    }

    pub fn new_with_default_model_prompt_options_and_verbose(
        store: ModelStore,
        backend: Arc<dyn GenerationBackend>,
        default_model: Option<String>,
        prompt_options_resolver: Option<PromptOptionsResolver>,
        verbose: bool,
    ) -> Self {
        let inference_service = Arc::new(InferenceService::new(store.clone()));
        let job_manager = Arc::new(JobManager::new(inference_service.as_ref().clone()));
        let runtime_adapter = Arc::new(RoutedRuntimeAdapter::new(backend.clone()));
        let backend: Arc<dyn GenerationBackend> = Arc::new(RuntimeRoutedGenerationBackend::new(
            backend,
            runtime_adapter.clone(),
        ));
        let local_control = LocalWerkControl::new(store.clone(), runtime_adapter);
        let principal_deriver = local_control.principal_deriver();
        Self {
            store: Arc::new(store),
            backend,
            werk_control: Arc::new(local_control),
            server_persistence: ServerPersistenceConfig::default(),
            default_model,
            default_image_model: None,
            chat_context_size: 4096,
            prompt_options_resolver,
            chat_sessions: Arc::new(Mutex::new(ChatSessionCache::default())),
            api_keys: Arc::new(Vec::new()),
            principal_deriver,
            principal_derivation_gate: Arc::new(Semaphore::new(MAX_PRINCIPAL_DERIVATIONS)),
            cors_origins: Arc::new(Vec::new()),
            automatic1111: Arc::new(Automatic1111State::default()),
            verbose,
            inference_service,
            job_manager,
        }
    }

    pub fn with_api_keys(mut self, api_keys: Vec<String>) -> Self {
        self.api_keys = Arc::new(api_keys);
        self
    }

    /// Replaces the local Werk control implementation. Primarily useful for
    /// transport contract tests and embedders with their own control service.
    pub fn with_werk_control(mut self, werk_control: Arc<dyn WerkControl>) -> Self {
        self.werk_control = werk_control;
        self
    }

    pub(crate) fn with_server_persistence(
        mut self,
        server_persistence: ServerPersistenceConfig,
    ) -> Self {
        self.server_persistence = server_persistence;
        self
    }

    pub fn with_default_image_model(mut self, model: Option<String>) -> Self {
        self.automatic1111.set_selected_checkpoint(model.clone());
        self.default_image_model = model;
        self
    }

    pub fn with_chat_context_size(mut self, context_size: Option<usize>) -> Self {
        self.chat_context_size = context_size.unwrap_or(4096);
        self
    }

    pub fn with_cors_origins(mut self, cors_origins: Vec<CorsOrigin>) -> Self {
        self.cors_origins = Arc::new(cors_origins);
        self
    }

    /// Replaces the media inference pipeline while keeping the chat backend
    /// untouched. This is useful for embedders and deterministic test backends.
    pub fn with_inference_service(mut self, inference_service: InferenceService) -> Self {
        let inference_service = Arc::new(inference_service);
        self.job_manager = Arc::new(JobManager::new(inference_service.as_ref().clone()));
        self.inference_service = inference_service;
        self
    }

    pub fn api_key_auth_enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }

    pub(super) fn cors_origins(&self) -> &[CorsOrigin] {
        &self.cors_origins
    }

    #[allow(clippy::result_large_err)]
    pub(super) fn authorize(&self, headers: &HeaderMap) -> Result<(), Response> {
        if self.api_keys.is_empty() {
            return Ok(());
        }

        if let Some(token) = headers.get("x-api-key") {
            let Ok(token) = token.to_str() else {
                return Err(auth_error("invalid X-API-Key header"));
            };
            if self.api_key_matches(token.trim()) {
                return Ok(());
            }
            if headers.get(header::AUTHORIZATION).is_none() {
                return Err(auth_error("invalid API key"));
            }
        }

        let Some(header_value) = headers.get(header::AUTHORIZATION) else {
            return Err(auth_error("missing bearer token"));
        };
        let Ok(header_value) = header_value.to_str() else {
            return Err(auth_error("invalid authorization header"));
        };
        let Some((scheme, token)) = header_value.split_once(' ') else {
            return Err(auth_error("expected Authorization: Bearer <token>"));
        };
        if !scheme.eq_ignore_ascii_case("bearer") {
            return Err(auth_error("expected Authorization: Bearer <token>"));
        }
        let token = token.trim();
        if token.is_empty() {
            return Err(auth_error("empty bearer token"));
        }
        if self.api_key_matches(token) {
            Ok(())
        } else {
            Err(auth_error("invalid bearer token"))
        }
    }

    pub(super) fn api_key_matches(&self, token: &str) -> bool {
        if token.is_empty() {
            return false;
        }
        let mut matched = false;
        for key in self.api_keys.iter() {
            // Evaluate every configured key so the position of a match does
            // not determine how many comparisons authentication performs.
            matched |= constant_time_eq(key.as_bytes(), token.as_bytes());
        }
        matched
    }

    /// Authenticates a Werk Protocol request and returns only its opaque
    /// storage namespace. Credentials never cross into the control service.
    pub(super) async fn werk_principal(
        &self,
        headers: &HeaderMap,
    ) -> Result<String, ProtocolError> {
        if self.api_keys.is_empty() {
            return Ok("local".to_string());
        }

        if let Some(token) = headers.get("x-api-key") {
            let token = token.to_str().map_err(|_| unauthorized())?;
            let token = token.trim();
            if self.api_key_matches(token) {
                return self.derive_principal(token.to_string()).await;
            }
            if headers.get(header::AUTHORIZATION).is_none() {
                return Err(unauthorized());
            }
        }

        let header_value = headers
            .get(header::AUTHORIZATION)
            .ok_or_else(unauthorized)?
            .to_str()
            .map_err(|_| unauthorized())?;
        let (scheme, token) = header_value.split_once(' ').ok_or_else(unauthorized)?;
        let token = token.trim();
        if !scheme.eq_ignore_ascii_case("bearer")
            || token.is_empty()
            || !self.api_key_matches(token)
        {
            return Err(unauthorized());
        }
        self.derive_principal(token.to_string()).await
    }

    async fn derive_principal(&self, credential: String) -> Result<String, ProtocolError> {
        let permit = self
            .principal_derivation_gate
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| principal_unavailable())?;
        let deriver = self.principal_deriver.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            deriver.derive(&credential)
        })
        .await
        .map_err(|_| principal_unavailable())?
        .map_err(|_| principal_unavailable())
    }

    pub(super) fn prompt_options(
        &self,
        manifest: &ModelManifest,
        has_images: bool,
    ) -> anyhow::Result<ChatTemplateOptions<'static>> {
        self.prompt_options_resolver
            .as_ref()
            .map(|resolver| resolver(&self.store, manifest, has_images))
            .unwrap_or_else(|| Ok(ChatTemplateOptions::default()))
    }

    pub(super) fn log_verbose(&self, message: impl AsRef<str>) {
        if self.verbose {
            eprintln!("{}", message.as_ref());
        }
    }

    pub(super) fn chat_session(
        &self,
        manifest: &ModelManifest,
        seed: Option<u64>,
    ) -> anyhow::Result<Option<Arc<dyn ChatGenerationSession>>> {
        let key = ChatSessionKey {
            model: ModelRuntimeIdentity::from_manifest(manifest)?,
            seed,
        };
        if let Some(session) = self
            .chat_sessions
            .lock()
            .map_err(|_| anyhow::anyhow!("chat session cache mutex poisoned"))?
            .get(key)
        {
            return Ok(Some(session));
        }

        let Some(session) = self.backend.start_chat_session(manifest, seed)? else {
            return Ok(None);
        };
        let session: Arc<dyn ChatGenerationSession> = Arc::from(session);
        let session = self
            .chat_sessions
            .lock()
            .map_err(|_| anyhow::anyhow!("chat session cache mutex poisoned"))?
            .insert(key, session);
        Ok(Some(session))
    }
}

fn unauthorized() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Unauthorized,
        "authentication credentials are missing or invalid",
    )
}

fn principal_unavailable() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Unavailable,
        "secure principal derivation is unavailable",
    )
    .retryable(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{
            GenerateRequest, GenerateResponse, GenerateStream, GenerationBackend, GenerationTimings,
        },
        model_store::{ModelFile, ModelFormat, ModelMetadata, ModelSource},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestChatSession;

    impl ChatGenerationSession for TestChatSession {
        fn generate(&self, _request: GenerateRequest) -> anyhow::Result<GenerateResponse> {
            unreachable!("chat-session cache tests do not generate tokens")
        }

        fn generate_stream(&self, _request: GenerateRequest) -> GenerateStream {
            Box::pin(tokio_stream::empty())
        }
    }

    struct SessionBackend {
        starts: Arc<AtomicUsize>,
    }

    impl GenerationBackend for SessionBackend {
        fn start_chat_session(
            &self,
            _manifest: &ModelManifest,
            _seed: Option<u64>,
        ) -> anyhow::Result<Option<Box<dyn ChatGenerationSession>>> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(Some(Box::new(TestChatSession)))
        }

        fn generate(
            &self,
            _manifest: &ModelManifest,
            _request: GenerateRequest,
        ) -> anyhow::Result<GenerateResponse> {
            Ok(GenerateResponse {
                text: String::new(),
                assistant_message: None,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".to_string(),
                timings: GenerationTimings::default(),
                backend_diagnostics: Vec::new(),
            })
        }

        fn generate_stream(
            &self,
            _manifest: ModelManifest,
            _request: GenerateRequest,
        ) -> GenerateStream {
            Box::pin(tokio_stream::empty())
        }
    }

    fn manifest(id: impl Into<String>) -> ModelManifest {
        ModelManifest {
            id: id.into(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format: ModelFormat::SafeTensors,
            architecture: Some("qwen3".to_string()),
            tokenizer_path: Some("tokenizer.json".to_string()),
            config_path: Some("config.json".to_string()),
            model_path: Some("model.safetensors".to_string()),
            backend: "test".to_string(),
            created_unix: 1,
            files: vec![ModelFile {
                path: "model.safetensors".to_string(),
                size: 42,
                checksum: "sha256:original".to_string(),
            }],
            artifacts: Vec::new(),
            metadata: ModelMetadata::default(),
        }
    }

    fn state(starts: Arc<AtomicUsize>) -> ApiState {
        let store = ModelStore::resolve(Some(
            std::env::temp_dir().join(format!("werk1112-api-session-cache-{}", std::process::id())),
        ))
        .unwrap();
        ApiState::new(store, Arc::new(SessionBackend { starts }))
    }

    #[test]
    fn chat_session_cache_uses_exact_manifest_identity() {
        let starts = Arc::new(AtomicUsize::new(0));
        let state = state(starts.clone());
        let original = manifest("same-id");

        state.chat_session(&original, Some(7)).unwrap().unwrap();
        state.chat_session(&original, Some(7)).unwrap().unwrap();
        assert_eq!(starts.load(Ordering::SeqCst), 1);

        let mut replacement = original;
        replacement.files[0].checksum = "sha256:replacement".to_string();
        state.chat_session(&replacement, Some(7)).unwrap().unwrap();
        assert_eq!(starts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn chat_session_cache_is_lru_bounded() {
        let starts = Arc::new(AtomicUsize::new(0));
        let state = state(starts.clone());

        for index in 0..=MAX_CHAT_SESSIONS {
            state
                .chat_session(&manifest(format!("model-{index}")), Some(1))
                .unwrap()
                .unwrap();
        }
        assert_eq!(
            state.chat_sessions.lock().unwrap().entries.len(),
            MAX_CHAT_SESSIONS
        );
        assert_eq!(starts.load(Ordering::SeqCst), MAX_CHAT_SESSIONS + 1);

        state
            .chat_session(&manifest("model-0"), Some(1))
            .unwrap()
            .unwrap();
        assert_eq!(starts.load(Ordering::SeqCst), MAX_CHAT_SESSIONS + 2);
    }
}

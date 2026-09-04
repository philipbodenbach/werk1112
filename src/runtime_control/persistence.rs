//! Server-owned defaults for Werk Protocol prefill persistence.
//!
//! Request-policy defaults are deliberately applied at the HTTP boundary,
//! where Werk can still distinguish an omitted member from an explicit value.
//! They do not rewrite OpenAI-compatible, media, or other legacy requests.
//! The serve command may additionally translate the enabled/reuse setting
//! into a validated automatic-prefix-caching launch default for local vLLM.

use crate::werk_protocol::{PersistencePolicy, PrefillRequest};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct ServerPersistenceConfig {
    enabled: bool,
    defaults: PersistencePolicy,
}

impl ServerPersistenceConfig {
    pub(crate) fn enabled(defaults: PersistencePolicy) -> Self {
        Self {
            enabled: true,
            defaults,
        }
    }

    pub(crate) const fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn defaults(&self) -> &PersistencePolicy {
        &self.defaults
    }

    /// Applies server defaults only when the corresponding top-level member
    /// was absent from the wire request. An explicit policy object owns the
    /// complete policy, including its protocol-level field defaults.
    pub(crate) fn apply_prefill_defaults(
        &self,
        request: &mut PrefillRequest,
        policy_was_supplied: bool,
        experimental_decision_was_supplied: bool,
    ) {
        if !self.enabled {
            return;
        }
        if !policy_was_supplied {
            request.policy = self.defaults.clone();
        }
        if !experimental_decision_was_supplied {
            request.allow_experimental = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::werk_protocol::{PersistenceMode, PrefillInput, ReuseMode};

    fn prefill_request() -> PrefillRequest {
        PrefillRequest {
            model_id: "model".to_string(),
            input: PrefillInput::Text {
                text: "prefix".to_string(),
            },
            policy: PersistencePolicy::default(),
            allow_experimental: false,
        }
    }

    fn enabled_config() -> ServerPersistenceConfig {
        ServerPersistenceConfig::enabled(PersistencePolicy {
            mode: PersistenceMode::Disk,
            reuse: ReuseMode::Required,
            ttl_seconds: Some(900),
            pin: true,
        })
    }

    #[test]
    fn disabled_configuration_preserves_protocol_defaults() {
        let mut request = prefill_request();
        ServerPersistenceConfig::default().apply_prefill_defaults(&mut request, false, false);

        assert_eq!(request.policy, PersistencePolicy::default());
        assert!(!request.allow_experimental);
    }

    #[test]
    fn enabled_configuration_defaults_only_absent_top_level_members() {
        let config = enabled_config();
        let mut defaulted = prefill_request();
        config.apply_prefill_defaults(&mut defaulted, false, false);

        assert_eq!(defaulted.policy, *config.defaults());
        assert!(defaulted.allow_experimental);

        let mut explicit = prefill_request();
        explicit.policy.mode = PersistenceMode::Memory;
        explicit.allow_experimental = false;
        config.apply_prefill_defaults(&mut explicit, true, true);

        assert_eq!(explicit.policy.mode, PersistenceMode::Memory);
        assert_eq!(explicit.policy.reuse, ReuseMode::Prefer);
        assert_eq!(explicit.policy.ttl_seconds, None);
        assert!(!explicit.policy.pin);
        assert!(!explicit.allow_experimental);
    }
}

//! Content-free snapshots of already running llama.cpp workers.
use super::LlamaServerBackend;
use crate::observability::{BackendSnapshot, ReasoningDefaults};
use serde_json::Value;
use std::{io::Read, time::Duration};

/// Native arguments are authoritative, including trailing WERK_LLAMA_ARGS.
pub(super) fn configured_slots(args: &[String]) -> Option<usize> {
    let mut slots = None;
    for (index, arg) in args.iter().enumerate() {
        let value = if matches!(arg.as_str(), "--parallel" | "-np") {
            Some(args.get(index + 1).map(String::as_str).unwrap_or(""))
        } else {
            arg.strip_prefix("--parallel=")
                .or_else(|| arg.strip_prefix("-np="))
        };
        if let Some(value) = value {
            slots = value.parse().ok().filter(|n| (1..=65536).contains(n));
        }
    }
    slots
}

pub(super) fn sample(backend: &LlamaServerBackend) -> Vec<BackendSnapshot> {
    // Never hold the registry or generation/state gate during monitoring I/O.
    let servers = backend.servers.snapshot();
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .ok();
    servers
        .into_iter()
        .take(8)
        .map(|server| {
            let mut sample = BackendSnapshot {
                backend: format!("llama.cpp / {}", super::display_name(server.mode)),
                instance: format!("llama-{}-{}", server.pid, server.url),
                model: server.model_id.clone(),
                ..Default::default()
            };
            if !server.is_running() {
                return sample;
            }
            sample.reasoning_defaults = server.reasoning_defaults.clone();
            let pid = sysinfo::Pid::from_u32(server.pid);
            let mut system = sysinfo::System::new();
            system.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[pid]),
                true,
                sysinfo::ProcessRefreshKind::nothing().with_memory(),
            );
            if let Some(process) = system.process(pid) {
                sample
                    .gauges
                    .insert("process_resident_bytes".into(), process.memory() as f64);
            }
            placement(&mut sample, &server.args);
            #[cfg(target_os = "linux")]
            server.expert_memory.sample(&server.model_path, &mut sample);
            if let Some(slots) = client
                .as_ref()
                .and_then(|client| fetch_slots(client, &server.url))
            {
                sample.available = parse_slots(&mut sample, &slots);
            }
            sample
        })
        .collect()
}

/// Capture the native launch configuration once. No subprocess, GPU probe or
/// additional HTTP request is needed. Environment defaults precede CLI options.
pub(super) fn reasoning_defaults(
    args: &[String],
    env: impl Fn(&str) -> Option<String>,
) -> Option<ReasoningDefaults> {
    let mut defaults = ReasoningDefaults {
        source: "worker_launch_configuration".into(),
        ..Default::default()
    };
    for (variable, option) in [
        ("LLAMA_ARG_CHAT_TEMPLATE_KWARGS", "--chat-template-kwargs"),
        ("LLAMA_ARG_REASONING", "--reasoning"),
        ("LLAMA_ARG_REASONING_EFFORT", "--reasoning-effort"),
        ("LLAMA_ARG_THINK_BUDGET", "--reasoning-budget"),
    ] {
        if let Some(value) = env(variable) {
            apply_reasoning_option(&mut defaults, option, &value);
        }
    }
    for (index, arg) in args.iter().enumerate() {
        if let Some((key, value)) = arg.split_once('=') {
            apply_reasoning_option(&mut defaults, key, value);
        } else if let Some(value) = args.get(index + 1) {
            apply_reasoning_option(&mut defaults, arg, value);
        }
    }
    (defaults.effort.is_some() || defaults.thinking.is_some() || defaults.budget_tokens.is_some())
        .then_some(defaults)
}

fn apply_reasoning_option(defaults: &mut ReasoningDefaults, key: &str, value: &str) {
    match key {
        "--reasoning" | "-rea" => {
            defaults.thinking = match value {
                "on" | "true" | "1" => Some(true),
                "off" | "false" | "0" => Some(false),
                _ => None,
            };
        }
        "--reasoning-effort" => {
            defaults.effort = (value != "default"
                && !value.is_empty()
                && value.len() <= 256
                && !value.chars().any(char::is_control))
            .then(|| value.into());
        }
        "--reasoning-budget" => {
            defaults.budget_tokens = value
                .parse::<i32>()
                .ok()
                .filter(|n| *n >= -1)
                .map(i64::from);
        }
        "--chat-template-kwargs" if value.len() <= 65536 => {
            if let Ok(Value::Object(kwargs)) = serde_json::from_str::<Value>(value) {
                if let Some(value) = kwargs.get("enable_thinking") {
                    defaults.thinking = value.as_bool();
                }
                if let Some(value) = kwargs.get("reasoning_effort") {
                    defaults.effort = (value.is_number()
                        || value.as_str().is_some_and(|s| {
                            !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control)
                        }))
                    .then(|| value.clone());
                }
            }
        }
        _ => {}
    }
}

fn fetch_slots(client: &reqwest::blocking::Client, url: &str) -> Option<Value> {
    let response = client
        .get(format!("{url}/slots"))
        .send()
        .ok()?
        .error_for_status()
        .ok()?;
    let mut bytes = Vec::new();
    response
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > 1024 * 1024 {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

fn placement(sample: &mut BackendSnapshot, args: &[String]) {
    for pair in args.windows(2) {
        let key = match pair[0].as_str() {
            "-ngl" | "--gpu-layers" | "--n-gpu-layers" => "gpu_layers_requested",
            "--n-cpu-moe" | "-ncmoe" => "cpu_moe_layers",
            _ => continue,
        };
        if let Ok(value) = pair[1].parse::<u32>() {
            sample.gauges.insert(key.into(), value as f64);
        }
    }
    if args.iter().any(|arg| arg == "--cpu-moe" || arg == "-cmoe") {
        sample.gauges.insert("cpu_moe_all".into(), 1.);
    }
}

fn parse_slots(sample: &mut BackendSnapshot, value: &Value) -> bool {
    let Some(slots) = value.as_array() else {
        return false;
    };
    if slots.is_empty() {
        return false;
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut per_request_min = u64::MAX;
    let mut active = 0.;
    let mut capacity = 0.;
    let mut context = 0.;
    let mut cached = Some(0.);
    let mut processed = Some(0.);
    let mut decoded = Some(0.);
    for slot in slots {
        let (Some(id), Some(processing), Some(n_ctx)) = (
            slot["id"].as_u64(),
            slot["is_processing"].as_bool(),
            slot["n_ctx"].as_u64(),
        ) else {
            return false;
        };
        if n_ctx == 0 || !ids.insert(id) {
            return false;
        }
        per_request_min = per_request_min.min(n_ctx);
        capacity += n_ctx as f64;
        context += slot["n_prompt_tokens"].as_u64().unwrap_or(0) as f64;
        if !processing {
            continue;
        }
        active += 1.;
        cached = cached
            .zip(slot["n_prompt_tokens_cache"].as_u64())
            .map(|(sum, n)| sum + n as f64);
        processed = processed
            .zip(slot["n_prompt_tokens_processed"].as_u64())
            .map(|(sum, n)| sum + n as f64);
        decoded = decoded
            .zip(slot["next_token"][0]["n_decoded"].as_u64())
            .map(|(sum, n)| sum + n as f64);
        if let (Some(task), Some(tokens)) = (
            slot["id_task"].as_u64(),
            slot["next_token"][0]["n_decoded"].as_u64(),
        ) {
            sample.gauges.insert(format!("slot_{id}_task"), task as f64);
            sample
                .gauges
                .insert(format!("slot_{id}_decoded"), tokens as f64);
        }
    }
    for (key, value) in [
        ("requests_active", active),
        ("slots_total", slots.len() as f64),
        ("context_capacity_tokens", capacity),
        // The aggregate cannot be divided safely when slots differ in size.
        // Clients use this minimum to admit a request into any native slot.
        ("context_per_request_tokens", per_request_min as f64),
        ("context_used_tokens", context),
    ] {
        sample.gauges.insert(key.into(), value);
    }
    let prompt = cached.zip(processed).map(|(c, p)| c + p);
    let hit_ratio = cached
        .zip(prompt)
        .filter(|(_, total)| *total > 0.)
        .map(|(cached, total)| cached / total);
    for (key, value) in [
        ("active_prompt_tokens", prompt),
        ("prompt_cache_hit_ratio", hit_ratio),
        ("active_cached_tokens", cached),
        ("active_output_tokens", decoded),
    ] {
        if let Some(value) = value {
            sample.gauges.insert(key.into(), value);
        }
    }
    true
}

#[cfg(test)]
mod observability_tests {
    use super::*;
    #[test]
    fn observability_reasoning_defaults_use_launch_precedence_and_exclude_other_settings() {
        let args = super::super::split_args(
            r#"--reasoning-budget 2048 --reasoning-effort=low --reasoning off --chat-template-kwargs '{"reasoning_effort":"high","enable_thinking":true,"private":"do-not-expose"}'"#,
        );
        let defaults = reasoning_defaults(&args, |name| match name {
            "LLAMA_ARG_THINK_BUDGET" => Some("99".into()),
            "LLAMA_ARG_REASONING_EFFORT" => Some("medium".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(defaults.budget_tokens, Some(2048));
        assert_eq!(defaults.effort, Some(serde_json::json!("high")));
        assert_eq!(defaults.thinking, Some(true));
        assert!(
            !serde_json::to_string(&defaults)
                .unwrap()
                .contains("do-not-expose")
        );
        assert_eq!(defaults.source, "worker_launch_configuration");
        assert!(reasoning_defaults(&[], |_| None).is_none());
        assert!(
            reasoning_defaults(
                &super::super::split_args("--reasoning-effort high --reasoning-effort default"),
                |_| None
            )
            .is_none()
        );
        let numeric = reasoning_defaults(
            &super::super::split_args(
                r#"--chat-template-kwargs '{"reasoning_effort":0.75}' --reasoning-budget=-1"#,
            ),
            |_| None,
        )
        .unwrap();
        assert_eq!(numeric.effort, Some(serde_json::json!(0.75)));
        assert_eq!(numeric.budget_tokens, Some(-1));
        assert!(numeric.thinking.is_none());
        let from_env = reasoning_defaults(&[], |name| {
            (name == "LLAMA_ARG_REASONING").then(|| "off".into())
        })
        .unwrap();
        assert_eq!(from_env.thinking, Some(false));
        assert!(from_env.effort.is_none());
    }
    #[test]
    fn observability_enables_slots_without_persistence() {
        for persistence in [None, Some(std::path::Path::new("/tmp/state"))] {
            let args = super::super::llama_server_args_with_state(
                crate::backend::LlamaCppMode::Cuda,
                std::path::Path::new("model.gguf"),
                None,
                1234,
                &Default::default(),
                &super::super::SupportedArgs {
                    slots: true,
                    ..Default::default()
                },
                false,
                persistence,
            );
            assert_eq!(args.iter().filter(|arg| *arg == "--slots").count(), 1);
        }
    }

    #[test]
    #[ignore = "requires WERK_TEST_LLAMA_URL pointing to a running llama-server"]
    fn observability_live_slots() {
        let url = std::env::var("WERK_TEST_LLAMA_URL").unwrap();
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let mut a = BackendSnapshot {
            instance: "live".into(),
            ..Default::default()
        };
        a.available = parse_slots(&mut a, &fetch_slots(&client, &url).unwrap());
        assert!(a.available);
        let start = std::time::Instant::now();
        std::thread::sleep(Duration::from_secs(2));
        let mut b = BackendSnapshot {
            instance: "live".into(),
            ..Default::default()
        };
        b.available = parse_slots(&mut b, &fetch_slots(&client, &url).unwrap());
        assert!(b.available);
        assert!(b.gauges["context_capacity_tokens"] > 0.);
        crate::ui_eprintln!(
            "live gauges: {:?}; rates: {:?}",
            b.gauges,
            crate::observability::Rates::between(&a, &b, start.elapsed().as_secs_f64())
        );
    }

    #[test]
    fn observability_missing_slot_counts_are_not_zeroes() {
        let mut sample = BackendSnapshot::default();
        assert!(parse_slots(
            &mut sample,
            &serde_json::json!([
                {"id":0,"id_task":42,"is_processing":true,"n_ctx":32768,"n_prompt_tokens":10}
            ])
        ));
        assert!(!sample.gauges.contains_key("active_cached_tokens"));
        assert!(!sample.gauges.contains_key("active_output_tokens"));
        assert!(!sample.gauges.contains_key("slot_0_decoded"));
    }

    #[test]
    fn native_slots_provide_live_counts_without_content() {
        let mut sample = BackendSnapshot::default();
        assert!(parse_slots(
            &mut sample,
            &serde_json::json!([
                {"id":0,"id_task":42,"is_processing":true,"n_ctx":32768,
                 "n_prompt_tokens":1234,"n_prompt_tokens_cache":1000,"n_prompt_tokens_processed":200,
                 "next_token":[{"n_decoded":34}],"prompt":"private","generated":"private"},
                {"id":1,"is_processing":false,"n_ctx":32768}
            ])
        ));
        assert_eq!(sample.gauges["requests_active"], 1.);
        assert_eq!(sample.gauges["context_capacity_tokens"], 65536.);
        assert_eq!(sample.gauges["context_per_request_tokens"], 32768.);
        assert_eq!(sample.gauges["active_output_tokens"], 34.);
        assert_eq!(sample.gauges["slot_0_task"], 42.);
        assert_eq!(sample.gauges["active_prompt_tokens"], 1200.);
        assert_eq!(sample.gauges["prompt_cache_hit_ratio"], 1000. / 1200.);
        assert!(!serde_json::to_string(&sample).unwrap().contains("private"));
        assert!(!parse_slots(
            &mut BackendSnapshot::default(),
            &serde_json::json!({"error":501})
        ));
    }

    #[test]
    fn per_request_context_is_minimum_not_average_and_requires_all_slots() {
        let mut sample = BackendSnapshot::default();
        assert!(parse_slots(
            &mut sample,
            &serde_json::json!([
                {"id":0,"is_processing":false,"n_ctx":8192},
                {"id":1,"is_processing":true,"n_ctx":32768}
            ])
        ));
        assert_eq!(sample.gauges["context_capacity_tokens"], 40960.);
        assert_eq!(sample.gauges["context_per_request_tokens"], 8192.);
        for slots in [
            serde_json::json!([]),
            serde_json::json!([{"id":0,"is_processing":false,"n_ctx":0}]),
            serde_json::json!([{"id":0,"is_processing":false,"n_ctx":8192},{"id":1,"is_processing":false}]),
            serde_json::json!([{"id":0,"is_processing":false,"n_ctx":8192},{"id":0,"is_processing":false,"n_ctx":8192}]),
        ] {
            let mut invalid = BackendSnapshot::default();
            assert!(!parse_slots(&mut invalid, &slots));
            assert!(!invalid.gauges.contains_key("context_per_request_tokens"));
        }
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::configured_slots;
    #[test]
    fn captured_parallel_arguments_respect_native_overrides_and_unknowns() {
        for (args, expected) in [
            (vec!["-np", "2"], Some(2)),
            (vec!["--parallel=4"], Some(4)),
            (vec!["-np", "2", "--parallel", "1"], Some(1)),
            (vec!["-np", "2", "--parallel=0"], None),
            (vec!["--parallel"], None),
            (vec![], None),
        ] {
            assert_eq!(
                configured_slots(&args.into_iter().map(str::to_owned).collect::<Vec<_>>()),
                expected
            );
        }
    }
}

//! Content-free snapshots of already running llama.cpp workers.
use super::LlamaServerBackend;
use crate::observability::BackendSnapshot;
use serde_json::Value;
use std::{io::Read, time::Duration};

pub(super) fn sample(backend: &LlamaServerBackend) -> Vec<BackendSnapshot> {
    // Never hold the registry or generation/state gate during monitoring I/O.
    let servers: Vec<_> = backend
        .servers
        .lock()
        .map(|servers| servers.values().cloned().collect())
        .unwrap_or_default();
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
        eprintln!(
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
}

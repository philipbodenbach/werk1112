use super::{
    http::{ProviderClient, ProviderError, read_json},
    media::*,
};
use crate::{
    model_store::ModelStore,
    model_support::{self, Provider, SupportStatus},
};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{Arc, Mutex},
};

type MockServer = (
    String,
    Arc<Mutex<Vec<(String, Value)>>>,
    std::thread::JoinHandle<()>,
);

pub(crate) fn mock(responses: Vec<(u16, &'static str, Vec<u8>)>) -> MockServer {
    mock_with(move |_| responses)
}
pub(crate) fn mock_with(
    build: impl FnOnce(&str) -> Vec<(u16, &'static str, Vec<u8>)>,
) -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let responses = build(&url);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = requests.clone();
    let thread = std::thread::spawn(move || {
        for (status, content_type, body) in responses {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut raw = Vec::new();
            let mut byte = [0];
            while !raw.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                raw.push(byte[0]);
            }
            let headers = String::from_utf8(raw).unwrap();
            let length = headers
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .and_then(|n| n.parse::<usize>().ok())
                })
                .unwrap_or(0);
            let mut bytes = vec![0; length];
            stream.read_exact(&mut bytes).unwrap();
            captured.lock().unwrap().push((
                headers,
                serde_json::from_slice(&bytes).unwrap_or(Value::Null),
            ));
            write!(stream,"HTTP/1.1 {status} status\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\nretry-after: 0\r\n\r\n",body.len()).unwrap();
            stream.write_all(&body).unwrap();
        }
    });
    (url, requests, thread)
}
#[test]
fn provider_registration_preserves_identity_and_never_claims_verification() {
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::resolve(Some(dir.path().into())).unwrap();
    for id in ["mistral-large-4", "flux-3-image", "eleven_v4"] {
        let model = model_support::register_remote(&store, id, None).unwrap();
        let loaded = store.get(id).unwrap();
        assert_eq!(loaded.source, model.source);
        assert_eq!(loaded.metadata.tasks, model.metadata.tasks);
        let data = serde_json::to_string(&loaded).unwrap();
        assert!(!data.contains("api_key"));
        assert!(model_support::register_remote(&store, id, None).is_err());
    }
    assert!(model_support::register_remote(&store, "mistral-large-latest", None).is_err());
    assert!(
        model_support::CATALOG
            .iter()
            .all(|m| m.local != SupportStatus::VerifiedLocal
                && m.remote != SupportStatus::VerifiedRemote)
    );
}
#[test]
fn flux3_rejects_flux1_parameters_and_validates_native_fields() {
    for field in [
        "seed",
        "width",
        "height",
        "negative_prompt",
        "output_format",
        "mode",
    ] {
        let mut value = json!({"prompt":"a bird"});
        value[field] = json!(42);
        assert!(
            serde_json::from_value::<FluxImageRequest>(value).is_err(),
            "{field}"
        );
    }
    let request: FluxImageRequest = serde_json::from_value(
        json!({"prompt":"a bird","resolution":"4k","aspect_ratio":"16:9","grounding":false}),
    )
    .unwrap();
    request.validate().unwrap();
    for value in [
        json!({"prompt":" "}),
        json!({"prompt":"x","images":[]}),
        json!({"prompt":"x","aspect_ratio":"17:8"}),
        json!({"prompt":"x","safety_tolerance":5}),
    ] {
        assert!(
            serde_json::from_value::<FluxImageRequest>(value)
                .unwrap()
                .validate()
                .is_err()
        );
    }
}
#[test]
fn speech_contract_preserves_voice_language_seed_and_raw_audio_mime() {
    let request:SpeechRequest=serde_json::from_value(json!({"inputs":[{"text":"Hallo","voice_id":"voice-one"},{"text":"Guten Tag","voice_id":"voice-two"}],"language_code":"de","settings":{"stability":0.5,"similarity":0.75},"seed":123,"output_format":"wav_24000","stream":true})).unwrap();
    assert_eq!(request.validate().unwrap(), "audio/wav");
    let body = request.body().unwrap();
    assert_eq!(body["model_id"], "eleven_v4");
    assert!(body.get("stream").is_none());
    assert!(body.get("output_format").is_none());
    assert_eq!(body["inputs"][1]["voice_id"], "voice-two");
    assert_eq!(audio_mime("pcm_24000").unwrap(), "audio/pcm");
    assert!(audio_mime("wav_99999").is_err());
    let request: SpeechRequest =
        serde_json::from_value(json!({"inputs":[{"text":"x".repeat(2001),"voice_id":"one"}]}))
            .unwrap();
    assert!(request.validate().is_err());
}
#[test]
fn transport_retries_rate_limits_but_never_replays_ambiguous_generation() {
    let (url, requests, thread) = mock(vec![
        (429, "application/json", b"{}".to_vec()),
        (200, "application/json", b"{\"ok\":true}".to_vec()),
    ]);
    let client = ProviderClient::for_test(Provider::Mistral, &url).unwrap();
    assert_eq!(
        read_json(
            client
                .post("/v1/chat/completions", &json!({"model":"mistral-large-4"}))
                .unwrap()
        )
        .unwrap()["ok"],
        true
    );
    thread.join().unwrap();
    assert_eq!(requests.lock().unwrap().len(), 2);
    let (url, requests, thread) = mock(vec![(
        503,
        "application/json",
        b"test-secret upstream internal".to_vec(),
    )]);
    let client = ProviderClient::for_test(Provider::Mistral, &url).unwrap();
    let e = client.post("/v1/chat/completions", &json!({})).unwrap_err();
    assert_eq!(e.downcast_ref::<ProviderError>().unwrap().status, Some(503));
    assert!(!format!("{e:?}").contains("test-secret"));
    thread.join().unwrap();
    assert_eq!(requests.lock().unwrap().len(), 1);
}
#[test]
fn provider_credentials_cannot_follow_redirects_or_foreign_poll_urls() {
    let client = ProviderClient::for_test(Provider::Bfl, "https://api.bfl.ai").unwrap();
    for url in [
        "http://api.bfl.ai/task",
        "https://bfl.ai.evil.test/task",
        "https://evil.test/task",
        "https://key@api.bfl.ai/task",
    ] {
        assert!(client.validate_poll_url(url).is_err());
    }
    assert!(
        client
            .validate_poll_url("https://api.eu1.bfl.ai/v1/get_result?id=x")
            .is_ok()
    );
    let (url, _, thread) = mock(vec![(302, "text/plain", b"test-secret".to_vec())]);
    let client = ProviderClient::for_test(Provider::ElevenLabs, &url).unwrap();
    assert!(client.post("/v1/text-to-dialogue", &json!({})).is_err());
    thread.join().unwrap();
}
#[test]
fn flux_polling_preserves_request_and_moderation_is_terminal() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    // Separate tiny server because the returned polling URL must include its port.
    let listener = TcpListener::bind(addr).unwrap();
    let url = format!("http://{addr}");
    let poll = format!("{url}/task/one");
    let server = std::thread::spawn(move || {
        for result in [
            json!({"id":"one","polling_url":poll}),
            json!({"status":"Content Moderated"}),
        ] {
            let (mut stream, _) = listener.accept().unwrap();
            let mut bytes = [0; 8192];
            let _ = stream.read(&mut bytes).unwrap();
            let body = result.to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        }
    });
    let client = ProviderClient::for_test(Provider::Bfl, &url).unwrap();
    let request = serde_json::from_value(json!({"prompt":"a bird"})).unwrap();
    let e = image_with(&client, &request).err().unwrap();
    assert_eq!(
        e.downcast_ref::<ProviderError>().unwrap().code,
        "content_moderated"
    );
    server.join().unwrap();
}

#[test]
fn flux_async_success_downloads_binary_without_credentials() {
    let png = b"\x89PNG\r\n\x1a\nmock-image".to_vec();
    let expected = png.clone();
    let (url, requests, server) = mock_with(|url| {
        vec![
        (200,"application/json",serde_json::to_vec(&json!({"id":"job","polling_url":format!("{url}/task"),"cost":4})).unwrap()),
        (200,"application/json",serde_json::to_vec(&json!({"status":"Ready","result":{"sample":format!("{url}/image"),"duration":2.0}})).unwrap()),
        (200,"image/png",png),
    ]
    });
    let client = ProviderClient::for_test(Provider::Bfl, &url).unwrap();
    let request = serde_json::from_value(
        json!({"prompt":"rabbit","images":["base64-reference"],"resolution":"2k"}),
    )
    .unwrap();
    let result = image_with(&client, &request).unwrap();
    assert_eq!(result.bytes, expected);
    assert_eq!(result.mime, "image/png");
    assert_eq!(result.metadata["cost"], 4);
    server.join().unwrap();
    let requests = requests.lock().unwrap();
    assert!(requests[0].0.contains("test-secret"));
    assert!(requests[1].0.contains("test-secret"));
    assert!(!requests[2].0.contains("test-secret"));
    assert_eq!(requests[0].1["images"], json!(["base64-reference"]));
}
#[test]
fn eleven_audio_and_stream_paths_preserve_binary_contract() {
    let bytes = vec![0, 255, 128, 1, 2, 3];
    for stream in [false, true] {
        let (url, requests, server) = mock(vec![(200, "audio/mpeg", bytes.clone())]);
        let client = ProviderClient::for_test(Provider::ElevenLabs, &url).unwrap();
        let request = serde_json::from_value(
            json!({"inputs":[{"text":"Hello","voice_id":"voice"}],"stream":stream}),
        )
        .unwrap();
        assert_eq!(
            super::http::read_bytes(speech_response_with(&client, &request).unwrap(), 1024)
                .unwrap(),
            bytes
        );
        server.join().unwrap();
        let requests = requests.lock().unwrap();
        assert!(requests[0].0.starts_with(if stream {
            "POST /v1/text-to-dialogue/stream?output_format=mp3_44100_128"
        } else {
            "POST /v1/text-to-dialogue?output_format=mp3_44100_128"
        }));
        assert_eq!(requests[0].1["model_id"], "eleven_v4");
    }
}

#[test]
fn stalled_provider_times_out_without_replaying_or_exposing_credentials() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .unwrap();
        let mut buf = [0; 4096];
        assert!(stream.read(&mut buf).unwrap() > 0);
        std::thread::sleep(std::time::Duration::from_millis(250));
        listener.set_nonblocking(true).unwrap();
        assert!(
            listener.accept().is_err(),
            "ambiguous POST must not be replayed"
        );
    });
    let mut client = ProviderClient::for_test(Provider::Mistral, &url).unwrap();
    client.request_timeout = std::time::Duration::from_millis(100);
    let error = client
        .post("/v1/chat/completions?private=test-secret", &json!({}))
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<ProviderError>().unwrap().code,
        "upstream_timeout"
    );
    assert!(!format!("{error:?}").contains("test-secret"));
    server.join().unwrap();
}

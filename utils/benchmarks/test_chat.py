import copy
import email.message
import http.client
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chat


def chunk(value):
    return "data: " + json.dumps(value) + "\n\n"


def successful_stream(text="Hello", usage=None):
    data = ": heartbeat\n\n"
    data += chunk({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    data += chunk({"choices": [{"index": 0, "delta": {"content": text}}]})
    data += chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if usage is not None:
        data += chunk({"choices": [], "usage": usage})
    return data + "data: [DONE]\n\n"


class FakeResponse(io.BytesIO):
    def __init__(self, data, content_type="text/event-stream"):
        super().__init__(data.encode())
        self.headers = email.message.Message()
        self.headers["Content-Type"] = content_type


class StreamTests(unittest.TestCase):
    def test_multiline_crlf_and_comments(self):
        lines = io.BytesIO(b": keepalive\r\nevent: message\r\ndata: {\r\ndata: \"a\": 1}\r\n\r\n")
        self.assertEqual(list(chat.sse_events(lines)), [("message", '{\n"a": 1}')])

    def test_usage_after_finish_and_single_chunk_is_not_one_token(self):
        usage = {"prompt_tokens": 27, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 20}, "completion_tokens_details": {"reasoning_tokens": 0}}
        clock_value = [0]

        def clock():
            clock_value[0] += 0.1
            return clock_value[0]

        result = chat.consume_stream(io.StringIO(successful_stream("Several words in one chunk.", usage)), 0, clock)
        self.assertIsNone(result["error"])
        self.assertEqual(result["completion_tokens"], 12)
        self.assertEqual(result["cached_tokens"], 20)
        self.assertIs(result["received_done"], True)
        self.assertIsNotNone(result["first_text_seconds"])
        self.assertIsNotNone(result["client_decode_tokens_per_second_estimate"])

    def test_missing_usage_remains_unknown(self):
        result = chat.consume_stream(io.StringIO(successful_stream()), 0, lambda: 1)
        self.assertIsNone(result["error"])
        self.assertIsNone(result["usage"])
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(result["cached_tokens"])
        self.assertIsNone(result["client_decode_tokens_per_second_estimate"])

    def test_reasoning_suppresses_client_decode_estimate(self):
        prefix = chunk({"choices": [{"delta": {"reasoning_content": "Let me think"}}]})
        result = chat.consume_stream(io.StringIO(prefix + successful_stream("Hi", {"completion_tokens": 100})), 0, lambda: 1)
        self.assertTrue(result["has_reasoning"])
        self.assertEqual(result["answer"], "Hi")
        self.assertIsNone(result["client_decode_tokens_per_second_estimate"])
        self.assertIsNone(chat.summarize_usage({"completion_tokens": 100, "completion_tokens_details": {"reasoning_tokens": 90}}, 1, 10)["client_decode_tokens_per_second_estimate"])

    def test_missing_reasoning_details_requires_explicit_estimate_opt_in(self):
        usage = {"completion_tokens": 54}
        result = chat.consume_stream(io.StringIO(successful_stream("12", usage)), 0, lambda: 1)
        self.assertFalse(result["has_reasoning"])
        self.assertIsNone(result["client_decode_tokens_per_second_estimate"])
        self.assertIsNone(chat.summarize_usage(usage, 1, 2)["client_decode_tokens_per_second_estimate"])
        self.assertEqual(chat.summarize_usage(usage, 1, 2, estimate_decode_rate=True)["client_decode_tokens_per_second_estimate"], 53)
        for details, detected in [({"reasoning_tokens": 52}, False), ({"reasoning_tokens": 0}, True)]:
            with self.subTest(details=details, detected=detected):
                measured = chat.summarize_usage({**usage, "completion_tokens_details": details}, 1, 2, detected, True)
                self.assertIsNone(measured["client_decode_tokens_per_second_estimate"])

    def test_request_propagates_decode_estimate_opt_in(self):
        with patch.object(chat.urllib.request, "urlopen", return_value=FakeResponse(successful_stream())), patch.object(chat, "consume_stream", return_value={}) as consumed:
            chat.request_chat("http://localhost/v1/chat/completions", "", {}, 10, 20, estimate_decode_rate=True)
        self.assertTrue(consumed.call_args.kwargs["estimate_decode_rate"])

    def test_incomplete_stream_preserves_partial_answer_and_error(self):
        stream = chunk({"choices": [{"delta": {"content": "Partial"}}]})
        result = chat.consume_stream(io.StringIO(stream), 0, lambda: 1)
        self.assertEqual(result["answer"], "Partial")
        self.assertIn("without [DONE]", result["error"])
        self.assertFalse(result["received_done"])

    def test_done_without_finish_is_error(self):
        result = chat.consume_stream(io.StringIO("data: [DONE]\n\n"), 0, lambda: 1)
        self.assertIn("finish reason", result["error"])

    def test_server_error_and_malformed_json(self):
        for data, message in [(chunk({"error": {"message": "failed"}}), "streaming error"), ("data: nope\n\n", "invalid JSON")]:
            with self.subTest(data=data):
                result = chat.consume_stream(io.StringIO(data), 0, lambda: 1)
                self.assertIn(message, result["error"])

    def test_connection_failure_and_total_deadline(self):
        def broken_connection():
            yield from io.StringIO(chunk({"choices": [{"delta": {"content": "Hi"}}]}))
            raise http.client.IncompleteRead(b"", 20)

        result = chat.consume_stream(broken_connection(), 0, lambda: 1)
        self.assertEqual(result["answer"], "Hi")
        self.assertIsNotNone(result["error"])
        result = chat.consume_stream(io.StringIO(successful_stream()), 0, lambda: 601)
        self.assertIn("deadline", result["error"])

    def test_error_echoing_secret_is_redacted(self):
        secret = 'secret-with-"quote'
        data = chunk({"error": {"message": "received " + secret}})
        with patch.object(chat.urllib.request, "urlopen", return_value=FakeResponse(data)) as opened:
            result = chat.request_chat("http://localhost/v1/chat/completions", secret, {}, 10, 20)
        request = opened.call_args.args[0]
        self.assertEqual(request.headers["Authorization"], "Bearer " + secret)
        self.assertNotIn(secret, json.dumps(result))
        self.assertIn("[REDACTED]", result["error"])


class FixtureTests(unittest.TestCase):
    def test_json_check_rejects_markdown_extra_fields_and_wrong_type(self):
        checks = [{"type": "json_equals", "value": {"safe": True}}]
        self.assertTrue(chat.check_answer('{"safe":true}', checks)[0]["passed"])
        for answer in ['```json\n{"safe":true}\n```', '{"safe":1}', '{"safe":true,"extra":1}']:
            self.assertFalse(chat.check_answer(answer, checks)[0]["passed"])

    def test_builtin_fixtures_are_valid(self):
        cases = chat.load_cases(Path(chat.__file__).with_name("fixtures.json"))
        self.assertEqual(len(cases), 5)
        self.assertTrue(any(len(case["turns"]) > 1 for case in cases))

    def test_main_replays_history_and_resets_repetitions(self):
        requests = []

        def fake_request(url, secret, payload, timeout, deadline, estimate_decode_rate=False):
            requests.append(copy.deepcopy(payload))
            answer = "Cedar uses Rust." if len(payload["messages"]) == 1 else '{"project":"Cedar","language":"Rust"}'
            return chat.consume_stream(io.StringIO(successful_stream(answer, {"completion_tokens": 5})), 0, lambda: 1)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(chat, "request_chat", side_effect=fake_request), patch("sys.stdout", new=io.StringIO()):
                status = chat.main(["--model", "test", "--case", "conversation", "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(status, 0)
            self.assertEqual(len(report["samples"]), 4)
            self.assertEqual([len(item["messages"]) for item in requests], [1, 3, 1, 3])
            self.assertEqual(requests[1]["messages"][1], {"role": "assistant", "content": "Cedar uses Rust."})
            self.assertEqual(requests[0]["stream_options"], {"include_usage": True})
            self.assertEqual(requests[0]["temperature"], 0)
            self.assertEqual(requests[0]["top_p"], 0.95)
            self.assertEqual(requests[0]["seed"], 42)
            self.assertEqual([sample["sample_kind"] for sample in report["samples"]], ["first", "first", "repeat", "repeat"])
            self.assertEqual(list(Path(directory).iterdir()), [output])
            self.assertEqual(report["measurement_options"], {"estimate_decode_rate": False})

    def test_sampling_profiles_match_payload_and_report(self):
        profiles = [
            (["--temperature", "0.6", "--top-p", "0.8", "--seed", "7"],
             {"temperature": 0.6, "top_p": 0.8, "seed": 7}),
            (["--temperature", "1", "--top-p", "1", "--unseeded"],
             {"temperature": 1, "top_p": 1}),
        ]
        for arguments, expected in profiles:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "report.json"
                result = chat.consume_stream(io.StringIO(successful_stream()), 0, lambda: 1)
                with patch.object(chat, "request_chat", return_value=result) as requested, patch("sys.stdout", new=io.StringIO()):
                    status = chat.main(["--model", "test", "--case", "rust-english", "--repeats", "1", "--output", str(output), *arguments])
                self.assertEqual(status, 0)
                payload = requested.call_args.args[2]
                controls = {key: payload[key] for key in ("temperature", "top_p", "seed") if key in payload}
                self.assertEqual(controls, expected)
                self.assertEqual(json.loads(output.read_text())["settings"], {**expected, "repeats": 1})

    def test_cli_decode_estimate_opt_in_is_recorded_and_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            result = chat.consume_stream(io.StringIO(successful_stream()), 0, lambda: 1)
            with patch.object(chat, "request_chat", return_value=result) as requested, patch("sys.stdout", new=io.StringIO()):
                status = chat.main(["--model", "test", "--case", "rust-english", "--repeats", "1", "--estimate-decode-rate", "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertTrue(requested.call_args.kwargs["estimate_decode_rate"])
            self.assertEqual(json.loads(output.read_text())["measurement_options"], {"estimate_decode_rate": True})

    def test_invalid_sampling_arguments_fail_before_http(self):
        invalid = [
            ["--temperature", "-0.1"], ["--temperature", "nan"], ["--temperature", "inf"],
            ["--top-p", "0"], ["--top-p", "1.01"], ["--top-p", "nan"], ["--top-p", "inf"],
            ["--seed", "1.5"], ["--seed", "42", "--unseeded"],
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), patch.object(chat, "request_chat") as requested, patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    chat.main(["--model", "test", "--output", "/unused/report.json", *arguments])
                self.assertEqual(raised.exception.code, 2)
                requested.assert_not_called()


if __name__ == "__main__":
    unittest.main()

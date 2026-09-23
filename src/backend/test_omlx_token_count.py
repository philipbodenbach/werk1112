"""Native token counting contract without loading oMLX or model weights."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("token_supervisor", Path(__file__).with_name("omlx_supervisor.py"))
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


class NativeCountTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.releases = 0
        self.calls = []
        self.count = 37
        owner = self

        class Lease:
            async def release(self):
                owner.releases += 1

        class Request:
            @staticmethod
            def model_validate(body):
                return SimpleNamespace(**dict(dict(chat_template_kwargs=None,
                    reasoning_effort=None, tools=None, tool_choice=None), **copy.deepcopy(body)))

        def count(messages, tools, **kwargs):
            self.calls.append((messages, tools, kwargs))
            if isinstance(self.count, Exception):
                raise self.count
            return self.count

        self.engine = SimpleNamespace(tokenizer=object(), count_chat_tokens=count)

        async def get_engine(model, lease):
            self.assertEqual(model, "physical-model")
            return self.engine

        async def noop(*args):
            pass

        self.server = SimpleNamespace(ChatCompletionRequest=Request, _LLMEngineLease=Lease,
            get_engine_for_model=get_engine, _raise_if_llm_lease_abort_requested=noop,
            VLMBatchedEngine=type("VLM", (), {}), _server_state=SimpleNamespace(mcp_manager=None),
            get_model_settings_for_request=lambda model: None,
            merge_chat_template_request_kwargs=lambda settings, kwargs: kwargs,
            merge_reasoning_effort_chat_template_kwargs=lambda kwargs, effort: kwargs,
            extract_text_content=lambda messages, *args, **kwargs: messages,
            detect_and_strip_partial=lambda messages: False,
            convert_tools_for_template=lambda tools: tools,
            _ensure_tokenizer_for_system_probe=noop,
            prepare_system_messages_for_template=lambda messages, *args, **kwargs: messages,
            _unsupported_mid_system_policy=lambda: "error")
        self.body = dict(model="physical-model", messages=[dict(role="user", content="hello")],
            chat_template_kwargs=dict(enable_thinking=False),
            tools=[dict(type="function", function=dict(name="add", parameters=dict(type="object")))])

    async def test_native_count_preserves_template_controls_and_tools_without_generate(self):
        result = await supervisor.count_chat_tokens(self.server, self.body)
        self.assertEqual(result, dict(input_tokens=37))
        messages, tools, kwargs = self.calls[0]
        self.assertEqual(messages, self.body["messages"])
        self.assertEqual(tools, self.body["tools"])
        self.assertEqual(kwargs, dict(chat_template_kwargs=dict(enable_thinking=False), is_partial=False))
        self.assertEqual(self.releases, 1)

    async def test_tool_choice_none_excludes_tools_from_template(self):
        self.body["tool_choice"] = "none"
        await supervisor.count_chat_tokens(self.server, self.body)
        self.assertIsNone(self.calls[0][1])

    async def test_tokenizer_failures_release_the_lease_and_never_estimate(self):
        self.count = ValueError("template failed")
        with self.assertRaisesRegex(ValueError, "template failed"):
            await supervisor.count_chat_tokens(self.server, self.body)
        self.assertEqual(self.releases, 1)

    async def test_noninteger_or_negative_counts_are_rejected(self):
        for count in (-1, 1.5, True, None):
            self.count = count
            with self.assertRaisesRegex(ValueError, "invalid token count"):
                await supervisor.count_chat_tokens(self.server, self.body)
        self.assertEqual(self.releases, 4)

    async def test_unverified_multimodal_counter_is_rejected(self):
        self.engine.supports_multimodal_fallback = True
        with self.assertRaisesRegex(ValueError, "multimodal"):
            await supervisor.count_chat_tokens(self.server, self.body)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.releases, 1)


if __name__ == "__main__":
    unittest.main()

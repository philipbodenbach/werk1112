#!/usr/bin/env python3
"""Official-SDK contract test, usable against the Rust fixture or a real Werk model.

Install the tested dependency from tests/anthropic-requirements.txt.
Tools are deliberately local integer addition; no shell/network tool execution.
"""
import argparse
import json
import time

import anthropic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--api-key", default="werk-local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--catalog-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    assert anthropic.__version__ == "0.86.0", "Use tests/anthropic-requirements.txt"
    assert 1 <= args.catalog_size <= 256
    client = anthropic.Anthropic(base_url=args.base_url, api_key=args.api_key,
                                 max_retries=0, timeout=600)
    count = client.messages.count_tokens(model=args.model,
        messages=[{"role":"user","content":"Say hello in one sentence."}])
    assert count.input_tokens > 0
    if args.fixture:
        assert count.input_tokens == 23
    print(json.dumps({"count_tokens":count.input_tokens}))
    schema = {"type": "object", "properties": {"a": {"type": "integer"},
              "b": {"type": "integer"}}, "required": ["a", "b"]}
    tools = [{"name": "add", "description": "Add two integers. Use this for addition.",
              "input_schema": schema}]
    tools.extend({"name": f"unused_{i}", "description": "Reserved; do not use.",
                  "input_schema": schema} for i in range(1, args.catalog_size))

    def send(messages, streaming, with_tools=False):
        kwargs = dict(model=args.model, max_tokens=args.max_tokens, temperature=0,
                      messages=messages)
        if with_tools:
            kwargs.update(tools=tools, tool_choice={"type": "auto"})
        start = time.monotonic()
        first_text = None
        event_names = []
        if streaming:
            with client.messages.stream(**kwargs) as stream:
                for event in stream:
                    event_names.append(event.type)
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        if first_text is None:
                            first_text = time.monotonic() - start
                result = stream.get_final_message()
            assert event_names[0] == "message_start"
            assert event_names[-1] == "message_stop"
        else:
            result = client.messages.create(**kwargs)
            assert result._request_id.startswith("req_werk_")
        assert result.type == "message" and result.role == "assistant"
        assert result.model == args.model
        assert result.usage.input_tokens > 0 and result.usage.output_tokens > 0
        if args.fixture:
            expected = (17, 8) if result.stop_reason == "tool_use" else (2, 1)
            assert (result.usage.input_tokens, result.usage.output_tokens) == expected
        print(json.dumps({"stream": streaming, "tools": with_tools,
                          "stop_reason": result.stop_reason,
                          "seconds": round(time.monotonic() - start, 4),
                          "first_text_seconds": first_text,
                          "usage": result.usage.model_dump(exclude_none=True)}, ensure_ascii=False))
        return result

    for streaming in (False, True):
        text = send([{"role": "user", "content": "Say hello in one sentence."}], streaming)
        assert text.stop_reason == "end_turn"
        assert any(block.type == "text" and block.text for block in text.content)
        prompt = ("Use the add tool to calculate 2+3 and 5+6. Call it for both sums. "
                  "After receiving the tool results, give a brief final answer.")
        history = [{"role": "user", "content": prompt}]
        seen_ids = set()
        pairs = set()
        completed_rounds = 0
        for _ in range(12):
            reply = send(history, streaming, with_tools=True)
            assert reply.stop_reason != "max_tokens", "Incomplete output; increase --max-tokens"
            history.append({"role": "assistant", "content": [
                block.model_dump(exclude_none=True) for block in reply.content]})
            if reply.stop_reason == "end_turn":
                assert {(2, 3), (5, 6)} <= pairs, "Model did not call add for both sums"
                completed_rounds += 1
                if completed_rounds == 2:
                    break
                pairs.clear()
                history.append({"role": "user", "content": prompt})
                continue
            assert reply.stop_reason == "tool_use"
            results = []
            for block in reply.content:
                if block.type != "tool_use":
                    continue
                assert block.id not in seen_ids
                seen_ids.add(block.id)
                assert block.name == "add", "Unexpected tool; nothing executed"
                a, b = block.input["a"], block.input["b"]
                assert type(a) is int and type(b) is int
                pairs.add((a, b))
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(a + b)})
            assert results
            history.append({"role": "user", "content": results})
        else:
            raise AssertionError("Model did not finish two tool rounds within twelve turns")
    client.close()
    print(f"PASS: anthropic {anthropic.__version__}, text/stream/client tool loop, catalog={args.catalog_size}")


if __name__ == "__main__":
    main()

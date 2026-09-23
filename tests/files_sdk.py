"""Official SDK contracts against an already-running local Werk server."""
import argparse
import anthropic
import openai

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", required=True)
parser.add_argument("--api-key", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()
oa = openai.OpenAI(base_url=args.base_url + "/v1", api_key=args.api_key, max_retries=0)
an = anthropic.Anthropic(base_url=args.base_url, api_key=args.api_key, max_retries=0)
ids = []
try:
    first = oa.files.create(file=("sdk.txt", b"Document number 42", "text/plain"), purpose="user_data")
    ids.append(first.id)
    assert oa.files.retrieve(first.id).bytes == 18
    assert oa.files.content(first.id).read() == b"Document number 42"
    assert first.id in [f.id for f in oa.files.list(limit=1)]
    meta = an.beta.files.retrieve_metadata(first.id)
    assert meta.filename == "sdk.txt" and meta.size_bytes == 18
    second = an.beta.files.upload(file=("anthropic.txt", b"Second document", "text/plain"))
    ids.append(second.id)
    assert set(ids) <= {f.id for f in an.beta.files.list(limit=1)}
    result = an.messages.create(model=args.model, max_tokens=32,
        messages=[{"role":"user","content":[{"type":"document","source":{"type":"file","file_id":first.id}},
            {"type":"text","text":"What number appears in the document?"}]}])
    assert result.type == "message"
    reply = oa.chat.completions.create(model=args.model, max_tokens=32,
        messages=[{"role":"user","content":[{"type":"file","file":{"file_id":second.id}},
            {"type":"text","text":"Summarize this document."}]}])
    assert reply.choices
finally:
    for file_id in ids:
        oa.files.delete(file_id)
print("OpenAI and Anthropic SDK file/reference contracts passed")

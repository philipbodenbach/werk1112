# Documents and file references

Document processing belongs to the shared HTTP request layer. Both OpenAI
Chat Completions and Anthropic Messages expand documents before prompt building
and backend execution. Anthropic token counting uses the same expansion.
This works with every text-generation backend, including CUDA, vLLM,
llama.cpp, MLX and oMLX; no GLM/Qwen-specific document adapter is involved.
It does not add document inputs to unrelated image-generation or audio routes.

Text extraction runs on the CPU. Normal requests without documents do not
start Python or read file storage. The inference kernels and expert-offload
implementation are unchanged. Hardware performance still needs console testing.

## Formats and representation

| Input | What reaches the model | Requirements |
| --- | --- | --- |
| UTF-8 text, Markdown, CSV, JSON, source code | Original text | Any text model; no Python |
| HTML/XML as UTF-8 | Original markup, not a rendered webpage | Any text model; no Python |
| PDF | Extracted text, with page labels; optionally rendered page images | Python and pypdf; rendering additionally needs pypdfium2/Pillow |
| Scanned PDF | Rendered pages for vision, or explicit local OCR text | Vision model/backend or Tesseract OCR |
| DOCX | Body, footnotes, endnotes, headers and footers as text | Python standard library |
| PPTX | Slide text in presentation order | Python standard library |
| XLSX | Sheets and cell references/values; formulas shown with cached results | Python standard library |
| OpenDocument | Extracted content text | Python standard library |

Office documents do not render embedded pictures, charts or layout. Spreadsheet
formulas/macros are never executed; a cached result may be stale. DOC/PPT/XLS,
encrypted PDFs, arbitrary ZIP archives and unsupported binary formats fail
explicitly. A PDF page with no text fails in text mode unless OCR produces text;
this also applies to blank pages. There is no retrieval index or automatic
chunk-and-summarize operation: the expanded input must fit the context.

`werk.documents` is a **local Werk option** accepted by both chat APIs and the
count endpoint:

```json
{"werk":{"documents":{"mode":"auto","ocr":false}}}
```

- `auto` (default): PDF text plus images when the selected model/backend reports
  available image understanding; otherwise PDF text only.
- `text`: extracted text for any text model, including CUDA models.
- `vision`: requires available image understanding; PDFs include text and images.
- `ocr: true`: additionally render PDF pages and run Tesseract. OCR remains
  usable with text-only models. It is never enabled implicitly.

The prompt labels each document, page and representation. Text mode explicitly
labels omitted visual content. A model declaring image understanding alone is
insufficient when its selected runtime cannot process images. Office files stay
textual in every mode.

Install optional PDF dependencies in a separate environment, then select it
when starting your normal server (no model installation is involved):

```bash
python3 -m venv .venv-documents
.venv-documents/bin/python -m pip install -r docs/document-requirements.txt
env WERK_DOCUMENT_PYTHON="$PWD/.venv-documents/bin/python" \
  werk serve --model YOUR_MODEL
```

On Windows use the environment's `Scripts/python.exe`. Office extraction needs
only a Python 3.10+ interpreter. OCR requires separately installed Tesseract and
its language files. `WERK_DOCUMENT_TESSERACT` selects the executable;
`WERK_DOCUMENT_OCR_LANG` selects languages (default `eng`, for example `deu+eng`).

## Upload once, reference through either API

Use a generated Werk API key in place of `YOUR_WERK_KEY`. OpenAI upload:

```bash
curl -fsS http://127.0.0.1:11434/v1/files \
  -H 'Authorization: Bearer YOUR_WERK_KEY' \
  -F purpose=user_data -F 'file=@/absolute/path/report.pdf;type=application/pdf'
```

Copy the returned `id` into either request. OpenAI:

```bash
curl -fsS http://127.0.0.1:11434/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_WERK_KEY' -H 'Content-Type: application/json' \
  -d '{"model":"YOUR_MODEL","max_tokens":512,
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Summarize this document."},
      {"type":"file","file":{"file_id":"file_RETURNED_ID"}}
    ]}],"werk":{"documents":{"mode":"text"}}}'
```

Anthropic, referencing the **same file and key**:

```bash
curl -fsS http://127.0.0.1:11434/v1/messages \
  -H 'x-api-key: YOUR_WERK_KEY' -H 'anthropic-version: 2023-06-01' \
  -H 'Content-Type: application/json' \
  -d '{"model":"YOUR_MODEL","max_tokens":512,
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Summarize this document."},
      {"type":"document","source":{"type":"file","file_id":"file_RETURNED_ID"}}
    ]}],"werk":{"documents":{"mode":"text"}}}'
```

An Anthropic upload uses the same `/v1/files`, `x-api-key` and
`anthropic-version` headers, and only the `file` multipart field (no `purpose`).
Legacy SDKs can additionally send `anthropic-beta: files-api-2025-04-14`.
That beta selects the legacy pagination/metadata shape; other betas fail.

Inline alternatives:

- OpenAI `{"type":"file","file":{"filename":"report.pdf","file_data":"data:application/pdf;base64,..."}}`;
  bare Base64 is also accepted. Exactly one of `file_id` and `file_data` is required.
- Anthropic `document.source`: `base64` with `media_type`/`data`, `text` with
  `media_type: text/plain`/`data`, `url` with an HTTP(S) URL, or `content` with
  a string/text-block array. Optional `title` and `context` are preserved.
- Documents also work inside tool results. File references belong to document
  blocks; image-source `file` references and image blocks inside document
  `content` are not implemented. Enabled provider document citations are rejected.

URLs are bounded, redirect-checked and resolved to pinned public addresses.
Local/private addresses, credentials in URLs and non-HTTP schemes are rejected;
upload a local file instead. No arbitrary server filesystem paths are accepted.

## File lifecycle and limits

| Operation | Route |
| --- | --- |
| Upload / list | `POST /v1/files` / `GET /v1/files` |
| Metadata / delete | `GET /v1/files/{id}` / `DELETE /v1/files/{id}` |
| Download | `GET /v1/files/{id}/content` |

The Anthropic headers select that wire contract; otherwise responses use OpenAI
file shapes. OpenAI lists accept `limit`, `order`, `after` and `purpose`.
Anthropic lists accept `limit`/`page` and return `next_page`; the legacy beta
accepts `after_id`/`before_id` and returns `has_more`, `first_id`, `last_id`.
Unsupported query parameters fail explicitly. Uploaded files are downloadable
through OpenAI; Anthropic metadata reports `downloadable: false` and that
contract rejects downloading input files, consistent with its upload semantics.

Files persist under `<werk-home>/api-files`, isolated by authenticated API key.
Bearer and `x-api-key` with the same key address the same files. Changing keys
does not grant access to another key's files. Unauthenticated clients share a
single namespace. File IDs survive server restarts. Nothing is uploaded to an
external provider. Upload `purpose` is metadata, not a training/batch service.

- 50 MiB per file and total document source bytes per request; 16 documents.
- 256 files / 256 MiB per key, 1024 files / 1 GiB globally. Full storage rejects
  new uploads; it never evicts live files to make room.
- OpenAI `expires_after[anchor]=created_at` and
  `expires_after[seconds]=3600..2592000` set expiry. `batch` defaults to 30 days;
  other uploads persist until deletion. Expiry is checked on access and reclaimed
  during inventory/upload operations. There is no background expiry timer.
- At most two concurrent document preparations and four combined uploads/downloads.
- 200 PDF pages, sheets or slides; 8 MiB extracted text; 64 MiB combined expanded
  content; PDF pages render at most 1600 pixels on the longest side.
- Worker timeout 90 seconds, OCR timeout 15 seconds per page; ZIP expansion and
  parser output are bounded. Linux/Windows workers additionally have a 2 GiB
  address-space/job bound; macOS relies on input/parser limits and timeout.

There is no persistent decoded-document cache. PDF/Office intermediate data
stays in a short-lived worker process. Request cancellation terminates that
process group/job, including OCR children. In-progress bounded blocking file/URL
reads retain admission until they return. OS DNS lookup latency remains
platform-dependent. Stale upload staging directories are removed on inventory;
successful uploads remain until deletion/expiry.

```bash
curl -fsS -X DELETE http://127.0.0.1:11434/v1/files/file_RETURNED_ID \
  -H 'Authorization: Bearer YOUR_WERK_KEY'
```

Deleting an upload does not erase text already included in a client's history
or the backend's existing prompt cache; those retain their own lifecycle.

## Verification

HTTP tests cover cross-protocol references, key isolation, restart persistence,
download admission/disconnect cleanup, prompt order, native-count parity and
GGUF/SafeTensors/MLX layouts. PDF extraction/rendering and Office/OCR tests use
generated fixtures; OCR subprocess results are mocked (no Tesseract installed).
Layout tests use a capture backend, not CUDA hardware.

```bash
python3 -m unittest discover -s src -p test_document_worker.py -v
cargo test --locked --offline api:: --lib
cargo test --locked --offline file_store:: --lib
cargo test --locked --offline real_pdf_is_expanded --lib -- --ignored
python3 -m pip install -r tests/files-requirements.txt
env WERK_TEST_ANTHROPIC_PYTHON=python3 cargo test --locked --offline \
  official_anthropic_sdk_text_stream_and_tool_loop --lib -- --ignored
```

The PDF HTTP test requires the optional dependencies. SDK tests use a local
fixture server and both pinned official SDKs; they do not load a real model.

Provider references: [OpenAI file inputs](https://developers.openai.com/api/docs/guides/file-inputs),
[OpenAI uploads](https://developers.openai.com/api/reference/resources/files/methods/create),
[Anthropic Files](https://platform.claude.com/docs/en/build-with-claude/files).
These describe the wire formats, not a promise of identical local capabilities.

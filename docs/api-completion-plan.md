# Local API coverage work

Constraints: provider wire semantics and Werk runtime settings stay separate.
Documents and file references are shared infrastructure above model/backend
routing, including CUDA/vLLM/llama.cpp, not model-specific adapters. Ordinary
text requests must not load document runtimes. Existing decode/offload kernels
remain unchanged; real-model performance acceptance is performed separately.

Implementation sequence:

1. Persistent, principal-scoped, bounded file storage; upload/list/retrieve/
   download/delete contracts for OpenAI and Anthropic on the same listener.
2. Shared document expansion for inline inputs and stored references, including
   plain text, PDF and office documents. Preserve order and page/source identity;
   expose text/visual processing explicitly and clean temporary resources.
3. Connect both chat adapters and native token counting to the same expansion.
4. Strict validation of unsupported API fields; structured output and sampling
   controls using explicit backend capabilities; matched-stop metadata and
   richer result transport where supported.
5. SDK/HTTP contracts, malformed inputs, access isolation, storage limits,
   cancellation and shared OpenAI/Anthropic regression tests; document actual
   capability coverage and remaining provider-only boundaries.

No claim of full provider-service parity follows from endpoint availability.

Implemented: all five steps above, with backend capability restrictions made
explicit in [the API matrix](api.md#post-v1chatcompletions) and format/lifecycle
details in [Documents and files](integrations/documents.md). Files and document
extraction are backend-independent. Native extended output currently targets
vLLM, llama.cpp server and oMLX; matched client stop sequences require vLLM's
actual metadata. Other runtimes reject extended fields without affecting their
document/text support. Provider-only services remain separate.

Verification results are recorded in
[Anthropic client verification](integrations/anthropic-clients.md#document-and-extended-output-verification-2026-09-23).
Real-model console performance acceptance and CUDA hardware execution remain
outside these fixture tests.

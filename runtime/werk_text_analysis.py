"""Werk's resident CUDA/CPU pooling and decision worker (JSONL transport).

One process owns one model/runtime. Rust discards failed processes before fallback,
so vLLM allocations and CUDA graphs cannot leak into the replacement worker.
"""
import contextlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
_loaded = None


class InputError(ValueError):
    pass


def check_lengths(tokenizer, texts, opts, pairs=None):
    if opts.get("truncate", False):
        return
    encoded = tokenizer(texts, text_pair=pairs, truncation=False)
    if any(len(ids) > opts["max_length"] for ids in encoded["input_ids"]):
        raise InputError(f"Input exceeds max_length={opts['max_length']} tokens. Split the text, increase werk.max_length within the model limit, or explicitly set werk.truncate=true")


def probe():
    import torch
    from packaging.version import Version
    versions = {}
    for package in ("transformers", "sentence-transformers", "laya", "pillow", "torchvision"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    ready = {
        "xlm-roberta": bool(versions["transformers"]),
        "laya": bool(versions["laya"] and Version(versions["laya"]) >= Version("0.3.29")),
        "embedding_gemma2": bool(versions["transformers"] and Version(versions["transformers"]) >= Version("5.19")
            and versions["sentence-transformers"] and Version(versions["sentence-transformers"]) >= Version("6.1")
            and versions["pillow"] and versions["torchvision"]),
    }
    return {"ok": True, "cuda": torch.cuda.is_available(), "architectures": ready,
            "torch": torch.__version__,
            "packages": versions,
            "vllm": importlib.util.find_spec("vllm") is not None,
            "tilelang": importlib.util.find_spec("tilelang") is not None,
            "wsl": "microsoft" in platform.release().lower()}


def load(request):
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("WERK_TEXT_THREADS", min(8, os.cpu_count() or 1)))))
    path = request["model"]
    arch, runtime, device = (request[k] for k in ("architecture", "runtime", "device"))
    opts = request["options"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install CUDA-enabled torch or select device=auto/cpu")
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if device == "cuda" and arch != "embedding_gemma2" else torch.float32)
    if runtime == "vllm":
        if device != "cuda":
            raise RuntimeError("Werk's vLLM pooling path requires CUDA; use transformers for CPU")
        from vllm import LLM
        model = LLM(model=path, runner="pooling", dtype=str(dtype).removeprefix("torch."),
                    trust_remote_code=False, max_model_len=opts["max_length"],
                    gpu_memory_utilization=0.35, enforce_eager=False)
        return {"model": model, "dtype": str(dtype), "device": device}
    if arch == "xlm-roberta":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, torch_dtype=dtype,
            attn_implementation="eager" if runtime == "transformers-eager" else "sdpa",
        ).to(device).eval()
        if request["task"] == "text-reranking" and model.config.num_labels != 1:
            raise ValueError("Reranking requires a single relevance score; this checkpoint has multiple labels")
        return {"model": model, "tokenizer": tokenizer, "dtype": str(dtype), "device": device,
                "max_length": min(opts["max_length"], model.config.max_position_embeddings - model.config.pad_token_id - 1)}
    if arch == "embedding_gemma2":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(path, device=device, local_files_only=True,
            model_kwargs={"torch_dtype": dtype, "attn_implementation": "eager" if runtime == "transformers-eager" else "sdpa"},
            config_kwargs={"vision_config": None, "audio_config": None})
        model.max_seq_length = opts["max_length"]
        return {"model": model, "dtype": str(dtype), "device": device}
    if arch == "laya":
        import laya
        checkpoint = opts["checkpoint"]
        if checkpoint == "auto":
            checkpoint = "multilingual" if (Path(path) / "multilingual").is_dir() else "english"
        local = Path(path) if checkpoint == "english" else Path(path) / checkpoint
        if not local.is_dir():
            raise ValueError(f"Laya checkpoint {checkpoint!r} missing at {local}; download the complete repository")
        agent = laya.load(str(local), device=device, fast=runtime == "laya-fast")
        actual = str(agent.device)
        if device == "cuda" and not actual.startswith("cuda"):
            raise RuntimeError("Laya silently fell back to CPU; CUDA request refused. Free VRAM or permit device=auto")
        if runtime == "laya-fast" and getattr(agent, "_fast", None) is None:
            raise RuntimeError("Laya TileLang acceleration did not activate; install a compatible laya[fast]/CUDA toolchain")
        return {"model": agent, "dtype": str(agent.dtype), "device": actual, "checkpoint": checkpoint}
    raise ValueError(f"Unsupported text-analysis architecture {arch}; use a supported checkpoint")


def checked_numbers(values):
    if not all(math.isfinite(x) for row in values for x in (row if isinstance(row, list) else [row])):
        raise RuntimeError("Model returned non-finite scores/embeddings. EmbeddingGemma 2 requires BF16 or FP32; update the runtime and check checkpoint integrity")
    return values


def infer(request, loaded):
    import torch
    body, opts = request["payload"], dict(request["options"])
    opts["max_length"] = loaded.get("max_length", opts["max_length"])
    task, runtime = request["task"], request["runtime"]
    model, device = loaded["model"], loaded["device"]
    warnings = []
    if task == "text-reranking":
        docs, query = body["documents"], body["query"]
        if runtime == "vllm":
            if opts.get("truncate", False):
                raise InputError("Explicit truncation is supported by backend=transformers or candle; vLLM pooling requires inputs within werk.max_length")
            check_lengths(model.get_tokenizer(), [query] * len(docs), opts, docs)
            outputs = model.score(query, docs, use_tqdm=False)
            scores = [float(output.outputs.score) for output in outputs]
            tokens = sum(len(output.prompt_token_ids) for output in outputs)
        else:
            tokenizer = loaded["tokenizer"]
            scores, tokens = [], 0
            for offset in range(0, len(docs), opts["batch_size"]):
                batch = docs[offset:offset + opts["batch_size"]]
                check_lengths(tokenizer, [query] * len(batch), opts, batch)
                encoded = tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                                    max_length=opts["max_length"], return_tensors="pt")
                tokens += int(encoded["attention_mask"].sum())
                with torch.inference_mode():
                    logits = model(**encoded.to(device)).logits.float().view(-1)
                    scores.extend(torch.sigmoid(logits).cpu().tolist())
        checked_numbers(scores)
        results = [{"index": i, "relevance_score": score} for i, score in enumerate(scores)]
        results.sort(key=lambda entry: entry["relevance_score"], reverse=True)
        if body.get("return_documents", False):
            for entry in results:
                entry["document"] = {"text": docs[entry["index"]]}
        return {"results": results[:body.get("top_n", len(results))],
                "usage": {"total_tokens": tokens}}, warnings
    if task == "text-embedding":
        texts = body["input"]
        if isinstance(texts, str):
            texts = [texts]
        prefix = "task: search result | query: " if body.get("input_type", "document") == "query" else "title: none | text: "
        prompts = [prefix + text for text in texts]
        dimensions = body.get("dimensions", 768)
        if runtime == "vllm":
            if opts.get("truncate", False):
                raise InputError("Explicit truncation is supported by backend=transformers; vLLM pooling requires inputs within werk.max_length")
            check_lengths(model.get_tokenizer(), prompts, opts)
            outputs = model.embed(prompts, use_tqdm=False)
            vectors = torch.tensor([output.outputs.embedding for output in outputs], dtype=torch.float32)
            tokens = sum(len(output.prompt_token_ids) for output in outputs)
            vectors = torch.nn.functional.normalize(vectors[:, :dimensions], p=2, dim=1).tolist()
        else:
            tokens = 0
            for offset in range(0, len(prompts), opts["batch_size"]):
                check_lengths(model.tokenizer, prompts[offset:offset + opts["batch_size"]], opts)
                encoded = model.tokenize(prompts[offset:offset + opts["batch_size"]])
                tokens += int(encoded["attention_mask"].sum())
            with torch.inference_mode():
                vectors = model.encode(prompts, batch_size=opts["batch_size"],
                    truncate_dim=dimensions, normalize_embeddings=False,
                    show_progress_bar=False, convert_to_tensor=True).float()
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1).cpu().tolist()
        checked_numbers(vectors)
        return {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": vector}
                 for i, vector in enumerate(vectors)],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}, warnings
    if task == "text-classification":
        if request["architecture"] == "xlm-roberta":
            texts = body["input"]
            if isinstance(texts, str):
                texts = [texts]
            tokenizer = loaded["tokenizer"]
            data, tokens = [], 0
            for offset in range(0, len(texts), opts["batch_size"]):
                check_lengths(tokenizer, texts[offset:offset + opts["batch_size"]], opts)
                encoded = tokenizer(texts[offset:offset + opts["batch_size"]], padding=True,
                    truncation=True, max_length=opts["max_length"], return_tensors="pt")
                tokens += int(encoded["attention_mask"].sum())
                with torch.inference_mode():
                    logits = model(**encoded.to(device)).logits.float()
                    probs = torch.sigmoid(logits) if model.config.problem_type == "multi_label_classification" else torch.softmax(logits, dim=-1)
                    for row in checked_numbers(probs.cpu().tolist()):
                        data.append({"index": len(data), "scores": [
                            {"label": model.config.id2label.get(i, str(i)), "score": score}
                            for i, score in enumerate(row)]})
            return {"data": data, "usage": {"total_tokens": tokens}}, warnings
        result = model.predict(body["state"], body["questions"], max_len=opts["max_length"])
        if result.get("usage", {}).get("truncated"):
            if not opts.get("truncate", False):
                raise InputError("Laya truncated state or question text. Reduce the input/options, increase werk.max_length within the checkpoint limit, or explicitly set werk.truncate=true")
            warnings.append("Laya truncated input; inspect usage.state_tokens_dropped and usage.truncated_questions")
        actual = str(model.device)
        if request["device"] == "cuda" and not actual.startswith("cuda"):
            raise RuntimeError("Laya switched to CPU during inference; free VRAM or use device=auto to permit CPU fallback")
        return {"answers": result["answers"], "usage": result.get("usage", {}),
                "checkpoint": loaded["checkpoint"]}, warnings
    raise ValueError(f"Unsupported task {task}")


def execute(request):
    global _loaded
    started = time.monotonic()
    hit = _loaded is not None
    if not hit:
        _loaded = load(request)
    loaded_at = time.monotonic()
    result, warnings = infer(request, _loaded)
    finished_at = time.monotonic()
    result.update(ok=True, warnings=warnings, werk={
        "runtime": request["runtime"], "device": _loaded["device"],
        "dtype": _loaded["dtype"], "model_cache_hit": hit,
        "load_seconds": loaded_at - started, "inference_seconds": finished_at - loaded_at,
        "total_seconds": finished_at - started,
        "max_length": _loaded.get("max_length", request["options"]["max_length"]),
        "truncate": request["options"].get("truncate", False),
    })
    return result


def dispatch(operation, request):
    try:
        if operation == "transport-handshake":
            return {"ok": True}
        if operation == "probe-model":
            return probe()
        if operation == "execute":
            return execute(request)
        raise ValueError(f"Unsupported operation {operation}")
    except InputError as error:
        return {"ok": True, "invalid_input": str(error)}
    except Exception as error:
        return {"ok": False, "error": {"code": "text_analysis_failed", "message":
            f"{type(error).__name__}: {error}. Install torch, transformers>=5.19, "
            "sentence-transformers>=6.1 and laya[serve,fast] in WERK_TEXT_PYTHON. "
            "For vLLM update to a release supporting this architecture; use backend=transformers as fallback."}}


def main():
    # Libraries may print during load. Keep the framed stdout channel pristine.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    operation = sys.argv[1]
    if operation == "serve":
        for line in sys.stdin:
            frame = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                response = dispatch(frame["operation"], frame["payload"])
            output.write(json.dumps({"transport_version": 1, "request_id": frame["request_id"],
                                     "response": response}, allow_nan=False) + "\n")
    else:
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            response = dispatch(operation, request)
        output.write(json.dumps(response, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

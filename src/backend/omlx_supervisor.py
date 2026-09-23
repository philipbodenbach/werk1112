"""Run a validated oMLX console script while its Werk parent holds stdin open.

Rust starts this interpreter as a new process-group leader. Private routes are
registered on oMLX's app; the launcher retains its original argv and import path.
"""

import os
from pathlib import Path
import runpy
import signal
import stat
import sys
import threading


def retain_lifetime_locks():
    """Keep Rust's locked open descriptions until interpreter exit.

    Never reopen the paths or unlock these descriptors: a duplicated flock is
    shared with the parent, and its final close is what proves worker exit.
    """
    raw = os.environ.pop("WERK_OMLX_LIFETIME_FDS", None)
    if raw is None:
        return ()
    values = raw.split(",")
    if not 1 <= len(values) <= 2 or any(not value.isascii() or not value.isdecimal() for value in values):
        raise SystemExit("invalid Werk worker lifetime descriptors")
    descriptors = tuple(int(value) for value in values)
    if len(set(descriptors)) != len(descriptors) or any(fd < 3 for fd in descriptors):
        raise SystemExit("invalid Werk worker lifetime descriptors")
    for descriptor in descriptors:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SystemExit("Werk worker lifetime descriptor is not a regular file")
        # oMLX subprocesses must not retain the lease after this worker exits.
        os.set_inheritable(descriptor, False)
    return descriptors


def install_expert_routes(manager):
    """Private authenticated endpoints; mutations use their model's executor."""
    import asyncio
    import json
    from fastapi import Depends, HTTPException, Request
    from omlx import server

    dependencies = [Depends(server.verify_api_key)]
    action_tasks = set()

    async def run_owned_action(body):
        # acquire()/get_engine() may load a model. Actions only operate on an
        # existing owner, with a lease preventing eviction while work is queued.
        try:
            pool = server.get_engine_pool()
        except HTTPException as error:
            raise RuntimeError("streamed model engine is unavailable") from error
        lock = getattr(pool, "_lock", None)
        if lock is None or not callable(getattr(pool, "get_entry", None)):
            raise RuntimeError("streamed model engine ownership is unavailable")
        model_id = manager.model_id
        async with lock:
            entry = pool.get_entry(model_id)
            engine = getattr(entry, "engine", None)
            core = getattr(getattr(engine, "_engine", None), "engine", None)
            executor = getattr(core, "_mlx_executor", None)
            model = manager._model_ref() if manager._model_ref is not None else None
            if (model is None or getattr(core, "model", None) is not model
                    or getattr(core, "_closed", False)
                    or not callable(getattr(executor, "submit", None))
                    or getattr(entry, "pending_unload_reason", None)
                    or getattr(entry, "abort_requested", False)):
                raise RuntimeError("streamed model has no available owning executor")
            entry.in_use += 1

        def action_on_owner():
            if (manager._model_ref is None or manager._model_ref() is not model
                    or getattr(core, "_closed", False)):
                raise RuntimeError("streamed model owner changed before expert action")
            return manager.action(body)

        try:
            return await asyncio.get_running_loop().run_in_executor(
                executor, action_on_owner
            )
        finally:
            await pool.release_engine(model_id)

    def action_done(task):
        action_tasks.discard(task)
        if not task.cancelled():
            # A disconnected HTTP client may no longer await the result.
            task.exception()

    @server.app.get("/werk/experts/status", dependencies=dependencies)
    async def expert_status():
        return await asyncio.to_thread(manager.status)

    @server.app.get("/werk/experts", dependencies=dependencies)
    async def expert_list(request: Request):
        if len(request.scope.get("query_string", b"")) > 65536:
            raise HTTPException(413, "expert query exceeds 64 KiB")
        try:
            pairs = list(request.query_params.multi_items())
            if len(pairs) != len(dict(pairs)):
                raise ValueError("duplicate expert filter")
            values = dict(pairs)
            if "limit" in values:
                values["limit"] = int(values["limit"])
            if "allow_experimental" in values:
                if values["allow_experimental"] not in ("true", "false"):
                    raise ValueError("allow_experimental must be boolean")
                values["allow_experimental"] = values["allow_experimental"] == "true"
            return await asyncio.to_thread(manager.list_experts, values)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @server.app.post("/werk/experts/actions", dependencies=dependencies)
    async def expert_action(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > 65536:
                raise HTTPException(413, "expert action exceeds 64 KiB")
            raw.extend(chunk)
        try:
            body = json.loads(raw)
            task = asyncio.create_task(run_owned_action(body))
            action_tasks.add(task)
            task.add_done_callback(action_done)
            # Retain the owner lease until the queued/running mutation finishes,
            # even when cancellation disconnects the HTTP caller.
            return await asyncio.shield(task)
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        except (MemoryError, RuntimeError) as error:
            raise HTTPException(409, str(error)) from error


def install_persistence_routes(manager):
    """Expose only native-cache readiness and aggregate persistence counters."""
    import asyncio
    from fastapi import Depends
    from omlx import server

    @server.app.get("/werk/persistence/status", dependencies=[Depends(server.verify_api_key)])
    async def persistence_status():
        return await asyncio.to_thread(manager.status)


async def count_chat_tokens(server, body):
    """Use the running text engine's real chat tokenizer, never run inference.

    Preserve request/model template controls. Do not use the native Anthropic
    counter, which does not forward enable_thinking/reasoning_effort.
    """
    import inspect
    request = server.ChatCompletionRequest.model_validate(body)
    lease = server._LLMEngineLease()
    try:
        engine = await server.get_engine_for_model(request.model, lease=lease)
        await server._raise_if_llm_lease_abort_requested(lease)
        if isinstance(engine, server.VLMBatchedEngine) or getattr(engine, "supports_multimodal_fallback", False):
            raise ValueError("native multimodal token counting is not verified for this runtime")
        if server._server_state.mcp_manager and server.mcp_tools_exposed():
            raise ValueError("token counting with implicit server MCP tools is not supported")
        settings = server.get_model_settings_for_request(request.model)
        kwargs = server.merge_chat_template_request_kwargs(settings,
            server.merge_reasoning_effort_chat_template_kwargs(
                request.chat_template_kwargs, request.reasoning_effort))
        limit = getattr(settings, "max_tool_result_tokens", None)
        extractor = getattr(engine, "message_extractor", None)
        if extractor is not None:
            options = {}
            if "consolidate_system_messages" in inspect.signature(extractor).parameters:
                options["consolidate_system_messages"] = False
            messages = extractor(request.messages, limit, engine.tokenizer, **options)
        else:
            messages = server.extract_text_content(request.messages, limit, engine.tokenizer,
                consolidate_system_messages=False)
        partial = server.detect_and_strip_partial(messages)
        tools = None if request.tool_choice == "none" else request.tools
        tools = server.convert_tools_for_template(tools) if tools else None
        if tools and "gemma" in request.model.lower():
            tools = server.enrich_tool_params_for_gemma4(tools)
        await server._ensure_tokenizer_for_system_probe(engine, messages)
        messages = server.prepare_system_messages_for_template(messages, engine.tokenizer,
            tools=tools, chat_template_kwargs=kwargs or None, is_partial=partial,
            merge_consecutive_roles=True,
            unsupported_mid_system_policy=server._unsupported_mid_system_policy())
        count = engine.count_chat_tokens(messages, tools,
            chat_template_kwargs=kwargs or None, is_partial=partial)
        if type(count) is not int or count < 0:
            raise ValueError("native tokenizer returned an invalid token count")
        return {"input_tokens": count}
    finally:
        await lease.release()


def install_token_count_route():
    import json
    # Keep the generic lifetime supervisor usable with diagnostic launchers
    # that do not import oMLX (including the tiny process-lifecycle fixtures).
    try:
        from omlx import server
    except ModuleNotFoundError as error:
        if error.name == "omlx":
            return
        raise
    from fastapi import Depends, HTTPException, Request

    @server.app.post("/werk/tokenize", dependencies=[Depends(server.verify_api_key)])
    async def tokenize(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > 128 * 1024 * 1024:
                raise HTTPException(413, "token count request exceeds 128 MiB")
            raw.extend(chunk)
        try:
            return await count_chat_tokens(server, json.loads(raw))
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error


def stop_owned_worker():
    """Never signal a process group that also belongs to the parent or a shell."""
    pid = os.getpid()
    termination = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        if hasattr(os, "killpg") and os.getpgrp() == pid:
            os.killpg(pid, termination)
            return
    except OSError:
        pass
    try:
        os.kill(pid, termination)
    except OSError:
        os._exit(1)


def watch_parent(fd):
    try:
        while True:
            try:
                if not os.read(fd, 4096):
                    break
            except InterruptedError:
                continue
            except OSError:
                break
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    # SIGKILL is deliberate: launcher signal handlers or blocked generation
    # cannot keep the worker/its descendants alive after Werk has exited.
    stop_owned_worker()


def main():
    lifetime_descriptors = retain_lifetime_locks()
    if len(sys.argv) < 2:
        raise SystemExit("oMLX supervisor requires a validated launcher path")
    launcher = sys.argv[1]
    sys.argv = sys.argv[1:]
    if not (getattr(sys.flags, "safe_path", False) or sys.flags.isolated):
        sys.path[0] = str(Path(launcher).resolve().parent)
    parent_fd = os.dup(sys.stdin.fileno())
    os.set_inheritable(parent_fd, False)
    threading.Thread(target=watch_parent, args=(parent_fd,), daemon=True).start()
    expert_bytes = os.environ.pop("WERK_OMLX_EXPERT_CACHE_BYTES", None)
    expert_model = os.environ.pop("WERK_OMLX_EXPERT_MODEL_DIR", None)
    ngram_bytes = os.environ.pop("WERK_OMLX_NGRAM_CACHE_BYTES", None)
    if expert_bytes is not None:
        if not expert_model:
            raise SystemExit("Werk expert offload requires an exact local model path")
        import json
        with open(Path(expert_model) / "config.json", "rb") as source:
            raw = source.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("offload metadata exceeds 4 MiB")
        config = json.loads(raw)
        if config.get("model_type") in ("qwen4_exp", "glm5_next"):
            from _werk_omlx_text_offload import install
            manager = install(expert_model, None if expert_bytes == "native" else int(expert_bytes),
                              None if ngram_bytes is None else int(ngram_bytes))
        else:
            from _werk_omlx_experts import install
            manager = install(expert_model, int(expert_bytes))
        install_expert_routes(manager)
    persistence_directory = os.environ.pop("WERK_OMLX_PERSISTENCE_DIR", None)
    persistence_model = os.environ.pop("WERK_OMLX_PERSISTENCE_MODEL_DIR", None)
    if persistence_directory is not None:
        if not persistence_model:
            raise SystemExit("Werk native persistence requires an exact local model path")
        from _werk_omlx_persistence import install
        manager = install(persistence_model, persistence_directory)
        install_persistence_routes(manager)
    install_token_count_route()
    runpy.run_path(launcher, run_name="__main__")


if __name__ == "__main__":
    main()

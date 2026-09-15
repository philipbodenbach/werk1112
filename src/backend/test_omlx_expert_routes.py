"""Private expert route contracts; stdlib tests plus optional real ASGI checks."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch
from urllib.parse import parse_qsl


SOURCE_PATH = Path(__file__).with_name("omlx_supervisor.py")
SPEC = importlib.util.spec_from_file_location("omlx_expert_supervisor", SOURCE_PATH)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


class RouteError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class RequestStub:
    def __init__(self, query=b"", chunks=()):
        self.scope = {"query_string": query}
        self.query_params = types.SimpleNamespace(
            multi_items=lambda: parse_qsl(query.decode("ascii"), keep_blank_values=True)
        )
        self.chunks = chunks

    async def stream(self):
        for chunk in self.chunks:
            yield chunk


class AppStub:
    def __init__(self):
        self.routes = {}

    def register(self, verb, path, dependencies):
        def decorate(handler):
            self.routes[(verb, path)] = (handler, dependencies)
            return handler
        return decorate

    def get(self, path, dependencies):
        return self.register("GET", path, dependencies)

    def post(self, path, dependencies):
        return self.register("POST", path, dependencies)


class ManagerStub:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.model_id = "org/model"
        self.model = object()
        self._model_ref = lambda: self.model

    def status(self):
        self.calls.append(("status",))
        return {"active": True}

    def list_experts(self, filters):
        self.calls.append(("list", filters))
        if self.failure:
            raise self.failure
        return {"experts": [], "next_cursor": None}

    def action(self, body):
        self.calls.append(("action", body, threading.get_ident()))
        if self.failure:
            raise self.failure
        return {"experts": [], "changed": 0, "dry_run": body["dry_run"]}


class PoolStub:
    def __init__(self, manager, executor):
        self._lock = asyncio.Lock()
        self.model_id = manager.model_id
        self.lookups = []
        self.releases = []
        self.core = types.SimpleNamespace(
            model=manager.model, _mlx_executor=executor, _closed=False)
        self.entry = types.SimpleNamespace(
            engine=types.SimpleNamespace(_engine=types.SimpleNamespace(engine=self.core)),
            in_use=0, pending_unload_reason=None, abort_requested=False)

    def get_entry(self, model_id):
        self.lookups.append(model_id)
        return self.entry if model_id == self.model_id else None

    async def release_engine(self, model_id):
        async with self._lock:
            assert model_id == self.model_id and self.entry.in_use > 0
            self.entry.in_use -= 1
            self.releases.append(model_id)

    async def get_engine(self, *args, **kwargs):
        raise AssertionError("expert actions must never load a model")


def installed_modules(app, verify_api_key, pool):
    server = types.ModuleType("omlx.server")
    server.app = app
    server.verify_api_key = verify_api_key
    server.get_engine_pool = lambda: pool
    engine = types.ModuleType("omlx.engine_core")
    def forbidden_global_executor():
        raise AssertionError("expert actions must use the owning model executor")
    engine.get_mlx_executor = forbidden_global_executor
    package = types.ModuleType("omlx")
    package.server = server
    return {"omlx": package, "omlx.server": server, "omlx.engine_core": engine}


class ExpertRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = AppStub()
        self.manager = ManagerStub()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.executor.shutdown)
        self.pool = PoolStub(self.manager, self.executor)
        self.verify_api_key = lambda: None
        fastapi = types.ModuleType("fastapi")
        fastapi.Depends = lambda dependency: types.SimpleNamespace(dependency=dependency)
        fastapi.HTTPException = RouteError
        fastapi.Request = RequestStub
        modules = installed_modules(self.app, self.verify_api_key, self.pool)
        modules["fastapi"] = fastapi
        with patch.dict(sys.modules, modules):
            supervisor.install_expert_routes(self.manager)

    def route(self, verb, path):
        return self.app.routes[(verb, path)][0]

    async def test_every_route_uses_the_private_worker_auth_dependency(self):
        self.assertEqual(set(self.app.routes), {
            ("GET", "/werk/experts/status"),
            ("GET", "/werk/experts"),
            ("POST", "/werk/experts/actions"),
        })
        for _, dependencies in self.app.routes.values():
            self.assertEqual(len(dependencies), 1)
            self.assertIs(dependencies[0].dependency, self.verify_api_key)
        result = await self.route("GET", "/werk/experts/status")()
        self.assertEqual(result, {"active": True})

    async def test_expert_filters_preserve_ids_and_parse_bounded_typed_values(self):
        result = await self.route("GET", "/werk/experts")(RequestStub(
            b"model_id=org%2Fmodel&tier=external&limit=2&allow_experimental=true"
        ))
        self.assertEqual(result, {"experts": [], "next_cursor": None})
        self.assertEqual(self.manager.calls, [("list", {
            "model_id": "org/model", "tier": "external", "limit": 2,
            "allow_experimental": True,
        })])

    async def test_duplicate_and_malformed_query_values_never_reach_manager(self):
        for query in (
            b"tier=ram&tier=external", b"limit=1&limit=2", b"limit=not-an-int",
            b"allow_experimental=1", b"allow_experimental=True",
        ):
            with self.subTest(query=query), self.assertRaises(RouteError) as raised:
                await self.route("GET", "/werk/experts")(RequestStub(query))
            self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(self.manager.calls, [])

    async def test_oversized_query_is_rejected_before_parsing(self):
        with self.assertRaises(RouteError) as raised:
            await self.route("GET", "/werk/experts")(RequestStub(b"x" * 65537))
        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(self.manager.calls, [])

    async def test_streamed_action_body_limit_applies_across_chunks(self):
        request = RequestStub(chunks=(b" " * 40000, b" " * 25537))
        with self.assertRaises(RouteError) as raised:
            await self.route("POST", "/werk/experts/actions")(request)
        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(self.manager.calls, [])

    async def test_malformed_json_is_rejected_before_scheduling_mutation(self):
        for raw in (b"{", b"\xff", b""):
            with self.subTest(raw=raw), self.assertRaises(RouteError) as raised:
                await self.route("POST", "/werk/experts/actions")(
                    RequestStub(chunks=(raw,))
                )
            self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(self.manager.calls, [])

    async def test_action_executes_on_owning_model_executor_preserving_dry_run(self):
        executor_thread = self.executor.submit(threading.get_ident).result()
        body = {"expert_ids": ["expert_1"], "action": "prefetch",
                "target_tier": "ram", "dry_run": True}
        raw = json.dumps(body).encode()
        result = await self.route("POST", "/werk/experts/actions")(
            RequestStub(chunks=(raw[:15], raw[15:]))
        )
        self.assertEqual(result["dry_run"], True)
        self.assertEqual(self.manager.calls, [("action", body, executor_thread)])
        self.assertNotEqual(executor_thread, threading.get_ident())
        self.assertEqual(self.pool.lookups, [self.manager.model_id])
        self.assertEqual(self.pool.releases, [self.manager.model_id])
        self.assertEqual(self.pool.entry.in_use, 0)

    async def test_unloaded_unowned_and_stopping_engines_fail_without_mutation(self):
        entry = self.pool.entry
        core = self.pool.core
        cases = (
            (entry, "engine", None),
            (core, "model", object()),
            (core, "_mlx_executor", None),
            (core, "_closed", True),
            (entry, "pending_unload_reason", "memory pressure"),
            (entry, "abort_requested", True),
            (self.manager, "_model_ref", None),
            (self.pool, "entry", None),
        )
        for target, name, value in cases:
            with self.subTest(name=name), patch.object(target, name, value):
                with self.assertRaises(RouteError) as raised:
                    await self.route("POST", "/werk/experts/actions")(
                        RequestStub(chunks=(b'{"dry_run":false}',)))
                self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self.manager.calls, [])
        self.assertEqual(self.pool.releases, [])
        self.assertEqual(entry.in_use, 0)

    async def test_action_waits_behind_model_work_and_retains_lease(self):
        release = threading.Event()
        self.addCleanup(release.set)
        blocker = self.executor.submit(release.wait, 5)
        task = asyncio.create_task(self.route("POST", "/werk/experts/actions")(
            RequestStub(chunks=(b'{"dry_run":false}',))))
        try:
            for _ in range(100):
                if self.pool.entry.in_use:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(self.pool.entry.in_use, 1)
            self.assertEqual(self.manager.calls, [])
            self.assertFalse(task.done())
        finally:
            release.set()
        await task
        await asyncio.wrap_future(blocker)
        self.assertEqual(self.pool.entry.in_use, 0)

    async def test_client_cancellation_retains_lease_until_action_finishes(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original = self.manager.action
        def slow_action(body):
            started.set()
            release.wait(5)
            return original(body)
        self.manager.action = slow_action
        task = asyncio.create_task(self.route("POST", "/werk/experts/actions")(
            RequestStub(chunks=(b'{"dry_run":false}',))))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(self.pool.entry.in_use, 1)
            self.assertEqual(self.pool.releases, [])
        finally:
            release.set()
        await asyncio.wrap_future(self.executor.submit(lambda: None))
        for _ in range(100):
            if not self.pool.entry.in_use:
                break
            await asyncio.sleep(0.001)
        self.assertEqual(self.pool.entry.in_use, 0)
        self.assertEqual(self.pool.releases, [self.manager.model_id])

    async def test_owner_is_rechecked_after_waiting_on_model_executor(self):
        release = threading.Event()
        self.addCleanup(release.set)
        blocker = self.executor.submit(release.wait, 5)
        task = asyncio.create_task(self.route("POST", "/werk/experts/actions")(
            RequestStub(chunks=(b'{"dry_run":false}',))))
        try:
            for _ in range(100):
                if self.pool.entry.in_use:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(self.pool.entry.in_use, 1)
            self.pool.core._closed = True
        finally:
            release.set()
        with self.assertRaises(RouteError) as raised:
            await task
        await asyncio.wrap_future(blocker)
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self.manager.calls, [])
        self.assertEqual(self.pool.entry.in_use, 0)

    async def test_executor_submission_failure_releases_model_lease(self):
        self.executor.shutdown()
        with self.assertRaises(RouteError) as raised:
            await self.route("POST", "/werk/experts/actions")(
                RequestStub(chunks=(b'{"dry_run":false}',)))
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self.manager.calls, [])
        self.assertEqual(self.pool.entry.in_use, 0)
        self.assertEqual(self.pool.releases, [self.manager.model_id])

    async def test_manager_validation_and_capacity_failures_return_errors(self):
        for failure, status in (
            (ValueError("unknown expert"), 400),
            (TypeError("invalid action"), 400),
            (MemoryError("cache full"), 409),
            (RuntimeError("worker unavailable"), 409),
        ):
            self.manager.failure = failure
            with self.subTest(failure=failure), self.assertRaises(RouteError) as raised:
                await self.route("POST", "/werk/experts/actions")(
                    RequestStub(chunks=(b'{"dry_run":false}',))
                )
            self.assertEqual(raised.exception.status_code, status)
            self.assertEqual(self.pool.entry.in_use, 0)
        self.assertEqual(len(self.pool.releases), 4)


try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.testclient import TestClient
except ImportError:
    FastAPI = None


@unittest.skipUnless(FastAPI, "optional FastAPI/httpx ASGI dependencies not installed")
class ExpertRouteAsgiTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.manager = ManagerStub()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.executor.shutdown)
        self.pool = PoolStub(self.manager, self.executor)

        async def verify_api_key(request: Request):
            if request.headers.get("authorization") != "Bearer private-worker-key":
                raise HTTPException(401, "invalid private worker key")

        with patch.dict(sys.modules, installed_modules(
            self.app, verify_api_key, self.pool
        )):
            supervisor.install_expert_routes(self.manager)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.auth = {"Authorization": "Bearer private-worker-key"}

    def test_auth_denies_every_route_before_manager_access(self):
        for method, url in (
            ("GET", "/werk/experts/status"),
            ("GET", "/werk/experts"),
            ("POST", "/werk/experts/actions"),
        ):
            for headers in ({}, {"Authorization": "Bearer incorrect"}):
                with self.subTest(method=method, url=url, headers=headers):
                    result = self.client.request(method, url, headers=headers)
                    self.assertEqual(result.status_code, 401)
        self.assertEqual(self.manager.calls, [])

    def test_authenticated_http_list_and_mutation_use_real_fastapi_request_binding(self):
        result = self.client.get("/werk/experts/status", headers=self.auth)
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["active"])
        result = self.client.get(
            "/werk/experts?tier=external&limit=3&allow_experimental=true",
            headers=self.auth,
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.manager.calls[-1], ("list", {
            "tier": "external", "limit": 3, "allow_experimental": True,
        }))
        result = self.client.post("/werk/experts/actions", headers=self.auth,
                                  json={"action": "evict", "dry_run": True})
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["dry_run"])

    def test_authenticated_http_rejects_duplicate_query_and_oversized_body(self):
        result = self.client.get("/werk/experts?tier=ram&tier=external", headers=self.auth)
        self.assertEqual(result.status_code, 400)
        result = self.client.post("/werk/experts/actions", headers=self.auth,
                                  content=b"x" * 65537)
        self.assertEqual(result.status_code, 413)
        self.assertEqual(self.manager.calls, [])


if __name__ == "__main__":
    unittest.main()

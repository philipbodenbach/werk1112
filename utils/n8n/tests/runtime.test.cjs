const test = require('node:test');
const assert = require('node:assert/strict');
const { WerkProtocolClient, WerkProtocolError, requireCapability, CAPABILITY_STATUSES } = require('../dist/shared/protocol');
const { runRuntime, persistencePolicy, runtimeIds } = require('../dist/shared/runtime');
const { sanitize } = require('../dist/shared/validation');

const info = () => ({ service: 'werk1112', service_version: '1.5.1', protocol: { major: 1, minor: 0 }, active_backend: 'test', limits: { max_page_size: 100, max_state_ids_per_operation: 100, max_expert_ids_per_operation: 256, max_request_bytes: 1048576, max_handoff_bytes: 4096, max_ttl_seconds: 2592000 } });
const cap = (id, status = 'supported') => ({ id, status, detail: `Status: ${status}`, operations: ['read'] });
const caps = () => ({ capabilities: ['runtime.pd.prefill', 'runtime.pd.decode', 'runtime.pd.handoff', 'runtime.experts.residency'].map(id => cap(id)) });
const state = (id = 'st_one') => ({ id, model_id: 'model', tier: 'ram', status: 'ready', bytes: 100, created_unix_ms: 1, last_accessed_unix_ms: 2, expires_unix_ms: null, pinned: true, backend: 'test', reusable: true });
const expert = (id = 'expert_one') => ({ id, model_id: 'model', tier: 'vram', bytes: 100, hotness: 1.2, pinned: false, last_used_unix_ms: null });
const token = 'private-single-use-handoff-value-1234567890';
const replacement = 'new-private-single-use-handoff-1234567890';
const envelope = (data, requestId = 'req_fixture') => ({ protocol: { major: 1, minor: 0 }, request_id: requestId, data });
const response = data => ({ statusCode: 200, headers: { 'content-type': 'application/json', 'x-werk-protocol-version': '1.0' }, body: envelope(data) });
function fixture(overrides = {}) {
  const calls = [];
  const transport = {
    calls,
    redact: value => sanitize(value, ['fixture-api-key']),
    async raw(method, path, body, query, protocol) {
      calls.push({ method, path, body, query, protocol });
      assert.equal(protocol, true);
      const custom = overrides[path];
      if (custom) return custom({ method, path, body, query, calls });
      let data;
      if (path.endsWith('/info')) data = info();
      else if (path.endsWith('/capabilities')) data = caps();
      else if (path.endsWith('/states')) data = { states: [state()], next_cursor: null };
      else if (path.includes('/states/') && path.endsWith('/actions')) data = { state: state(decodeURIComponent(path.split('/')[4])), changed: true, dry_run: body.dry_run };
      else if (path.endsWith('/prune')) data = { matched: 2, removed: body.dry_run ? 0 : 2, bytes: 200, dry_run: body.dry_run };
      else if (path.endsWith('/experts')) data = { experts: [expert()], next_cursor: null };
      else if (path.endsWith('/experts/actions')) data = { experts: body.expert_ids.map(expert), changed: body.expert_ids.length, dry_run: body.dry_run };
      else if (path.endsWith('/prefill')) data = { handoff: token, state_id: 'st_one', prompt_tokens: 12, reused: false, tier: 'ram', expires_unix_ms: 500 };
      else if (path.endsWith('/decode')) data = { text: 'Hello', handoff: replacement, completion_tokens: 3, finish_reason: 'stop' };
      else if (path.endsWith('/memory')) data = { observed_at_unix_ms: 10, overall_pressure: 'soft', topology: 'discrete', host: { capacity_bytes: 1000, available_bytes: 600, managed_bytes: 300, reserved_bytes: 100, pressure: 'soft' }, accelerator: { capacity_bytes: null, available_bytes: null, managed_bytes: 0, reserved_bytes: 0, pressure: 'unknown' }, last_action_unix_ms: null, counters: { demotions: 1 } };
      else throw new Error('Unexpected test endpoint');
      return response(data);
    },
  };
  return transport;
}
const run = (transport, params = {}) => runRuntime(transport, (name, fallback) => name in params ? params[name] : fallback);
const posts = t => t.calls.filter(call => call.method === 'POST');

test('runtime validates service and protocol versions independently, all six capability statuses and memory', async () => {
  const f = fixture({ '/werk/v1/capabilities': async () => response({ capabilities: CAPABILITY_STATUSES.map((status, i) => cap(`cap_${i}`, status)) }) });
  const result = await run(f);
  assert.equal(result.info.service_version, '1.5.1');
  assert.deepEqual(result.capabilities.map(c => c.status), [...CAPABILITY_STATUSES]);
  assert.equal((await run(f, { operation: 'memory' })).accelerator.capacity_bytes, null);
  assert.equal((await run(f, { operation: 'memory' })).counters.demotions, 1);
  assert.equal(posts(f).length, 0);
});

test('optional header remains compatible; present response header must be unique, valid and compatible', async () => {
  for (const header of [undefined, '1.0', ' 1.0 ', '01.00']) {
    const f = fixture({ '/werk/v1/info': async () => ({ statusCode: 200, headers: { 'content-type': 'application/json', ...(header === undefined ? {} : { 'x-werk-protocol-version': header }) }, body: envelope(info()) }) });
    assert.equal((await new WerkProtocolClient(f).info()).protocol.major, 1);
  }
  for (const header of ['1.1', '2.0', '1.0,1.0', ['1.0', '1.0'], '1', '-1.0', '65536.0', '1.０']) {
    const f = fixture({ '/werk/v1/info': async () => ({ ...response(info()), headers: { 'content-type': 'application/json', 'X-Werk-Protocol-Version': header } }) });
    await assert.rejects(new WerkProtocolClient(f).info(), WerkProtocolError);
  }
  const duplicate = fixture({ '/werk/v1/info': async () => ({ ...response(info()), headers: { 'content-type': 'application/json', 'x-werk-protocol-version': '1.0', 'X-Werk-Protocol-Version': '1.0' } }) });
  await assert.rejects(new WerkProtocolClient(duplicate).info(), /Duplicate/);
});

test('invalid envelopes, IDs, unsafe numbers, content types and mismatched versions fail closed', async () => {
  const invalid = [
    { ...response(info()), body: { data: info() } },
    { ...response(info()), body: envelope(info(), 'contains spaces') },
    { ...response(info()), body: { ...envelope(info()), protocol: { major: 1, minor: 1 } } },
    { ...response(info()), body: { protocol: { major: 1, minor: 0 }, request_id: 'req' } },
    { ...response(info()), body: { ...envelope(info()), error: {} } },
    { ...response(info()), headers: { 'content-type': 'text/html' } },
    { ...response(info()), body: '{"protocol":{"major":1,"major":1,"minor":0},"request_id":"req","data":{}}' },
  ];
  for (const value of invalid) {
    const f = fixture({ '/werk/v1/info': async () => value });
    await assert.rejects(new WerkProtocolClient(f).info(), WerkProtocolError);
    assert.equal(f.calls.length, 1);
  }
  const f = fixture({ '/werk/v1/info': async () => response({ ...info(), limits: { ...info().limits, max_page_size: Number.MAX_SAFE_INTEGER + 1 } }) });
  await assert.rejects(new WerkProtocolClient(f).info(), /safe integer/);
});

test('protocol errors preserve code, request ID, retryable, status and redact credential details', async () => {
  const f = fixture({ '/werk/v1/info': async () => ({ statusCode: 503, headers: { 'content-type': 'application/json', 'x-werk-protocol-version': '1.0' }, body: { protocol: { major: 1, minor: 0 }, request_id: 'req_error', error: { code: 'unavailable', message: 'Backend missing fixture-api-key at /home/operator/private.json https://host/a?signature=private', retryable: true, details: { credentials: 'fixture-api-key' } } } }) });
  await assert.rejects(new WerkProtocolClient(f).info(), error => {
    assert.equal(error.code, 'unavailable'); assert.equal(error.requestId, 'req_error'); assert.equal(error.retryable, true); assert.equal(error.statusCode, 503);
    assert.doesNotMatch(JSON.stringify(error) + error.stack, /fixture-api-key|operator|signature=|credentials/);
    return true;
  });
});

test('malformed error DTO, unknown error code, redirects and transport exceptions cannot trigger legacy calls or retries', async () => {
  for (const badError of [{ code: 'made_up', message: 'bad', retryable: false }, { code: 'internal', message: 'bad', retryable: 'false' }]) {
    const f = fixture({ '/werk/v1/info': async () => ({ ...response(null), statusCode: 500, body: { protocol: { major: 1, minor: 0 }, request_id: 'req', error: badError } }) });
    await assert.rejects(new WerkProtocolClient(f).info(), error => error.code === 'invalid_error_envelope');
    assert.equal(f.calls.length, 1);
  }
  const redirect = fixture({ '/werk/v1/info': async () => ({ ...response(null), statusCode: 302 }) });
  await assert.rejects(new WerkProtocolClient(redirect).info(), error => error.code === 'redirect_rejected');
  const broken = fixture({ '/werk/v1/info': async () => { throw new Error(`transport ${token} fixture-api-key`); } });
  await assert.rejects(new WerkProtocolClient(broken).info(), error => !error.message.includes(token) && error.code === 'transport_error');
  assert.equal(broken.calls.length, 1);
});

test('strict DTO validation rejects duplicate capabilities, invalid state enums, expert identities and memory counters', async () => {
  const duplicates = fixture({ '/werk/v1/capabilities': async () => response({ capabilities: [cap('same'), cap('same')] }) });
  await assert.rejects(run(duplicates), /Duplicate capability/);
  const wrongState = fixture({ '/werk/v1/states': async () => response({ states: [{ ...state(), status: 'completed' }], next_cursor: null }) });
  await assert.rejects(run(wrongState, { operation: 'states' }), /state.status/);
  const wrongExpert = fixture({ '/werk/v1/experts': async () => response({ experts: [expert(), expert()], next_cursor: null }) });
  await assert.rejects(run(wrongExpert, { operation: 'experts' }), /Duplicate expert/);
  const wrongCounter = fixture({ '/werk/v1/memory': async () => response({ counters: { count: -1 }, observed_at_unix_ms: 1, overall_pressure: 'normal', topology: 'unknown', host: {}, accelerator: {} }) });
  await assert.rejects(run(wrongCounter, { operation: 'memory' }));
});

test('capability gates honor statuses and narrowly limit probe and external read exceptions', () => {
  for (const status of CAPABILITY_STATUSES) {
    const list = [cap('runtime.pd.decode', status)];
    if (status === 'supported' || status === 'experimental') assert.doesNotThrow(() => requireCapability(list, 'runtime.pd.decode', true));
    else assert.throws(() => requireCapability(list, 'runtime.pd.decode', true));
  }
  assert.throws(() => requireCapability([cap('runtime.pd.decode', 'unavailable')], 'runtime.pd.decode', true, 'prefillProbe'));
  assert.doesNotThrow(() => requireCapability([cap('runtime.pd.prefill', 'unavailable')], 'runtime.pd.prefill', true, 'prefillProbe'));
  assert.throws(() => requireCapability([cap('runtime.pd.prefill', 'unavailable')], 'runtime.pd.prefill', false, 'prefillProbe'));
  assert.doesNotThrow(() => requireCapability([cap('runtime.experts.residency', 'externally_managed')], 'runtime.experts.residency', false, 'expertRead'));
  assert.throws(() => requireCapability([cap('runtime.experts.residency', 'metadata_only')], 'runtime.experts.residency', true, 'expertRead'));
  assert.throws(() => requireCapability([], 'runtime.pd.decode', true), /did not declare/);
});

test('state action defaults to dry run and supports every exact action/target contract', async () => {
  for (const [action, tier] of [['pin', 'unchanged'], ['unpin', 'unchanged'], ['evict', 'unchanged'], ['promote', 'ram'], ['promote', 'vram'], ['demote', 'ram'], ['demote', 'disk']]) {
    for (const dryRun of [undefined, false]) {
      const f = fixture();
      const params = { operation: 'stateAction', stateId: 'st_one', stateAction: action, targetTier: tier, ...(dryRun === undefined ? {} : { dryRun }) };
      const result = await run(f, params);
      assert.equal(result.dry_run, dryRun ?? true);
      assert.equal(posts(f).length, 1);
      assert.equal(posts(f)[0].body.action, action);
      assert.equal(posts(f)[0].body.target_tier, tier === 'unchanged' ? undefined : tier);
    }
  }
  for (const [action, tier] of [['pin', 'ram'], ['promote', 'disk'], ['demote', 'vram'], ['promote', 'unchanged']]) {
    const f = fixture();
    await assert.rejects(run(f, { operation: 'stateAction', stateId: 'st_one', stateAction: action, targetTier: tier }));
    assert.equal(posts(f).length, 0);
  }
});

test('prune requires explicit bounded selection and defaults to dry run', async () => {
  for (const params of [{ selector: 'ids', stateIds: 'st_one,st_two,st_one' }, { selector: 'filter', modelId: 'model' }, { selector: 'filter', tier: 'disk' }, { selector: 'filter', olderThanUnixMs: 1 }, { selector: 'all', confirmAll: true }]) {
    const f = fixture();
    assert.equal((await run(f, { operation: 'pruneStates', ...params })).dry_run, true);
    assert.equal(posts(f)[0].body.dry_run, true);
  }
  const live = fixture();
  assert.equal((await run(live, { operation: 'pruneStates', selector: 'ids', stateIds: 'st_one', dryRun: false })).removed, 2);
  for (const params of [{ selector: 'ids', stateIds: '' }, { selector: 'filter' }, { selector: 'filter', modelId: ' ', olderThanUnixMs: 0 }, { selector: 'all', confirmAll: false }]) {
    const f = fixture();
    await assert.rejects(run(f, { operation: 'pruneStates', ...params }));
    assert.equal(posts(f).length, 0);
  }
  const tiny = fixture({ '/werk/v1/info': async () => response({ ...info(), limits: { ...info().limits, max_state_ids_per_operation: 1 } }) });
  await assert.rejects(run(tiny, { operation: 'pruneStates', stateIds: 'st_one,st_two' }), /at most 1 state/);
  assert.equal(posts(tiny).length, 0);
});

test('state and expert action replies must match explicit selection and dry run', async () => {
  const f = fixture({ '/werk/v1/states/st_one/actions': async () => response({ state: state('st_other'), changed: true, dry_run: true }) });
  await assert.rejects(run(f, { operation: 'stateAction', stateId: 'st_one' }), /does not match/);
  const e = fixture({ '/werk/v1/experts/actions': async () => response({ experts: [{ ...expert(), model_id: 'other' }], changed: 1, dry_run: true }) });
  await assert.rejects(run(e, { operation: 'expertAction', modelId: 'model', expertIds: 'expert_one' }), /explicit selection/);
  const prune = fixture({ '/werk/v1/states/prune': async () => response({ matched: 1, removed: 1, bytes: 1, dry_run: true }) });
  await assert.rejects(run(prune, { operation: 'pruneStates', stateIds: 'st_one' }), /dry-run contract/);
});

test('experts allow external reads but no metadata or externally-managed mutations; selectors and targets are strict', async () => {
  for (const status of ['unsupported', 'unavailable', 'metadata_only', 'externally_managed', 'experimental']) {
    const f = fixture({ '/werk/v1/capabilities': async () => response({ capabilities: [cap('runtime.experts.residency', status)] }) });
    if (status === 'externally_managed') assert.equal((await run(f, { operation: 'experts' })).count, 1);
    else await assert.rejects(run(f, { operation: 'experts' }));
    await assert.rejects(run(f, { operation: 'expertAction', modelId: 'model', expertIds: 'expert_one' }));
    assert.equal(posts(f).length, 0);
  }
  for (const expertAction of ['pin', 'unpin', 'evict', 'prefetch']) {
    const f = fixture();
    const result = await run(f, { operation: 'expertAction', modelId: 'model', expertIds: 'expert_one', expertAction, ...(expertAction === 'prefetch' ? { expertTargetTier: 'ram' } : {}) });
    assert.equal(result.dry_run, true);
    assert.equal(posts(f)[0].body.target_tier, expertAction === 'prefetch' ? 'ram' : undefined);
  }
  for (const params of [{ expertIds: 'expert_one,expert_one' }, { modelId: '' }, { expertIds: '../bad' }, { expertAction: 'prefetch' }, { expertAction: 'pin', expertTargetTier: 'ram' }]) {
    const f = fixture();
    await assert.rejects(run(f, { operation: 'expertAction', modelId: 'model', expertIds: 'expert_one', ...params }));
    assert.equal(posts(f).length, 0);
  }
});

test('manual pages preserve cursors, validate server limits, and bounded automatic pagination detects loops', async () => {
  const f = fixture({ '/werk/v1/states': async ({ query }) => response({ states: [state(query.cursor ? 'st_two' : 'st_one')], next_cursor: query.cursor ? null : 'next_1' }) });
  const first = await run(f, { operation: 'states', limit: 2, tier: 'disk', modelId: 'model' });
  assert.equal(first.nextCursor, 'next_1');
  assert.equal(f.calls.at(-1).query.tier, 'disk');
  const all = await run(f, { operation: 'states', returnAll: true });
  assert.equal(all.count, 2); assert.equal(all.nextCursor, null);
  const loop = fixture({ '/werk/v1/states': async ({ calls }) => response({ states: [state(`st_${calls.length}`)], next_cursor: 'repeat' }) });
  await assert.rejects(run(loop, { operation: 'states', returnAll: true }), /repeated cursor/);
  assert.equal(loop.calls.filter(c => c.path.endsWith('/states')).length, 2);
  await assert.rejects(run(f, { operation: 'states', limit: 101 }), /must not exceed 100/);
  await assert.rejects(run(f, { operation: 'states', returnAll: true, maxPages: 1 }), /page bound/);
  const excessive = fixture({ '/werk/v1/states': async () => response({ states: [state('st_1'), state('st_2')], next_cursor: null }) });
  await assert.rejects(run(excessive, { operation: 'states', limit: 1 }), /requested limit/);
});

test('persistence policy omission, complete policy defaults and zero TTL retain intended semantics', () => {
  assert.equal(persistencePolicy({}, 10), undefined);
  assert.equal(persistencePolicy({ enabled: false, mode: 'disk', ttlSeconds: 4 }, 10), undefined);
  assert.deepEqual(persistencePolicy({ enabled: true }, 10), { mode: 'auto', reuse: 'prefer', pin: false });
  assert.deepEqual(persistencePolicy({ enabled: true, ttlSeconds: 0, pin: false, mode: 'memory', reuse: 'disabled' }, 10), { mode: 'memory', reuse: 'disabled', pin: false });
  assert.equal(persistencePolicy({ enabled: true, ttlSeconds: 10 }, 10).ttl_seconds, 10);
  assert.throws(() => persistencePolicy({ enabled: true, ttlSeconds: 11 }, 10), /server limit/);
  assert.throws(() => persistencePolicy({ enabled: true, ttlSeconds: Number.MAX_SAFE_INTEGER + 1 }, 10), /safe integer/);
  assert.throws(() => persistencePolicy({ enabled: true, unknown: true }, 10), /unknown fields/);
  assert.throws(() => runtimeIds('["expert_one","expert_one"]', 'expert'), /unique/);
});

test('prefill/decode probes unavailable only with explicit opt-in and re-reads capabilities before one decode', async () => {
  let discovery = 0;
  const f = fixture({ '/werk/v1/capabilities': async () => {
    discovery++;
    return response({ capabilities: ['runtime.pd.prefill', 'runtime.pd.handoff', 'runtime.pd.decode'].map(id => cap(id, discovery === 1 ? 'unavailable' : 'experimental')) });
  } });
  const result = await run(f, { operation: 'prefillDecode', modelId: 'model', text: 'Hello', allowExperimental: true, temperature: 0, seed: 0 });
  assert.equal(result.text, 'Hello'); assert.equal(result.stateId, 'st_one'); assert.equal(result.promptTokens, 12); assert.equal(result.completionTokens, 3);
  assert.deepEqual(f.calls.map(call => call.path), ['/werk/v1/info', '/werk/v1/capabilities', '/werk/v1/prefill', '/werk/v1/capabilities', '/werk/v1/decode']);
  assert.equal(posts(f)[0].body.policy, undefined);
  assert.equal(posts(f)[1].body.handoff, token);
  assert.equal(posts(f)[1].body.temperature, 0); assert.equal(posts(f)[1].body.seed, 0); assert.equal(posts(f)[1].body.top_p, undefined);
  assert.doesNotMatch(JSON.stringify(result), /handoff|private-single-use/);
  const noOpt = fixture({ '/werk/v1/capabilities': async () => response({ capabilities: ['runtime.pd.prefill', 'runtime.pd.handoff'].map(id => cap(id, 'unavailable')) }) });
  await assert.rejects(run(noOpt, { operation: 'prefillDecode', modelId: 'model', text: 'Hello' }), /unavailable/);
  assert.equal(posts(noOpt).length, 0);
});

test('decode unavailable remains fail-closed even after successful prefill', async () => {
  const f = fixture({ '/werk/v1/capabilities': async () => response({ capabilities: [cap('runtime.pd.prefill'), cap('runtime.pd.handoff'), cap('runtime.pd.decode', 'unavailable')] }) });
  await assert.rejects(run(f, { operation: 'prefillDecode', modelId: 'model', text: 'Hello', allowExperimental: true }), /unavailable/);
  assert.equal(posts(f).length, 1);
  assert.equal(posts(f)[0].path, '/werk/v1/prefill');
});

test('handoffs are absent from outputs even when echoed in metadata, text, finish reason or request IDs', async () => {
  const f = fixture({
    '/werk/v1/prefill': async () => ({ ...response(null), body: envelope({ handoff: token, state_id: token, prompt_tokens: 12, reused: false, tier: 'ram', expires_unix_ms: 500 }, token) }),
    '/werk/v1/decode': async () => ({ ...response(null), body: envelope({ handoff: replacement, text: `Text ${token} ${replacement}`, completion_tokens: 1, finish_reason: replacement }, replacement) }),
  });
  const result = await run(f, { operation: 'prefillDecode', modelId: 'model', text: 'Hello' });
  assert.ok(!JSON.stringify(result).includes(token));
  assert.ok(!JSON.stringify(result).includes(replacement));
  assert.equal(result.stateId, '[redacted]');
  assert.equal(posts(f).length, 2);
});

test('decode error redacts handoff from message/request ID and never retries single-use requests', async () => {
  const f = fixture({ '/werk/v1/decode': async () => ({ ...response(null), statusCode: 410, body: { protocol: { major: 1, minor: 0 }, request_id: token, error: { code: 'expired_handoff', message: `Expired ${token}; fixture-api-key`, retryable: false, nested: { handoff: token } } } }) });
  await assert.rejects(run(f, { operation: 'prefillDecode', modelId: 'model', text: 'Hello' }), error => {
    assert.equal(error.code, 'expired_handoff'); assert.equal(error.retryable, false); assert.equal(error.requestId, undefined);
    assert.ok(!(error.stack + JSON.stringify(error)).includes(token)); assert.ok(!error.message.includes('fixture-api-key'));
    return true;
  });
  assert.equal(posts(f).filter(call => call.path.endsWith('/decode')).length, 1);
  const transport = fixture({ '/werk/v1/decode': async () => { throw Object.assign(new Error(`lost response ${token}`), { cause: { body: { handoff: token } } }); } });
  await assert.rejects(run(transport, { operation: 'prefillDecode', modelId: 'model', text: 'Hello' }), error => error.code === 'transport_error' && !error.stack.includes(token));
  assert.equal(posts(transport).length, 2);
});

test('prefill/decode inputs preserve message order/whitespace, reject unsafe seeds and enforce local bounds before POST', async () => {
  const f = fixture();
  const messages = [{ role: 'system', content: '  Keep spaces  ' }, { role: 'user', content: 'Ask' }];
  await run(f, { operation: 'prefillDecode', modelId: 'model', inputType: 'messages', messages, persistencePolicy: { enabled: true, mode: 'ephemeral', reuse: 'disabled', ttlSeconds: 0 } });
  assert.deepEqual(posts(f)[0].body.input.messages, messages);
  assert.deepEqual(posts(f)[0].body.policy, { mode: 'ephemeral', reuse: 'disabled', pin: false });
  for (const params of [{ seed: Number.MAX_SAFE_INTEGER + 1 }, { temperature: NaN }, { topP: 0 }, { maxTokens: 32769 }, { text: 'a'.repeat(512 * 1024 + 1) }, { stopSequences: '[""]' }, { inputType: 'messages', messages: '[{"role":"user","role":"system","content":"a"}]' }, { inputType: 'messages', messages: [{ role: 'user', content: 'a', handoff: 'bad' }] }]) {
    const invalid = fixture();
    await assert.rejects(run(invalid, { operation: 'prefillDecode', modelId: 'model', text: 'Hello', ...params }));
    assert.equal(posts(invalid).length, 0);
  }
});

test('discovered request and handoff size limits are enforced without a second POST', async () => {
  const smallRequest = fixture({ '/werk/v1/info': async () => response({ ...info(), limits: { ...info().limits, max_request_bytes: 50 } }) });
  await assert.rejects(run(smallRequest, { operation: 'prefillDecode', modelId: 'model', text: 'Hello' }), /request_too_large/);
  assert.equal(posts(smallRequest).length, 0);
  const smallHandoff = fixture({ '/werk/v1/info': async () => response({ ...info(), limits: { ...info().limits, max_handoff_bytes: 32 } }) });
  await assert.rejects(run(smallHandoff, { operation: 'prefillDecode', modelId: 'model', text: 'Hello' }), /handoff/);
  assert.equal(posts(smallHandoff).length, 1);
});

test('native Runtime execute evaluates each item separately, pairs outputs and stores no handoffs on continueOnFail', async () => {
  const { WerkRuntime } = require('../dist/nodes/WerkRuntime/WerkRuntime.node');
  const issued = new Map();
  const requests = [];
  let sequence = 0;
  const f = fixture({
    '/werk/v1/prefill': async ({ body }) => {
      const handoff = `local-runtime-item-${++sequence}-handoff-private-1234567890`;
      issued.set(handoff, body.input.text);
      return response({ handoff, state_id: 'st_one', prompt_tokens: 1, reused: false, tier: 'ram', expires_unix_ms: 500 });
    },
    '/werk/v1/decode': async ({ body }) => {
      assert.ok(issued.has(body.handoff));
      const text = issued.get(body.handoff);
      issued.delete(body.handoff);
      return response({ text, handoff: null, completion_tokens: 1, finish_reason: 'stop' });
    },
  });
  const parameters = [
    { operation: 'prefillDecode', modelId: '', text: 'invalid item' },
    { operation: 'prefillDecode', modelId: 'model', text: 'first valid item' },
    { operation: 'prefillDecode', modelId: 'model', text: 'second valid item' },
  ];
  const context = {
    getInputData: () => parameters.map(() => ({ json: {} })),
    getNodeParameter: (name, index, fallback) => name in parameters[index] ? parameters[index][name] : fallback,
    getCredentials: async () => ({ baseUrl: 'http://127.0.0.1:11434', authMode: 'none', apiKey: '', verifyTls: true }),
    getNode: () => ({ id: 'runtime', name: 'Runtime', type: 'CUSTOM.werkRuntime', typeVersion: 1, position: [0, 0], parameters: {} }),
    getExecutionCancelSignal: () => undefined,
    continueOnFail: () => true,
    helpers: {
      async httpRequest(options) {
        assert.equal(options.headers.Accept, 'application/json');
        assert.equal(options.headers['X-Werk-Protocol-Version'], '1.0');
        assert.equal(options.disableFollowRedirect, true);
        requests.push(options);
        const raw = await f.raw(options.method, new URL(options.url).pathname, options.body ? JSON.parse(options.body) : undefined, options.qs, true);
        return { ...raw, body: Buffer.from(JSON.stringify(raw.body)) };
      },
    },
  };
  const [output] = await new WerkRuntime().execute.call(context);
  assert.equal(output.length, 3);
  assert.equal(output[0].json.itemIndex, 0);
  assert.equal(output[1].json.text, 'first valid item');
  assert.equal(output[2].json.text, 'second valid item');
  assert.deepEqual(output.map(item => item.pairedItem), [{ item: 0 }, { item: 1 }, { item: 2 }]);
  assert.equal(issued.size, 0);
  assert.equal(requests.filter(request => request.method === 'POST').length, 4);
  assert.doesNotMatch(JSON.stringify(output), /local-runtime-item|handoff-private/);
});

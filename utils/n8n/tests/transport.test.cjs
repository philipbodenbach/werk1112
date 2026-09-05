const { test } = require('node:test');
const assert = require('node:assert/strict');
const { Readable } = require('node:stream');
const { WerkClient, authenticate, normalizeBaseUrl, sameOrigin, readBounded } = require('../dist/shared/client');
const { werkApiTest } = require('../dist/shared/credential-test');
const { parseJson, parseObject, sanitize, safeMessage } = require('../dist/shared/validation');
const { executeItems, commonMethods } = require('../dist/shared/common');
const { mergeModels, requireModelTask, requireChatModel, parameterSchema } = require('../dist/shared/discovery');
const { readBinary, binaryItem } = require('../dist/shared/binary');

const secret = 'fixture-private-key';
function context(replies = [], config = {}) {
  const calls = [];
  const credentials = { baseUrl: 'http://127.0.0.1:11434/prefix', authMode: 'apiKey', apiKey: secret, verifyTls: true, ...config.credentials };
  const ctx = {
    getCredentials: async () => credentials,
    getNode: () => ({ name: 'Fixture', type: 'CUSTOM.werkImage', typeVersion: 1, position: [0, 0], parameters: {} }),
    getInputData: () => config.items ?? [{ json: {} }],
    getNodeParameter: (name, index, fallback) => config.parameters?.[index]?.[name] ?? fallback,
    getExecutionCancelSignal: () => config.signal,
    continueOnFail: () => config.continueOnFail ?? false,
    helpers: {
      async httpRequestWithAuthentication(type, options) { assert.equal(type, 'werkApi'); return this.helpers.httpRequest(await authenticate(credentials, options)); },
      async httpRequest(options) {
        calls.push(options);
        const result = replies.shift();
        if (result instanceof Error) throw result;
        const reply = typeof result === 'function' ? await result(options) : result ?? {};
        return { statusCode: reply.statusCode ?? 200, headers: reply.headers ?? { 'content-type': 'application/json' }, body: reply.stream ?? Readable.from([Buffer.from(typeof reply.body === 'string' ? reply.body : JSON.stringify(reply.body ?? { data: [] }))]) };
      },
      assertBinaryData(index, property) { return config.items[index].binary[property]; },
      async getBinaryDataBuffer(index, property) { calls.push({ binaryRead: [index, property] }); return config.binaryBytes[index][property]; },
      async prepareBinaryData(bytes, filename, mimeType) { calls.push({ binaryWrite: Buffer.from(bytes), filename, mimeType }); return { id: `filesystem-v2:fixture-${calls.length}`, data: 'filesystem-v2', mimeType, fileName: filename }; },
    },
  };
  return { ctx, calls, credentials };
}

test('base URL validates scheme, embedded credentials and query while preserving proxy prefix', () => {
  assert.equal(normalizeBaseUrl('http://LOCALHOST:80/prefix/'), 'http://localhost/prefix');
  for (const url of ['file:///tmp/x', 'http://a:b@localhost', 'http://localhost/?api_key=x', 'http://localhost/#x', 'garbage']) assert.throws(() => normalizeBaseUrl(url));
  assert.ok(sameOrigin('http://localhost:80/a', 'http://LOCALHOST/b'));
  assert.ok(!sameOrigin('https://localhost', 'http://localhost'));
  assert.ok(!sameOrigin('http://localhost:11434', 'http://localhost:11435'));
});

test('credentials require explicit authentication mode and only authenticate exact origin', async () => {
  const { credentials } = context();
  for (const url of ['https://127.0.0.1:11434/x', 'http://127.0.0.1:80/x', 'http://localhost:11434/x', 'http://user@127.0.0.1:11434/x']) await assert.rejects(authenticate(credentials, { url }));
  await assert.rejects(authenticate({ ...credentials, authMode: 'apiKey', apiKey: '' }, { url: credentials.baseUrl }));
  await assert.rejects(authenticate({ ...credentials, authMode: undefined }, { url: credentials.baseUrl }));
  const options = await authenticate(credentials, { url: `${credentials.baseUrl}/v1/models` });
  assert.equal(options.headers.Authorization, `Bearer ${secret}`);
  assert.equal(options.disableFollowRedirect, true);
  assert.equal(options.skipSslCertificateValidation, false);
  const unauth = await authenticate({ ...credentials, authMode: 'none' }, { url: credentials.baseUrl });
  assert.equal(unauth.headers.Authorization, undefined);
});

test('transport uses n8n authentication helper, bounded streaming, finite timeout and protocol headers', async () => {
  const { ctx, calls } = context([{ body: { data: [] } }, { body: { protocol: { major: 1, minor: 0 }, data: {} } }]);
  const client = await WerkClient.create(ctx);
  await client.api('GET', '/v1/models');
  assert.equal(calls[0].url, 'http://127.0.0.1:11434/prefix/v1/models');
  assert.equal(calls[0].headers.Authorization, `Bearer ${secret}`);
  assert.equal(calls[0].encoding, 'stream');
  assert.equal(calls[0].timeout, 120000);
  assert.equal(calls[0].disableFollowRedirect, true);
  const raw = await client.raw('GET', '/werk/v1/info', undefined, undefined, true);
  assert.equal(typeof raw.body, 'string');
  assert.equal(calls[1].headers.Accept, 'application/json');
  assert.equal(calls[1].headers['X-Werk-Protocol-Version'], '1.0');
  await assert.rejects(client.api('GET', '/werk/v1/info'), /separate/);
  await assert.rejects(client.raw('GET', '/v1/models', undefined, undefined, true), /separate/);
  assert.equal(calls.length, 2);
});

test('authenticated helper receives exact per-item credentials and scoped deadline/cancel signals', async () => {
  const controller = new AbortController();
  const { ctx } = context([], { signal: controller.signal });
  const credentials = [
    { baseUrl: 'http://127.0.0.1:11434/first', authMode: 'apiKey', apiKey: 'first-private', verifyTls: true },
    { baseUrl: 'http://127.0.0.1:11435/second', authMode: 'apiKey', apiKey: 'second-private', verifyTls: true },
  ];
  const seen = [];
  ctx.getCredentials = async (_, index = 0) => credentials[index];
  ctx.helpers.httpRequestWithAuthentication = async function (_, options, additional) {
    // Match the pinned n8n helper: it replaces options.abortSignal from its call context.
    options.abortSignal = this.getExecutionCancelSignal();
    assert.ok(additional.credentialsDecrypted.data);
    const authenticated = await authenticate(additional.credentialsDecrypted.data, options);
    seen.push({ url: authenticated.url, key: authenticated.headers.Authorization, signal: authenticated.abortSignal, timeout: authenticated.timeout });
    if (authenticated.abortSignal.aborted) throw new Error('cancelled');
    return { statusCode: 200, headers: {}, body: Readable.from(['{"id":"job-1","status":"queued"}']) };
  };
  const first = await WerkClient.create(ctx, 0);
  const second = await WerkClient.create(ctx, 1);
  await first.api('GET', '/v1/models');
  await second.api('GET', '/v1/models');
  assert.deepEqual(seen.map(item => item.key), ['Bearer first-private', 'Bearer second-private']);
  assert.match(seen[1].url, /11435\/second\//);
  assert.notEqual(seen[0].signal, controller.signal);
  controller.abort();
  await assert.rejects(second.api('GET', '/v1/jobs/job-1'), /cancelled/);
  await second.cancelJob('job-1');
  assert.equal(seen.at(-1).signal.aborted, false);
  assert.ok(seen.at(-1).timeout <= 10000);
  assert.equal(ctx.getExecutionCancelSignal(), controller.signal, 'original context was not mutated');
});

test('invalid UTF-8 is rejected before strict protocol evaluation', async () => {
  const { ctx } = context([{ stream: Readable.from([Buffer.from([0xc3, 0x28])]) }]);
  await assert.rejects((await WerkClient.create(ctx)).raw('GET', '/werk/v1/info', undefined, undefined, true), /UTF-8/);
});

test('continueOnFail keeps sanitized structured protocol error identifiers', async () => {
  const { WerkProtocolError } = require('../dist/shared/protocol');
  const { ctx } = context([], { continueOnFail: true });
  const [items] = await executeItems(ctx, async () => { throw new WerkProtocolError('unavailable', `private ${secret}`, 503, 'request-123', true); });
  assert.deepEqual(items[0].json.werk.error, { code: 'unavailable', requestId: 'request-123', retryable: true, statusCode: 503 });
  assert.ok(!JSON.stringify(items).includes(secret));
});

test('external media uses unauthenticated n8n helper; internal output retains proxy path', async () => {
  const { ctx, calls } = context([{ body: 'bytes', headers: { 'content-type': 'image/png' } }, { body: 'bytes' }]);
  const client = await WerkClient.create(ctx);
  await client.download('https://cdn.example/media?signature=private');
  await client.download('/v1/outputs/output-id');
  assert.equal(calls[0].headers.Authorization, undefined);
  assert.equal(calls[1].headers.Authorization, `Bearer ${secret}`);
  assert.equal(calls[1].url, 'http://127.0.0.1:11434/prefix/v1/outputs/output-id');
});

test('all redirects are rejected without second request, including same-origin', async () => {
  for (const target of ['/v1/models', 'https://evil.example/?private=yes']) {
    const { ctx, calls } = context([{ statusCode: 302, headers: { location: target } }]);
    const client = await WerkClient.create(ctx);
    await assert.rejects(client.api('POST', '/v1/images/generations', { prompt: 'x' }), /redirect rejected/);
    assert.equal(calls.length, 1);
  }
});

test('transport errors and nested Werk HTTP errors redact credentials, signed URL queries and internal paths', async () => {
  const input = `failure ${secret} https://cdn.example/a?signature=private /home/private/model.bin`;
  for (const reply of [new Error(input), { statusCode: 400, body: { error: { error: { message: input } } } }]) {
    const { ctx, calls } = context([reply]);
    const client = await WerkClient.create(ctx);
    await assert.rejects(client.api('POST', '/v1/audio/speech', { model: 'm', input: 'hi', async: true }), error => {
      const serial = `${error.stack} ${JSON.stringify(error)}`;
      for (const forbidden of [secret, 'signature=', '/home/private']) assert.ok(!serial.includes(forbidden));
      return true;
    });
    assert.equal(calls.length, 1, 'no ambiguous POST retry');
  }
});

test('response limits reject headers and streamed overflow before full buffering', async () => {
  const stream = Readable.from([Buffer.alloc(20)]);
  await assert.rejects(readBounded(stream, { 'content-length': '20' }, 10, 100), /exceeds/);
  assert.ok(stream.destroyed);
  const chunked = Readable.from([Buffer.alloc(8), Buffer.alloc(8), Buffer.alloc(100)]);
  await assert.rejects(readBounded(chunked, {}, 10, 100), /exceeds/);
  assert.ok(chunked.destroyed);
  const hanging = new Readable({ read() {} });
  await assert.rejects(readBounded(hanging, {}, 10, 5), /timed out/);
});

test('JSON parser rejects nested duplicates, prototype pollution, unsafe seeds, NaN and invalid syntax', () => {
  for (const source of ['{"seed":9007199254740993}', '{"x":1,"x":2}', '{"x":{"a":1,"a":2}}', '{"__proto__":{}}', '{"x":1e309}', '{"x":NaN}', '[1,]', '{"x":1} trailing']) assert.throws(() => parseJson(source));
  assert.deepEqual(parseObject('{"zero":0,"off":false,"items":[1,"a"]}', 'test'), { zero: 0, off: false, items: [1, 'a'] });
  assert.throws(() => parseObject({ seed: 9007199254740992 }, 'seed'));
});

test('metadata sanitizer removes nested input Base64, secrets, handoffs and internal paths', () => {
  const result = sanitize({ werk: { effective_request: { inputs: [{ source: { kind: 'base64', data: 'PRIVATEBYTES' } }] }, path: '/tmp/private', b64_json: 'PRIVATEBYTES', api_key: secret, handoff: 'TOKEN', message: `Bearer ${secret} data:image/png;base64,PRIVATEBYTES https://cdn/a?signature=private` }, usage: { completion_tokens: 1 } }, [secret]);
  const wire = JSON.stringify(result);
  for (const forbidden of [secret, 'PRIVATEBYTES', 'TOKEN', '/tmp/private', 'signature=']) assert.ok(!wire.includes(forbidden));
  assert.equal(result.usage.completion_tokens, 1);
  assert.equal(safeMessage({ request: { headers: { Authorization: secret } } }).includes(secret), false);
});

test('real read-only credential-test contract uses /v1/models, credentials and redirect protection', async () => {
  let called;
  const { credentials } = context();
  const testContext = { helpers: { request: async opts => { called = opts; return { statusCode: 200, headers: {}, body: Readable.from(['{"data":[]}']) }; } } };
  assert.equal((await werkApiTest.call(testContext, { data: credentials })).status, 'OK');
  assert.equal(called.method, 'GET');
  assert.equal(called.url, 'http://127.0.0.1:11434/prefix/v1/models');
  assert.equal(called.headers.Authorization, `Bearer ${secret}`);
  assert.equal(called.followRedirect, false);
  const failure = await werkApiTest.call({ helpers: { request: async () => { throw new Error(`nested ${secret}`); } } }, { data: credentials });
  assert.equal(failure.status, 'Error'); assert.ok(!failure.message.includes(secret));
});

test('discovery merges exact IDs and task spelling while retaining unavailable metadata', async () => {
  const models = { data: [{ id: 'Model-A' }, { id: 'model-a' }, { id: 'Unrelated' }] };
  const caps = { models: [
    { id: 'Model-A', tasks: ['image_generation'], available_tasks: ['image-generation'], task_statuses: { image_generation: { status: 'available' } } },
    { id: 'model-a', tasks: ['image-generation'], available_tasks: [], task_statuses: { 'image-generation': { status: 'not_implemented', message: 'No registered adapter' } } },
  ] };
  const info = mergeModels(models, caps, 'image_generation');
  assert.deepEqual(info.installed, ['Model-A', 'model-a', 'Unrelated']);
  assert.deepEqual(info.declared, ['Model-A', 'model-a']);
  assert.deepEqual(info.available, ['Model-A']);
  assert.equal(info.models[1].task_statuses['image-generation'].status, 'not_implemented');
  const fake = { api: async (_, path) => path === '/v1/models' ? models : caps, safeMessage };
  await requireModelTask(fake, 'Model-A', 'image-generation');
  await assert.rejects(requireModelTask(fake, 'model-a', 'image-generation'), /No registered adapter/);
  await assert.rejects(requireModelTask(fake, 'Model-A-derived-name', 'image-generation'), /not installed/);
  assert.throws(() => mergeModels({ models: [] }, caps));
});

test('parameter discovery forwards actual task/model/backend and full schema limits', async () => {
  let query;
  const schema = { parameters: [{ path: 'image.width', value_type: 'integer', minimum: 64, maximum: 2048 }], model_constraints: { family: 'fixture' } };
  const client = { api: async (method, path, body, qs) => { assert.equal(path, '/v1/parameters'); query = qs; return schema; } };
  assert.equal(await parameterSchema(client, 'image_generation', 'exact/model', 'cuda'), schema);
  assert.deepEqual(query, { task: 'image-generation', model: 'exact/model', backend: 'cuda' });
});

test('chat uses installed/declaration checks without a false MediaBackend availability gate', async () => {
  const models = { data: [{ id: 'chat' }, { id: 'vision' }, { id: 'legacy' }, { id: 'image-only' }] };
  const capabilities = { models: [
    { id: 'chat', tasks: ['text_generation'], available_tasks: [], task_statuses: { text_generation: { status: 'not_implemented', detail: 'Media adapter does not implement chat' } } },
    { id: 'vision', tasks: ['image_understanding'], available_tasks: [] },
    { id: 'legacy', tasks: [], available_tasks: [] },
    { id: 'image-only', tasks: ['image_generation'], available_tasks: ['image_generation'] },
  ] };
  const client = { api: async (_, path) => path === '/v1/models' ? models : capabilities, safeMessage };
  for (const model of ['chat', 'vision', 'legacy']) await requireChatModel(client, model);
  await assert.rejects(requireChatModel(client, 'image-only'), /chat-compatible/);
  await assert.rejects(requireChatModel(client, 'missing'), /not installed/);
  const { ctx } = context([{ body: models }, { body: capabilities }]);
  ctx.getCurrentNodeParameter = () => '';
  ctx.getNode = () => ({ type: 'CUSTOM.werkText' });
  const options = await commonMethods.listSearch.searchModels.call(ctx);
  assert.deepEqual(options.results.map(option => option.value), ['chat', 'vision', 'legacy']);
  assert.ok(options.results.every(option => option.name.includes('chat readiness checked')));
});

test('offline model picker returns empty without invalidating editable ID/expressions', async () => {
  const result = await commonMethods.listSearch.searchModels.call({ getCredentials: async () => { throw new Error('offline'); } });
  assert.deepEqual(result, { results: [] });
});

test('official binary helpers support an external reference and prepare real n8n binary outputs', async () => {
  const { ctx, calls } = context([], { items: [{ json: {}, binary: { image: { id: 's3:opaque-reference', data: 'not-base64', mimeType: 'image/png' } } }], binaryBytes: [{ image: Buffer.from('original-image-bytes') }] });
  const input = await readBinary(ctx, 0, 'image', 'image');
  assert.equal(input.data.toString(), 'original-image-bytes');
  const result = await binaryItem(ctx, 0, input.data, input.mimeType, { outputId: 'image-1' });
  assert.equal(result.binary.data.data, 'filesystem-v2');
  assert.equal(result.binary.data.fileName, 'werk-image-1.png');
  assert.deepEqual(result.pairedItem, { item: 0 });
  assert.equal(calls[0].binaryRead[1], 'image');
  assert.equal(calls[1].binaryWrite.toString(), 'original-image-bytes');
});

test('application/ogg binary input preserves bytes and normalizes the audio MIME alias', async () => {
  const bytes = Buffer.from('OggS fixture');
  const { ctx } = context([], { items: [{ json: {}, binary: { data: { id: 's3:audio', data: 'reference', mimeType: 'application/ogg' } } }], binaryBytes: [{ data: bytes }] });
  const input = await readBinary(ctx, 0, 'data', 'audio');
  assert.equal(input.mimeType, 'audio/ogg');
  assert.deepEqual(input.data, bytes);
});

test('execution isolates sequential items, evaluates per-item parameters, pairs multiple results and continues per item', async () => {
  const { ctx } = context([], { items: [{ json: {} }, { json: {} }, { json: {} }], parameters: [{ httpTimeoutSeconds: 11 }, { httpTimeoutSeconds: 22 }, { httpTimeoutSeconds: 33 }], continueOnFail: true });
  const order = [];
  const [items] = await executeItems(ctx, async (_, index) => {
    order.push(index);
    if (index === 1) throw new Error(`failure ${secret}`);
    return [{ json: { index, value: secret } }, { json: { index, value: 'second' } }];
  });
  assert.deepEqual(order, [0, 1, 2]);
  assert.deepEqual(items.map(item => item.pairedItem.item), [0, 0, 1, 2, 2]);
  assert.equal(items[2].json.itemIndex, 1);
  assert.ok(!JSON.stringify(items).includes(secret));
  const { ctx: failCtx } = context();
  await assert.rejects(executeItems(failCtx, async () => { throw new Error(secret); }), error => error.name === 'NodeOperationError' && error.context.itemIndex === 0 && !error.message.includes(secret));
});

module.exports = { context };

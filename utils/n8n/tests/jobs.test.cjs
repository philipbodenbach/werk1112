const test = require('node:test');
const assert = require('node:assert/strict');
const { JOB_STATUSES, jobRecord, waitForJob, submitJob, downloadedOutput, jobOutputs } = require('../dist/shared/jobs');
const { audioAnalysisTasks, audioProcessTasks } = require('../dist/shared/mediaRequests');
const { safeMessage, sanitize } = require('../dist/shared/validation');
const { WerkJobs } = require('../dist/nodes/WerkJobs/WerkJobs.node');

const completed = (id = 'job-1', task = 'video-generation', outputs = [{ id: 'output-1', mime_type: 'video/mp4' }]) => ({ id, status: 'completed', result: { id: 'result-1', model: 'model-1', task, outputs } });
const active = (status = 'queued', id = 'job-1') => ({ id, status });

function fakeClient(responses = [], downloads = {}) {
	const calls = []; const cancellations = []; const downloadCalls = [];
	const client = {
		calls, cancellations, downloadCalls,
		api: async (method, path, body, query, limits) => {
			calls.push({ method, path, body, query, limits });
			const response = responses.shift();
			if (response instanceof Error) throw response;
			if (typeof response === 'function') return response({ method, path, body, query, limits });
			assert.notEqual(response, undefined, `Unexpected ${method} ${path}`);
			return response;
		},
		cancelJob: async (id) => { cancellations.push(id); },
		download: async (path) => { downloadCalls.push(path); assert.ok(downloads[path], `Unexpected download ${path}`); return downloads[path]; },
		safeMessage: (value) => safeMessage(value, ['job-secret']),
		redact: (value) => sanitize(value, ['job-secret']),
	};
	return client;
}

function context(params = {}, controller = new AbortController()) {
	const prepared = [];
	return {
		prepared,
		getNodeParameter: (name, _index, fallback) => params[name] ?? fallback,
		getExecutionCancelSignal: () => controller.signal,
		helpers: {
			prepareBinaryData: async (data, fileName, mimeType) => { prepared.push({ data, fileName, mimeType }); return { data: 'filesystem-v2', id: `external:${prepared.length}`, fileName, mimeType }; },
		},
	};
}

test('job contract accepts all seven states, rejects unknown status and changed ID', () => {
	assert.deepEqual(JOB_STATUSES, ['queued', 'loading', 'running', 'encoding', 'completed', 'failed', 'cancelled']);
	for (const status of JOB_STATUSES) assert.equal(jobRecord(status === 'completed' ? completed() : active(status), 'job-1').status, status);
	for (const value of [null, [], {}, { id: '', status: 'queued' }, active('pending'), active('done'), active('running', 'different-job')]) assert.throws(() => jobRecord(value, 'job-1'));
});

test('completed records require concrete result outputs, output IDs and MIME types', () => {
	for (const result of [undefined, null, [], {}, { outputs: [] }, { outputs: {} }, { outputs: [null] }, { outputs: [{}] }, { outputs: [{ id: 'o' }] }, { outputs: [{ id: '', mime_type: 'audio/wav' }] }]) assert.throws(() => jobRecord({ id: 'job-1', status: 'completed', result }));
	assert.equal(jobRecord(completed()).result.outputs[0].id, 'output-1');
});

test('polling traverses queued/loading/running/encoding/completed using one stable GET path', async () => {
	const client = fakeClient([active('loading'), active('running'), active('encoding'), completed()]);
	const result = await waitForJob(client, active(), { waitSeconds: 1, pollSeconds: 0.001, cancelOnAbort: true });
	assert.equal(result.status, 'completed'); assert.equal(client.calls.length, 4);
	for (const call of client.calls) { assert.equal(call.method, 'GET'); assert.equal(call.path, '/v1/jobs/job-1'); assert.equal(call.body, undefined); assert.ok(call.limits.timeoutMs > 0 && call.limits.timeoutMs <= 1000); }
	assert.deepEqual(client.cancellations, []);
});

test('failed and cancelled terminal states stop immediately without cancellation or retry', async () => {
	for (const status of ['failed', 'cancelled']) {
		const client = fakeClient();
		await assert.rejects(waitForJob(client, { ...active(status), error: { message: 'backend job-secret failure', code: 'task_failed' } }, { waitSeconds: 1, pollSeconds: 0.001, cancelOnAbort: true }), (error) => {
			assert.match(error.message, new RegExp(status)); assert.match(error.message, /job-1/); assert.doesNotMatch(error.message, /job-secret/); return true;
		});
		assert.deepEqual(client.calls, []); assert.deepEqual(client.cancellations, []);
	}
});

test('polling rejects changed identity and invalid completed outputs, preserving original job ID', async () => {
	for (const response of [active('running', 'foreign-job'), { id: 'job-1', status: 'completed', result: { outputs: [] } }, active('invented-status')]) {
		const client = fakeClient([response]);
		await assert.rejects(waitForJob(client, active(), { waitSeconds: 1, pollSeconds: 0.001, cancelOnAbort: true }), /job-1/);
		assert.equal(client.calls.length, 1); assert.deepEqual(client.cancellations, ['job-1']);
	}
});

test('wait deadline keeps known job ID and best-effort cancellation applies only when selected', async () => {
	for (const cancelOnAbort of [false, true]) {
		// Fractional timers may wake just before the deadline, permitting one last bounded GET.
		const client = fakeClient(Array.from({ length: 100 }, () => active()));
		const started = performance.now();
		await assert.rejects(waitForJob(client, active(), { waitSeconds: 0.01, pollSeconds: 0.1, cancelOnAbort }), (error) => {
			assert.match(error.message, /timed out/); assert.match(error.message, /jobId: job-1/); return true;
		});
		assert.ok(performance.now() - started < 1000, 'bounded wait');
		assert.ok(client.calls.every((call) => call.method === 'GET')); assert.deepEqual(client.cancellations, cancelOnAbort ? ['job-1'] : []);
	}
});

test('remaining wait deadline is passed to an in-flight GET and transport failure is never retried', async () => {
	const client = fakeClient([({ limits }) => { assert.ok(limits.timeoutMs > 0 && limits.timeoutMs < 50); throw new Error('HTTP timeout'); }]);
	await assert.rejects(waitForJob(client, active(), { waitSeconds: 0.05, pollSeconds: 0.001, cancelOnAbort: false }), /HTTP timeout.*jobId: job-1/);
	assert.equal(client.calls.length, 1); assert.deepEqual(client.cancellations, []);
});

test('execution abort while polling triggers one cleanup and cleanup failure preserves original error', async () => {
	const controller = new AbortController();
	const client = fakeClient();
	client.cancelJob = async (id) => { client.cancellations.push(id); throw new Error('cleanup failed job-secret'); };
	const waiting = waitForJob(client, active(), { waitSeconds: 1, pollSeconds: 0.5, cancelOnAbort: true, signal: controller.signal });
	controller.abort();
	await assert.rejects(waiting, (error) => { assert.match(error.message, /cancelled.*jobId: job-1/); assert.doesNotMatch(error.message, /cleanup failed|job-secret/); return true; });
	assert.deepEqual(client.cancellations, ['job-1']); assert.deepEqual(client.calls, []);
});

test('submit-only returns acceptance metadata immediately and never polls or downloads', async () => {
	const client = fakeClient([active()]);
	const output = await submitJob(context({ waitMode: 'submitOnly' }), client, 3, '/v1/videos/generations', { model: 'selected-model', prompt: 'clip' }, 'video-generation');
	assert.equal(output[0].json.jobId, 'job-1'); assert.equal(output[0].json.status, 'queued'); assert.equal(output[0].json.model, 'selected-model'); assert.equal(output[0].json.task, 'video-generation'); assert.deepEqual(output[0].pairedItem, { item: 3 });
	assert.deepEqual(client.calls.map((call) => call.method), ['POST']); assert.deepEqual(client.downloadCalls, []); assert.deepEqual(client.cancellations, []);
});

test('self-started job uses best-effort cancellation on abort; ambiguous submission never repeats', async () => {
	const controller = new AbortController();
	const client = fakeClient([() => { controller.abort(); return active(); }]);
	await assert.rejects(submitJob(context({}, controller), client, 0, '/v1/audio/speech', { model: 'm', input: 'hello', async: true }, 'text-to-speech'), /jobId: job-1/);
	assert.equal(client.calls.filter((call) => call.method === 'POST').length, 1); assert.deepEqual(client.cancellations, ['job-1']);
	const ambiguous = fakeClient([new Error('connection lost after submit')]);
	await assert.rejects(submitJob(context(), ambiguous, 0, '/v1/jobs', { model: 'm', task: 'audio-embedding' }, 'audio-embedding'), /connection lost/);
	assert.equal(ambiguous.calls.length, 1); assert.deepEqual(ambiguous.cancellations, [], 'unknown job ID cannot be cancelled safely');
});

test('malformed known-ID submission is cancelled by submit-and-wait; missing ID cannot be cancelled', async () => {
	for (const response of [active('unknown-status'), { id: 'job-1', status: 'completed', result: { outputs: [] } }]) {
		const client = fakeClient([response]);
		await assert.rejects(submitJob(context(), client, 0, '/v1/videos/generations', { model: 'm', prompt: 'clip' }, 'video-generation'), /job-1/);
		assert.deepEqual(client.cancellations, ['job-1']); assert.deepEqual(client.calls.map((call) => call.method), ['POST']);
	}
	for (const response of [{ status: 'queued' }, { id: '', status: 'queued' }]) {
		const client = fakeClient([response]);
		await assert.rejects(submitJob(context(), client, 0, '/v1/videos/generations', { model: 'm', prompt: 'clip' }, 'video-generation'), /ID/);
		assert.deepEqual(client.cancellations, []); assert.equal(client.calls.length, 1);
	}
});

test('submit-and-wait performs one POST then only GET and output downloads', async () => {
	const client = fakeClient([active(), completed()], { '/v1/outputs/output-1': { data: Buffer.from('MP4 fixture'), mimeType: 'video/mp4' } });
	const output = await submitJob(context({ pollSeconds: 0.1, waitSeconds: 1 }), client, 2, '/v1/videos/generations', { model: 'model-1', prompt: 'clip' }, 'video-generation');
	assert.deepEqual(client.calls.map((call) => call.method), ['POST', 'GET']); assert.deepEqual(client.downloadCalls, ['/v1/outputs/output-1']); assert.deepEqual(output[0].pairedItem, { item: 2 });
	assert.equal(output[0].binary.data.mimeType, 'video/mp4');
});

test('invalid wait configuration fails before generation submission', async () => {
	for (const params of [{ waitMode: 'other' }, { waitSeconds: 0 }, { waitSeconds: Infinity }, { waitSeconds: 86401 }, { pollSeconds: 0 }, { pollSeconds: 301 }]) {
		const client = fakeClient(); await assert.rejects(submitJob(context(params), client, 0, '/v1/jobs', { model: 'm' }, 'speech-to-text')); assert.equal(client.calls.length, 0);
	}
});

test('output downloads use each output ID, preserve bytes and pair multiple results with input item', async () => {
	const first = Buffer.from('first'); const second = Buffer.from('second'); const ctx = context();
	const client = fakeClient([], { '/v1/outputs/output%2Fone': { data: first, mimeType: 'audio/wav' }, '/v1/outputs/output-two': { data: second, mimeType: 'audio/flac' } });
	const record = completed('job-do-not-download', 'voice-conversion', [{ id: 'output/one', mime_type: 'audio/wav' }, { id: 'output-two', mime_type: 'audio/flac' }]);
	const output = await jobOutputs(ctx, client, 4, record);
	assert.deepEqual(client.downloadCalls, ['/v1/outputs/output%2Fone', '/v1/outputs/output-two']); assert.equal(client.calls.length, 0);
	assert.deepEqual(output.map((item) => item.pairedItem), [{ item: 4 }, { item: 4 }]); assert.deepEqual(ctx.prepared.map((entry) => entry.data), [first, second]); assert.match(ctx.prepared[1].fileName, /\.flac$/);
});

test('all four transformation tasks produce audio binary; GIF remains a valid video output', async () => {
	for (const task of audioProcessTasks) {
		const client = fakeClient([], { '/v1/outputs/o': { data: Buffer.from('audio'), mimeType: 'audio/wav' } });
		const output = await downloadedOutput(context(), client, 0, 'o', { task }, 'audio/wav'); assert.equal(output.binary.data.mimeType, 'audio/wav');
	}
	for (const task of ['video-generation', 'image-to-video']) {
		const client = fakeClient([], { '/v1/outputs/gif': { data: Buffer.from('GIF89a'), mimeType: 'image/gif' } });
		const output = await downloadedOutput(context(), client, 0, 'gif', { task }, 'image/gif'); assert.equal(output.binary.data.mimeType, 'image/gif'); assert.match(output.binary.data.fileName, /\.gif$/);
	}
});

test('text, JSON, NDJSON and embeddings retain structured values instead of becoming audio files', async () => {
	for (const [mimeType, body, expected] of [
		['text/plain', 'hello', 'hello'],
		['application/json', '{"text":"hallo","segments":[{"start":0,"end":1.25}]}', { text: 'hallo', segments: [{ start: 0, end: 1.25 }] }],
		['application/x-ndjson', '{"label":"voice"}\n\n{"confidence":0.75}\n', [{ label: 'voice' }, { confidence: 0.75 }]],
		['application/json', '{"embedding":[0,0.125,-0.5],"dimensions":3}', { embedding: [0, 0.125, -0.5], dimensions: 3 }],
	]) {
		const ctx = context(); const client = fakeClient([], { '/v1/outputs/analysis': { data: Buffer.from(body), mimeType } });
		const item = await downloadedOutput(ctx, client, 1, 'analysis', { task: 'audio-embedding' }, mimeType);
		assert.equal(item.binary, undefined); assert.deepEqual(item.json.result, expected); assert.deepEqual(item.pairedItem, { item: 1 }); assert.equal(ctx.prepared.length, 0);
	}
});

test('every analysis task refuses audio files and generated media refuses text or wrong modality', async () => {
	for (const task of audioAnalysisTasks) {
		const client = fakeClient([], { '/v1/outputs/o': { data: Buffer.from('audio'), mimeType: 'audio/wav' } });
		await assert.rejects(downloadedOutput(context(), client, 0, 'o', { task }), /non-text\/JSON/);
	}
	for (const task of ['video-generation', 'image-to-video', 'text-to-speech', 'audio-generation', ...audioProcessTasks]) {
		const client = fakeClient([], { '/v1/outputs/o': { data: Buffer.from('not media'), mimeType: 'text/plain' } });
		await assert.rejects(downloadedOutput(context(), client, 0, 'o', { task }), /non-(audio|video)/);
	}
});

test('structured output validates UTF-8, JSON, NDJSON, safe numbers and finite size', async () => {
	for (const [mimeType, data] of [['text/plain', Buffer.from([0xc3, 0x28])], ['application/json', Buffer.from('{bad json')], ['application/x-ndjson', Buffer.from('{}\n{bad json')], ['application/json', Buffer.from('{"seed":9007199254740993}')], ['text/plain', Buffer.alloc(16 * 1024 * 1024 + 1)]]) {
		const client = fakeClient([], { '/v1/outputs/o': { data, mimeType } });
		await assert.rejects(downloadedOutput(context(), client, 0, 'o', { task: 'speech-to-text' }));
	}
	const mismatch = fakeClient([], { '/v1/outputs/o': { data: Buffer.from('{}'), mimeType: 'application/json' } });
	await assert.rejects(downloadedOutput(context(), mismatch, 0, 'o', {}, 'video/mp4'), /differs/);
});

function nodeFixture(parameters, responder, controller = new AbortController()) {
	const ctx = context(parameters, controller); const requests = []; const node = new WerkJobs();
	ctx.getInputData = () => [{ json: {} }];
	ctx.getNode = () => ({ id: 'jobs', name: 'WERK Jobs (Beta)', type: 'CUSTOM.werkJobs', typeVersion: 1, parameters, position: [0, 0] });
	ctx.getCredentials = async () => ({ baseUrl: 'http://werk.test', authMode: 'apiKey', apiKey: 'job-secret', verifyTls: true });
	ctx.continueOnFail = () => false;
	const handle = async (options) => {
		const request = { method: options.method, path: new URL(options.url).pathname, timeout: options.timeout, signal: options.abortSignal };
		requests.push(request); const result = await responder(request);
		return { statusCode: 200, headers: { 'content-type': result.mimeType ?? 'application/json' }, body: result.data ?? Buffer.from(JSON.stringify(result.json ?? result)) };
	};
	ctx.helpers.httpRequestWithAuthentication = async (_type, options) => handle(options);
	ctx.helpers.httpRequest = handle;
	return { run: () => node.execute.call(ctx), requests, ctx };
}

test('WERK Jobs Get is read-only and does not generate, wait or download', async () => {
	const f = nodeFixture({ operation: 'get', jobId: 'job/opaque' }, ({ method, path }) => { assert.equal(method, 'GET'); assert.equal(path, '/v1/jobs/job%2Fopaque'); return active('encoding', 'job/opaque'); });
	const [[item]] = await f.run(); assert.equal(item.json.jobId, 'job/opaque'); assert.equal(item.json.status, 'encoding'); assert.equal(f.requests.length, 1); assert.equal(f.ctx.prepared.length, 0);
});

test('WERK Jobs Wait gets existing job once and optionally downloads completed outputs', async () => {
	for (const download of [true, false]) {
		const f = nodeFixture({ operation: 'wait', jobId: 'job-1', download }, ({ path }) => path === '/v1/jobs/job-1' ? completed() : { mimeType: 'video/mp4', data: Buffer.from('video') });
		const [[item]] = await f.run(); assert.equal(item.json.jobId, 'job-1'); assert.equal(Boolean(item.binary), download); assert.ok(f.requests.every((request) => request.method === 'GET')); assert.equal(f.requests.length, download ? 2 : 1);
	}
});

test('WERK Jobs existing-job cancellation on interrupted wait is opt-in, with independent bounded cleanup', async () => {
	for (const cancelOnAbort of [undefined, true]) {
		const controller = new AbortController();
		const f = nodeFixture({ operation: 'wait', jobId: 'job-1', cancelOnAbort }, ({ method, signal, timeout }) => {
			if (method === 'GET') { controller.abort(); return active('running'); }
			assert.equal(method, 'DELETE'); assert.equal(signal.aborted, false); assert.ok(timeout > 0 && timeout <= 10000);
			return active('cancelled');
		}, controller);
		await assert.rejects(f.run(), /job-1/);
		assert.deepEqual(f.requests.map((request) => request.method), cancelOnAbort ? ['GET', 'DELETE'] : ['GET']);
	}
});

test('WERK Jobs Wait handles malformed initial known-ID response under explicit cancellation policy', async () => {
	for (const cancelOnAbort of [false, true]) {
		const f = nodeFixture({ operation: 'wait', jobId: 'job-1', cancelOnAbort }, ({ method }) => method === 'GET' ? active('unknown-status') : active('cancelled'));
		await assert.rejects(f.run(), /unknown status/);
		assert.deepEqual(f.requests.map((request) => request.method), cancelOnAbort ? ['GET', 'DELETE'] : ['GET']);
	}
});

test('WERK Jobs Cancel sends cooperative DELETE and returns actual server state', async () => {
	const f = nodeFixture({ operation: 'cancel', jobId: 'job-1' }, ({ method }) => { assert.equal(method, 'DELETE'); return active('running'); });
	const [[item]] = await f.run(); assert.equal(item.json.status, 'running'); assert.equal(f.requests.length, 1);
});

test('WERK Jobs Download uses supplied Output ID only, never job or result ID', async () => {
	const f = nodeFixture({ operation: 'download', outputId: 'output/encoded', jobId: 'job-must-not-be-used', resultId: 'result-must-not-be-used' }, ({ path, method }) => {
		assert.equal(method, 'GET'); assert.equal(path, '/v1/outputs/output%2Fencoded'); return { mimeType: 'audio/wav', data: Buffer.from('audio') };
	});
	const [[item]] = await f.run(); assert.equal(item.json.outputId, 'output/encoded'); assert.equal(f.requests.length, 1);
	const missing = nodeFixture({ operation: 'download', jobId: 'job-1' }, () => { throw new Error('must not perform HTTP'); });
	await assert.rejects(missing.run(), /Output ID/); assert.equal(missing.requests.length, 0);
});

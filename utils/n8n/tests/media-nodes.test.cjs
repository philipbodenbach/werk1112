const test = require('node:test');
const assert = require('node:assert/strict');
const { WerkImage } = require('../dist/nodes/WerkImage/WerkImage.node');
const { WerkVision } = require('../dist/nodes/WerkVision/WerkVision.node');
const { WerkText } = require('../dist/nodes/WerkText/WerkText.node');
const { WerkVideo } = require('../dist/nodes/WerkVideo/WerkVideo.node');
const { WerkAudio } = require('../dist/nodes/WerkAudio/WerkAudio.node');
const { audioTasks } = require('../dist/shared/mediaRequests');

// An actual 1x1 PNG. Binary references deliberately do not contain Base64 data.
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==', 'base64');
const wav = Buffer.from('RIFF....WAVEfixture bytes');
const allTasks = ['image-generation', 'image-understanding', 'text-generation', 'video-generation', 'image-to-video', ...audioTasks];

function fixture(NodeClass, parameterItems, responder, inputs = [], continueOnFail = false) {
	const requests = []; const binaryReads = []; const prepared = []; const events = [];
	const node = new NodeClass();
	const signal = new AbortController().signal;
	const handle = async (options, authenticated) => {
		const url = new URL(options.url); const body = options.body ? JSON.parse(options.body) : undefined;
		const request = { method: options.method, path: url.pathname, url: options.url, body, authenticated };
		requests.push(request);
		let result;
		if (url.pathname === '/v1/models') result = { data: [{ id: 'model-0' }, { id: 'model-1' }] };
		else if (url.pathname === '/v1/capabilities') result = { models: [0, 1].map((index) => ({ id: `model-${index}`, tasks: allTasks, available_tasks: allTasks, task_statuses: {} })) };
		else result = await responder(request);
		return { statusCode: result.statusCode ?? 200, headers: { 'content-type': result.mimeType ?? 'application/json' }, body: Buffer.isBuffer(result.bytes) ? result.bytes : Buffer.from(JSON.stringify(result.json ?? result)) };
	};
	const context = {
		getInputData: () => parameterItems.map((_, index) => inputs[index] ?? { json: {} }),
		getNode: () => ({ id: 'node1', name: node.description.displayName, type: `CUSTOM.${node.description.name}`, typeVersion: 1, parameters: {}, position: [0, 0] }),
		getNodeParameter(name, index, fallback, options) {
			const value = parameterItems[index]?.[name] ?? fallback;
			return options?.extractValue && value && typeof value === 'object' && '__rl' in value ? value.value : value;
		},
		getCredentials: async () => ({ baseUrl: 'http://werk.test', authMode: 'apiKey', apiKey: 'fixture-secret', verifyTls: true }),
		continueOnFail: () => continueOnFail,
		getExecutionCancelSignal: () => signal,
		helpers: {
			httpRequest: async (options) => handle(options, false),
			httpRequestWithAuthentication: async (credentialType, options) => { assert.equal(credentialType, 'werkApi'); return handle(options, true); },
			assertBinaryData: (index, property) => { const reference = inputs[index]?.binary?.[property]; assert.ok(reference, `binary ${index}/${property}`); return reference; },
			getBinaryDataBuffer: async (index, property) => { binaryReads.push([index, property]); return inputs[index].stored[property]; },
			prepareBinaryData: async (data, fileName, mimeType) => { prepared.push({ data, fileName, mimeType }); return { data: 'filesystem-v2', id: `external-store:${prepared.length}`, mimeType, fileName }; },
		},
	};
	return { run: () => node.execute.call(context), node, context, requests, binaryReads, prepared, events };
}

function parameters(extra = {}, index = 0) { return { model: { __rl: true, mode: 'id', value: `model-${index}` }, ...extra }; }
function binaryInput(values) {
	return { json: {}, stored: Object.fromEntries(Object.entries(values).map(([key, entry]) => [key, entry.bytes])), binary: Object.fromEntries(Object.entries(values).map(([key, entry]) => [key, { data: 'filesystem-v2', id: `external-store:${key}`, mimeType: entry.mimeType }])) };
}
function completed(id, model, task, outputs) { return { id, status: 'completed', result: { id: 'result1', model, task, outputs } }; }

test('image uses per-item parameters, sequential generations, pairedItem and real binary helpers', async () => {
	let active = 0; let maximum = 0;
	const f = fixture(WerkImage, [parameters({ operation: 'generate', prompt: 'first' }), parameters({ operation: 'generate', prompt: 'second' }, 1)], async ({ body, method, path }) => {
		assert.equal(method, 'POST'); assert.equal(path, '/v1/images/generations');
		active++; maximum = Math.max(maximum, active); await new Promise((resolve) => setTimeout(resolve, 2)); active--;
		return { data: [1, 2].map((i) => ({ id: `temporary-${body.model}-${i}`, mime_type: 'image/png', b64_json: png.toString('base64') })), werk: { model: body.model, effective_request: { inputs: [{ source: { kind: 'base64', data: 'never-output' } }] }, output_path: '/tmp/private.png' } };
	});
	const [items] = await f.run();
	assert.equal(items.length, 4); assert.equal(maximum, 1);
	assert.deepEqual(items.map((item) => item.pairedItem), [{ item: 0 }, { item: 0 }, { item: 1 }, { item: 1 }]);
	assert.deepEqual(f.requests.filter((request) => request.method === 'POST').map((request) => [request.body.model, request.body.prompt]), [['model-0', 'first'], ['model-1', 'second']]);
	for (const item of items) { assert.equal(item.binary.data.id.startsWith('external-store:'), true); assert.equal(item.json.outputId, null); assert.equal(item.json.downloadable, false); }
	for (const output of f.prepared) { assert.deepEqual(output.data, png); assert.equal(output.mimeType, 'image/png'); assert.match(output.fileName, /\.png$/); }
	assert.equal(f.requests.filter((request) => request.path.startsWith('/v1/outputs/')).length, 0);
	assert.doesNotMatch(JSON.stringify(items), /never-output|fixture-secret|private\.png|iVBOR/);
});

test('image URL output is downloaded without credentials at an external origin', async () => {
	const f = fixture(WerkImage, [parameters({ operation: 'generate', prompt: 'URL image', configuration: { responseFormat: 'url' } })], ({ method, url, authenticated }) => {
		if (method === 'POST') return { data: [{ id: 'retained-output', url: 'https://media.test/signed.png?token=secret-signature' }] };
		assert.equal(url, 'https://media.test/signed.png?token=secret-signature'); assert.equal(authenticated, false);
		return { bytes: png, mimeType: 'image/png' };
	});
	const [[item]] = await f.run(); assert.equal(item.json.outputId, 'retained-output'); assert.equal(item.json.downloadable, true);
	assert.doesNotMatch(JSON.stringify(item.json), /secret-signature/);
});

test('multi-MiB Base64 image validation remains bounded without regex stack overflow', async () => {
	const large = Buffer.concat([png, Buffer.alloc(4 * 1024 * 1024)]);
	const f = fixture(WerkImage, [parameters({ operation: 'generate', prompt: 'large image' })], () => ({ data: [{ b64_json: large.toString('base64') }] }));
	const [[item]] = await f.run(); assert.ok(item.binary.data); assert.deepEqual(f.prepared[0].data, large);
});

test('vision reads external binary storage in defined image order and sanitizes embedded requests', async () => {
	const first = Buffer.concat([png, Buffer.from('one')]); const second = Buffer.concat([png, Buffer.from('two')]);
	const f = fixture(WerkVision, [parameters({ operation: 'analyze', prompt: 'compare images', images: { image: [{ binaryProperty: 'second' }, { binaryProperty: 'first' }] }, options: { imageDetail: 'low' } })], ({ body }) => {
		assert.equal(body.messages.length, 1);
		const content = body.messages[0].content;
		assert.equal(content[0].image_url.url, `data:image/png;base64,${second.toString('base64')}`);
		assert.equal(content[1].image_url.url, `data:image/png;base64,${first.toString('base64')}`);
		assert.equal(content[2].text, 'compare images');
		return { id: 'chat1', model: body.model, choices: [{ index: 0, message: { role: 'assistant', content: 'Two images' }, finish_reason: 'stop' }], usage: { prompt_tokens: 12, completion_tokens: 3, total_tokens: 15 } };
	}, [binaryInput({ first: { bytes: first, mimeType: 'image/png' }, second: { bytes: second, mimeType: 'image/png' } })]);
	const [[item]] = await f.run(); assert.equal(item.json.text, 'Two images'); assert.equal(item.json.usage.total_tokens, 15);
	assert.deepEqual(f.binaryReads, [[0, 'second'], [0, 'first']]); assert.doesNotMatch(JSON.stringify(item.json), /iVBOR|fixture-secret/);
});

test('text preserves null-content tool calls, finish reason and usage without executing tools', async () => {
	const toolCalls = [{ id: 'call1', type: 'function', function: { name: 'lookup', arguments: '{"query":"hello"}' } }];
	const f = fixture(WerkText, [parameters({ operation: 'complete', messages: { message: [{ role: 'user', content: 'hello' }] } })], ({ path, body }) => {
		assert.equal(path, '/v1/chat/completions'); assert.equal(body.stream, false);
		return { id: 'chat2', model: body.model, choices: [{ index: 0, message: { role: 'assistant', content: null, tool_calls: toolCalls }, finish_reason: 'tool_calls' }], usage: { prompt_tokens: 1, completion_tokens: 3, total_tokens: 4 } };
	});
	const [[item]] = await f.run(); assert.equal(item.json.text, ''); assert.deepEqual(item.json.toolCalls, toolCalls); assert.equal(item.json.finishReason, 'tool_calls');
	assert.equal(f.requests.filter((request) => request.method === 'POST').length, 1);
});

test('image-to-video submit only embeds one image and performs exactly one submission with no polling', async () => {
	const f = fixture(WerkVideo, [parameters({ operation: 'imageToVideo', prompt: 'move', binaryProperty: 'firstFrame', waitMode: 'submitOnly' })], ({ path, body }) => {
		assert.equal(path, '/v1/videos/generations'); assert.deepEqual(body.initial_image, { base64: png.toString('base64'), mime_type: 'image/png' });
		return { statusCode: 202, json: { id: 'video-job', status: 'queued' } };
	}, [binaryInput({ firstFrame: { bytes: png, mimeType: 'image/png' } })]);
	const [[item]] = await f.run(); assert.equal(item.json.jobId, 'video-job'); assert.equal(item.json.status, 'queued'); assert.equal(item.json.task, 'image-to-video');
	assert.equal(f.requests.filter((request) => request.method === 'POST').length, 1); assert.equal(f.requests.filter((request) => request.path.startsWith('/v1/jobs/')).length, 0); assert.equal(f.prepared.length, 0);
});

test('TTS posts async true and returns unchanged audio bytes via n8n helper', async () => {
	const f = fixture(WerkAudio, [parameters({ operation: 'generate', task: 'text-to-speech', prompt: 'Guten Tag', configuration: { seed: 0, sampleRate: 0, channels: 0 }, waitMode: 'wait' })], ({ method, path, body }) => {
		if (method === 'POST') {
			assert.equal(path, '/v1/audio/speech'); assert.equal(body.async, true); assert.equal(body.input, 'Guten Tag'); assert.equal(body.prompt, undefined); assert.equal(body.parameters, undefined);
			return completed('tts-job', body.model, 'text-to-speech', [{ id: 'speech-output', mime_type: 'audio/wav' }]);
		}
		assert.equal(path, '/v1/outputs/speech-output'); return { bytes: wav, mimeType: 'audio/wav' };
	});
	const [[item]] = await f.run(); assert.equal(item.json.jobId, 'tts-job'); assert.equal(item.json.outputId, 'speech-output'); assert.equal(item.binary.data.mimeType, 'audio/wav'); assert.deepEqual(f.prepared[0].data, wav);
});

test('audio transcription uses canonical input and emits structured JSON instead of audio binary', async () => {
	const f = fixture(WerkAudio, [parameters({ operation: 'analyze', task: 'speech-to-text', binaryProperty: 'recording', waitMode: 'wait' })], ({ method, path, body }) => {
		if (method === 'POST') {
			assert.equal(path, '/v1/jobs'); assert.deepEqual(body.inputs, [{ modality: 'audio', role: 'input_audio', source: { kind: 'base64', data: wav.toString('base64') }, mime_type: 'audio/wav' }]);
			return completed('transcribe-job', body.model, 'speech-to-text', [{ id: 'transcript', mime_type: 'application/json' }]);
		}
		assert.equal(path, '/v1/outputs/transcript'); return { bytes: Buffer.from('{"text":"Hallo","segments":[{"start":0,"end":1.5}]}'), mimeType: 'application/json' };
	}, [binaryInput({ recording: { bytes: wav, mimeType: 'audio/wav' } })]);
	const [[item]] = await f.run(); assert.equal(item.binary, undefined); assert.equal(item.json.text, 'Hallo'); assert.deepEqual(item.json.result.segments, [{ start: 0, end: 1.5 }]); assert.equal(f.prepared.length, 0);
});

test('continueOnFail isolates item validation failures and later items still execute', async () => {
	const f = fixture(WerkImage, [parameters({ operation: 'generate', prompt: '' }), parameters({ operation: 'generate', prompt: 'valid' }, 1)], () => ({ data: [{ b64_json: png.toString('base64') }] }), [], true);
	const [items] = await f.run(); assert.match(items[0].json.error, /Prompt/); assert.equal(items[0].json.itemIndex, 0); assert.equal(items[1].json.model, 'model-1'); assert.deepEqual(items[1].pairedItem, { item: 1 });
	assert.equal(f.requests.filter((request) => request.method === 'POST').length, 1);
});

test('ambiguous generation transport failure never retries a POST', async () => {
	const f = fixture(WerkImage, [parameters({ operation: 'generate', prompt: 'valid' })], () => { throw new Error('lost response fixture-secret'); });
	await assert.rejects(f.run(), (error) => { assert.doesNotMatch(error.message, /fixture-secret/); return /lost response/.test(error.message); });
	assert.equal(f.requests.filter((request) => request.method === 'POST').length, 1);
});

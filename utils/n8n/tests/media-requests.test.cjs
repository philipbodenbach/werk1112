const test = require('node:test');
const assert = require('node:assert/strict');
const { normalizeParameters, tristate } = require('../dist/shared/parameters');
const { buildRouting } = require('../dist/shared/routing');
const { buildImageRequest, buildVideoRequest, buildAudioRequest, buildAudioInputRequest, buildTextRequest, buildVisionRequest, audioAnalysisTasks, audioProcessTasks } = require('../dist/shared/mediaRequests');

const base = { model: 'model/exact_ID', prompt: 'A test prompt' };
const wav = { data: Buffer.from('RIFF....WAVE'), mimeType: 'audio/wav' };
const png = { data: Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), mimeType: 'image/png' };

test('parameter namespaces flatten and preserve false, zero and list overrides', () => {
	assert.deepEqual(normalizeParameters({ image: { custom: false }, zero: 0, 'image.loras': { operation: 'add', values: [{ name: 'test', weight: 0 }] } }, 'image'), {
		'image.custom': false, 'image.zero': 0, 'image.loras': { operation: 'add', values: [{ name: 'test', weight: 0 }] },
	});
	for (const operation of ['inherit', 'replace', 'add', 'clear']) assert.deepEqual(normalizeParameters({ loras: { operation } }, 'image'), { 'image.loras': { operation } });
});

test('parameters reject foreign namespaces, exact duplicates, normalized duplicates, forbidden dedicated and transport fields', () => {
	for (const value of [
		'{"foo":1,"foo":2}', '{"foo":1,"image.foo":2}', '{"image":{"foo":1},"image.foo":2}',
		'{"tts.foo":2}', '{"url":"http://example.test"}', '{"image.headers":{"Authorization":"x"}}',
		'{"image":{"api_key":"x"}}', '{"__proto__":{}}',
	]) assert.throws(() => normalizeParameters(value, 'image'));
	assert.throws(() => normalizeParameters({ steps: 2 }, 'image', ['image.steps']), /dedicated/);
	assert.throws(() => normalizeParameters({ seed: 9007199254740992 }, 'image'), /unsafe|safe/);
	assert.throws(() => normalizeParameters('{"seed":9007199254740993}', 'image'), /unsafe/);
	assert.throws(() => normalizeParameters({ guidance: NaN }, 'image'), /finite/);
});

test('list operations enforce genuine empty inherit/clear and arrays for replace/add', () => {
	assert.throws(() => normalizeParameters({ loras: { operation: 'clear', values: ['x'] } }, 'image'), /does not accept/);
	assert.throws(() => normalizeParameters({ loras: { operation: 'inherit', values: ['x'] } }, 'image'), /does not accept/);
	assert.throws(() => normalizeParameters({ loras: { operation: 'replace', values: 'x' } }, 'image'), /array/);
	assert.throws(() => normalizeParameters({ loras: { operation: 'add', url: 'x' } }, 'image'), /unknown fields/);
	assert.throws(() => normalizeParameters({ loras: { operation: 'erase' } }, 'image'), /one of/);
});

test('routing preserves all reference controls and three-state booleans', () => {
	const input = { backend: ' diffusers ', accelerator: 'cuda', device: 'cuda:1', precision: 'bf16', quantization: 'int8', profile: 'custom', quality: 'high', performance_preference: 'memory', fallback_policy: 'backend', parameter_policy: 'strict', allow_cpu_offload: 'disabled', allow_sequential_offload: 'enabled', allow_component_offload: 'inherit', allow_disk_offload: 'disabled', attention_backend: 'sdpa', compile: 'disabled', inferenceTimeoutSeconds: 450, additionalRoutingParameters: { extension: 0 } };
	const result = buildRouting(input);
	assert.deepEqual(result.request, { backend: 'diffusers', accelerator: 'cuda', device: 'cuda:1', precision: 'bf16', quantization: 'int8', profile: 'custom', attention_backend: 'sdpa', quality: 'high', performance_preference: 'memory', fallback_policy: 'backend', parameter_policy: 'strict', allow_cpu_offload: false, allow_sequential_offload: true, allow_disk_offload: false, compile: false, timeout_seconds: 450 });
	assert.deepEqual(result.parameters, { 'routing.extension': 0 });
	assert.deepEqual(buildRouting({ compile: 'inherit', inferenceTimeoutSeconds: 0, backend: '' }), { request: {}, parameters: {} });
	assert.equal(tristate('disabled', 'test'), false);
	assert.throws(() => buildRouting({ compile: false }), /one of/);
	assert.throws(() => buildRouting({ timeout_seconds: 22 }), /unsupported/);
	assert.throws(() => buildRouting({ additionalRoutingParameters: { timeout: 22 } }), /dedicated/);
	assert.throws(() => buildRouting({ quality: 'imaginary' }), /one of/);
});

test('omitted image options really inherit; explicit zero and false survive', () => {
	assert.deepEqual(buildImageRequest(base), base);
	assert.deepEqual(buildImageRequest({ ...base, configuration: { guidance: 0, seed: 0, vaeTiling: 'disabled' }, routing: { compile: 'disabled' } }), { ...base, compile: false, parameters: { 'image.guidance': 0, 'image.seed': 0, 'image.vae_tiling': false } });
});

test('image separates format/response transport and count from batch size', () => {
	const request = buildImageRequest({ ...base, negativePrompt: 'fog', configuration: { width: 512, height: 768, count: 2, batchSize: 1, steps: 12, outputFormat: 'jpeg', responseFormat: 'url', style: 'natural' } });
	assert.equal(request.n, 2); assert.equal(request.size, '512x768'); assert.equal(request.output_format, 'jpeg'); assert.equal(request.response_format, 'url'); assert.equal(request.style, 'natural'); assert.equal(request.negative_prompt, 'fog');
	const batched = buildImageRequest({ ...base, configuration: { count: 1, batchSize: 3 } });
	assert.equal(batched.n, undefined); assert.equal(batched.parameters['image.batch_size'], 3);
	assert.throws(() => buildImageRequest({ ...base, configuration: { count: 2, batchSize: 2 } }), /cannot both/);
	assert.throws(() => buildImageRequest({ ...base, additionalParameters: { seed: 1 } }), /dedicated/);
	assert.throws(() => buildImageRequest({ ...base, configuration: { seed: 9007199254740992 } }), /safe integer/);
	assert.throws(() => buildImageRequest({ ...base, configuration: { guidance: Infinity } }), /finite/);
	assert.throws(() => buildImageRequest({ ...base, configuration: { url: 'http://evil.test' } }), /unsupported/);
});

test('video derives task from exactly one initial image; container uses response_format', () => {
	const request = buildVideoRequest({ ...base, configuration: { outputFormat: 'gif', frames: 1, fps: 0.5, temporalVaeTiling: 'disabled' } }, png);
	assert.deepEqual(request.initial_image, { base64: png.data.toString('base64'), mime_type: 'image/png' });
	assert.equal(request.response_format, 'gif'); assert.equal(request.output_format, undefined); assert.equal(request.task, undefined);
	assert.equal(request.parameters['video.temporal_vae_tiling'], false);
	assert.equal(buildVideoRequest(base).initial_image, undefined);
	assert.throws(() => buildVideoRequest(base, wav), /image MIME/);
	assert.throws(() => buildVideoRequest({ ...base, configuration: { count: 2, batchSize: 3 } }), /cannot both/);
});

test('TTS uses input and async; its portable sentinel values stay absent', () => {
	const request = buildAudioRequest('text-to-speech', { ...base, configuration: { seed: 0, sampleRate: 0, channels: 0, speed: 1, duration: 30, variations: 1, voice: '', instrumental: 'inherit' } });
	assert.deepEqual(request, { model: base.model, input: base.prompt, async: true });
	const populated = buildAudioRequest('text-to-speech', { ...base, configuration: { voice: 'speaker1', speed: 1.5, seed: 42, sampleRate: 24000, channels: 1, language: 'de', speakingStyle: 'neutral', outputFormat: 'flac' } });
	assert.equal(populated.response_format, 'flac'); assert.equal(populated.voice, 'speaker1'); assert.equal(populated.speed, 1.5);
	assert.deepEqual(populated.parameters, { 'tts.seed': 42, 'tts.sample_rate': 24000, 'tts.channels': 1, 'tts.language': 'de', 'tts.speaking_style': 'neutral' });
	assert.throws(() => buildAudioRequest('text-to-speech', { ...base, negativePrompt: 'noise' }), /Negative prompt/);
	for (const configuration of [{ duration: 4 }, { variations: 2 }, { instrumental: 'disabled' }]) assert.throws(() => buildAudioRequest('text-to-speech', { ...base, configuration }));
	assert.throws(() => buildAudioRequest('text-to-speech', { ...base, configuration: { language: 'de' }, additionalParameters: { language: 'en' } }), /duplicates/);
});

test('audio/music generation preserve zero seed, explicit false and correct namespace', () => {
	for (const task of ['audio-generation', 'music-generation']) {
		const request = buildAudioRequest(task, { ...base, negativePrompt: 'noise', configuration: { seed: 0, instrumental: 'disabled', duration: 10, variations: 2, outputFormat: 'ogg' } });
		assert.equal(request.task, task); assert.equal(request.n, 2); assert.equal(request.response_format, 'ogg'); assert.equal(request.negative_prompt, 'noise');
		assert.deepEqual(request.parameters, { 'audio.duration': 10, 'audio.seed': 0, 'audio.instrumental': false });
	}
	for (const configuration of [{ voice: 'x' }, { speed: 1.1 }, { language: 'de' }, { speakingStyle: 'x' }, { sampleRate: 100 }]) assert.throws(() => buildAudioRequest('audio-generation', { ...base, configuration }));
});

test('all concrete audio analysis/transformation tasks use canonical binary input roles', () => {
	assert.equal(audioAnalysisTasks.length, 12); assert.equal(audioProcessTasks.length, 4);
	for (const task of [...audioAnalysisTasks, ...audioProcessTasks]) {
		const request = buildAudioInputRequest(task, base, wav);
		assert.equal(request.task, task);
		assert.deepEqual(request.inputs, [{ modality: 'audio', role: 'input_audio', source: { kind: 'base64', data: wav.data.toString('base64') }, mime_type: 'audio/wav' }]);
	}
	const converted = buildAudioInputRequest('voice-conversion', base, wav, wav);
	assert.equal(converted.inputs[1].role, 'reference_audio');
	assert.throws(() => buildAudioInputRequest('speech-to-text', base, wav, wav), /only for voice-conversion/);
	assert.throws(() => buildAudioInputRequest('audio-understanding', { ...base, prompt: '' }, wav), /Prompt/);
	assert.throws(() => buildAudioInputRequest('audio-editing', { ...base, prompt: '' }, wav), /Prompt/);
	assert.equal(buildAudioInputRequest('speech-to-text', { ...base, prompt: '', additionalParameters: { language: 'de', temperature: 0 } }, wav).parameters['stt.temperature'], 0);
	assert.throws(() => buildAudioInputRequest('speech-to-text', { ...base, additionalParameters: { 'audio.foo': 1 } }, wav), /namespace/);
});

test('vision preserves image order before text in exactly one user message', () => {
	const second = { data: Buffer.from('second image'), mimeType: 'image/jpeg' };
	const request = buildVisionRequest('vision', 'compare', 'instruction', [png, second], { imageDetail: 'high', temperature: 0, topP: 0, seed: 0, maxCompletionTokens: 1, stopSequences: '["STOP"]' });
	assert.equal(request.stream, false); assert.equal(request.messages.length, 2); assert.equal(request.messages[0].role, 'system');
	assert.deepEqual(request.messages[1], { role: 'user', content: [{ type: 'image_url', image_url: { url: `data:image/png;base64,${png.data.toString('base64')}`, detail: 'high' } }, { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${second.data.toString('base64')}`, detail: 'high' } }, { type: 'text', text: 'compare' }] });
	assert.equal(request.temperature, 0); assert.equal(request.top_p, 0); assert.equal(request.seed, 0);
	assert.throws(() => buildVisionRequest('vision', 'test', '', [png], { backend: 'ignored' }), /unsupported/);
	assert.throws(() => buildVisionRequest('vision', 'test', '', [], {}), /at least one/);
});

test('text supports ordered messages and structural tools but fixes stream false', () => {
	const toolCall = { id: 'call1', type: 'function', function: { name: 'lookup', arguments: '{"q":"test"}' } };
	const messages = { message: [{ role: 'system', content: 'help' }, { role: 'user', content: 'test' }, { role: 'assistant', content: '', toolCalls: JSON.stringify([toolCall]) }, { role: 'tool', content: 'result', toolCallId: 'call1' }] };
	const tools = [{ type: 'function', function: { name: 'lookup', parameters: { type: 'object' }, strict: false } }];
	const request = buildTextRequest('text', messages, { tools: JSON.stringify(tools), toolChoice: '"auto"', parallelToolCalls: 'disabled' });
	assert.equal(request.stream, false); assert.deepEqual(request.tools, tools); assert.equal(request.parallel_tool_calls, false);
	assert.deepEqual(request.messages.map((message) => message.role), ['system', 'user', 'assistant', 'tool']);
	assert.deepEqual(request.messages[2].tool_calls, [toolCall]); assert.equal(request.messages[3].tool_call_id, 'call1');
	assert.throws(() => buildTextRequest('text', messages, { stream: true }), /unsupported/);
	assert.throws(() => buildTextRequest('text', messages, { frequencyPenalty: 1 }), /unsupported/);
	assert.throws(() => buildTextRequest('text', { message: [{ role: 'tool', content: 'x' }] }), /tool call ID/);
	assert.throws(() => buildTextRequest('text', { message: [] }), /at least one/);
});

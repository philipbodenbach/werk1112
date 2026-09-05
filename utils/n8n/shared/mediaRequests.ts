import { choice, ensureKeys, finite, integer, jsonValue, mergeFields, normalizeParameters, record, textValue, tristate, type Fields } from './parameters';
import { buildRouting } from './routing';

export const audioGenerationTasks = ['audio-generation', 'music-generation', 'text-to-speech'] as const;
export const audioProcessTasks = ['voice-conversion', 'stem-separation', 'audio-enhancement', 'audio-editing'] as const;
export const audioAnalysisTasks = ['speech-to-text', 'speech-translation', 'audio-event-detection', 'voice-activity-detection', 'speaker-identification', 'language-identification', 'speech-emotion-recognition', 'audio-captioning', 'speaker-diarization', 'audio-classification', 'audio-understanding', 'audio-embedding'] as const;
export const audioTasks = [...audioGenerationTasks, ...audioProcessTasks, ...audioAnalysisTasks];
export const MAX_INPUT_BYTES = 64 * 1024 * 1024;

export interface MediaOptions {
	model: unknown; prompt: unknown; negativePrompt?: unknown; configuration?: unknown; routing?: unknown; additionalParameters?: unknown;
}
export interface MediaBytes { data: Buffer; mimeType: string }

function base(input: MediaOptions, namespace: string, dedicated: string[], allowedConfig: string[]): { request: Fields; config: Fields; parameters: Fields } {
	const config = record(input.configuration ?? {}, 'Configuration');
	ensureKeys(config, allowedConfig, 'Configuration');
	const routing = buildRouting(input.routing);
	const request: Fields = { model: textValue(input.model, 'Model').trim(), prompt: textValue(input.prompt, 'Prompt'), ...routing.request };
	if (input.negativePrompt !== undefined && textValue(input.negativePrompt, 'Negative prompt', true).trim()) request.negative_prompt = input.negativePrompt;
	return { request, config, parameters: mergeFields(routing.parameters, normalizeParameters(input.additionalParameters, namespace, dedicated.map((key) => `${namespace}.${key}`))) };
}

function setNumeric(config: Fields, params: Fields, mappings: Array<[string, string, number, boolean]>): void {
	for (const [ui, wire, min, isInteger] of mappings) if (config[ui] !== undefined) params[wire] = isInteger ? integer(config[ui], ui, min) : finite(config[ui], ui, min);
}

function setSizeCount(config: Fields, request: Fields, params: Fields, ns: string): void {
	const width = config.width === undefined ? undefined : integer(config.width, 'Width', 1);
	const height = config.height === undefined ? undefined : integer(config.height, 'Height', 1);
	if (width !== undefined && height !== undefined) request.size = `${width}x${height}`;
	else {
		if (width !== undefined) params[`${ns}.width`] = width;
		if (height !== undefined) params[`${ns}.height`] = height;
	}
	const count = config.count === undefined ? undefined : integer(config.count, 'Count', 1);
	const batch = config.batchSize === undefined ? undefined : integer(config.batchSize, 'Batch size', 1);
	if ((count ?? 1) > 1 && (batch ?? 1) > 1) throw new Error('Count and batch size cannot both be greater than 1; Werk treats them as alternative count controls');
	if ((batch ?? 1) > 1) params[`${ns}.batch_size`] = batch;
	else if (count !== undefined) request.n = count;
}

export function buildImageRequest(input: MediaOptions): Fields {
	const { request, config, parameters } = base(input, 'image', ['width', 'height', 'num_images', 'batch_size', 'steps', 'guidance', 'seed', 'output_format', 'vae_tiling', 'vae_slicing'], ['width', 'height', 'count', 'batchSize', 'steps', 'guidance', 'seed', 'outputFormat', 'responseFormat', 'style', 'vaeTiling', 'vaeSlicing']);
	setSizeCount(config, request, parameters, 'image');
	setNumeric(config, parameters, [['steps', 'image.steps', 1, true], ['guidance', 'image.guidance', 0, false], ['seed', 'image.seed', 0, true]]);
	if (config.outputFormat !== undefined) request.output_format = choice(config.outputFormat, 'Image output format', ['png', 'jpeg', 'webp']);
	if (config.responseFormat !== undefined) request.response_format = choice(config.responseFormat, 'Image response format', ['b64_json', 'url']);
	if (config.style !== undefined && config.style !== 'none') request.style = choice(config.style, 'Style', ['vivid', 'natural']);
	for (const [ui, wire] of [['vaeTiling', 'image.vae_tiling'], ['vaeSlicing', 'image.vae_slicing']] as const) {
		const bool = tristate(config[ui], ui);
		if (bool !== undefined) parameters[wire] = bool;
	}
	if (Object.keys(parameters).length) request.parameters = parameters;
	return request;
}

export function buildVideoRequest(input: MediaOptions, initialImage?: MediaBytes): Fields {
	const { request, config, parameters } = base(input, 'video', ['width', 'height', 'num_videos', 'batch_size', 'frames', 'fps', 'steps', 'guidance', 'seed', 'output_format', 'temporal_vae_tiling'], ['width', 'height', 'count', 'batchSize', 'frames', 'fps', 'steps', 'guidance', 'seed', 'outputFormat', 'temporalVaeTiling']);
	setSizeCount(config, request, parameters, 'video');
	setNumeric(config, parameters, [['frames', 'video.frames', 1, true], ['fps', 'video.fps', 0.1, false], ['steps', 'video.steps', 1, true], ['guidance', 'video.guidance', 0, false], ['seed', 'video.seed', 0, true]]);
	if (config.outputFormat !== undefined) request.response_format = choice(config.outputFormat, 'Video output format', ['mp4', 'gif']);
	const tiling = tristate(config.temporalVaeTiling, 'Temporal VAE tiling');
	if (tiling !== undefined) parameters['video.temporal_vae_tiling'] = tiling;
	if (initialImage) {
		validateMediaInput(initialImage, 'image');
		request.initial_image = { base64: initialImage.data.toString('base64'), mime_type: initialImage.mimeType };
	}
	if (Object.keys(parameters).length) request.parameters = parameters;
	return request;
}

export function buildAudioRequest(task: string, input: MediaOptions): Fields {
	choice(task, 'Audio generation task', audioGenerationTasks);
	const tts = task === 'text-to-speech';
	const ns = tts ? 'tts' : 'audio';
	const dedicated = tts ? ['voice', 'speed', 'seed', 'sample_rate', 'channels', 'output_format'] : ['duration', 'variations', 'seed', 'sample_rate', 'channels', 'instrumental', 'output_format'];
	const { request, config, parameters } = base(input, ns, dedicated, ['duration', 'variations', 'seed', 'sampleRate', 'channels', 'outputFormat', 'instrumental', 'voice', 'speed', 'language', 'speakingStyle']);
	if (config.outputFormat !== undefined) request.response_format = choice(config.outputFormat, 'Audio output format', ['wav', 'flac', 'ogg']);
	if (config.sampleRate !== undefined && integer(config.sampleRate, 'Sample rate') !== 0) parameters[`${ns}.sample_rate`] = integer(config.sampleRate, 'Sample rate', 8000);
	if (config.channels !== undefined && integer(config.channels, 'Channels') !== 0) parameters[`${ns}.channels`] = integer(config.channels, 'Channels', 1);
	const seed = config.seed === undefined ? undefined : integer(config.seed, 'Seed');
	if (tts) {
		if (request.negative_prompt !== undefined) throw new Error('Negative prompt is not supported for text-to-speech');
		if (config.duration !== undefined && finite(config.duration, 'Duration', 0.1) !== 30) throw new Error('Duration applies only to audio/music generation');
		if (config.variations !== undefined && integer(config.variations, 'Variations', 1) !== 1) throw new Error('Text-to-speech supports exactly one variation');
		if (tristate(config.instrumental, 'Instrumental') !== undefined) throw new Error('Instrumental applies only to audio/music generation');
		if (config.voice !== undefined && textValue(config.voice, 'Voice', true).trim()) request.voice = (config.voice as string).trim();
		if (config.speed !== undefined && finite(config.speed, 'Speed', 0.1) !== 1) request.speed = config.speed;
		if (seed) parameters['tts.seed'] = seed;
		for (const [ui, wire] of [['language', 'tts.language'], ['speakingStyle', 'tts.speaking_style']] as const) {
			if (config[ui] !== undefined && textValue(config[ui], ui, true).trim()) {
				if (Object.hasOwn(parameters, wire)) throw new Error(`Additional parameter ${wire} duplicates a populated input`);
				parameters[wire] = (config[ui] as string).trim();
			}
		}
		request.input = request.prompt;
		delete request.prompt;
		request.async = true;
	} else {
		for (const ui of ['voice', 'language', 'speakingStyle']) if (config[ui] !== undefined && textValue(config[ui], ui, true).trim()) throw new Error(`${ui} applies only to text-to-speech`);
		if (config.speed !== undefined && finite(config.speed, 'Speed', 0.1) !== 1) throw new Error('Speed applies only to text-to-speech');
		if (config.duration !== undefined) parameters['audio.duration'] = finite(config.duration, 'Duration', 0.1);
		if (config.variations !== undefined) request.n = integer(config.variations, 'Variations', 1);
		if (seed !== undefined) parameters['audio.seed'] = seed;
		const instrumental = tristate(config.instrumental, 'Instrumental');
		if (instrumental !== undefined) parameters['audio.instrumental'] = instrumental;
		request.task = task;
	}
	if (Object.keys(parameters).length) request.parameters = parameters;
	return request;
}

export function validateMediaInput(media: MediaBytes, kind: 'image' | 'audio'): void {
	if (!Buffer.isBuffer(media.data) || !media.data.length || media.data.length > MAX_INPUT_BYTES) throw new Error(`The ${kind} input must contain between 1 and ${MAX_INPUT_BYTES} bytes`);
	if (!new RegExp(`^${kind}/[a-z0-9.+-]+$`, 'i').test(media.mimeType) && !(kind === 'audio' && media.mimeType === 'application/ogg')) throw new Error(`The input requires a valid ${kind} MIME type`);
}

export function buildAudioInputRequest(task: string, input: MediaOptions, audio: MediaBytes, referenceAudio?: MediaBytes): Fields {
	choice(task, 'Audio input task', [...audioProcessTasks, ...audioAnalysisTasks]);
	const model = textValue(input.model, 'Model').trim();
	const prompt = textValue(input.prompt ?? '', 'Prompt', !['audio-understanding', 'audio-editing'].includes(task));
	if (referenceAudio && task !== 'voice-conversion') throw new Error('Reference audio is supported only for voice-conversion');
	validateMediaInput(audio, 'audio');
	if (referenceAudio) validateMediaInput(referenceAudio, 'audio');
	if (audio.data.length + (referenceAudio?.data.length ?? 0) > MAX_INPUT_BYTES) throw new Error(`Combined audio inputs exceed ${MAX_INPUT_BYTES} bytes`);
	const toInput = (media: MediaBytes, role: string): Fields => ({ modality: 'audio', role, source: { kind: 'base64', data: media.data.toString('base64') }, mime_type: media.mimeType });
	const routing = buildRouting(input.routing);
	const ns = ['speech-to-text', 'speech-translation'].includes(task) ? 'stt' : 'audio';
	const parameters = mergeFields(routing.parameters, normalizeParameters(input.additionalParameters, ns));
	const request: Fields = { model, task, inputs: [toInput(audio, 'input_audio'), ...(referenceAudio ? [toInput(referenceAudio, 'reference_audio')] : [])], ...routing.request };
	if (prompt.trim()) request.prompt = prompt;
	if (input.negativePrompt !== undefined && textValue(input.negativePrompt, 'Negative prompt', true).trim()) request.negative_prompt = input.negativePrompt;
	if (Object.keys(parameters).length) request.parameters = parameters;
	return request;
}

function chatOptions(value: unknown, vision: boolean): Fields {
	const options = record(value ?? {}, 'Chat options');
	ensureKeys(options, ['temperature', 'topP', 'maxCompletionTokens', 'seed', 'stopSequences', ...(vision ? ['imageDetail'] : ['tools', 'toolChoice', 'parallelToolCalls'])], 'Chat options');
	const request: Fields = {};
	if (options.temperature !== undefined) request.temperature = finite(options.temperature, 'Temperature');
	if (options.topP !== undefined) request.top_p = finite(options.topP, 'Top P', 0, 1);
	if (options.maxCompletionTokens !== undefined) request.max_completion_tokens = integer(options.maxCompletionTokens, 'Maximum completion tokens', 1);
	if (options.seed !== undefined) request.seed = integer(options.seed, 'Seed');
	if (options.stopSequences !== undefined) {
		const stop = jsonValue(options.stopSequences, 'Stop sequences');
		if (!Array.isArray(stop) || stop.some((entry) => typeof entry !== 'string' || !entry)) throw new Error('Stop sequences must be an array of non-empty strings');
		if (stop.length) request.stop = stop;
	}
	if (!vision) {
		if (options.tools !== undefined) {
			const tools = jsonValue(options.tools, 'Tools');
			if (!Array.isArray(tools)) throw new Error('Tools must be an array');
			for (const tool of tools) {
				const entry = record(tool, 'Tool'); ensureKeys(entry, ['type', 'function'], 'Tool');
				choice(entry.type, 'Tool type', ['function']);
				const fn = record(entry.function, 'Tool function'); ensureKeys(fn, ['name', 'description', 'parameters', 'strict'], 'Tool function');
				textValue(fn.name, 'Tool name');
				if (fn.description !== undefined) textValue(fn.description, 'Tool description', true);
				if (fn.strict !== undefined && typeof fn.strict !== 'boolean') throw new Error('Tool strict must be a boolean');
			}
			request.tools = tools;
		}
		if (options.toolChoice !== undefined) {
			const selected = jsonValue(options.toolChoice, 'Tool choice');
			if (typeof selected === 'string') choice(selected, 'Tool choice', ['none', 'auto', 'required']);
			else {
				const tool = record(selected, 'Tool choice'); ensureKeys(tool, ['type', 'function'], 'Tool choice'); choice(tool.type, 'Tool choice type', ['function']);
				const fn = record(tool.function, 'Tool choice function'); ensureKeys(fn, ['name'], 'Tool choice function'); textValue(fn.name, 'Tool choice function name');
			}
			request.tool_choice = selected;
		}
		const parallel = tristate(options.parallelToolCalls, 'Parallel tool calls');
		if (parallel !== undefined) request.parallel_tool_calls = parallel;
	}
	return request;
}

export function buildTextRequest(model: unknown, messages: unknown, options?: unknown): Fields {
	const group = record(messages, 'Messages'); ensureKeys(group, ['message'], 'Messages');
	if (!Array.isArray(group.message) || !group.message.length) throw new Error('Provide at least one message');
	const normalized = group.message.map((value) => {
		const entry = record(value, 'Message'); ensureKeys(entry, ['role', 'content', 'name', 'toolCallId', 'toolCalls'], 'Message');
		const role = choice(entry.role, 'Message role', ['system', 'user', 'assistant', 'tool']);
		const result: Fields = { role, content: textValue(entry.content ?? '', 'Message content', true) };
		if (entry.name !== undefined && textValue(entry.name, 'Message name', true).trim()) result.name = entry.name;
		if (entry.toolCallId !== undefined && textValue(entry.toolCallId, 'Tool call ID', true).trim()) result.tool_call_id = entry.toolCallId;
		if (role === 'tool' && !result.tool_call_id) throw new Error('Tool messages require a tool call ID');
		if (entry.toolCalls !== undefined) {
			const calls = jsonValue(entry.toolCalls, 'Tool calls');
			if (!Array.isArray(calls)) throw new Error('Tool calls must be an array');
			if (calls.length && role !== 'assistant') throw new Error('Only assistant messages may contain tool calls');
			for (const call of calls) {
				const entry = record(call, 'Tool call'); ensureKeys(entry, ['id', 'type', 'function'], 'Tool call');
				textValue(entry.id, 'Tool call ID'); choice(entry.type, 'Tool call type', ['function']);
				const fn = record(entry.function, 'Tool call function'); ensureKeys(fn, ['name', 'arguments'], 'Tool call function');
				textValue(fn.name, 'Tool call function name'); textValue(fn.arguments, 'Tool call arguments', true);
			}
			if (calls.length) result.tool_calls = calls;
		}
		return result;
	});
	return { model: textValue(model, 'Model').trim(), messages: normalized, stream: false, ...chatOptions(options, false) };
}

export function buildVisionRequest(model: unknown, prompt: unknown, systemPrompt: unknown, images: MediaBytes[], options?: unknown): Fields {
	if (!images.length) throw new Error('Vision requires at least one image');
	if (images.reduce((sum, media) => sum + media.data.length, 0) > MAX_INPUT_BYTES) throw new Error(`Combined vision images exceed ${MAX_INPUT_BYTES} bytes`);
	const config = record(options ?? {}, 'Vision options');
	const detail = choice(config.imageDetail ?? 'auto', 'Image detail', ['auto', 'low', 'high']);
	const content: Fields[] = images.map((media) => {
		validateMediaInput(media, 'image');
		return { type: 'image_url', image_url: { url: `data:${media.mimeType};base64,${media.data.toString('base64')}`, detail } };
	});
	content.push({ type: 'text', text: textValue(prompt, 'Prompt') });
	const system = textValue(systemPrompt ?? '', 'System prompt', true).trim();
	return { model: textValue(model, 'Model').trim(), messages: [...(system ? [{ role: 'system', content: system }] : []), { role: 'user', content }], stream: false, ...chatOptions(config, true) };
}

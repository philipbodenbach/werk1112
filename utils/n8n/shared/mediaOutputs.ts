import type { IDataObject, IExecuteFunctions, INodeExecutionData } from 'n8n-workflow';
import { binaryItem } from './binary';
import type { WerkClient } from './client';
import { integer, record, textValue, type Fields } from './parameters';
import { sanitize } from './validation';

/** Preserve bytes: checking signatures does not decode or transcode an image. */
export function imageMime(data: Buffer): string {
	if (data.length >= 8 && data.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) return 'image/png';
	if (data.length >= 3 && data[0] === 255 && data[1] === 216 && data[2] === 255) return 'image/jpeg';
	if (data.length >= 12 && data.toString('ascii', 0, 4) === 'RIFF' && data.toString('ascii', 8, 12) === 'WEBP') return 'image/webp';
	throw new Error('Werk returned an unsupported image signature (expected PNG, JPEG, or WebP)');
}

export async function imageOutputs(ctx: IExecuteFunctions, client: WerkClient, index: number, response: Fields, request: Fields): Promise<INodeExecutionData[]> {
	if (!Array.isArray(response.data) || !response.data.length) throw new Error('Werk image generation returned no images');
	const results: INodeExecutionData[] = [];
	for (const raw of response.data) {
		const entry = record(raw, 'Image output');
		let data: Buffer;
		let retained = false;
		if (typeof entry.b64_json === 'string') {
			const encoded = entry.b64_json.replace(/^data:image\/[a-z0-9.+-]+;base64,/i, '');
			// Bounded independently of the HTTP envelope, before allocating decoded bytes.
			if (encoded.length > Math.ceil(256 * 1024 * 1024 / 3) * 4) throw new Error('Werk image output exceeds the 256 MiB byte limit');
			// A repeated four-character regex group overflows V8's stack on ordinary multi-MiB images.
			if (!encoded || encoded.length % 4 !== 0 || !/^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) throw new Error('Werk image output contains invalid Base64');
			data = Buffer.from(encoded, 'base64');
		} else if (typeof entry.url === 'string') {
			const downloaded = await client.download(entry.url);
			if (downloaded.mimeType && !downloaded.mimeType.toLowerCase().startsWith('image/')) throw new Error('Werk image download returned a non-image content type');
			data = downloaded.data;
			retained = true;
		} else throw new Error('Werk image output contains neither Base64 nor URL');
		const mimeType = imageMime(data);
		const json = sanitize({ model: request.model, task: 'image-generation', status: 'completed', outputId: retained && typeof entry.id === 'string' ? entry.id : null, downloadable: retained, werk: { response, request, output: entry } }) as IDataObject;
		results.push(await binaryItem(ctx, index, data, mimeType, json));
	}
	return results;
}

export function chatOutputs(index: number, response: Fields, request: Fields, task: string): INodeExecutionData[] {
	if (!Array.isArray(response.choices) || !response.choices.length) throw new Error('Werk chat response contains no choices');
	return response.choices.map((raw) => {
		const choice = record(raw, 'Chat choice');
		const message = record(choice.message, 'Assistant message');
		if (message.content !== null && message.content !== undefined && typeof message.content !== 'string') throw new Error('Werk chat message content must be text or null');
		if (message.tool_calls !== undefined && !Array.isArray(message.tool_calls)) throw new Error('Werk chat tool calls must be an array');
		if (typeof message.content !== 'string' && (!Array.isArray(message.tool_calls) || !message.tool_calls.length)) throw new Error('Werk chat choice contains neither text nor tool calls');
		const toolCalls = message.tool_calls ?? [];
		for (const rawCall of toolCalls as unknown[]) {
			const call = record(rawCall, 'Tool call'); textValue(call.id, 'Tool call ID');
			const fn = record(call.function, 'Tool call function'); textValue(fn.name, 'Tool call function name'); textValue(fn.arguments, 'Tool call arguments', true);
		}
		if (choice.finish_reason !== null && choice.finish_reason !== undefined && typeof choice.finish_reason !== 'string') throw new Error('Werk finish reason must be a string or null');
		return { json: sanitize({ model: typeof response.model === 'string' ? response.model : request.model, task, text: message.content ?? '', usage: response.usage ?? null, finishReason: choice.finish_reason ?? null, completionId: response.id ?? null, choiceIndex: integer(choice.index ?? 0, 'Choice index'), toolCalls, werk: { response, request } }) as IDataObject, pairedItem: { item: index } };
	});
}

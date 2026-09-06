import type { INodeProperties } from 'n8n-workflow';
import { collectionProperty, numberProperty, optionsProperty, stringProperty, triProperty } from './parameters';

export const promptProperty: INodeProperties = { ...stringProperty('prompt', 'Prompt'), typeOptions: { rows: 4 } };
export const negativePromptProperty: INodeProperties = { ...stringProperty('negativePrompt', 'Negative Prompt'), typeOptions: { rows: 2 } };
export const binaryProperty: INodeProperties = { name: 'binaryProperty', displayName: 'Input Binary Field', type: 'string', default: 'data', required: true, description: 'Name of the n8n binary field. Reads bytes through the n8n Binary Data helper, including external storage.' };
const seedProperty: INodeProperties = { ...numberProperty('seed', 'Seed', 0), description: 'Safe integers up to 9007199254740991 only. For TTS, 0 inherits the backend default.' };
const dimensions = (): INodeProperties[] => [numberProperty('width', 'Width', 1024, 1), numberProperty('height', 'Height', 1024, 1), numberProperty('count', 'Count', 1, 1), { ...numberProperty('batchSize', 'Batch Size', 1, 1), description: 'Count and batch size are alternative controls; they cannot both exceed 1' }];
const sampling = (): INodeProperties[] => [numberProperty('steps', 'Steps', 28, 1), numberProperty('guidance', 'Guidance', 7), seedProperty];

export const imageConfigurationProperty = collectionProperty('configuration', 'Image Configuration', [
	...dimensions(), ...sampling(), optionsProperty('outputFormat', 'Output File Format', ['png', 'jpeg', 'webp']),
	optionsProperty('responseFormat', 'Response Transport', ['b64_json', 'url']), optionsProperty('style', 'Style', ['none', 'vivid', 'natural']),
	triProperty('vaeTiling', 'VAE Tiling'), triProperty('vaeSlicing', 'VAE Slicing'),
]);
export const videoConfigurationProperty = collectionProperty('configuration', 'Video Configuration', [
	...dimensions().map((property) => property.name === 'width' ? { ...property, default: 832 } : property.name === 'height' ? { ...property, default: 480 } : property),
	numberProperty('frames', 'Frames', 81, 1), numberProperty('fps', 'Frames Per Second', 24, 0.1),
	...sampling().map((property) => property.name === 'steps' ? { ...property, default: 30 } : property.name === 'guidance' ? { ...property, default: 6 } : property),
	optionsProperty('outputFormat', 'Output File Format', ['mp4', 'gif']), triProperty('temporalVaeTiling', 'Temporal VAE Tiling'),
]);
export const audioConfigurationProperty = collectionProperty('configuration', 'Audio Configuration', [
	{ ...numberProperty('duration', 'Duration (Seconds)', 30, 0.1), description: 'Audio/music generation only; TTS accepts the neutral value 30 without sending it' },
	numberProperty('variations', 'Variations', 1, 1), seedProperty,
	{ ...numberProperty('sampleRate', 'Sample Rate', 0), description: '0 inherits; otherwise at least 8000 Hz' },
	{ ...numberProperty('channels', 'Channels', 0), description: '0 inherits the backend default' },
	optionsProperty('outputFormat', 'Output File Format', ['wav', 'flac', 'ogg']), triProperty('instrumental', 'Instrumental'),
	stringProperty('voice', 'Voice (TTS)'), numberProperty('speed', 'Speed (TTS)', 1, 0.1), stringProperty('language', 'Language (TTS)'), stringProperty('speakingStyle', 'Speaking Style (TTS)'),
]);

export function chatOptionsProperty(vision: boolean): INodeProperties {
	return collectionProperty('options', vision ? 'Vision Options' : 'Chat Options', [
		numberProperty('temperature', 'Temperature', 0.2), { ...numberProperty('topP', 'Top P', 1), typeOptions: { minValue: 0, maxValue: 1 } },
		numberProperty('maxCompletionTokens', 'Maximum Completion Tokens', 1024, 1), seedProperty,
		{ name: 'stopSequences', displayName: 'Stop Sequences (JSON)', type: 'json', default: '[]', description: 'Ordered array of non-empty strings' },
		...(vision ? [optionsProperty('imageDetail', 'Image Detail', ['auto', 'low', 'high'])] : [
			{ name: 'tools', displayName: 'Tool Definitions (JSON)', type: 'json' as const, default: '[]', description: 'OpenAI function tool definitions. Requires a compatible Werk backend. Returned tool calls are not executed by this node.' },
			{ name: 'toolChoice', displayName: 'Tool Choice (JSON)', type: 'json' as const, default: '"auto"', description: '"none", "auto", "required", or a named function selector object' },
			triProperty('parallelToolCalls', 'Parallel Tool Calls'),
		]),
	]);
}

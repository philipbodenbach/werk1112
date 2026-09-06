import type { INodeProperties } from 'n8n-workflow';
import { parseJson, validateJson } from './validation';

export type Fields = Record<string, unknown>;

export function record(value: unknown, label: string): Fields {
	if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
	return value as Fields;
}

export function jsonValue(value: unknown, label: string): unknown {
	const parsed = typeof value === 'string' ? parseJson(value, label) : value;
	validateJson(parsed);
	return parsed;
}

export function textValue(value: unknown, label: string, allowEmpty = false): string {
	if (typeof value !== 'string' || (!allowEmpty && !value.trim())) throw new Error(`${label} must be ${allowEmpty ? 'a string' : 'a non-empty string'}`);
	return value;
}

export function integer(value: unknown, label: string, min = 0): number {
	if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min) throw new Error(`${label} must be a safe integer from ${min} to ${Number.MAX_SAFE_INTEGER}`);
	return value;
}

export function finite(value: unknown, label: string, min = 0, max = Infinity): number {
	if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) throw new Error(`${label} must be a finite number between ${min} and ${max}`);
	return value;
}

export function choice(value: unknown, label: string, values: readonly string[]): string {
	if (typeof value !== 'string' || !values.includes(value)) throw new Error(`${label} must be one of: ${values.join(', ')}`);
	return value;
}

export function tristate(value: unknown, label: string): boolean | undefined {
	if (value === undefined || value === 'inherit') return undefined;
	choice(value, label, ['enabled', 'disabled']);
	return value === 'enabled';
}

const RESERVED = new Set(['model', 'prompt', 'negative_prompt', 'initial_image', 'input', 'inputs', 'task', 'async', 'background', 'job', 'voice', 'speed', 'n', 'size', 'response_format', 'output_format', 'style', 'quality', 'parameter_policy', 'routing', 'backend', 'accelerator', 'device', 'precision', 'quantization', 'profile', 'performance_preference', 'fallback_policy', 'allow_cpu_offload', 'allow_sequential_offload', 'allow_component_offload', 'allow_disk_offload', 'attention_backend', 'compile', 'timeout_seconds', 'user', 'server_url', 'base_url', 'api_key', 'authorization', 'headers', 'url', 'path', 'method', 'auth', 'credentials', 'timeout', 'http_timeout', 'max_response_bytes', 'protocol', 'request_id']);

/** Canonical names stay in their task namespace. Values retain server list operations. */
export function normalizeParameters(value: unknown, namespace: string, dedicated: readonly string[] = []): Fields {
	const source = record(jsonValue(value ?? '{}', 'Additional parameters'), 'Additional parameters');
	const flattened: Array<[string, unknown]> = [];
	for (const [key, child] of Object.entries(source)) {
		if (key === namespace) flattened.push(...Object.entries(record(child, `${namespace} parameters`)).map(([name, entry]): [string, unknown] => [`${namespace}.${name}`, entry]));
		else flattened.push([key, child]);
	}
	const result: Fields = {};
	for (const [rawName, child] of flattened) {
		const name = rawName.trim();
		const canonical = name.includes('.') ? name : `${namespace}.${name}`;
		const lowered = name.toLowerCase();
		const first = lowered.split('.')[0]!;
		const unprefixed = lowered.startsWith(`${namespace}.`) ? lowered.slice(namespace.length + 1) : lowered;
		if (!name || !canonical.startsWith(`${namespace}.`) || !canonical.slice(namespace.length + 1)) throw new Error(`Additional parameters must use the ${namespace}. namespace`);
		if (dedicated.includes(canonical)) throw new Error(`Additional parameter ${canonical} duplicates a dedicated input`);
		if ((lowered !== namespace && RESERVED.has(lowered)) || (first !== namespace && RESERVED.has(first)) || RESERVED.has(unprefixed)) throw new Error('Additional parameters contain a reserved request or transport field');
		if (Object.hasOwn(result, canonical)) throw new Error(`Additional parameter ${canonical} is duplicated after namespace normalization`);
		if (child && typeof child === 'object' && !Array.isArray(child) && Object.hasOwn(child, 'operation')) {
			const override = record(child, 'List override');
			const operation = choice(override.operation, 'List operation', ['inherit', 'replace', 'add', 'clear']);
			if (Object.keys(override).some((key) => !['operation', 'values'].includes(key))) throw new Error('List override contains unknown fields');
			if (override.values !== undefined && !Array.isArray(override.values)) throw new Error('List override values must be an array');
			if (['inherit', 'clear'].includes(operation) && Array.isArray(override.values) && override.values.length > 0) throw new Error(`${operation} does not accept list values`);
		}
		result[canonical] = child;
	}
	return result;
}

export function mergeFields(...objects: Fields[]): Fields {
	const result: Fields = {};
	for (const source of objects) for (const [key, value] of Object.entries(source)) {
		if (Object.hasOwn(result, key)) throw new Error(`Parameter ${key} duplicates a populated input`);
		result[key] = value;
	}
	return result;
}

export function ensureKeys(fields: Fields, allowed: readonly string[], label: string): void {
	if (Object.keys(fields).some((key) => !allowed.includes(key))) throw new Error(`${label} contains unsupported fields`);
}

export const additionalParametersProperty: INodeProperties = {
	displayName: 'Additional Model Parameters (JSON)', name: 'additionalParameters', type: 'json', default: '{}',
	description: 'Schema-discovered parameters in the current task namespace. Dedicated inputs and transport fields cannot be overridden. List values support operation inherit, replace, add, or clear.',
};

export function stringProperty(name: string, displayName: string, description = ''): INodeProperties {
	return { name, displayName, type: 'string', default: '', description };
}
export function numberProperty(name: string, displayName: string, defaultValue: number, min = 0): INodeProperties {
	return { name, displayName, type: 'number', default: defaultValue, typeOptions: { minValue: min } };
}
export function optionsProperty(name: string, displayName: string, values: readonly string[], defaultValue = values[0]!): INodeProperties {
	return { name, displayName, type: 'options', default: defaultValue, options: values.map((value) => ({ name: value, value })) };
}
export function triProperty(name: string, displayName: string): INodeProperties {
	return optionsProperty(name, displayName, ['inherit', 'enabled', 'disabled']);
}
export function collectionProperty(name: string, displayName: string, options: INodeProperties[]): INodeProperties {
	return { name, displayName, type: 'collection', default: {}, placeholder: 'Add Option', options,
		description: 'Only added options are sent. Absent options inherit Werk defaults.' };
}

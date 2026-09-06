import type { INodeProperties } from 'n8n-workflow';
import { choice, collectionProperty, ensureKeys, integer, normalizeParameters, numberProperty, optionsProperty, record, stringProperty, textValue, triProperty, tristate, type Fields } from './parameters';

const TEXT_FIELDS = ['backend', 'accelerator', 'device', 'precision', 'quantization', 'profile', 'attention_backend'];
const BOOL_FIELDS = ['allow_cpu_offload', 'allow_sequential_offload', 'allow_component_offload', 'allow_disk_offload', 'compile'];
const ENUM_FIELDS: Record<string, string[]> = {
	quality: ['draft', 'balanced', 'high', 'maximum'],
	performance_preference: ['quality', 'balanced', 'speed', 'latency', 'throughput', 'memory'],
	fallback_policy: ['none', 'backend', 'degrade'], parameter_policy: ['strict', 'warn', 'permissive'],
};
const dedicated = [...TEXT_FIELDS, ...BOOL_FIELDS, ...Object.keys(ENUM_FIELDS)].map((key) => `routing.${key}`).concat('routing.timeout');

export function buildRouting(value: unknown = {}): { request: Fields; parameters: Fields } {
	const fields = record(value, 'Routing');
	ensureKeys(fields, [...TEXT_FIELDS, ...BOOL_FIELDS, ...Object.keys(ENUM_FIELDS), 'inferenceTimeoutSeconds', 'additionalRoutingParameters'], 'Routing');
	const request: Fields = {};
	for (const key of TEXT_FIELDS) if (fields[key] !== undefined) {
		const text = textValue(fields[key], key, true).trim();
		if (text) request[key] = text;
	}
	for (const [key, values] of Object.entries(ENUM_FIELDS)) if (fields[key] !== undefined && fields[key] !== 'inherit') request[key] = choice(fields[key], key, values);
	for (const key of BOOL_FIELDS) {
		const boolean = tristate(fields[key], key);
		if (boolean !== undefined) request[key] = boolean;
	}
	if (fields.inferenceTimeoutSeconds !== undefined) {
		const timeout = integer(fields.inferenceTimeoutSeconds, 'Inference timeout');
		if (timeout) request.timeout_seconds = timeout;
	}
	return { request, parameters: normalizeParameters(fields.additionalRoutingParameters, 'routing', dedicated) };
}

export const routingProperty: INodeProperties = collectionProperty('routing', 'Routing', [
	...TEXT_FIELDS.map((name) => stringProperty(name, name.replaceAll('_', ' '))),
	...Object.entries(ENUM_FIELDS).map(([name, values]) => optionsProperty(name, name.replaceAll('_', ' '), ['inherit', ...values])),
	...BOOL_FIELDS.map((name) => triProperty(name, name.replaceAll('_', ' '))),
	{ ...numberProperty('inferenceTimeoutSeconds', 'Inference Timeout (Seconds)', 0), description: '0 inherits Werk defaults. Independent of HTTP timeout and job wait duration.' },
	{ displayName: 'Additional Routing Parameters (JSON)', name: 'additionalRoutingParameters', type: 'json', default: '{}', description: 'Only routing namespace parameters; dedicated routing fields cannot be overridden' },
]);

import type { IExecuteFunctions, INodeExecutionData, INodeProperties, INodeType } from 'n8n-workflow';
import { commonMethods, executeItems, nodeDescription, transportProperties } from '../../shared/common';
import { runRuntime } from '../../shared/runtime';

const show = (...operations: string[]) => ({ show: { operation: operations } });
const options = (...values: string[]) => values.map((value) => ({ name: value === 'vram' ? 'VRAM' : value === 'ram' ? 'RAM' : value.charAt(0).toUpperCase() + value.slice(1), value }));
const properties: INodeProperties[] = [
  { displayName: 'Operation', name: 'operation', type: 'options', noDataExpression: true, default: 'info', options: [
    { name: 'Runtime Info', value: 'info', action: 'Read runtime info and capabilities' },
    { name: 'Capabilities', value: 'capabilities', action: 'Read runtime capabilities' },
    { name: 'Memory', value: 'memory', action: 'Read memory status' },
    { name: 'List States', value: 'states', action: 'List runtime states' },
    { name: 'State Action', value: 'stateAction', action: 'Control a runtime state' },
    { name: 'Prune States', value: 'pruneStates', action: 'Prune selected runtime states' },
    { name: 'List Experts', value: 'experts', action: 'List expert residency metadata' },
    { name: 'Expert Action', value: 'expertAction', action: 'Control explicit runtime experts' },
    { name: 'Prefill & Decode', value: 'prefillDecode', action: 'Prefill and decode in one execution' },
  ] },
  { displayName: 'Model ID', name: 'modelId', type: 'string', default: '', displayOptions: show('states', 'pruneStates', 'experts', 'expertAction', 'prefillDecode'), description: 'Exact installed model ID. Required for expert actions and Prefill & Decode; optional as a list/prune filter.' },
  { displayName: 'Tier', name: 'tier', type: 'options', default: 'all', options: options('all', 'ram', 'vram', 'disk', 'external'), displayOptions: show('states', 'pruneStates') },
  { displayName: 'Tier', name: 'tier', type: 'options', default: 'all', options: options('all', 'ram', 'vram', 'external'), displayOptions: show('experts') },
  { displayName: 'Page Limit', name: 'limit', type: 'number', default: 50, typeOptions: { minValue: 1, maxValue: 4096, numberPrecision: 0 }, displayOptions: show('states', 'experts'), description: 'Must fit the discovered server max_page_size' },
  { displayName: 'Cursor', name: 'cursor', type: 'string', default: '', displayOptions: show('states', 'experts'), description: 'Opaque nextCursor from an earlier page' },
  { displayName: 'Return All', name: 'returnAll', type: 'boolean', default: false, displayOptions: show('states', 'experts'), description: 'Whether to follow cursors, bounded by maximum pages and 4096 total entries' },
  { displayName: 'Maximum Pages', name: 'maxPages', type: 'number', default: 10, typeOptions: { minValue: 1, maxValue: 100, numberPrecision: 0 }, displayOptions: { show: { operation: ['states', 'experts'], returnAll: [true] } } },
  { displayName: 'State ID', name: 'stateId', type: 'string', default: '', required: true, displayOptions: show('stateAction') },
  { displayName: 'State Action', name: 'stateAction', type: 'options', default: 'pin', options: options('pin', 'unpin', 'promote', 'demote', 'evict'), displayOptions: show('stateAction') },
  { displayName: 'Target Tier', name: 'targetTier', type: 'options', default: 'unchanged', options: options('unchanged', 'ram', 'vram', 'disk'), displayOptions: show('stateAction'), description: 'Promote: RAM/VRAM; demote: RAM/disk; other actions: unchanged' },
  { displayName: 'Selector', name: 'selector', type: 'options', default: 'ids', options: options('ids', 'filter', 'all'), displayOptions: show('pruneStates') },
  { displayName: 'State IDs', name: 'stateIds', type: 'string', default: '', typeOptions: { rows: 3 }, displayOptions: { show: { operation: ['pruneStates'], selector: ['ids'] } }, description: 'JSON array or newline/comma-separated explicit state IDs; duplicates are collapsed' },
  { displayName: 'Older Than (Unix Milliseconds)', name: 'olderThanUnixMs', type: 'number', default: 0, typeOptions: { minValue: 0, numberPrecision: 0 }, displayOptions: { show: { operation: ['pruneStates'], selector: ['filter'] } }, description: '0 omits this constraint. A filter needs at least one model, tier or nonzero timestamp.' },
  { displayName: 'Confirm All', name: 'confirmAll', type: 'boolean', default: false, displayOptions: { show: { operation: ['pruneStates'], selector: ['all'] } }, description: 'Whether to explicitly select every visible runtime state' },
  { displayName: 'Expert IDs', name: 'expertIds', type: 'string', default: '', required: true, typeOptions: { rows: 3 }, displayOptions: show('expertAction'), description: 'Unique explicit IDs as JSON array or newline/comma-separated strings' },
  { displayName: 'Expert Action', name: 'expertAction', type: 'options', default: 'pin', options: options('prefetch', 'pin', 'unpin', 'evict'), displayOptions: show('expertAction') },
  { displayName: 'Expert Target Tier', name: 'expertTargetTier', type: 'options', default: 'unchanged', options: options('unchanged', 'ram', 'vram'), displayOptions: show('expertAction'), description: 'Prefetch requires RAM or VRAM; other actions require unchanged' },
  { displayName: 'Dry Run', name: 'dryRun', type: 'boolean', default: true, displayOptions: show('stateAction', 'pruneStates', 'expertAction'), description: 'Whether to preview the action without applying it' },
  { displayName: 'Allow Experimental', name: 'allowExperimental', type: 'boolean', default: false, displayOptions: show('stateAction', 'experts', 'expertAction', 'prefillDecode'), description: 'Whether to explicitly opt in to advertised experimental operations. Beta naming does not enable this.' },
  { displayName: 'Expert operations depend on declared capabilities. Current production adapters do not implement operational expert residency.', name: 'expertsNotice', type: 'notice', default: '', displayOptions: show('experts', 'expertAction') },
  { displayName: 'Prefill & Decode keeps its single-use handoff only inside this execution. No persistent split workflow or automatic decode retry is provided.', name: 'prefillNotice', type: 'notice', default: '', displayOptions: show('prefillDecode') },
  { displayName: 'Input Type', name: 'inputType', type: 'options', default: 'text', options: options('text', 'messages'), displayOptions: show('prefillDecode') },
  { displayName: 'Text', name: 'text', type: 'string', default: '', typeOptions: { rows: 5 }, displayOptions: { show: { operation: ['prefillDecode'], inputType: ['text'] } } },
  { displayName: 'Messages', name: 'messages', type: 'json', default: '[{"role":"user","content":"Hello"}]', displayOptions: { show: { operation: ['prefillDecode'], inputType: ['messages'] } }, description: 'Ordered array of objects containing exactly role and non-empty content' },
  { displayName: 'Max Tokens', name: 'maxTokens', type: 'number', default: 256, typeOptions: { minValue: 1, maxValue: 32768, numberPrecision: 0 }, displayOptions: show('prefillDecode') },
  { displayName: 'Temperature', name: 'temperature', type: 'number', default: -1, typeOptions: { minValue: -1, maxValue: 2 }, displayOptions: show('prefillDecode'), description: '-1 inherits the server value; 0 is sent explicitly' },
  { displayName: 'Top P', name: 'topP', type: 'number', default: -1, typeOptions: { minValue: -1, maxValue: 1 }, displayOptions: show('prefillDecode'), description: '-1 inherits; otherwise greater than 0 and at most 1' },
  { displayName: 'Seed', name: 'seed', type: 'number', default: -1, typeOptions: { minValue: -1, maxValue: Number.MAX_SAFE_INTEGER, numberPrecision: 0 }, displayOptions: show('prefillDecode'), description: '-1 inherits. Seeds outside the JavaScript safe integer range are rejected.' },
  { displayName: 'Stop Sequences', name: 'stopSequences', type: 'json', default: '[]', displayOptions: show('prefillDecode'), description: 'Up to 16 non-empty strings of at most 1024 UTF-8 bytes each' },
  { displayName: 'Persistence Policy', name: 'persistencePolicy', type: 'collection', default: {}, placeholder: 'Add Policy Setting', displayOptions: show('prefillDecode'), description: 'Disabled policy is absent from the request. An enabled policy is a complete object.', options: [
    { displayName: 'Enabled', name: 'enabled', type: 'boolean', default: false, description: 'Whether to send an explicit complete persistence policy' },
    { displayName: 'Mode', name: 'mode', type: 'options', default: 'auto', options: options('auto', 'ephemeral', 'memory', 'disk') },
    { displayName: 'Reuse', name: 'reuse', type: 'options', default: 'prefer', options: options('prefer', 'disabled', 'required') },
    { displayName: 'TTL (Seconds)', name: 'ttlSeconds', type: 'number', default: 0, typeOptions: { minValue: 0, numberPrecision: 0 }, description: '0 omits TTL within this explicit policy and means no TTL' },
    { displayName: 'Pin', name: 'pin', type: 'boolean', default: false, description: 'Whether to pin the resulting state, subject to expiry' },
  ] },
  ...transportProperties,
];

export class WerkRuntime implements INodeType {
  description = nodeDescription('werkRuntime', 'WERK Runtime (Beta)', properties);
  methods = commonMethods;

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    return executeItems(this, async (client, index) => {
      const result = await runRuntime(client, (name, fallback) => this.getNodeParameter(name, index, fallback as string));
      return [{ json: result as INodeExecutionData['json'], pairedItem: { item: index } }];
    });
  }
}

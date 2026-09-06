import { object, parseJson, safeInteger, sanitize, string } from './validation';
import { boolean, boundedString, choice, cursorValue, EXPERT_TIERS, modelIdentifier, opaqueId, redactHandoffs, requireCapability, STATE_TIERS, WerkProtocolClient, WerkProtocolError } from './protocol';
import type { ProtocolTransport, RuntimeInfo } from './protocol';

type JsonObject = Record<string, unknown>;
export type RuntimeParameters = (name: string, fallback: unknown) => unknown;

function json(value: unknown, label: string): unknown {
  if (typeof value !== 'string') return value;
  try { return parseJson(value, label); } catch { throw new Error(`${label} must be valid JSON without duplicate keys or unsafe numbers`); }
}

function optionalString(value: unknown, label: string): string {
  if (typeof value !== 'string') throw new Error(`${label} must be a string`);
  return value.trim();
}

export function runtimeIds(value: unknown, kind: 'state' | 'expert'): string[] {
  let values: unknown = value;
  if (typeof value === 'string') {
    const raw = value.trim();
    values = raw.startsWith('[') ? json(raw, `${kind} IDs`) : raw ? raw.split(/[\n,]/).map((entry) => entry.trim()) : [];
  }
  if (!Array.isArray(values) || values.length < 1 || values.length > 4096) throw new Error(`${kind} IDs require 1 to 4096 explicit IDs`);
  const result: string[] = [];
  for (const value of values) {
    const id = kind === 'expert' ? opaqueId(value, 'expert ID') : boundedString(value, 'state ID', 256);
    if (/[\x00-\x1f\x7f]/.test(id)) throw new Error('Invalid state ID');
    if (result.includes(id)) {
      if (kind === 'expert') throw new Error('Expert IDs must be unique');
    } else result.push(id);
  }
  return result;
}

export function persistencePolicy(value: unknown, maxTtl: number): JsonObject | undefined {
  const source = object(value, 'persistence policy');
  if (!boolean(source.enabled ?? false, 'policy enabled')) return undefined;
  const allowed = ['enabled', 'mode', 'reuse', 'ttlSeconds', 'pin'];
  if (Object.keys(source).some((key) => !allowed.includes(key))) throw new Error('Persistence policy contains unknown fields');
  const mode = choice(source.mode ?? 'auto', 'policy mode', ['auto', 'ephemeral', 'memory', 'disk']);
  const reuse = choice(source.reuse ?? 'prefer', 'policy reuse', ['prefer', 'disabled', 'required']);
  const pin = boolean(source.pin ?? false, 'policy pin');
  const ttl = safeInteger(source.ttlSeconds ?? 0, 'policy TTL');
  if (ttl > maxTtl) throw new Error(`Persistence TTL exceeds the server limit of ${maxTtl} seconds`);
  return { mode, reuse, pin, ...(ttl ? { ttl_seconds: ttl } : {}) };
}

function prefillInput(p: RuntimeParameters): JsonObject {
  const kind = choice(p('inputType', 'text'), 'input type', ['text', 'messages']);
  if (kind === 'text') return { type: 'text', text: boundedString(p('text', ''), 'prefill text', 512 * 1024) };
  const source = json(p('messages', '[]'), 'messages');
  if (!Array.isArray(source) || source.length < 1 || source.length > 256) throw new Error('Prefill messages must contain 1 to 256 entries');
  let bytes = 0;
  const messages = source.map((entry) => {
    const value = object(entry, 'prefill message');
    if (Object.keys(value).sort().join(',') !== 'content,role') throw new Error('Prefill messages require exactly role and content');
    const role = string(value.role, 'message role');
    if (!/^[a-z_-]{1,32}$/.test(role)) throw new Error('Invalid prefill message role');
    const content = boundedString(value.content, 'message content', 512 * 1024);
    bytes += Buffer.byteLength(content);
    if (bytes > 512 * 1024) throw new Error('Prefill message content exceeds 512 KiB');
    return { role, content };
  });
  return { type: 'messages', messages };
}

function decodeOptions(p: RuntimeParameters, allowExperimental: boolean): JsonObject {
  const maxTokens = safeInteger(p('maxTokens', 256), 'max tokens', 1);
  if (maxTokens > 32768) throw new Error('Decode max tokens must not exceed 32768');
  const stop = json(p('stopSequences', '[]'), 'stop sequences');
  if (!Array.isArray(stop) || stop.length > 16) throw new Error('Decode stop requires at most 16 strings');
  const result: JsonObject = { max_tokens: maxTokens, stop: stop.map((entry) => boundedString(entry, 'stop sequence', 1024)), allow_experimental: allowExperimental };
  const temperature = p('temperature', -1);
  if (typeof temperature !== 'number' || !Number.isFinite(temperature) || (temperature !== -1 && (temperature < 0 || temperature > 2))) throw new Error('Temperature must be -1 (inherit) or between 0 and 2');
  if (temperature !== -1) result.temperature = temperature;
  const topP = p('topP', -1);
  if (typeof topP !== 'number' || !Number.isFinite(topP) || (topP !== -1 && (topP <= 0 || topP > 1))) throw new Error('Top P must be -1 (inherit) or greater than 0 and at most 1');
  if (topP !== -1) result.top_p = topP;
  const seed = p('seed', -1);
  // JSON numbers outside the safe integer range cannot be represented faithfully by JavaScript.
  safeInteger(seed, 'seed', -1);
  if ((seed as number) >= 0) result.seed = seed;
  return result;
}

function limit(p: RuntimeParameters, info: RuntimeInfo): number {
  const value = safeInteger(p('limit', 50), 'page limit', 1);
  const maximum = Math.min(4096, info.limits.max_page_size);
  if (value > maximum) throw new Error(`Runtime page limit must not exceed ${maximum}`);
  return value;
}

async function pages(client: WerkProtocolClient, kind: 'states' | 'experts', p: RuntimeParameters, info: RuntimeInfo, allowExperimental: boolean): Promise<JsonObject> {
  const model = optionalString(p('modelId', ''), 'model ID');
  const tier = choice(p('tier', 'all'), 'tier', ['all', ...(kind === 'experts' ? EXPERT_TIERS : STATE_TIERS)]);
  const cursor = optionalString(p('cursor', ''), 'cursor');
  const query: Record<string, string | number | boolean> = { limit: limit(p, info) };
  if (model) query.model_id = modelIdentifier(model);
  if (tier !== 'all') query.tier = tier;
  if (cursor) query.cursor = cursorValue(cursor);
  if (kind === 'experts') query.allow_experimental = allowExperimental;
  const returnAll = boolean(p('returnAll', false), 'return all');
  const maxPages = safeInteger(p('maxPages', 10), 'maximum pages', 1);
  if (maxPages > 100) throw new Error('Automatic pagination is limited to 100 pages');
  const cursors = new Set<string>(cursor ? [cursor] : []);
  const identities = new Set<string>();
  const entries: JsonObject[] = [];
  for (let pageIndex = 0; pageIndex < (returnAll ? maxPages : 1); pageIndex++) {
    const page = await client.page(kind, query);
    for (const entry of page.entries) {
      const id = kind === 'experts' ? JSON.stringify([entry.model_id, entry.id]) : string(entry.id, 'state ID');
      if (identities.has(id)) throw new Error('Runtime pagination returned a duplicate identity');
      identities.add(id);
      entries.push(entry);
      if (entries.length > 4096) throw new Error('Automatic pagination exceeded 4096 entries; use individual pages and cursors');
    }
    if (!returnAll || !page.next_cursor) return { [kind]: entries, count: entries.length, nextCursor: page.next_cursor };
    if (cursors.has(page.next_cursor)) throw new Error('Runtime pagination returned a repeated cursor');
    cursors.add(page.next_cursor);
    query.cursor = page.next_cursor;
  }
  throw new Error('Runtime pagination reached the configured page bound; use individual pages and cursors');
}

/** Capability reads remain per operation, and are repeated after prefill's model-scoped probe. */
export async function runRuntime(transport: ProtocolTransport, p: RuntimeParameters): Promise<JsonObject> {
  const operation = choice(p('operation', 'info'), 'runtime operation', ['info', 'capabilities', 'memory', 'states', 'stateAction', 'pruneStates', 'experts', 'expertAction', 'prefillDecode']);
  const client = new WerkProtocolClient(transport);
  const info = await client.info();
  let capabilities = await client.capabilities();
  const allowExperimental = boolean(p('allowExperimental', false), 'experimental opt-in');
  const werk = (): JsonObject => ({ protocol: info.protocol, serviceVersion: info.service_version, backend: info.active_backend, requestIds: client.requestIds });
  const output = (result: JsonObject): JsonObject => ({ ...result, werk: werk() });
  if (operation === 'info') return output({ info, capabilities });
  if (operation === 'capabilities') return output({ capabilities });
  if (operation === 'memory') return output(await client.memory());
  // As in ComfyUI, catalog/memory reads and state controls use mandatory discovery,
  // then the server gates the actual state. No backend-name capability inference.
  if (operation === 'states') return output(await pages(client, 'states', p, info, allowExperimental));
  if (operation === 'stateAction') {
    const stateId = boundedString(p('stateId', ''), 'state ID', 256);
    const action = choice(p('stateAction', 'pin'), 'state action', ['pin', 'unpin', 'promote', 'demote', 'evict']);
    const targetTier = choice(p('targetTier', 'unchanged'), 'target tier', ['unchanged', 'ram', 'vram', 'disk']);
    if (action === 'promote' && !['ram', 'vram'].includes(targetTier)) throw new Error('Promote requires an explicit RAM or VRAM target');
    if (action === 'demote' && !['ram', 'disk'].includes(targetTier)) throw new Error('Demote requires an explicit RAM or disk target');
    if (!['promote', 'demote'].includes(action) && targetTier !== 'unchanged') throw new Error('Only promote and demote accept a target tier');
    const payload: JsonObject = { action, dry_run: boolean(p('dryRun', true), 'dry run'), allow_experimental: allowExperimental };
    if (targetTier !== 'unchanged') payload.target_tier = targetTier;
    return output(await client.stateAction(stateId, payload));
  }
  if (operation === 'pruneStates') {
    const kind = choice(p('selector', 'ids'), 'prune selector', ['ids', 'filter', 'all']);
    const selector: JsonObject = { kind };
    if (kind === 'ids') {
      const ids = runtimeIds(p('stateIds', ''), 'state');
      if (ids.length > info.limits.max_state_ids_per_operation) throw new Error(`Server accepts at most ${info.limits.max_state_ids_per_operation} state IDs`);
      selector.ids = ids;
    } else if (kind === 'filter') {
      const model = optionalString(p('modelId', ''), 'model ID');
      const tier = choice(p('tier', 'all'), 'tier', ['all', ...STATE_TIERS]);
      const cutoff = safeInteger(p('olderThanUnixMs', 0), 'older-than timestamp');
      if (model) selector.model_id = modelIdentifier(model);
      if (tier !== 'all') selector.tier = tier;
      if (cutoff) selector.older_than_unix_ms = cutoff;
      if (Object.keys(selector).length === 1) throw new Error('Prune filter requires at least one real restriction');
    } else {
      if (!boolean(p('confirmAll', false), 'confirm all')) throw new Error('Prune All requires explicit Confirm All');
      selector.confirm = true;
    }
    return output(await client.pruneStates({ selector, dry_run: boolean(p('dryRun', true), 'dry run') }));
  }
  if (operation === 'experts') {
    requireCapability(capabilities, 'runtime.experts.residency', allowExperimental, 'expertRead');
    return output(await pages(client, 'experts', p, info, allowExperimental));
  }
  if (operation === 'expertAction') {
    const model = modelIdentifier(p('modelId', ''));
    const ids = runtimeIds(p('expertIds', ''), 'expert');
    if (ids.length > info.limits.max_expert_ids_per_operation) throw new Error(`Server accepts at most ${info.limits.max_expert_ids_per_operation} expert IDs`);
    const action = choice(p('expertAction', 'pin'), 'expert action', ['prefetch', 'pin', 'unpin', 'evict']);
    const tier = choice(p('expertTargetTier', 'unchanged'), 'expert target tier', ['unchanged', 'ram', 'vram']);
    if (action === 'prefetch' && !['ram', 'vram'].includes(tier)) throw new Error('Expert prefetch requires an explicit RAM or VRAM target');
    if (action !== 'prefetch' && tier !== 'unchanged') throw new Error('Only expert prefetch accepts a target tier');
    requireCapability(capabilities, 'runtime.experts.residency', allowExperimental);
    return output(await client.expertAction({ model_id: model, expert_ids: ids, action, dry_run: boolean(p('dryRun', true), 'dry run'), allow_experimental: allowExperimental, ...(tier !== 'unchanged' ? { target_tier: tier } : {}) }));
  }
  const modelId = modelIdentifier(p('modelId', ''));
  const input = prefillInput(p);
  const policy = persistencePolicy(p('persistencePolicy', {}), info.limits.max_ttl_seconds);
  const decode = decodeOptions(p, allowExperimental);
  requireCapability(capabilities, 'runtime.pd.prefill', allowExperimental, 'prefillProbe');
  requireCapability(capabilities, 'runtime.pd.handoff', allowExperimental, 'prefillProbe');
  const prefill = await client.prefill({ model_id: modelId, input, allow_experimental: allowExperimental, ...(policy ? { policy } : {}) });
  const handoff = prefill.handoff;
  try {
    capabilities = await client.capabilities();
    requireCapability(capabilities, 'runtime.pd.decode', allowExperimental);
    requireCapability(capabilities, 'runtime.pd.handoff', allowExperimental);
    const decoded = await client.decode({ ...decode, handoff }, handoff);
    return redactHandoffs(output({ model: modelId, task: 'prefill_decode', text: decoded.text, finishReason: decoded.finish_reason, promptTokens: prefill.prompt_tokens, completionTokens: decoded.completion_tokens, stateId: prefill.state_id, reused: prefill.reused, tier: prefill.tier, expiresUnixMs: prefill.expires_unix_ms }), [handoff]) as JsonObject;
  } catch (error) {
    const detail = error instanceof WerkProtocolError ? error.safeDetail : error instanceof Error ? error.message : 'Runtime decode failed';
    const message = String(redactHandoffs(sanitize(detail), [handoff]));
    if (error instanceof WerkProtocolError) throw new WerkProtocolError(error.code, message, error.statusCode, error.requestId === handoff ? undefined : error.requestId, error.retryable);
    throw new Error(message);
  }
}

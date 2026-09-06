import type { WerkClient } from './client';
import { object, parseJson, safeInteger, sanitize, string } from './validation';

type JsonObject = Record<string, unknown>;
export type ProtocolTransport = Pick<WerkClient, 'raw'> & { redact?: (value: unknown) => unknown };
export const CAPABILITY_STATUSES = ['supported', 'unsupported', 'unavailable', 'experimental', 'externally_managed', 'metadata_only'] as const;
const ERROR_CODES = ['invalid_request', 'unauthorized', 'forbidden', 'not_found', 'conflict', 'incompatible_state', 'expired_handoff', 'unsupported', 'unavailable', 'experimental_opt_in_required', 'resource_exhausted', 'corrupt_state', 'internal', 'incompatible_protocol'];
export const STATE_TIERS = ['vram', 'ram', 'disk', 'external'];
export const EXPERT_TIERS = ['vram', 'ram', 'external'];
const PRESSURES = ['normal', 'soft', 'hard', 'emergency', 'unknown'];
const LOCAL_BYTES = 1024 * 1024;

export class WerkProtocolError extends Error {
  readonly code: string;
  readonly requestId?: string;
  readonly retryable: boolean;
  readonly statusCode?: number;
  readonly safeDetail: string;

  constructor(code: string, message: string, statusCode?: number, requestId?: string, retryable = false) {
    super(`Werk Protocol: ${code}: ${message}${requestId ? ` [request ${requestId}]` : ''} (retryable: ${retryable})`);
    this.name = 'WerkProtocolError';
    this.code = code;
    this.requestId = requestId;
    this.retryable = retryable;
    this.statusCode = statusCode;
    this.safeDetail = message;
  }
}

export function boolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} must be a boolean`);
  return value;
}

export function choice(value: unknown, label: string, values: readonly string[]): string {
  const result = string(value, label);
  if (!values.includes(result)) throw new Error(`${label} has an unknown value`);
  return result;
}

export function boundedString(value: unknown, label: string, maxBytes: number, allowEmpty = false): string {
  if (typeof value !== 'string' || (!allowEmpty && !value) || Buffer.byteLength(value) > maxBytes) {
    throw new Error(`${label} must be ${allowEmpty ? 'a' : 'a non-empty'} string of at most ${maxBytes} UTF-8 bytes`);
  }
  return value;
}

function optionalInteger(value: unknown, label: string): number | null {
  return value == null ? null : safeInteger(value, label);
}

export function opaqueId(value: unknown, label: string): string {
  const id = boundedString(value, label, 128);
  if (!/^[A-Za-z0-9_.-]+$/.test(id) || id.includes('..')) throw new Error(`${label} is not a valid opaque ID`);
  return id;
}

export function cursorValue(value: unknown): string {
  const result = boundedString(value, 'cursor', 256);
  if (/[\x00-\x1f\x7f]/.test(result)) throw new Error('cursor contains invalid characters');
  return result;
}

export function modelIdentifier(value: unknown): string {
  const result = boundedString(value, 'model_id', 256);
  if (/[\x00-\x1f\x7f]/.test(result) || result.includes('..')) throw new Error('model_id contains invalid characters');
  return result;
}

function boundedArray(value: unknown, label: string, maximum = 4096): unknown[] {
  if (!Array.isArray(value) || value.length > maximum) throw new Error(`${label} must be an array of at most ${maximum} entries`);
  return value;
}

function version(value: unknown): { major: number; minor: number } {
  const source = object(value, 'protocol');
  const major = safeInteger(source.major, 'protocol.major');
  const minor = safeInteger(source.minor, 'protocol.minor');
  if (major !== 1 || minor !== 0) throw new WerkProtocolError('incompatible_protocol', 'This client requires Werk Protocol 1.0');
  return { major, minor };
}

function header(headers: JsonObject, name: string): unknown {
  const found = Object.entries(headers).filter(([key]) => key.toLowerCase() === name);
  if (found.length > 1) throw new Error('Duplicate protocol response header');
  return found[0]?.[1];
}

function validateVersionHeader(headers: JsonObject): void {
  const declaration = header(headers, 'x-werk-protocol-version');
  if (declaration === undefined) return;
  if (typeof declaration !== 'string' || !/^\s*[0-9]+\.[0-9]+\s*$/.test(declaration)) throw new Error('Invalid or duplicate Werk Protocol version header');
  const [major, minor] = declaration.trim().split('.').map(Number);
  if (major > 65535 || minor > 65535) throw new Error('Invalid Werk Protocol version header');
  version({ major, minor });
}

export interface RuntimeInfo {
  service: string;
  service_version: string;
  protocol: { major: number; minor: number };
  active_backend: string;
  limits: Record<'max_page_size' | 'max_state_ids_per_operation' | 'max_expert_ids_per_operation' | 'max_request_bytes' | 'max_handoff_bytes' | 'max_ttl_seconds', number>;
}

export interface Capability {
  id: string;
  status: string;
  detail: string;
  operations: string[];
}

export function requireCapability(capabilities: Capability[], id: string, allowExperimental: boolean, mode: 'normal' | 'prefillProbe' | 'expertRead' = 'normal'): void {
  const capability = capabilities.find((entry) => entry.id === id);
  if (!capability) throw new Error(`Werk did not declare required capability ${id}`);
  const { status } = capability;
  if (status === 'supported' || (status === 'experimental' && allowExperimental)) return;
  if (status === 'externally_managed' && mode === 'expertRead' && id === 'runtime.experts.residency') return;
  if (status === 'unavailable' && allowExperimental && mode === 'prefillProbe' && ['runtime.pd.prefill', 'runtime.pd.handoff'].includes(id)) return;
  if (status === 'experimental') throw new Error(`Werk capability ${id} is experimental; enable explicit experimental opt-in`);
  throw new Error(`Werk capability ${id} is ${status}: ${String(sanitize(capability.detail))}`);
}

function stateSummary(value: unknown): JsonObject {
  const data = object(value, 'state');
  return {
    id: string(data.id, 'state.id'), model_id: string(data.model_id, 'state.model_id'),
    tier: choice(data.tier, 'state.tier', STATE_TIERS),
    status: choice(data.status, 'state.status', ['ready', 'loading', 'moving', 'unavailable', 'quarantined']),
    bytes: optionalInteger(data.bytes, 'state.bytes'), created_unix_ms: safeInteger(data.created_unix_ms, 'state.created_unix_ms'),
    last_accessed_unix_ms: safeInteger(data.last_accessed_unix_ms, 'state.last_accessed_unix_ms'),
    expires_unix_ms: optionalInteger(data.expires_unix_ms, 'state.expires_unix_ms'), pinned: boolean(data.pinned, 'state.pinned'),
    backend: string(data.backend, 'state.backend'), reusable: boolean(data.reusable, 'state.reusable'),
  };
}

function expertSummaries(value: unknown): JsonObject[] {
  const seen = new Set<string>();
  return boundedArray(value, 'experts').map((item) => {
    const data = object(item, 'expert');
    const id = opaqueId(data.id, 'expert.id');
    const modelId = modelIdentifier(data.model_id);
    const identity = JSON.stringify([modelId, id]);
    if (seen.has(identity)) throw new Error('Duplicate expert identity');
    seen.add(identity);
    if (typeof data.hotness !== 'number' || !Number.isFinite(data.hotness) || data.hotness < 0) throw new Error('expert.hotness must be a finite non-negative number');
    return {
      id, model_id: modelId, tier: choice(data.tier, 'expert.tier', EXPERT_TIERS), bytes: optionalInteger(data.bytes, 'expert.bytes'),
      hotness: data.hotness, pinned: boolean(data.pinned, 'expert.pinned'), last_used_unix_ms: optionalInteger(data.last_used_unix_ms, 'expert.last_used_unix_ms'),
    };
  });
}

export function redactHandoffs(value: unknown, tokens: readonly string[]): unknown {
  if (typeof value === 'string') {
    let result = value;
    for (const token of tokens) if (token) result = result.split(token).join('[redacted]');
    return result;
  }
  if (Array.isArray(value)) return value.map((entry) => redactHandoffs(entry, tokens));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).filter(([key]) => !/handoff|authorization|api[_-]?key|secret|password/i.test(key)).map(([key, item]) => [String(redactHandoffs(key, tokens)), redactHandoffs(item, tokens)]));
  return value;
}

/** No cache, fallback, retries or stored handoff state. Each instance is local to one input item. */
export class WerkProtocolClient {
  private limits?: RuntimeInfo['limits'];
  readonly requestIds: string[] = [];

  constructor(private readonly transport: ProtocolTransport) {}

  private safe(value: unknown, tokens: readonly string[] = []): unknown {
    return redactHandoffs(this.transport.redact ? this.transport.redact(value) : sanitize(value), tokens);
  }

  async request(method: 'GET' | 'POST', path: string, payload?: JsonObject, query?: Record<string, string | number | boolean>, tokens: readonly string[] = []): Promise<JsonObject> {
    if (!/^\/werk\/v1\/(info|capabilities|memory|states|states\/prune|states\/[^/?#]+\/actions|experts|experts\/actions|prefill|decode)$/.test(path)) throw new WerkProtocolError('invalid_request', 'Invalid protocol endpoint');
    if (payload && Buffer.byteLength(JSON.stringify(payload)) > Math.min(LOCAL_BYTES, this.limits?.max_request_bytes ?? LOCAL_BYTES)) throw new WerkProtocolError('request_too_large', 'Protocol request exceeds the discovered or local byte limit');
    let response: Awaited<ReturnType<ProtocolTransport['raw']>>;
    try {
      response = await this.transport.raw(method, path, payload, query, true);
    } catch (error) {
      if (error instanceof Error && error.message.startsWith('WERK redirect rejected')) throw new WerkProtocolError('redirect_rejected', 'Protocol redirects are not allowed');
      // A helper/transport error can include serialized request bodies and bearer handoffs.
      throw new WerkProtocolError('transport_error', 'Protocol request failed or timed out; the operation may have executed. No automatic retry was made.', undefined, undefined, true);
    }
    const failed = response.statusCode < 200 || response.statusCode >= 300;
    if (response.statusCode >= 300 && response.statusCode < 400) throw new WerkProtocolError('redirect_rejected', 'Protocol redirects are not allowed', response.statusCode);
    let requestId: string | undefined;
    try {
      validateVersionHeader(response.headers);
      const contentType = header(response.headers, 'content-type');
      if (typeof contentType !== 'string' || contentType.split(';')[0].trim().toLowerCase() !== 'application/json') throw new Error('Werk Protocol response must be application/json');
      const serialized = typeof response.body === 'string' ? response.body : JSON.stringify(response.body);
      if (typeof serialized !== 'string' || Buffer.byteLength(serialized) > (failed ? 64 * 1024 : 8 * LOCAL_BYTES)) throw new Error('Protocol response exceeds the safe byte limit');
      let parsed: unknown;
      try { parsed = typeof response.body === 'string' ? parseJson(response.body, 'protocol response') : response.body; } catch { throw new Error('Invalid protocol JSON or unsafe numbers'); }
      const envelope = object(parsed, 'protocol envelope');
      version(envelope.protocol);
      const id = boundedString(envelope.request_id, 'request_id', 128);
      if (!/^[A-Za-z0-9_.-]+$/.test(id)) throw new Error('Invalid request_id');
      const safeId = String(this.safe(id, tokens));
      requestId = safeId === id ? id : undefined;
      if (failed) {
        if ('data' in envelope) throw new Error('Error envelope cannot contain data');
        const detail = object(envelope.error, 'protocol error');
        const code = choice(detail.code, 'error.code', ERROR_CODES);
        const retryable = boolean(detail.retryable, 'error.retryable');
        const message = string(detail.message, 'error.message');
        throw new WerkProtocolError(code, String(this.safe(message, tokens)).replace(/[\x00-\x1f\x7f]/g, ' ').slice(0, 1000), response.statusCode, requestId, retryable);
      }
      if (!('data' in envelope) || 'error' in envelope) throw new Error('Success envelope must contain data and no error');
      if (requestId) this.requestIds.push(requestId);
      return object(envelope.data, 'protocol data');
    } catch (error) {
      if (error instanceof WerkProtocolError) throw error;
      const message = error instanceof Error ? error.message : 'Invalid protocol response';
      throw new WerkProtocolError(failed ? 'invalid_error_envelope' : 'invalid_response', String(this.safe(message, tokens)), response.statusCode, requestId);
    }
  }

  async info(): Promise<RuntimeInfo> {
    const data = await this.request('GET', '/werk/v1/info');
    const rawLimits = object(data.limits, 'runtime limits');
    const limits: RuntimeInfo['limits'] = {
      max_page_size: safeInteger(rawLimits.max_page_size, 'limits.max_page_size', 1),
      max_state_ids_per_operation: safeInteger(rawLimits.max_state_ids_per_operation, 'limits.max_state_ids_per_operation', 1),
      max_expert_ids_per_operation: safeInteger(rawLimits.max_expert_ids_per_operation, 'limits.max_expert_ids_per_operation', 1),
      max_request_bytes: safeInteger(rawLimits.max_request_bytes, 'limits.max_request_bytes', 1),
      max_handoff_bytes: safeInteger(rawLimits.max_handoff_bytes, 'limits.max_handoff_bytes', 1),
      max_ttl_seconds: safeInteger(rawLimits.max_ttl_seconds, 'limits.max_ttl_seconds', 1),
    };
    const result = { service: string(data.service, 'service'), service_version: string(data.service_version, 'service_version'), protocol: version(data.protocol), active_backend: string(data.active_backend, 'active_backend'), limits };
    this.limits = limits;
    return result;
  }

  async capabilities(): Promise<Capability[]> {
    const data = await this.request('GET', '/werk/v1/capabilities');
    const seen = new Set<string>();
    return boundedArray(data.capabilities, 'capabilities').map((value) => {
      const entry = object(value, 'capability');
      const id = string(entry.id, 'capability.id');
      if (seen.has(id)) throw new Error('Duplicate capability ID');
      seen.add(id);
      return { id, status: choice(entry.status, 'capability.status', CAPABILITY_STATUSES), detail: boundedString(entry.detail, 'capability.detail', 1024 * 1024, true), operations: boundedArray(entry.operations ?? [], 'capability.operations', 128).map((operation) => string(operation, 'capability.operation')) };
    });
  }

  async memory(): Promise<JsonObject> {
    const data = await this.request('GET', '/werk/v1/memory');
    const tier = (value: unknown): JsonObject => {
      const entry = object(value, 'memory tier');
      return { capacity_bytes: optionalInteger(entry.capacity_bytes, 'capacity_bytes'), available_bytes: optionalInteger(entry.available_bytes, 'available_bytes'), managed_bytes: safeInteger(entry.managed_bytes, 'managed_bytes'), reserved_bytes: safeInteger(entry.reserved_bytes, 'reserved_bytes'), pressure: choice(entry.pressure, 'pressure', PRESSURES) };
    };
    const counters = object(data.counters, 'memory counters');
    if (Object.keys(counters).length > 1024) throw new Error('Too many memory counters');
    return { observed_at_unix_ms: safeInteger(data.observed_at_unix_ms, 'observed_at_unix_ms'), overall_pressure: choice(data.overall_pressure, 'overall_pressure', PRESSURES), topology: string(data.topology, 'topology'), host: tier(data.host), accelerator: tier(data.accelerator), last_action_unix_ms: optionalInteger(data.last_action_unix_ms, 'last_action_unix_ms'), counters: Object.fromEntries(Object.entries(counters).map(([key, value]) => [string(key, 'counter name'), safeInteger(value, 'memory counter')])) };
  }

  async page(kind: 'states' | 'experts', query: Record<string, string | number | boolean>): Promise<{ entries: JsonObject[]; next_cursor: string | null }> {
    const data = await this.request('GET', `/werk/v1/${kind}`, undefined, query);
    const entries = kind === 'states' ? boundedArray(data.states, 'states').map(stateSummary) : expertSummaries(data.experts);
    if (typeof query.limit === 'number' && entries.length > query.limit) throw new Error('Runtime page exceeded the requested limit');
    return { entries, next_cursor: data.next_cursor == null ? null : cursorValue(data.next_cursor) };
  }

  async stateAction(stateId: string, payload: JsonObject): Promise<JsonObject> {
    const data = await this.request('POST', `/werk/v1/states/${encodeURIComponent(stateId)}/actions`, payload);
    const state = stateSummary(data.state);
    const dryRun = boolean(data.dry_run, 'state action dry_run');
    if (state.id !== stateId || dryRun !== payload.dry_run) throw new Error('State action response does not match its request');
    return { state, changed: boolean(data.changed, 'state action changed'), dry_run: dryRun };
  }

  async pruneStates(payload: JsonObject): Promise<JsonObject> {
    const data = await this.request('POST', '/werk/v1/states/prune', payload);
    const matched = safeInteger(data.matched, 'prune matched');
    const removed = safeInteger(data.removed, 'prune removed');
    const dryRun = boolean(data.dry_run, 'prune dry_run');
    if (dryRun !== payload.dry_run || removed > matched || (dryRun && removed !== 0)) throw new Error('Prune response does not match its dry-run contract');
    return { matched, removed, bytes: optionalInteger(data.bytes, 'prune bytes'), dry_run: dryRun };
  }

  async expertAction(payload: JsonObject): Promise<JsonObject> {
    const data = await this.request('POST', '/werk/v1/experts/actions', payload);
    const experts = expertSummaries(data.experts);
    const changed = safeInteger(data.changed, 'expert action changed');
    const dryRun = boolean(data.dry_run, 'expert action dry_run');
    const ids = payload.expert_ids as string[];
    if (dryRun !== payload.dry_run || experts.length > ids.length || changed > ids.length || experts.some((entry) => entry.model_id !== payload.model_id || !ids.includes(entry.id as string))) throw new Error('Expert action response does not match its explicit selection');
    return { experts, changed, dry_run: dryRun };
  }

  private handoff(value: unknown): string {
    const token = boundedString(value, 'handoff', Math.min(4096, this.limits?.max_handoff_bytes ?? 4096));
    if (token.length < 32) throw new Error('Invalid handoff length');
    return token;
  }

  private discardSecretRequestIds(tokens: readonly string[]): void {
    for (let index = this.requestIds.length - 1; index >= 0; index--) {
      if (tokens.some((token) => token && this.requestIds[index].includes(token))) this.requestIds.splice(index, 1);
    }
  }

  async prefill(payload: JsonObject): Promise<{ handoff: string; state_id: string | null; prompt_tokens: number; reused: boolean; tier: string; expires_unix_ms: number }> {
    const data = await this.request('POST', '/werk/v1/prefill', payload);
    const handoff = this.handoff(data.handoff);
    this.discardSecretRequestIds([handoff]);
    return { handoff, state_id: data.state_id == null ? null : string(data.state_id, 'prefill state_id'), prompt_tokens: safeInteger(data.prompt_tokens, 'prompt_tokens'), reused: boolean(data.reused, 'prefill reused'), tier: choice(data.tier, 'prefill tier', STATE_TIERS), expires_unix_ms: safeInteger(data.expires_unix_ms, 'prefill expires_unix_ms') };
  }

  async decode(payload: JsonObject, token: string): Promise<{ text: string; completion_tokens: number; finish_reason: string }> {
    const data = await this.request('POST', '/werk/v1/decode', payload, undefined, [token]);
    const updated = data.handoff == null ? '' : this.handoff(data.handoff);
    this.discardSecretRequestIds([token, updated]);
    // An updated handoff is deliberately discarded, including echoed tokens in text/metadata.
    return { text: String(this.safe(boundedString(data.text, 'decode text', LOCAL_BYTES, true), [token, updated])), completion_tokens: safeInteger(data.completion_tokens, 'completion_tokens'), finish_reason: String(this.safe(string(data.finish_reason, 'finish_reason'), [token, updated])) };
  }
}

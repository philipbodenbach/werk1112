export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as Record<string, unknown>;
}

export function string(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a nonempty string`);
  return value.trim();
}

export function safeInteger(value: unknown, label: string, min = 0): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min) {
    throw new Error(`${label} must be a safe integer >= ${min} (maximum ${Number.MAX_SAFE_INTEGER})`);
  }
  return value;
}

export function finiteNumber(value: unknown, label: string, min = 0): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min) throw new Error(`${label} must be a finite number >= ${min}`);
  return value;
}

/** Reject duplicate keys before JSON.parse can silently erase a conflicting override. */
export function parseJson(text: string, label = 'JSON'): unknown {
  if (Buffer.byteLength(text) > 128 * 1024 * 1024) throw new Error(`${label} exceeds JSON size limit`);
  let pos = 0;
  const ws = () => { while (/\s/.test(text[pos] ?? '') && pos < text.length) pos++; };
  const fail = (): never => { throw new Error(`${label} is invalid JSON or contains duplicate keys/unsafe numbers`); };
  function tokenString(): string {
    const start = pos++;
    while (pos < text.length) {
      if (text[pos] === '\\') { pos += 2; continue; }
      if (text[pos++] === '"') {
        try { return JSON.parse(text.slice(start, pos)) as string; } catch { return fail(); }
      }
    }
    return fail();
  }
  function walk(depth: number): void {
    if (depth > 64) fail();
    ws();
    if (text[pos] === '"') { tokenString(); return; }
    if (text[pos] === '{' || text[pos] === '[') {
      const isObject = text[pos++] === '{';
      const end = isObject ? '}' : ']';
      const keys = new Set<string>();
      ws();
      if (text[pos] === end) { pos++; return; }
      while (pos < text.length) {
        if (isObject) {
          ws();
          if (text[pos] !== '"') fail();
          const key = tokenString();
          if (keys.has(key) || ['__proto__', 'prototype', 'constructor'].includes(key)) fail();
          keys.add(key);
          ws();
          if (text[pos++] !== ':') fail();
        }
        walk(depth + 1); ws();
        if (text[pos] === end) { pos++; return; }
        if (text[pos++] !== ',') fail();
      }
      fail();
    }
    const literal = /^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(text.slice(pos));
    if (!literal) fail();
    const value = literal![0];
    pos += value.length;
    if (/^-?\d/.test(value)) {
      const number = Number(value);
      if (!Number.isFinite(number) || (Number.isInteger(number) && !Number.isSafeInteger(number))) fail();
    }
  }
  walk(0); ws();
  if (pos !== text.length) fail();
  try { return JSON.parse(text) as unknown; } catch { return fail(); }
}

export function validateJson(value: unknown, depth = 0): void {
  if (depth > 64) throw new Error('JSON nesting exceeds limit');
  if (typeof value === 'number' && (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value)))) throw new Error('JSON contains a non-finite number or unsafe integer');
  if (value === null || ['string', 'boolean', 'number'].includes(typeof value)) return;
  if (Array.isArray(value)) { value.forEach(v => validateJson(v, depth + 1)); return; }
  if (value && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) {
      if (['__proto__', 'prototype', 'constructor'].includes(key)) throw new Error('JSON contains a forbidden key');
      validateJson(child, depth + 1);
    }
    return;
  }
  throw new Error('Expected JSON-compatible value');
}

export function parseObject(value: unknown, label: string): Record<string, unknown> {
  const result = object(typeof value === 'string' ? parseJson(value, label) : value, label);
  validateJson(result);
  return result;
}

export function safeText(value: string, secrets: readonly string[] = []): string {
  let text = value;
  for (const secret of secrets) if (secret) {
    text = text.split(secret).join('[redacted]').split(encodeURIComponent(secret)).join('[redacted]');
  }
  return text
    .replace(/data:[^\s"'<>]+/gi, '[embedded media]')
    .replace(/\bBearer\s+[^\s"',;]+/gi, 'Bearer [redacted]')
    .replace(/https?:\/\/[^\s"'<>]+/gi, value => {
      try { const url = new URL(value); return `${url.origin}${url.pathname}`; } catch { return '[URL]'; }
    })
    .replace(/(?:[A-Za-z]:\\|\/(?:home|Users|tmp|var|private|mnt|opt|srv)\/)[^\s"'<>]+/g, '[internal path]');
}

/** Sanitize metadata only; binary bytes are kept exclusively in n8n binary helpers. */
export function sanitize(value: unknown, secrets: readonly string[] = [], depth = 0): Json {
  if (depth > 64) return '[nesting limit]';
  if (value === null || value === undefined) return null;
  if (typeof value === 'string') return safeText(value, secrets);
  if (typeof value === 'number') return Number.isFinite(value) ? (Number.isInteger(value) && !Number.isSafeInteger(value) ? String(value) : value) : null;
  if (typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.map(v => sanitize(v, secrets, depth + 1));
  if (typeof value !== 'object') return '[removed]';
  const source = value as Record<string, unknown>;
  const result: Record<string, Json> = {};
  for (const [key, child] of Object.entries(source)) {
    if (/^(?:__proto__|prototype|constructor|path|output_path|local_path|filesystem_path|base64|b64_json|api[_-]?key|authorization|headers|credentials?|password|secret|handoff|handoff_token|token)$/i.test(key)) continue;
    if (key === 'data' && source.kind === 'base64') { result.embedded = true; continue; }
    result[safeText(key, secrets)] = sanitize(child, secrets, depth + 1);
  }
  return result;
}

export function safeMessage(value: unknown, secrets: readonly string[] = []): string {
  // Only extract human-readable details, never stringify request/config/cause objects.
  function detail(input: unknown, depth = 0): string {
    if (depth > 8) return '';
    if (typeof input === 'string') return input;
    if (!input || typeof input !== 'object') return '';
    for (const key of ['message', 'detail', 'error']) {
      const result = detail((input as Record<string, unknown>)[key], depth + 1);
      if (result) return result;
    }
    return '';
  }
  return safeText(detail(value) || 'Werk request failed; check server connectivity and capabilities', secrets).slice(0, 1500);
}

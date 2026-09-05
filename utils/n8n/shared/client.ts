import type { ICredentialDataDecryptedObject, IExecuteFunctions, IHttpRequestOptions, ILoadOptionsFunctions } from 'n8n-workflow';
import { Readable } from 'node:stream';
import { finiteNumber, object, parseJson, safeMessage, sanitize, string, validateJson } from './validation';

export type HttpContext = IExecuteFunctions | ILoadOptionsFunctions;
export type Method = 'GET' | 'POST' | 'DELETE';
export type Query = Record<string, string | number | boolean>;
export type RawResponse = { statusCode: number; headers: Record<string, unknown>; body: unknown };
type RequestLimits = { timeoutMs?: number; ignoreExecutionCancel?: boolean };
export const JSON_LIMIT = 128 * 1024 * 1024;
export const BINARY_LIMIT = 512 * 1024 * 1024;
export const INPUT_LIMIT = 64 * 1024 * 1024;

export function normalizeBaseUrl(value: unknown): string {
  const raw = string(value, 'WERK base URL');
  let url: URL;
  try { url = new URL(raw); } catch { throw new Error('WERK base URL must be a valid HTTP(S) URL'); }
  if (!['http:', 'https:'].includes(url.protocol) || !url.hostname || url.username || url.password || url.search || url.hash) {
    throw new Error('WERK base URL must use HTTP(S), without embedded credentials, query or fragment');
  }
  return url.toString().replace(/\/+$/, '');
}

export function sameOrigin(left: string, right: string): boolean {
  return new URL(left).origin === new URL(right).origin;
}

export function credentialSettings(credentials: ICredentialDataDecryptedObject): { baseUrl: string; apiKey: string; authenticated: boolean; verifyTls: boolean } {
  const baseUrl = normalizeBaseUrl(credentials.baseUrl);
  if (credentials.authMode !== 'apiKey' && credentials.authMode !== 'none') throw new Error('Select API Key or explicitly select unauthenticated operation');
  const authenticated = credentials.authMode === 'apiKey';
  const apiKey = typeof credentials.apiKey === 'string' ? credentials.apiKey.trim() : '';
  if (authenticated && !apiKey) throw new Error('WERK API Key is required for authenticated operation');
  if (/[\r\n]/.test(apiKey)) throw new Error('WERK API Key contains invalid characters');
  return { baseUrl, apiKey, authenticated, verifyTls: credentials.verifyTls !== false };
}

/** Invoked by n8n's credential helper, including the real credential test. */
export async function authenticate(credentials: ICredentialDataDecryptedObject, options: IHttpRequestOptions): Promise<IHttpRequestOptions> {
  const settings = credentialSettings(credentials);
  const target = new URL(options.url, options.baseURL || settings.baseUrl);
  if (!sameOrigin(target.href, settings.baseUrl) || target.username || target.password) throw new Error('WERK authentication may only be sent to the exact configured origin');
  const headers = { ...options.headers };
  // There is no fallback to environment keys or item fields.
  if (settings.authenticated) headers.Authorization = `Bearer ${settings.apiKey}`;
  return { ...options, headers, disableFollowRedirect: true, maxRedirects: 0, sendCredentialsOnCrossOriginRedirect: false, skipSslCertificateValidation: !settings.verifyTls };
}

export async function readBounded(body: unknown, headers: Record<string, unknown>, limit: number, timeoutMs: number, signal?: AbortSignal): Promise<Buffer> {
  const declared = Number(headers['content-length']);
  if (Number.isFinite(declared) && declared > limit) {
    if (body instanceof Readable) body.destroy();
    throw new Error(`WERK response exceeds ${limit} bytes`);
  }
  if (body instanceof Readable) {
    const chunks: Buffer[] = [];
    let size = 0;
    const abort = () => body.destroy(new Error('WERK HTTP request cancelled'));
    const timer = setTimeout(() => body.destroy(new Error('WERK HTTP response timed out')), timeoutMs);
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
    try {
      for await (const chunk of body) {
        const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array);
        size += bytes.length;
        if (size > limit) { body.destroy(); throw new Error(`WERK response exceeds ${limit} bytes`); }
        chunks.push(bytes);
      }
      return Buffer.concat(chunks, size);
    } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort); }
  }
  const bytes = Buffer.isBuffer(body) ? body : typeof body === 'string' ? Buffer.from(body) : Buffer.from(JSON.stringify(body));
  if (bytes.length > limit) throw new Error(`WERK response exceeds ${limit} bytes`);
  return bytes;
}

export class WerkClient {
  private readonly settings;
  private readonly credentials: ICredentialDataDecryptedObject;
  private readonly context: HttpContext;
  private readonly timeoutMs: number;
  private readonly signal?: AbortSignal;

  private constructor(context: HttpContext, credentials: ICredentialDataDecryptedObject, timeoutSeconds: number) {
    this.settings = credentialSettings(credentials);
    this.credentials = credentials;
    this.context = context;
    this.timeoutMs = finiteNumber(timeoutSeconds, 'HTTP timeout', 1) * 1000;
    if (timeoutSeconds > 3600) throw new Error('HTTP timeout cannot exceed 3600 seconds');
    this.signal = 'getExecutionCancelSignal' in context ? context.getExecutionCancelSignal() : undefined;
  }

  static async create(context: HttpContext, index = 0): Promise<WerkClient> {
    const credentials = await context.getCredentials('werkApi', index);
    const timeout = 'getInputData' in context ? Number(context.getNodeParameter('httpTimeoutSeconds', index, 120)) : 30;
    return new WerkClient(context, credentials, timeout);
  }

  safeMessage(value: unknown): string { return safeMessage(value, [this.settings.apiKey]); }
  redact(value: unknown): unknown { return sanitize(value, [this.settings.apiKey]); }

  private url(path: string): string {
    if (!/^\/(?:v1|werk\/v1)\//.test(path) || /[?#\\]/.test(path)) throw new Error('Invalid WERK API path');
    const target = `${this.settings.baseUrl}${path}`;
    if (!sameOrigin(target, this.settings.baseUrl)) throw new Error('WERK API path escaped its origin');
    return target;
  }

  private async request(options: IHttpRequestOptions, authenticated: boolean, limit: number, limits: RequestLimits = {}): Promise<{statusCode: number; headers: Record<string, unknown>; bytes: Buffer}> {
    try {
      const timeoutMs = Math.max(1, Math.ceil(Math.min(this.timeoutMs, limits.timeoutMs ?? this.timeoutMs)));
      const deadline = AbortSignal.timeout(timeoutMs);
      const signal = this.signal && !limits.ignoreExecutionCancel ? AbortSignal.any([this.signal, deadline]) : deadline;
      const request: IHttpRequestOptions = {
        ...options, encoding: 'stream', returnFullResponse: true, ignoreHttpStatusErrors: true,
        disableFollowRedirect: true, maxRedirects: 0, sendCredentialsOnCrossOriginRedirect: false,
        timeout: timeoutMs, abortSignal: signal,
        skipSslCertificateValidation: !this.settings.verifyTls,
      };
      // n8n's auth helper replaces abortSignal with its call context's signal.
      // Keep the original context intact while supplying this request's deadline
      // (or the independent, bounded best-effort cancellation signal).
      const authContext = Object.create(this.context) as HttpContext;
      Object.defineProperty(authContext, 'getExecutionCancelSignal', { value: () => signal });
      const credentialRef = this.context.getNode().credentials?.werkApi;
      const response: unknown = authenticated
        ? await this.context.helpers.httpRequestWithAuthentication.call(authContext, 'werkApi', request, {
          credentialsDecrypted: { id: credentialRef?.id ?? 'werk', name: credentialRef?.name ?? 'WERK', type: 'werkApi', data: this.credentials },
        })
        : await this.context.helpers.httpRequest(request);
      const full = object(response, 'WERK HTTP response');
      const headers = object(full.headers, 'WERK response headers');
      const statusCode = Number(full.statusCode);
      if (!Number.isInteger(statusCode) || statusCode < 100 || statusCode > 599) throw new Error('Invalid WERK HTTP status');
      if (statusCode >= 300 && statusCode < 400) {
        if (full.body instanceof Readable) full.body.destroy();
        throw new Error('WERK redirect rejected to protect credentials (including same-origin redirects)');
      }
      const bytes = await readBounded(full.body, headers, statusCode >= 400 ? 65536 : limit, timeoutMs, signal);
      return { statusCode, headers, bytes };
    } catch (error) {
      throw new Error(this.safeMessage(error));
    }
  }

  async raw(method: Method, path: string, body?: Record<string, unknown>, query?: Query, protocol = false, limits: RequestLimits = {}): Promise<RawResponse> {
    if (!(protocol ? path.startsWith('/werk/v1/') : path.startsWith('/v1/'))) throw new Error('WERK inference and strict runtime protocol transports must remain separate');
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (protocol) headers['X-Werk-Protocol-Version'] = '1.0';
    let wire: string | undefined;
    if (body !== undefined) {
      validateJson(body);
      wire = JSON.stringify(body);
      if (Buffer.byteLength(wire) > (protocol ? 1024 * 1024 : JSON_LIMIT)) throw new Error('WERK request exceeds size limit');
      headers['Content-Type'] = 'application/json';
    }
    const result = await this.request({ method, url: this.url(path), headers, body: wire, qs: query }, this.settings.authenticated, protocol ? 8 * 1024 * 1024 : JSON_LIMIT, limits);
    let decoded: string;
    try { decoded = new TextDecoder('utf-8', { fatal: true }).decode(result.bytes); }
    catch { throw new Error('WERK returned invalid UTF-8'); }
    if (protocol) return { statusCode: result.statusCode, headers: result.headers, body: decoded };
    let payload: unknown;
    try { payload = parseJson(decoded, 'WERK response'); }
    catch { throw new Error(`WERK returned invalid JSON or unsafe numbers (HTTP ${result.statusCode})`); }
    return { statusCode: result.statusCode, headers: result.headers, body: payload };
  }

  async api(method: Method, path: string, body?: Record<string, unknown>, query?: Query, limits: RequestLimits = {}): Promise<Record<string, unknown>> {
    const response = await this.raw(method, path, body, query, false, limits);
    if (response.statusCode < 200 || response.statusCode >= 300) throw new Error(`WERK HTTP ${response.statusCode}: ${this.safeMessage(response.body)}`);
    return object(response.body, 'WERK API response');
  }

  async cancelJob(id: string): Promise<Record<string, unknown>> {
    return this.api('DELETE', `/v1/jobs/${encodeURIComponent(string(id, 'Job ID'))}`, undefined, undefined, { timeoutMs: 10000, ignoreExecutionCancel: true });
  }

  async download(value: string): Promise<{ data: Buffer; mimeType: string }> {
    // Werk emits origin-relative /v1 paths; preserve configured reverse-proxy prefix.
    const target = value.startsWith('/v1/') ? this.url(value) : new URL(value, `${this.settings.baseUrl}/`).href;
    const url = new URL(target);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('WERK media URL must use HTTP(S) without embedded credentials');
    const response = await this.request({ method: 'GET', url: target, headers: { Accept: '*/*' } }, this.settings.authenticated && sameOrigin(target, this.settings.baseUrl), BINARY_LIMIT);
    if (response.statusCode < 200 || response.statusCode >= 300) throw new Error(`WERK output download failed (HTTP ${response.statusCode})`);
    return { data: response.bytes, mimeType: String(response.headers['content-type'] ?? 'application/octet-stream').split(';')[0].trim().toLowerCase() };
  }
}

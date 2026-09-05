import type { IDataObject, IExecuteFunctions, INodeExecutionData, INodeProperties } from 'n8n-workflow';
import { binaryItem } from './binary';
import type { WerkClient } from './client';
import { jsonItem } from './common';
import { audioAnalysisTasks, audioGenerationTasks, audioProcessTasks } from './mediaRequests';
import { finiteNumber, object, parseJson, sanitize, string } from './validation';

export const JOB_STATUSES = ['queued', 'loading', 'running', 'encoding', 'completed', 'failed', 'cancelled'] as const;
export const waitProperties: INodeProperties[] = [
  { displayName: 'Job Wait Time (Seconds)', name: 'waitSeconds', type: 'number', default: 900, typeOptions: { minValue: 1, maxValue: 86400 }, description: 'Maximum total time spent waiting for a job; separate from the HTTP and inference timeouts' },
  { displayName: 'Poll Interval (Seconds)', name: 'pollSeconds', type: 'number', default: 1, typeOptions: { minValue: 0.1, maxValue: 300 } },
];
export const jobProperties: INodeProperties[] = [
  { displayName: 'Job Handling', name: 'waitMode', type: 'options', default: 'wait', options: [
    { name: 'Submit and Wait', value: 'wait' }, { name: 'Submit Only', value: 'submitOnly' },
  ], description: 'Submit Only returns immediately. Use WERK Jobs to retrieve the same job later.' },
  ...waitProperties.map(property => ({ ...property, displayOptions: { show: { waitMode: ['wait'] } } })),
];

export function jobRecord(value: unknown, expectedId?: string): Record<string, unknown> {
  const record = object(value, 'WERK job');
  const id = string(record.id, 'WERK job ID');
  if (expectedId && id !== expectedId) throw new Error(`WERK job ID changed while polling job ${expectedId}`);
  if (!(JOB_STATUSES as readonly unknown[]).includes(record.status)) throw new Error(`WERK job ${id} returned an unknown status`);
  if (record.status === 'completed') {
    const result = object(record.result, `WERK completed job ${id} result`);
    if (!Array.isArray(result.outputs) || !result.outputs.length) throw new Error(`WERK completed job ${id} has no outputs`);
    for (const entry of result.outputs) {
      const output = object(entry, 'WERK output');
      string(output.id, 'WERK output ID'); string(output.mime_type, 'WERK output MIME type');
    }
  }
  return record;
}

export function jobMetadata(record: Record<string, unknown>, task?: string): Record<string, unknown> {
  const result = record.result && typeof record.result === 'object' ? object(record.result, 'result') : {};
  return {
    jobId: record.id, status: record.status, model: result.model ?? null, task: result.task ?? task ?? null,
    outputIds: Array.isArray(result.outputs) ? result.outputs.map(output => object(output, 'output').id) : [],
    werk: sanitize(record),
  };
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = () => { signal?.removeEventListener('abort', abort); resolve(); };
    const timer = setTimeout(finish, ms);
    const abort = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort); reject(new Error('WERK job wait cancelled')); };
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
  });
}

export async function waitForJob(client: WerkClient, initial: Record<string, unknown>, options: { waitSeconds: number; pollSeconds: number; cancelOnAbort: boolean; signal?: AbortSignal }): Promise<Record<string, unknown>> {
  const id = string(initial.id, 'WERK job ID');
  const waitSeconds = finiteNumber(options.waitSeconds, 'Job wait time', 0.001);
  const pollSeconds = finiteNumber(options.pollSeconds, 'Poll interval', 0.001);
  if (waitSeconds > 86400 || pollSeconds > 300) throw new Error('WERK job wait/poll interval exceeds limit');
  const deadline = performance.now() + waitSeconds * 1000;
  let active = true;
  let record = initial;
  try {
    while (true) {
      record = jobRecord(record, id);
      if (record.status === 'completed') { active = false; return record; }
      if (record.status === 'failed' || record.status === 'cancelled') {
        active = false;
        throw new Error(`WERK job ${id} ${record.status}: ${record.status === 'failed' ? client.safeMessage(record.error) : 'cooperative cancellation reported'}`);
      }
      if (options.signal?.aborted) throw new Error(`WERK job ${id} wait cancelled`);
      const remaining = deadline - performance.now();
      if (remaining <= 0) throw new Error(`WERK job ${id} wait timed out after ${waitSeconds} seconds; use WERK Jobs Get to inspect it`);
      await delay(Math.min(remaining, pollSeconds * 1000), options.signal);
      if (performance.now() >= deadline) throw new Error(`WERK job ${id} wait timed out after ${waitSeconds} seconds; use WERK Jobs Get to inspect it`);
      // Bound an in-flight GET by the remaining job deadline as well as the HTTP timeout.
      record = await client.api('GET', `/v1/jobs/${encodeURIComponent(id)}`, undefined, undefined, { timeoutMs: deadline - performance.now() });
    }
  } catch (error) {
    if (active && options.cancelOnAbort) {
      // A separate client request cancellation signal is omitted for cleanup by cancelJob.
      try { await client.cancelJob(id); } catch { /* Best effort; preserve original error and job ID. */ }
    }
    throw new Error(`${client.safeMessage(error)} [jobId: ${id}]`);
  }
}

function outputKind(task: string): 'audio' | 'video' | 'structured' | undefined {
  const normalized = task.replace(/_/g, '-');
  if (['video-generation', 'image-to-video'].includes(normalized)) return 'video';
  if (([...audioGenerationTasks, ...audioProcessTasks] as readonly string[]).includes(normalized)) return 'audio';
  if ((audioAnalysisTasks as readonly string[]).includes(normalized)) return 'structured';
  return undefined;
}

export async function downloadedOutput(context: IExecuteFunctions, client: WerkClient, index: number, outputId: string, metadata: Record<string, unknown> = {}, expectedMime?: string): Promise<INodeExecutionData> {
  const { data, mimeType: downloadedMime } = await client.download(`/v1/outputs/${encodeURIComponent(string(outputId, 'Output ID'))}`);
  const declaredMime = downloadedMime === 'application/octet-stream' && expectedMime ? expectedMime : downloadedMime;
  const mimeType = declaredMime === 'application/ogg' ? 'audio/ogg' : declaredMime;
  if (expectedMime === 'application/ogg') expectedMime = 'audio/ogg';
  if (expectedMime && mimeType !== expectedMime && !(mimeType.startsWith('audio/') && expectedMime.startsWith('audio/'))) throw new Error('WERK output download MIME type differs from job metadata');
  const json = { ...metadata, outputId, mimeType };
  const kind = outputKind(String(metadata.task ?? ''));
  const structured = mimeType.startsWith('text/') || ['application/json', 'application/x-ndjson', 'application/ndjson'].includes(mimeType);
  if ((kind === 'audio' || kind === 'video') && !mimeType.startsWith(`${kind}/`) && !(kind === 'video' && mimeType === 'image/gif')) throw new Error(`WERK ${kind} job returned a non-${kind} output`);
  if (kind === 'structured' && !structured) throw new Error('WERK audio analysis returned a non-text/JSON output');
  if (structured) {
    if (data.length > 16 * 1024 * 1024) throw new Error('WERK structured output exceeds 16 MiB');
    const text = new TextDecoder('utf-8', { fatal: true }).decode(data);
    let result: unknown = text;
    if (mimeType === 'application/json') result = parseJson(text, 'WERK structured output');
    else if (mimeType.endsWith('ndjson')) result = text.split(/\r?\n/).filter(line => line.trim()).map(line => parseJson(line, 'WERK NDJSON output'));
    const outputText = typeof result === 'object' && result !== null && !Array.isArray(result) && typeof (result as Record<string, unknown>).text === 'string' ? (result as Record<string, unknown>).text : text;
    return { json: client.redact({ ...json, text: outputText, result }) as IDataObject, pairedItem: { item: index } };
  }
  return binaryItem(context, index, data, mimeType, json);
}

export async function jobOutputs(context: IExecuteFunctions, client: WerkClient, index: number, record: Record<string, unknown>, task?: string): Promise<INodeExecutionData[]> {
  jobRecord(record);
  const result = object(record.result, 'completed result');
  const metadata = jobMetadata(record, task);
  const output: INodeExecutionData[] = [];
  for (const entry of result.outputs as unknown[]) {
    const item = object(entry, 'output');
    output.push(await downloadedOutput(context, client, index, string(item.id, 'Output ID'), metadata, string(item.mime_type, 'Output MIME type').split(';')[0].toLowerCase()));
  }
  return output;
}

export async function submitJob(context: IExecuteFunctions, client: WerkClient, index: number, path: string, request: Record<string, unknown>, task: string): Promise<INodeExecutionData[]> {
  const mode = String(context.getNodeParameter('waitMode', index, 'wait'));
  if (mode !== 'wait' && mode !== 'submitOnly') throw new Error('Unknown WERK job handling mode');
  // Validate wait settings before performing the only POST.
  const waitSeconds = finiteNumber(context.getNodeParameter('waitSeconds', index, 900), 'Job wait time', 1);
  const pollSeconds = finiteNumber(context.getNodeParameter('pollSeconds', index, 1), 'Poll interval', 0.1);
  if (waitSeconds > 86400 || pollSeconds > 300) throw new Error('WERK job wait settings exceed limits');
  const initial = await client.api('POST', path, request);
  if (mode === 'submitOnly') return [jsonItem(index, { ...jobMetadata(jobRecord(initial), task), model: request.model })];
  const record = await waitForJob(client, initial, { waitSeconds, pollSeconds, cancelOnAbort: true, signal: context.getExecutionCancelSignal?.() });
  return jobOutputs(context, client, index, record, task);
}

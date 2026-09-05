import type { IDataObject, IExecuteFunctions, INodeExecutionData } from 'n8n-workflow';
import { INPUT_LIMIT, BINARY_LIMIT } from './client';
import { string } from './validation';

export async function readBinary(context: IExecuteFunctions, index: number, property: string, kind: 'image' | 'audio' | 'video'): Promise<{data: Buffer; mimeType: string}> {
  string(property, 'Binary property');
  const reference = context.helpers.assertBinaryData(index, property);
  const declaredMime = reference.mimeType.toLowerCase().split(';')[0];
  const mimeType = kind === 'audio' && declaredMime === 'application/ogg' ? 'audio/ogg' : declaredMime;
  if (!mimeType.startsWith(`${kind}/`)) throw new Error(`Input binary '${property}' must have an ${kind} MIME type`);
  const data = await context.helpers.getBinaryDataBuffer(index, property);
  if (data.length === 0 || data.length > INPUT_LIMIT) throw new Error(`Input binary must contain 1 to ${INPUT_LIMIT} bytes`);
  return { data, mimeType };
}

const extensions: Record<string, string> = {
  'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/gif': 'gif', 'image/avif': 'avif',
  'audio/wav': 'wav', 'audio/x-wav': 'wav', 'audio/mpeg': 'mp3', 'audio/flac': 'flac', 'audio/ogg': 'ogg',
  'audio/mp4': 'm4a', 'audio/aac': 'aac', 'audio/webm': 'webm', 'video/mp4': 'mp4', 'video/webm': 'webm',
  'video/quicktime': 'mov', 'application/json': 'json', 'application/x-ndjson': 'ndjson', 'text/plain': 'txt',
};

export async function binaryItem(context: IExecuteFunctions, index: number, data: Buffer, mimeType: string, json: Record<string, unknown>, property = 'data'): Promise<INodeExecutionData> {
  if (!data.length || data.length > BINARY_LIMIT) throw new Error('WERK binary output is empty or exceeds size limit');
  const extension = extensions[mimeType] ?? (mimeType.split('/')[1]?.replace(/[^a-z0-9]/gi, '').slice(0, 12) || 'bin');
  const id = String(json.outputId ?? 'output').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 80);
  const binary = await context.helpers.prepareBinaryData(data, `werk-${id}.${extension}`, mimeType);
  return { json: json as IDataObject, binary: { [property]: binary }, pairedItem: { item: index } };
}

import type { WerkClient } from './client';
import { object, sanitize, string } from './validation';

export const taskName = (task: string): string => task.trim().toLowerCase().replace(/_/g, '-');
function tasks(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter((x): x is string => typeof x === 'string').map(taskName))];
}

export function mergeModels(models: Record<string, unknown>, capabilities: Record<string, unknown>, filter = ''): Record<string, unknown> {
  if (!Array.isArray(models.data) || !Array.isArray(capabilities.models)) throw new Error('WERK discovery requires models.data and capabilities.models arrays');
  const byId = new Map<string, Record<string, unknown>>();
  for (const entry of capabilities.models) {
    const cap = object(entry, 'model capability');
    const id = string(cap.id, 'capability model ID');
    if (byId.has(id)) throw new Error('WERK capabilities contain a duplicate exact model ID');
    byId.set(id, cap);
  }
  const installed: string[] = [];
  const declared: string[] = [];
  const available: string[] = [];
  const matched: Record<string, unknown>[] = [];
  const task = taskName(filter);
  for (const entry of models.data) {
    const model = object(entry, 'installed model');
    const id = string(model.id, 'model ID');
    if (installed.includes(id)) throw new Error('WERK model list contains a duplicate exact model ID');
    installed.push(id);
    const cap = byId.get(id) ?? model;
    const declaredTasks = tasks(cap.tasks ?? model.tasks);
    const availableTasks = tasks(cap.available_tasks ?? model.available_tasks);
    if (!task || declaredTasks.includes(task)) declared.push(id);
    if (!task ? availableTasks.length > 0 : availableTasks.includes(task)) available.push(id);
    if (!task || declaredTasks.includes(task) || availableTasks.includes(task)) matched.push({
      ...model, ...cap, id, installed: true, tasks: cap.tasks ?? model.tasks ?? [], available_tasks: cap.available_tasks ?? model.available_tasks ?? [],
      task_statuses: object(cap.task_statuses ?? model.task_statuses ?? {}, 'task statuses'),
    });
  }
  return { task: task || null, installed, declared, available, models: sanitize(matched) };
}

export async function discoverModels(client: WerkClient, task = ''): Promise<Record<string, unknown>> {
  // The two reads are independent and do not submit inference or mutate runtime state.
  const [models, capabilities] = await Promise.all([client.api('GET', '/v1/models'), client.api('GET', '/v1/capabilities')]);
  return mergeModels(models, capabilities, task);
}

export async function requireModelTask(client: WerkClient, model: string, task: string): Promise<void> {
  const info = await discoverModels(client, task);
  if (!(info.installed as string[]).includes(model)) throw new Error(`WERK model '${model}' is not installed (exact model ID required)`);
  if ((info.available as string[]).includes(model)) return;
  const entry = (info.models as Record<string, unknown>[]).find(entry => entry.id === model);
  if (!entry) throw new Error(`WERK model '${model}' does not declare task ${taskName(task)}`);
  const statuses = object(entry.task_statuses, 'task statuses');
  const status = Object.entries(statuses).find(([key]) => taskName(key) === taskName(task))?.[1];
  const details = status && typeof status === 'object' ? object(status, 'task status') : {};
  const statusName = typeof details.status === 'string' ? details.status : 'unavailable';
  const hint = statusName.replace(/_/g, '-') === 'not-implemented' ? 'No registered Werk adapter; installing a package alone does not add task support.'
    : statusName === 'installable' && typeof details.install_command === 'string' ? `Server-provided installation action: ${details.install_command}` : '';
  throw new Error(`WERK model '${model}' is not currently probe-eligible for ${taskName(task)} (${statusName}): ${client.safeMessage(status ?? 'no available task reported')}. ${client.safeMessage(hint || 'Use Discovery to inspect task_statuses.')} This node installs no adapter or model.`);
}

/** Chat has its own GenerationBackend; media-adapter readiness cannot gate it.
 * See service.rs capabilities_with_optional_generation_backend and chat.rs.
 */
export async function requireChatModel(client: WerkClient, model: string): Promise<void> {
  const info = await discoverModels(client);
  const entry = (info.models as Record<string, unknown>[]).find(entry => entry.id === model);
  if (!entry) throw new Error(`WERK model '${model}' is not installed (exact model ID required)`);
  const declared = tasks(entry.tasks);
  if (declared.length && !declared.includes('text-generation') && !declared.includes('image-understanding')) {
    throw new Error(`WERK model '${model}' does not declare a chat-compatible task`);
  }
}

export async function parameterSchema(client: WerkClient, task: string, model = '', backend = ''): Promise<Record<string, unknown>> {
  const query: Record<string, string> = { task: taskName(string(task, 'Task')) };
  if (model.trim()) query.model = model.trim();
  if (backend.trim()) query.backend = backend.trim();
  const response = await client.api('GET', '/v1/parameters', undefined, query);
  if (!Array.isArray(response.parameters)) throw new Error('WERK parameter schema must contain parameters array');
  for (const entry of response.parameters) {
    const descriptor = object(entry, 'parameter descriptor');
    string(descriptor.path, 'parameter path');
    string(descriptor.value_type, 'parameter value type');
  }
  return response;
}

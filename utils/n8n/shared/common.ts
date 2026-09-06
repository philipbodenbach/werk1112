import { NodeOperationError } from 'n8n-workflow';
import type { IDataObject, IExecuteFunctions, ILoadOptionsFunctions, INodeExecutionData, INodeProperties, INodeTypeDescription } from 'n8n-workflow';
import { WerkClient } from './client';
import { discoverModels } from './discovery';
import { safeMessage, sanitize } from './validation';
import { werkApiTest } from './credential-test';
import { WerkProtocolError } from './protocol';

export const transportProperties: INodeProperties[] = [{
  displayName: 'HTTP Timeout (Seconds)', name: 'httpTimeoutSeconds', type: 'number', default: 120,
  typeOptions: { minValue: 1, maxValue: 3600 }, description: 'Finite timeout for each HTTP request. Separate from inference timeout and total job wait time.',
}];

export function nodeDescription(name: string, title: string, properties: INodeProperties[]): INodeTypeDescription {
  return {
    displayName: title.includes('(Beta)') ? title : `WERK ${title} (Beta)`, name, version: 1,
    description: 'Beta Werk1112 inference server integration', group: ['transform'],
    icon: 'file:werk.png', defaults: { name: title.includes('(Beta)') ? title : `WERK ${title} (Beta)` },
    inputs: ['main'], outputs: ['main'],
    credentials: [{ name: 'werkApi', required: true, testedBy: 'werkApiTest' }],
    properties,
  };
}

export function modelProperty(task = ''): INodeProperties {
  return {
    displayName: 'Model', name: 'model', type: 'resourceLocator', default: { mode: 'id', value: '' }, required: true,
    modes: [
      { displayName: 'By ID', name: 'id', type: 'string' },
      { displayName: 'From List', name: 'list', type: 'list', typeOptions: { searchListMethod: 'searchModels', searchable: true } },
    ],
    description: 'Exact installed model ID; accepts expressions. The list filters models by the selected task.',
    // The free text ID remains editable even if the server cannot be reached.
    hint: task ? `Task: ${task}. Discovery lists declared and currently available models separately.` : 'Use the exact model ID returned by Werk.',
  };
}

export const commonMethods = {
  credentialTest: { werkApiTest },
  listSearch: {
    async searchModels(this: ILoadOptionsFunctions, filter?: string) {
      const models = await commonMethods.loadOptions.getModels.call(this);
      return { results: models.filter(model => !filter || model.value.toLowerCase().includes(filter.toLowerCase())) };
    },
  },
  loadOptions: {
    async getModels(this: ILoadOptionsFunctions) {
      try {
        const client = await WerkClient.create(this);
        let task = String(this.getCurrentNodeParameter('task') ?? '');
        if (!task) {
          const type = this.getNode().type.split('.').pop();
          task = ({ werkText: 'text-generation', werkVision: 'image-understanding', werkImage: 'image-generation', werkVideo: this.getCurrentNodeParameter('operation') === 'imageToVideo' ? 'image-to-video' : 'video-generation' } as Record<string, string>)[type ?? ''] ?? '';
        }
        const isText = task.replace(/_/g, '-') === 'text-generation';
        const info = await discoverModels(client, isText ? '' : task);
        const models = (info.models as Record<string, unknown>[]).filter(model => {
          if (!isText) return true;
          const tasks = Array.isArray(model.tasks) ? model.tasks.map(task => String(task).replace(/_/g, '-')) : [];
          return !tasks.length || tasks.includes('text-generation') || tasks.includes('image-understanding');
        });
        return models.map(model => ({ name: `${model.id}${isText ? ' (chat readiness checked at execution)' : (info.available as string[]).includes(String(model.id)) ? '' : ' (unavailable)'}`, value: String(model.id) }));
      } catch { return []; }
    },
  },
};

export async function executeItems(context: IExecuteFunctions, handler: (client: WerkClient, index: number) => Promise<INodeExecutionData[]>): Promise<INodeExecutionData[][]> {
  const output: INodeExecutionData[] = [];
  // Deliberately sequential: each item can launch a heavy inference job.
  for (let index = 0; index < context.getInputData().length; index++) {
    let client: WerkClient | undefined;
    try {
      client = await WerkClient.create(context, index);
      if (context.getExecutionCancelSignal?.()?.aborted) throw new Error('WERK execution cancelled');
      for (const item of await handler(client, index)) output.push({ ...item, json: client.redact(item.json) as IDataObject, pairedItem: { item: index } });
    } catch (error) {
      const message = client ? client.safeMessage(error) : safeMessage(error);
      const protocolError = error instanceof WerkProtocolError ? {
        code: error.code, requestId: error.requestId ?? null, retryable: error.retryable, statusCode: error.statusCode ?? null,
      } : undefined;
      if (context.continueOnFail()) output.push({ json: {
        error: message, itemIndex: index,
        ...(protocolError ? { werk: { error: client ? client.redact(protocolError) : sanitize(protocolError) } } : {}),
      }, pairedItem: { item: index } });
      else throw new NodeOperationError(context.getNode(), message, { itemIndex: index, description: 'See Discovery for supported tasks, readiness and runtime details.' });
    }
  }
  return [output];
}

export function jsonItem(index: number, value: unknown): INodeExecutionData {
  return { json: sanitize(value) as IDataObject, pairedItem: { item: index } };
}

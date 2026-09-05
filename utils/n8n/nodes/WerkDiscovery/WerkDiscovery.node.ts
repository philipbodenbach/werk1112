import type { IExecuteFunctions, INodeExecutionData, INodeType } from 'n8n-workflow';
import { commonMethods, executeItems, jsonItem, nodeDescription, transportProperties } from '../../shared/common';
import { discoverModels, parameterSchema } from '../../shared/discovery';
import { string } from '../../shared/validation';

export class WerkDiscovery implements INodeType {
  description = nodeDescription('werkDiscovery', 'WERK Discovery (Beta)', [
    { displayName: 'Operation', name: 'operation', type: 'options', noDataExpression: true, default: 'serverInfo', options: [
      { name: 'Server Information', value: 'serverInfo', action: 'Get server information' },
      { name: 'List Models', value: 'models', action: 'List installed and available models' },
      { name: 'Model Information', value: 'model', action: 'Get model information' },
      { name: 'Capabilities', value: 'capabilities', action: 'Get model capabilities' },
      { name: 'Parameter Schema', value: 'parameters', action: 'Get task parameter schema' },
    ] },
    { displayName: 'Task', name: 'task', type: 'string', default: '', displayOptions: { show: { operation: ['models', 'parameters'] } }, description: 'Canonical task, for example image-generation or speech-to-text. Required for parameter schemas; empty lists all models.' },
    { displayName: 'Model ID', name: 'model', type: 'string', default: '', displayOptions: { show: { operation: ['model', 'parameters'] } }, description: 'Exact installed ID; optional for a task-only parameter schema' },
    { displayName: 'Backend', name: 'backend', type: 'string', default: '', displayOptions: { show: { operation: ['parameters'] } } },
    ...transportProperties,
  ]);
  methods = commonMethods;
  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    return executeItems(this, async (client, index) => {
      const operation = String(this.getNodeParameter('operation', index));
      let result: unknown;
      switch (operation) {
        case 'serverInfo': {
          const [models, capabilities] = await Promise.all([discoverModels(client), client.api('GET', '/v1/capabilities')]);
          result = { models, capabilities }; break;
        }
        case 'models': result = await discoverModels(client, String(this.getNodeParameter('task', index, ''))); break;
        case 'model': result = await client.api('GET', `/v1/models/${encodeURIComponent(string(this.getNodeParameter('model', index), 'Model ID'))}`); break;
        case 'capabilities': result = await client.api('GET', '/v1/capabilities'); break;
        case 'parameters': result = await parameterSchema(client, String(this.getNodeParameter('task', index, '')), String(this.getNodeParameter('model', index, '')), String(this.getNodeParameter('backend', index, ''))); break;
        default: throw new Error('Unknown WERK Discovery operation');
      }
      return [jsonItem(index, result)];
    });
  }
}

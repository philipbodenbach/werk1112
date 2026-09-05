import type { IExecuteFunctions, INodeExecutionData, INodeType } from 'n8n-workflow';
import { commonMethods, executeItems, modelProperty, nodeDescription, transportProperties } from '../../shared/common';
import { requireChatModel } from '../../shared/discovery';
import { chatOutputs } from '../../shared/mediaOutputs';
import { chatOptionsProperty } from '../../shared/mediaProperties';
import { buildTextRequest } from '../../shared/mediaRequests';
import { choice, optionsProperty, stringProperty } from '../../shared/parameters';

export class WerkText implements INodeType {
	description = nodeDescription('werkText', 'WERK Text (Beta)', [
		optionsProperty('operation', 'Operation', ['complete']), modelProperty('text-generation'),
		{ displayName: 'Messages', name: 'messages', type: 'fixedCollection', default: { message: [{ role: 'user', content: '' }] }, typeOptions: { multipleValues: true, sortable: true }, options: [{ name: 'message', displayName: 'Message', values: [
			optionsProperty('role', 'Role', ['system', 'user', 'assistant', 'tool'], 'user'),
			{ ...stringProperty('content', 'Content'), typeOptions: { rows: 4 } }, stringProperty('name', 'Name (Optional)'), stringProperty('toolCallId', 'Tool Call ID (Tool Result)'),
			{ name: 'toolCalls', displayName: 'Tool Calls (Assistant, JSON)', type: 'json', default: '[]', description: 'Previously returned structured tool calls; this node does not execute functions' },
		] }], description: 'Sent in this order. This is a normal workflow node, not an AI Agent chat-model subnode.' },
		chatOptionsProperty(false), ...transportProperties,
	]);
	methods = commonMethods;
	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		return executeItems(this, async (client, index) => {
			choice(this.getNodeParameter('operation', index), 'Operation', ['complete']);
			const request = buildTextRequest(this.getNodeParameter('model', index, undefined, { extractValue: true }), this.getNodeParameter('messages', index), this.getNodeParameter('options', index, {}));
			await requireChatModel(client, request.model as string);
			return chatOutputs(index, await client.api('POST', '/v1/chat/completions', request), request, 'text-generation');
		});
	}
}

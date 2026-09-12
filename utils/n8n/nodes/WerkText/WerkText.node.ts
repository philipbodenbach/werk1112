import type { IExecuteFunctions, INodeExecutionData, INodeType } from 'n8n-workflow';
import { commonMethods, executeItems, modelProperty, nodeDescription, transportProperties } from '../../shared/common';
import { requireChatModel } from '../../shared/discovery';
import { chatOutputs } from '../../shared/mediaOutputs';
import { chatOptionsProperty } from '../../shared/mediaProperties';
import { buildTextRequest } from '../../shared/mediaRequests';
import { choice, optionsProperty, record, stringProperty } from '../../shared/parameters';
import { WerkProtocolClient } from '../../shared/protocol';

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
			if (request.werk !== undefined) {
				const requested = Object.keys(record(record(request.werk, 'Werk chat options').omlx, 'oMLX chat options'));
				const capabilities = await new WerkProtocolClient(client).capabilities();
				const capability = capabilities.find((entry) => entry.id === 'api.chat.omlx_options');
				if (capability?.status !== 'supported' || requested.some((option) => !capability.operations.includes(option))) {
					throw new Error('oMLX request options are unsupported by this Werk server; update and restart Werk before using these options');
				}
			}
			return chatOutputs(index, await client.api('POST', '/v1/chat/completions', request), request, 'text-generation');
		});
	}
}

import type { IExecuteFunctions, INodeExecutionData, INodeType } from 'n8n-workflow';
import { readBinary } from '../../shared/binary';
import { commonMethods, executeItems, modelProperty, nodeDescription, transportProperties } from '../../shared/common';
import { requireModelTask } from '../../shared/discovery';
import { chatOutputs } from '../../shared/mediaOutputs';
import { chatOptionsProperty, promptProperty } from '../../shared/mediaProperties';
import { buildVisionRequest, type MediaBytes, MAX_INPUT_BYTES } from '../../shared/mediaRequests';
import { choice, optionsProperty, record, stringProperty, textValue } from '../../shared/parameters';

export class WerkVision implements INodeType {
	description = nodeDescription('werkVision', 'WERK Vision (Beta)', [
		optionsProperty('operation', 'Operation', ['analyze']), modelProperty('image-understanding'), promptProperty,
		{ ...stringProperty('systemPrompt', 'System Prompt'), typeOptions: { rows: 3 } },
		{ displayName: 'Images', name: 'images', type: 'fixedCollection', default: { image: [{ binaryProperty: 'data' }] }, typeOptions: { multipleValues: true, sortable: true }, options: [{ name: 'image', displayName: 'Image', values: [{ displayName: 'Input Binary Field', name: 'binaryProperty', type: 'string', default: 'data', required: true }] }], description: 'Images are sent in this order, followed by the prompt, in one user message' },
		chatOptionsProperty(true), ...transportProperties,
	]);
	methods = commonMethods;
	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		return executeItems(this, async (client, index) => {
			choice(this.getNodeParameter('operation', index), 'Operation', ['analyze']);
			const group = record(this.getNodeParameter('images', index), 'Images');
			if (!Array.isArray(group.image) || !group.image.length) throw new Error('Provide at least one image binary field');
			const images: MediaBytes[] = [];
			let total = 0;
			for (const entry of group.image) {
				const media = await readBinary(this, index, textValue(record(entry, 'Image').binaryProperty, 'Binary field'), 'image');
				total += media.data.length;
				if (total > MAX_INPUT_BYTES) throw new Error(`Combined vision images exceed ${MAX_INPUT_BYTES} bytes`);
				images.push(media);
			}
			const request = buildVisionRequest(this.getNodeParameter('model', index, undefined, { extractValue: true }), this.getNodeParameter('prompt', index), this.getNodeParameter('systemPrompt', index, ''), images, this.getNodeParameter('options', index, {}));
			await requireModelTask(client, request.model as string, 'image-understanding');
			return chatOutputs(index, await client.api('POST', '/v1/chat/completions', request), request, 'image-understanding');
		});
	}
}

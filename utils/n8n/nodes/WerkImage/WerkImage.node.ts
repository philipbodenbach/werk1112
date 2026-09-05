import type { IExecuteFunctions, INodeExecutionData, INodeType } from 'n8n-workflow';
import { commonMethods, executeItems, modelProperty, nodeDescription, transportProperties } from '../../shared/common';
import { requireModelTask } from '../../shared/discovery';
import { imageOutputs } from '../../shared/mediaOutputs';
import { imageConfigurationProperty, negativePromptProperty, promptProperty } from '../../shared/mediaProperties';
import { buildImageRequest } from '../../shared/mediaRequests';
import { additionalParametersProperty, choice, optionsProperty } from '../../shared/parameters';
import { routingProperty } from '../../shared/routing';

export class WerkImage implements INodeType {
	description = nodeDescription('werkImage', 'WERK Image (Beta)', [
		optionsProperty('operation', 'Operation', ['generate']), modelProperty('image-generation'), promptProperty,
		negativePromptProperty, imageConfigurationProperty, routingProperty, additionalParametersProperty, ...transportProperties,
	]);
	methods = commonMethods;
	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		return executeItems(this, async (client, index) => {
			choice(this.getNodeParameter('operation', index), 'Operation', ['generate']);
			const request = buildImageRequest({ model: this.getNodeParameter('model', index, undefined, { extractValue: true }), prompt: this.getNodeParameter('prompt', index), negativePrompt: this.getNodeParameter('negativePrompt', index, ''), configuration: this.getNodeParameter('configuration', index, {}), routing: this.getNodeParameter('routing', index, {}), additionalParameters: this.getNodeParameter('additionalParameters', index, '{}') });
			await requireModelTask(client, request.model as string, 'image-generation');
			return imageOutputs(this, client, index, await client.api('POST', '/v1/images/generations', request), request);
		});
	}
}

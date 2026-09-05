import type { IExecuteFunctions, INodeExecutionData, INodeType } from 'n8n-workflow';
import { readBinary } from '../../shared/binary';
import { commonMethods, executeItems, modelProperty, nodeDescription, transportProperties } from '../../shared/common';
import { requireModelTask } from '../../shared/discovery';
import { jobProperties, submitJob } from '../../shared/jobs';
import { binaryProperty, negativePromptProperty, promptProperty, videoConfigurationProperty } from '../../shared/mediaProperties';
import { buildVideoRequest } from '../../shared/mediaRequests';
import { additionalParametersProperty, choice } from '../../shared/parameters';
import { routingProperty } from '../../shared/routing';

export class WerkVideo implements INodeType {
	description = nodeDescription('werkVideo', 'WERK Video (Beta)', [
		{ displayName: 'Operation', name: 'operation', type: 'options', default: 'generate', options: [{ name: 'Text to Video', value: 'generate' }, { name: 'Image to Video', value: 'imageToVideo' }] },
		modelProperty(), promptProperty, negativePromptProperty,
		{ ...binaryProperty, displayOptions: { show: { operation: ['imageToVideo'] } } },
		videoConfigurationProperty, routingProperty, additionalParametersProperty, ...jobProperties, ...transportProperties,
	]);
	methods = commonMethods;
	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		return executeItems(this, async (client, index) => {
			const operation = choice(this.getNodeParameter('operation', index), 'Operation', ['generate', 'imageToVideo']);
			const task = operation === 'imageToVideo' ? 'image-to-video' : 'video-generation';
			const initialImage = operation === 'imageToVideo' ? await readBinary(this, index, this.getNodeParameter('binaryProperty', index, 'data') as string, 'image') : undefined;
			const request = buildVideoRequest({ model: this.getNodeParameter('model', index, undefined, { extractValue: true }), prompt: this.getNodeParameter('prompt', index), negativePrompt: this.getNodeParameter('negativePrompt', index, ''), configuration: this.getNodeParameter('configuration', index, {}), routing: this.getNodeParameter('routing', index, {}), additionalParameters: this.getNodeParameter('additionalParameters', index, '{}') }, initialImage);
			await requireModelTask(client, request.model as string, task);
			return submitJob(this, client, index, '/v1/videos/generations', request, task);
		});
	}
}

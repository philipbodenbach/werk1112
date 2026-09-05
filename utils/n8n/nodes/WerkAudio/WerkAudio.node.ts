import type { IExecuteFunctions, INodeExecutionData, INodeType } from 'n8n-workflow';
import { readBinary } from '../../shared/binary';
import { commonMethods, executeItems, modelProperty, nodeDescription, transportProperties } from '../../shared/common';
import { requireModelTask } from '../../shared/discovery';
import { jobProperties, submitJob } from '../../shared/jobs';
import { audioConfigurationProperty, binaryProperty, negativePromptProperty, promptProperty } from '../../shared/mediaProperties';
import { audioAnalysisTasks, audioGenerationTasks, audioProcessTasks, buildAudioInputRequest, buildAudioRequest } from '../../shared/mediaRequests';
import { additionalParametersProperty, choice, optionsProperty, stringProperty, textValue } from '../../shared/parameters';
import { routingProperty } from '../../shared/routing';

export class WerkAudio implements INodeType {
	description = nodeDescription('werkAudio', 'WERK Audio (Beta)', [
		optionsProperty('operation', 'Operation', ['generate', 'process', 'analyze']),
		{ ...optionsProperty('task', 'Task', audioGenerationTasks), displayOptions: { show: { operation: ['generate'] } } },
		{ ...optionsProperty('task', 'Task', audioProcessTasks), displayOptions: { show: { operation: ['process'] } }, description: 'A declared task does not guarantee an available execution adapter' },
		{ ...optionsProperty('task', 'Task', audioAnalysisTasks), displayOptions: { show: { operation: ['analyze'] } }, description: 'A declared task does not guarantee an available execution adapter' },
		modelProperty(), { ...promptProperty, description: 'Text input for generation/TTS; required for audio-understanding and audio-editing; optional for other audio input tasks' },
		{ ...negativePromptProperty, displayOptions: { hide: { task: ['text-to-speech'] } } },
		{ ...binaryProperty, displayOptions: { show: { operation: ['process', 'analyze'] } } },
		{ ...stringProperty('referenceBinaryProperty', 'Reference Audio Binary Field'), displayOptions: { show: { operation: ['process'], task: ['voice-conversion'] } }, description: 'Optional second audio input; only supported for voice-conversion' },
		{ ...audioConfigurationProperty, displayOptions: { show: { operation: ['generate'] } } },
		routingProperty, additionalParametersProperty, ...jobProperties, ...transportProperties,
	]);
	methods = commonMethods;
	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		return executeItems(this, async (client, index) => {
			const operation = choice(this.getNodeParameter('operation', index), 'Operation', ['generate', 'process', 'analyze']);
			const task = choice(this.getNodeParameter('task', index), 'Task', operation === 'generate' ? audioGenerationTasks : operation === 'process' ? audioProcessTasks : audioAnalysisTasks);
			const input = { model: this.getNodeParameter('model', index, undefined, { extractValue: true }), prompt: this.getNodeParameter('prompt', index, ''), negativePrompt: this.getNodeParameter('negativePrompt', index, ''), routing: this.getNodeParameter('routing', index, {}), additionalParameters: this.getNodeParameter('additionalParameters', index, '{}') };
			let request;
			if (operation === 'generate') request = buildAudioRequest(task, { ...input, configuration: this.getNodeParameter('configuration', index, {}) });
			else {
				const audio = await readBinary(this, index, textValue(this.getNodeParameter('binaryProperty', index, 'data'), 'Input binary field'), 'audio');
				const referenceField = textValue(this.getNodeParameter('referenceBinaryProperty', index, ''), 'Reference binary field', true).trim();
				const reference = referenceField ? await readBinary(this, index, referenceField, 'audio') : undefined;
				request = buildAudioInputRequest(task, input, audio, reference);
			}
			await requireModelTask(client, request.model as string, task);
			return submitJob(this, client, index, operation === 'generate' ? task === 'text-to-speech' ? '/v1/audio/speech' : '/v1/audio/generations' : '/v1/jobs', request, task);
		});
	}
}

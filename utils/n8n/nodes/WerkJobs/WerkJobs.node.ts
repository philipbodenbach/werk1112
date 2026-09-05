import type { IExecuteFunctions, INodeExecutionData, INodeType } from 'n8n-workflow';
import { commonMethods, executeItems, jsonItem, nodeDescription, transportProperties } from '../../shared/common';
import { downloadedOutput, jobMetadata, jobOutputs, jobRecord, waitForJob, waitProperties } from '../../shared/jobs';
import { finiteNumber, string } from '../../shared/validation';

export class WerkJobs implements INodeType {
  description = nodeDescription('werkJobs', 'WERK Jobs (Beta)', [
    { displayName: 'Operation', name: 'operation', type: 'options', noDataExpression: true, default: 'get', options: [
      { name: 'Get Job', value: 'get', action: 'Get an existing job' },
      { name: 'Wait for Job', value: 'wait', action: 'Wait for an existing job' },
      { name: 'Cancel Job', value: 'cancel', action: 'Request cooperative job cancellation' },
      { name: 'Download Output', value: 'download', action: 'Download an output by output ID' },
    ] },
    { displayName: 'Job ID', name: 'jobId', type: 'string', default: '', required: true, displayOptions: { show: { operation: ['get', 'wait', 'cancel'] } } },
    { displayName: 'Output ID', name: 'outputId', type: 'string', default: '', required: true, displayOptions: { show: { operation: ['download'] } }, description: 'The output ID from result.outputs, not a job or result ID. Retention applies.' },
    ...waitProperties.map(property => ({ ...property, displayOptions: { show: { operation: ['wait'] } } })),
    { displayName: 'Cancel Job if Waiting Aborts', name: 'cancelOnAbort', type: 'boolean', default: false, displayOptions: { show: { operation: ['wait'] } }, description: 'Whether to request best-effort cancellation of this existing job on timeout, interruption or polling error' },
    { displayName: 'Download Completed Outputs', name: 'download', type: 'boolean', default: true, displayOptions: { show: { operation: ['wait'] } } },
    ...transportProperties,
  ]);
  methods = commonMethods;
  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    return executeItems(this, async (client, index) => {
      const operation = String(this.getNodeParameter('operation', index));
      if (operation === 'download') return [await downloadedOutput(this, client, index, string(this.getNodeParameter('outputId', index), 'Output ID'))];
      const id = string(this.getNodeParameter('jobId', index), 'Job ID');
      if (operation === 'cancel') return [jsonItem(index, jobMetadata(jobRecord(await client.api('DELETE', `/v1/jobs/${encodeURIComponent(id)}`), id)))];
      if (!['get', 'wait'].includes(operation)) throw new Error('Unknown WERK Jobs operation');
      const initial = await client.api('GET', `/v1/jobs/${encodeURIComponent(id)}`);
      if (operation === 'get') return [jsonItem(index, jobMetadata(jobRecord(initial, id)))];
      if (initial.id !== id) throw new Error(`WERK job ID differs from requested job ${id}`);
      const record = await waitForJob(client, initial, {
        waitSeconds: finiteNumber(this.getNodeParameter('waitSeconds', index, 900), 'Job wait time', 1),
        pollSeconds: finiteNumber(this.getNodeParameter('pollSeconds', index, 1), 'Poll interval', 0.1),
        cancelOnAbort: this.getNodeParameter('cancelOnAbort', index, false) === true,
        signal: this.getExecutionCancelSignal?.(),
      });
      return this.getNodeParameter('download', index, true) ? jobOutputs(this, client, index, record) : [jsonItem(index, jobMetadata(record))];
    });
  }
}

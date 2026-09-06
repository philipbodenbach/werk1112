import type { ICredentialType, INodeProperties } from 'n8n-workflow';
import { authenticate } from '../shared/client';

export class WerkApi implements ICredentialType {
  name = 'werkApi';
  displayName = 'WERK API';
  documentationUrl = 'https://github.com/philipbodenbach/werk1112/tree/main/utils/n8n';
  properties: INodeProperties[] = [
    { displayName: 'Server Base URL', name: 'baseUrl', type: 'string', default: 'http://127.0.0.1:11434', required: true, description: 'HTTP or HTTPS URL; an optional reverse-proxy path prefix is supported' },
    { displayName: 'Authentication', name: 'authMode', type: 'options', default: 'apiKey', options: [
      { name: 'API Key', value: 'apiKey' },
      { name: 'Unauthenticated (Explicit Server Opt-In Required)', value: 'none' },
    ] },
    { displayName: 'API Key', name: 'apiKey', type: 'string', typeOptions: { password: true }, default: '', displayOptions: { show: { authMode: ['apiKey'] } } },
    { displayName: 'Verify TLS Certificates', name: 'verifyTls', type: 'boolean', default: true, description: 'Whether HTTPS certificates must pass the standard host certificate validation' },
  ];
  authenticate = authenticate;
}

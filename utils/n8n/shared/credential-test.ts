import type { ICredentialTestFunctions, ICredentialsDecrypted, INodeCredentialTestResult } from 'n8n-workflow';
import { credentialSettings, readBounded } from './client';
import { object, parseJson, safeMessage } from './validation';

export async function werkApiTest(this: ICredentialTestFunctions, credential: ICredentialsDecrypted): Promise<INodeCredentialTestResult> {
  const key = typeof credential.data?.apiKey === 'string' ? credential.data.apiKey : '';
  try {
    const settings = credentialSettings(credential.data ?? {});
    // Credential-test contexts expose the official request helper (not httpRequest).
    const response: unknown = await this.helpers.request({
      method: 'GET', url: `${settings.baseUrl}/v1/models`,
      headers: { Accept: 'application/json', ...(settings.authenticated ? { Authorization: `Bearer ${settings.apiKey}` } : {}) },
      timeout: 15000, useStream: true, resolveWithFullResponse: true, simple: false,
      followRedirect: false, followAllRedirects: false, maxRedirects: 0,
      sendCredentialsOnCrossOriginRedirect: false, rejectUnauthorized: settings.verifyTls,
    });
    const full = object(response, 'credential test HTTP response');
    const bytes = await readBounded(full.body, object(full.headers, 'credential test headers'), 16 * 1024 * 1024, 15000);
    if (full.statusCode !== 200) throw new Error(`WERK credential discovery failed (HTTP ${Number(full.statusCode)}); redirects are rejected`);
    const body = object(parseJson(bytes.toString('utf8')), 'models response');
    if (!Array.isArray(body.data)) throw new Error('Server did not return a WERK/OpenAI models list');
    return { status: 'OK', message: 'WERK discovery succeeded. Model inference and memory availability are checked separately.' };
  } catch (error) {
    return { status: 'Error', message: safeMessage(error, [key]) };
  }
}

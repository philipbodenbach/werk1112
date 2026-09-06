/* Real n8n process + imported workflow. No imports from the package's dist here. */
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');
const { validateExamples } = require('./example-validation.cjs');

const packageDirectory = path.resolve(__dirname, '..');
const expectedNames = ['Discovery', 'Text', 'Image', 'Vision', 'Video', 'Audio', 'Jobs', 'Runtime'].map((name) => `werk${name}`);
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZlN0AAAAASUVORK5CYII=', 'base64');
const testKey = `fixture-${crypto.randomBytes(16).toString('hex')}`;
const processes = new Set();
let temporaryDirectory;
let mock;

function log(message) { process.stdout.write(`${message}\n`); }
function safeLog(value) { return String(value).split(testKey).join('[REDACTED]'); }

function child(binary, args, environment, cwd) {
	const isScript = /(?:\.js|\.cjs|[\\/]bin[\\/]n8n)$/.test(binary);
	const processHandle = spawn(isScript ? process.execPath : binary, isScript ? [binary, ...args] : args, {
		cwd, env: environment, stdio: ['ignore', 'pipe', 'pipe'],
	});
	processes.add(processHandle);
	let output = '';
	processHandle.stdout.on('data', (data) => { output += data.toString(); });
	processHandle.stderr.on('data', (data) => { output += data.toString(); });
	const done = new Promise((resolve, reject) => {
		processHandle.once('error', reject);
		processHandle.once('close', (code, signal) => { processes.delete(processHandle); resolve({ code, signal, output }); });
	});
	return { process: processHandle, done, output: () => output };
}

async function stop(handle) {
	if (handle.process.exitCode !== null || handle.process.signalCode !== null) return handle.done;
	handle.process.kill('SIGTERM');
	const timer = setTimeout(() => handle.process.kill('SIGKILL'), 10000);
	try { return await handle.done; } finally { clearTimeout(timer); }
}

async function run(binary, args, environment, cwd) {
	const handle = child(binary, args, environment, cwd);
	const timer = setTimeout(() => handle.process.kill('SIGKILL'), 120000);
	try {
		const result = await handle.done;
		assert.equal(result.code, 0, `${args[0]} failed:\n${safeLog(result.output.slice(-14000))}`);
		return result.output;
	} finally { clearTimeout(timer); }
}

async function port() {
	const server = net.createServer();
	await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
	const assigned = server.address().port;
	await new Promise((resolve) => server.close(resolve));
	return assigned;
}

function environment(userFolder, portNumber) {
	const result = { ...process.env };
	for (const key of Object.keys(result)) if (/^(?:N8N_|DB_|QUEUE_|EXECUTIONS_|CREDENTIALS_|WEBHOOK_URL$|NODE_PATH$)/.test(key)) delete result[key];
	return {
		...result, N8N_USER_FOLDER: userFolder, N8N_PORT: String(portNumber),
		N8N_LISTEN_ADDRESS: '127.0.0.1', N8N_HOST: '127.0.0.1', N8N_PROTOCOL: 'http',
		N8N_ENCRYPTION_KEY: crypto.randomBytes(32).toString('hex'),
		N8N_DIAGNOSTICS_ENABLED: 'false', N8N_VERSION_NOTIFICATIONS_ENABLED: 'false',
		N8N_TEMPLATES_ENABLED: 'false', N8N_PERSONALIZATION_ENABLED: 'false',
		N8N_SECURE_COOKIE: 'false',
		N8N_DEFAULT_BINARY_DATA_MODE: 'filesystem',
		N8N_SSRF_PROTECTION_ENABLED: 'true', N8N_SSRF_ALLOWED_IP_RANGES: '127.0.0.1/32',
		N8N_LOG_LEVEL: 'warn', EXECUTIONS_DATA_PRUNE: 'false', NO_COLOR: '1',
	};
}

async function registration(binary, environment, cwd) {
	const handle = child(binary, ['start'], environment, cwd);
	const origin = `http://127.0.0.1:${environment.N8N_PORT}`;
	const deadline = Date.now() + 120000;
	let cookie = '';
	try {
		while (Date.now() < deadline) {
			if (handle.process.exitCode !== null) throw new Error(`n8n exited before loading: ${safeLog(handle.output().slice(-14000))}`);
			try {
				const response = await fetch(`${origin}/types/nodes.json`, { headers: cookie ? { Cookie: cookie } : {}, signal: AbortSignal.timeout(2000) });
				if (response.status === 401 && !cookie) {
					// The current host protects type manifests. Set up an owner only in
					// this fresh temporary instance and use its ordinary session cookie.
					const setup = await fetch(`${origin}/rest/owner/setup`, {
						method: 'POST', headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify({ email: 'loader-fixture@example.invalid', firstName: 'Loader', lastName: 'Fixture', password: `Fixture1!${crypto.randomBytes(20).toString('hex')}` }),
						signal: AbortSignal.timeout(5000),
					});
					assert.ok(setup.ok, `temporary owner setup failed (HTTP ${setup.status}): ${await setup.text()}`);
					cookie = setup.headers.getSetCookie().map((value) => value.split(';')[0]).join('; ');
					assert.ok(cookie, 'temporary instance owner session cookie');
					continue;
				}
				if (response.ok) {
					const nodes = await response.json();
					const credentialsResponse = await fetch(`${origin}/types/credentials.json`, { headers: { Cookie: cookie } });
					assert.ok(credentialsResponse.ok, 'real credential registry endpoint');
					const credentials = await credentialsResponse.json();
					const registered = nodes.filter((description) => expectedNames.some((name) => description.name.endsWith(`.${name}`)));
					assert.equal(registered.length, 8, `eight registered nodes; log: ${safeLog(handle.output().slice(-5000))}`);
					for (const description of registered) {
						assert.match(description.displayName, /\(Beta\)$/);
						assert.equal(description.version, 1);
						assert.equal(description.credentials[0].name, 'werkApi');
					}
					assert.equal(credentials.filter((description) => description.name === 'werkApi').length, 1);
					validateExamples(nodes, { requireBuiltins: true });
					return { nodes, credentials, ids: registered.map((description) => description.name).sort() };
				}
			} catch (error) {
				if (error instanceof assert.AssertionError) throw error;
			}
			await delay(300);
		}
		throw new Error(`n8n registration timeout: ${safeLog(handle.output().slice(-14000))}`);
	} finally { await stop(handle); }
}

async function startMock() {
	const requests = [];
	const server = http.createServer(async (request, response) => {
		try {
			assert.equal(request.headers.authorization, `Bearer ${testKey}`, 'official credential helper authenticates requests');
			const chunks = [];
			for await (const chunk of request) chunks.push(chunk);
			const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : undefined;
			const url = new URL(request.url, 'http://fixture');
			requests.push({ method: request.method, path: url.pathname, body });
			let payload;
			if (request.method === 'GET' && url.pathname === '/proxy/v1/models') {
				payload = { object: 'list', data: [{ id: 'fixture-image' }, { id: 'fixture-vision' }] };
			} else if (request.method === 'GET' && url.pathname === '/proxy/v1/capabilities') {
				payload = { object: 'werk.capabilities', models: [
					{ id: 'fixture-image', tasks: ['image_generation'], available_tasks: ['image_generation'] },
					{ id: 'fixture-vision', tasks: ['image-understanding'], available_tasks: ['image-understanding'] },
				] };
			} else if (request.method === 'GET' && url.pathname === '/proxy/v1/parameters') {
				payload = { task: url.searchParams.get('task'), model: url.searchParams.get('model'), parameters: [] };
			} else if (request.method === 'POST' && url.pathname === '/proxy/v1/images/generations') {
				assert.equal(body.model, 'fixture-image');
				assert.equal(body.prompt, 'Fixture image 0', 'the real n8n expression engine evaluates per-item parameters');
				payload = { created: 1, data: [{ b64_json: png.toString('base64') }], model: 'fixture-image', werk: { backend: 'mock', request_id: 'fixture-image-request' } };
			} else if (request.method === 'POST' && url.pathname === '/proxy/v1/chat/completions') {
				assert.equal(body.model, 'fixture-vision');
				assert.equal(body.stream, false);
				const parts = body.messages.at(-1).content;
				assert.equal(parts[0].type, 'image_url');
				assert.equal(parts[1].type, 'text');
				assert.equal(parts[0].image_url.url, `data:image/png;base64,${png.toString('base64')}`, 'filesystem-backed image bytes resolved by n8n helper');
				payload = { id: 'chat-fixture', model: 'fixture-vision', choices: [{ index: 0, message: { role: 'assistant', content: 'Binary image received intact.' }, finish_reason: 'stop' }], usage: { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 } };
			} else {
				throw new Error(`Unexpected fixture request ${request.method} ${url.pathname}`);
			}
			response.writeHead(200, { 'Content-Type': 'application/json' });
			response.end(JSON.stringify(payload));
		} catch (error) {
			requests.push({ error: safeLog(error.message) });
			response.writeHead(400, { 'Content-Type': 'application/json' });
			response.end(JSON.stringify({ error: safeLog(error.message) }));
		}
	});
	await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
	return { server, requests, baseUrl: `http://127.0.0.1:${server.address().port}/proxy` };
}

function executionJson(output) {
	const end = output.lastIndexOf('}') + 1;
	for (const match of output.matchAll(/^\{/gm)) {
		try { return JSON.parse(output.slice(match.index, end)); } catch { /* Skip non-JSON host logs. */ }
	}
	throw new Error(`n8n execute returned no JSON result: ${safeLog(output.slice(-10000))}`);
}

async function main() {
	const binary = process.env.N8N_BIN || 'n8n';
	temporaryDirectory = await fs.mkdtemp(path.join(os.tmpdir(), 'werk-n8n-loader-'));
	const userFolder = path.join(temporaryDirectory, 'native-user');
	const customDirectory = path.join(userFolder, '.n8n/custom/werk1112');
	await fs.mkdir(customDirectory, { recursive: true });
	await fs.cp(path.join(packageDirectory, 'dist'), customDirectory, { recursive: true });
	assert.ok(!(await fs.readdir(customDirectory)).includes('node_modules'), 'artifact does not contain development dependencies');
	const env = environment(userFolder, await port());
	const version = (await run(binary, ['--version'], env, temporaryDirectory)).trim();
	assert.equal(version, '2.37.10', 'the smoke host must match the documented pinned n8n version');
	log(`Testing n8n ${version}, Node.js ${process.versions.node}, complete dist only`);
	const registry = await registration(binary, env, temporaryDirectory);
	assert.deepEqual(registry.ids, expectedNames.map((name) => `CUSTOM.${name}`).sort());
	log(`Native custom-directory loader: ${registry.ids.join(', ')}`);

	mock = await startMock();
	const credentialId = 'WerkLoaderCred001';
	const credentialFile = path.join(temporaryDirectory, 'credentials.json');
	await fs.writeFile(credentialFile, JSON.stringify([{ id: credentialId, name: 'Loader fixture only', type: 'werkApi', data: { baseUrl: mock.baseUrl, authMode: 'apiKey', apiKey: testKey, verifyTls: true } }]), { mode: 0o600 });
	const imported = JSON.parse(await fs.readFile(path.join(packageDirectory, 'examples/02-image.json'), 'utf8'));
	imported.id = 'WerkLoaderSmoke1';
	const imageNode = imported.nodes.find((node) => node.type.endsWith('.werkImage'));
	imageNode.parameters.model = { __rl: true, mode: 'id', value: 'fixture-image' };
	imageNode.parameters.prompt = '={{ "Fixture image " + $itemIndex }}';
	imageNode.credentials = { werkApi: { id: credentialId, name: 'Loader fixture only' } };
	const visionTemplate = JSON.parse(await fs.readFile(path.join(packageDirectory, 'examples/03-vision.json'), 'utf8'));
	const visionNode = visionTemplate.nodes.find((node) => node.type.endsWith('.werkVision'));
	visionNode.parameters.model = { __rl: true, mode: 'id', value: 'fixture-vision' };
	visionNode.credentials = imageNode.credentials;
	imported.nodes.push(visionNode);
	imported.connections[imageNode.name] = { main: [[{ node: visionNode.name, type: 'main', index: 0 }]] };
	const workflowFile = path.join(temporaryDirectory, 'workflow.json');
	await fs.writeFile(workflowFile, JSON.stringify(imported));
	const credentialImport = await run(binary, ['import:credentials', `--input=${credentialFile}`], env, temporaryDirectory);
	assert.ok(!/error occurred/i.test(credentialImport), safeLog(credentialImport));
	const workflowImport = await run(binary, ['import:workflow', `--input=${workflowFile}`], env, temporaryDirectory);
	assert.ok(!/error occurred/i.test(workflowImport), safeLog(workflowImport));
	// n8n 2.37.10 emits --rawOutput through its info logger too.
	const executed = executionJson(await run(binary, ['execute', `--id=${imported.id}`, '--rawOutput'], { ...env, N8N_LOG_LEVEL: 'info' }, temporaryDirectory));
	assert.equal(executed.data.resultData.error, undefined, safeLog(JSON.stringify(executed.data.resultData.error)));
	const imageItem = executed.data.resultData.runData[imageNode.name][0].data.main[0][0];
	assert.equal(imageItem.binary.data.mimeType, 'image/png');
	assert.ok(imageItem.binary.data.id?.startsWith('filesystem'), 'real output is a filesystem binary reference');
	assert.notEqual(imageItem.binary.data.data, png.toString('base64'), 'output is not just inline base64');
	const visionItem = executed.data.resultData.runData[visionNode.name][0].data.main[0][0];
	assert.equal(visionItem.json.text, 'Binary image received intact.');
	assert.equal(mock.requests.filter((request) => request.method === 'POST' && request.path.endsWith('/images/generations')).length, 1);
	assert.equal(mock.requests.filter((request) => request.method === 'POST' && request.path.endsWith('/chat/completions')).length, 1);
	assert.equal(mock.requests.filter((request) => request.error).length, 0, safeLog(JSON.stringify(mock.requests.filter((request) => request.error))));
	assert.ok(!JSON.stringify(executed).includes(testKey), 'credentials never enter execution output');
	log('Imported Image → Vision workflow passed: authenticated proxy path, native item expression, actual filesystem binary output/input, no duplicate POSTs');

	const extensionDirectory = path.join(temporaryDirectory, 'extension-dist');
	await fs.cp(path.join(packageDirectory, 'dist'), extensionDirectory, { recursive: true });
	const extensionUser = path.join(temporaryDirectory, 'extension-user');
	await fs.mkdir(extensionUser);
	const extensionEnv = { ...environment(extensionUser, await port()), N8N_CUSTOM_EXTENSIONS: extensionDirectory };
	const extensionRegistry = await registration(binary, extensionEnv, temporaryDirectory);
	assert.deepEqual(extensionRegistry.ids, registry.ids);
	log('Absolute N8N_CUSTOM_EXTENSIONS path passed; all eight examples match actual host types and parameters');
}

main().catch((error) => { process.exitCode = 1; process.stderr.write(`${safeLog(error.stack)}\n`); }).finally(async () => {
	for (const handle of processes) handle.kill('SIGKILL');
	if (mock) await new Promise((resolve) => mock.server.close(resolve));
	if (temporaryDirectory) await fs.rm(temporaryDirectory, { recursive: true, force: true });
});

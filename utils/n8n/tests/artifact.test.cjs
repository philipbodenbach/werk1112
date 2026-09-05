const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const pkg = require('../package.json');
const root = path.resolve(__dirname, '..');

test('private Beta package registers all eight version-one nodes and one credential with complete dist', () => {
  assert.equal(pkg.private, true);
  assert.equal(pkg.version, '1.6.0');
  assert.match(pkg.description, /Beta/);
  assert.equal(pkg.license, 'Elastic-2.0');
  assert.equal(pkg.n8n.n8nNodesApiVersion, 1);
  assert.ok(pkg.keywords.includes('n8n-community-node-package'));
  assert.equal(pkg.peerDependencies['n8n-workflow'], '2.37.4');
  assert.deepEqual(pkg.dependencies ?? {}, {});
  assert.equal(pkg.n8n.nodes.length, 8);
  assert.equal(pkg.n8n.credentials.length, 1);
  const names = [];
  for (const file of pkg.n8n.nodes) {
    const NodeClass = Object.values(require(path.join(root, file))).find(value => typeof value === 'function');
    const node = new NodeClass();
    names.push(node.description.name);
    assert.match(node.description.displayName, /^WERK \w+ \(Beta\)$/);
    assert.equal(node.description.version, 1);
    assert.deepEqual(node.description.inputs, ['main']);
    assert.deepEqual(node.description.outputs, ['main']);
    assert.equal(node.description.credentials[0].name, 'werkApi');
    assert.equal(node.description.credentials[0].testedBy, 'werkApiTest');
    assert.equal(typeof node.methods.credentialTest.werkApiTest, 'function');
    assert.ok(fs.existsSync(path.join(path.dirname(path.join(root, file)), node.description.icon.replace('file:', ''))));
  }
  assert.deepEqual(names.sort(), ['werkAudio', 'werkDiscovery', 'werkImage', 'werkJobs', 'werkRuntime', 'werkText', 'werkVideo', 'werkVision']);
  const { WerkApi } = require('../dist/credentials/WerkApi.credentials');
  const credential = new WerkApi();
  assert.equal(credential.name, 'werkApi');
  assert.equal(credential.properties.find(p => p.name === 'apiKey').typeOptions.password, true);
  assert.equal(credential.properties.find(p => p.name === 'verifyTls').default, true);
  for (const file of fs.readdirSync(path.join(root, 'shared')).filter(file => file.endsWith('.ts'))) {
    assert.ok(fs.existsSync(path.join(root, 'dist/shared', file.replace(/\.ts$/, '.js'))), `shipped shared module ${file}`);
  }
  assert.equal(fs.readFileSync(path.join(root, 'LICENSE'), 'utf8'), fs.readFileSync(path.join(root, '../../LICENSE'), 'utf8'));
  assert.equal(fs.readFileSync(path.join(root, 'dist/LICENSE'), 'utf8'), fs.readFileSync(path.join(root, 'LICENSE'), 'utf8'));
  assert.equal(pkg.scripts.release, undefined);
  assert.equal(pkg.scripts.publish, undefined);
});

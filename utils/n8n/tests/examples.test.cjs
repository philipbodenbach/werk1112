const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { validateExamples } = require('./example-validation.cjs');

const packageDirectory = path.resolve(__dirname, '..');
const packageJson = require('../package.json');

test('all eight examples reference compiled custom node versions and real parameter IDs', () => {
	const descriptions = packageJson.n8n.nodes.map((file) => {
		const exports = require(path.join(packageDirectory, file));
		const NodeClass = Object.values(exports).find((value) => typeof value === 'function');
		const { description } = new NodeClass();
		return { ...description, name: `CUSTOM.${description.name}` };
	});
	assert.equal(validateExamples(descriptions).length, 8);
});

test('parity table accounts for the actual 30 public ComfyUI registrations', () => {
	const registrations = ['nodes.py', 'runtime_nodes.py'].flatMap((file) => {
		const source = fs.readFileSync(path.resolve(packageDirectory, '../comfyUI', file), 'utf8');
		const block = source.match(/\nNODE_CLASS_MAPPINGS = \{([\s\S]*?)\n\}/)?.[1];
		assert.ok(block, `${file} public mappings found`);
		return [...block.matchAll(/"(Werk\w+)":/g)].map((match) => match[1]);
	});
	assert.equal(registrations.length, 30);
	assert.equal(new Set(registrations).size, 30);
	const table = fs.readFileSync(path.join(packageDirectory, 'docs/comfyui-parity.md'), 'utf8');
	const rows = [...table.matchAll(/^\| `(Werk\w+)` \|/gm)].map((match) => match[1]);
	assert.deepEqual(rows.sort(), registrations.sort());
});

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

test('parity table covers every mapping provider exported by ComfyUI', () => {
    const root = path.resolve(packageDirectory, '../comfyUI');
    const init = fs.readFileSync(path.join(root, '__init__.py'), 'utf8');
    const providers = [...init.matchAll(/from \.(\w+) import \(([\s\S]*?)\)/g)]
        .filter((match) => /NODE_CLASS_MAPPINGS as \w+/.test(match[2]))
        .map((match) => `${match[1]}.py`);
    assert.ok(providers.length > 0, 'public mapping providers found in __init__.py');
    assert.ok(providers.includes('text_nodes.py'), 'dedicated text registrations included');
    const registrations = providers.flatMap((file) => {
        const source = fs.readFileSync(path.join(root, file), 'utf8');
        const block = source.match(/\nNODE_CLASS_MAPPINGS = \{([\s\S]*?)\n\}/)?.[1];
        assert.ok(block, `${file} public mappings found`);
        return [...block.matchAll(/"(Werk\w+)":/g)].map((match) => match[1]);
    });
    assert.equal(new Set(registrations).size, registrations.length, 'no silently overwritten registrations');
    const table = fs.readFileSync(path.join(packageDirectory, 'docs/comfyui-parity.md'), 'utf8');
    const rows = [...table.matchAll(/^\| `(Werk\w+)` \|/gm)].map((match) => match[1]);
    assert.deepEqual(rows.sort(), registrations.sort());
});

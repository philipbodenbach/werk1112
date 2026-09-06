const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const examplesDirectory = path.resolve(__dirname, '../examples');

function propertiesCheck(parameters, properties, label) {
	for (const [name, value] of Object.entries(parameters)) {
		const candidates = properties.filter((property) => property.name === name);
		assert.ok(candidates.length, `${label}: unknown parameter ${name}`);
		if (typeof value === 'string' && value.startsWith('=')) continue;
		const property = candidates.find((candidate) => candidate.type !== 'notice') ?? candidates[0];
		if (property.type === 'collection') {
			assert.equal(typeof value, 'object', `${label}.${name} collection`);
			propertiesCheck(value, candidates.flatMap((candidate) => candidate.options ?? []), `${label}.${name}`);
		} else if (property.type === 'fixedCollection') {
			for (const [group, groupValue] of Object.entries(value)) {
				const option = candidates.flatMap((candidate) => candidate.options ?? []).find((entry) => entry.name === group);
				assert.ok(option, `${label}.${name}: unknown collection group ${group}`);
				for (const entry of Array.isArray(groupValue) ? groupValue : [groupValue]) propertiesCheck(entry, option.values, `${label}.${name}.${group}`);
			}
		} else if (property.type === 'options' && property.options?.length && !property.typeOptions?.loadOptionsMethod) {
			const values = candidates.flatMap((candidate) => candidate.options ?? []).map((option) => option.value);
			assert.ok(values.includes(value), `${label}.${name}: invalid option ${String(value)}`);
		}
	}
}

function validateExamples(descriptions, { requireBuiltins = false } = {}) {
	const files = fs.readdirSync(examplesDirectory).filter((file) => file.endsWith('.json')).sort();
	assert.equal(files.length, 8, 'eight importable examples');
	const byName = new Map();
	for (const description of descriptions) byName.set(description.name, [...(byName.get(description.name) ?? []), description]);
	for (const file of files) {
		const contents = fs.readFileSync(path.join(examplesDirectory, file), 'utf8');
		const workflow = JSON.parse(contents);
		assert.equal(workflow.active, false);
		assert.ok(!/"(?:apiKey|accessToken|handoff|handoffToken)"/.test(contents), `${file}: no secrets/handoff fields`);
		const names = new Set(workflow.nodes.map((node) => node.name));
		assert.equal(names.size, workflow.nodes.length, `${file}: unique node names`);
		for (const node of workflow.nodes) {
			assert.ok(!node.credentials, `${file}: no embedded credential references`);
			const candidates = byName.get(node.type);
			if (!candidates && !requireBuiltins && node.type.startsWith('n8n-nodes-base.')) continue;
			assert.ok(candidates, `${file}: actual loader must register ${node.type}`);
			const description = candidates.find((candidate) => (Array.isArray(candidate.version) ? candidate.version : [candidate.version]).includes(node.typeVersion));
			assert.ok(description, `${file}: ${node.type} version ${node.typeVersion}`);
			propertiesCheck(node.parameters, description.properties, `${file}/${node.name}`);
		}
		for (const [source, connection] of Object.entries(workflow.connections)) {
			assert.ok(names.has(source));
			for (const output of connection.main) for (const target of output) assert.ok(names.has(target.node));
		}
	}
	return files;
}

module.exports = { propertiesCheck, validateExamples };

const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
for (const folder of fs.readdirSync(path.join(root, 'nodes'))) {
  fs.copyFileSync(path.join(root, 'icons/werk.png'), path.join(root, 'dist/nodes', folder, 'werk.png'));
}
fs.copyFileSync(path.join(root, 'LICENSE'), path.join(root, 'dist/LICENSE'));
for (const entry of [...require('../package.json').n8n.nodes, ...require('../package.json').n8n.credentials]) {
  if (!fs.existsSync(path.join(root, entry))) throw new Error(`Missing build artifact: ${entry}`);
}

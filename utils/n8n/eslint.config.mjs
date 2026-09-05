import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist/**', 'node_modules/**', 'tests/**', 'scripts/**'] },
  ...tseslint.configs.recommended,
  { files: ['**/*.ts'], rules: { '@typescript-eslint/no-explicit-any': 'error' } },
);

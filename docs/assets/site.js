(() => {
  const root = document.documentElement;
  const preference = matchMedia('(prefers-color-scheme: dark)');
  let theme;
  // Use the same preference as WerkWorker and philipbodenbach.de.
  try { theme = localStorage.getItem('pb-theme'); } catch {}

  const apply = () => {
    const dark = theme === 'dark' || (theme !== 'light' && preference.matches);
    root.classList.toggle('dark', dark);
    document.querySelector('meta[name="theme-color"]').content = dark ? '#0a0a0a' : '#fafafa';
    const button = document.querySelector('[data-theme-toggle]');
    if (!button) return;
    const label = dark ? button.dataset.lightLabel : button.dataset.darkLabel;
    button.setAttribute('aria-pressed', String(dark));
    button.setAttribute('aria-label', label);
    button.title = label;
  };

  apply();
  preference.addEventListener('change', apply);
  document.addEventListener('DOMContentLoaded', () => {
    const button = document.querySelector('[data-theme-toggle]');
    if (!button) return;
    button.hidden = false;
    button.addEventListener('click', () => {
      theme = root.classList.contains('dark') ? 'light' : 'dark';
      try { localStorage.setItem('pb-theme', theme); } catch {}
      apply();
    });
    apply();
  });
})();

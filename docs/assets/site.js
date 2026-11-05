/* VUGVA docs — client-side search + keyboard shortcuts. No dependencies. */
(() => {
  'use strict';

  const PAGES = {
    h: 'index.html',
    a: 'architecture.html',
    c: 'configuration.html',
    b: 'benchmarks.html',
    i: 'implementation.html',
    t: 'troubleshooting.html',
    s: 'search.html',
  };

  // ── theme ────────────────────────────────────────────────────────────────
  const root = document.documentElement;
  const saved = localStorage.getItem('vugva-theme');
  if (saved) root.dataset.theme = saved;

  const toggleTheme = () => {
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    root.dataset.theme = next;
    localStorage.setItem('vugva-theme', next);
  };

  // ── overlays ─────────────────────────────────────────────────────────────
  const searchOverlay = document.getElementById('search-overlay');
  const helpOverlay = document.getElementById('help-overlay');
  const input = document.getElementById('search-input');
  const results = document.getElementById('search-results');
  const closeSearchBtn = document.getElementById('search-close');
  const openSearchBtn = document.getElementById('search-open');
  const helpCloseBtn = document.getElementById('help-close');

  const isOpen = (el) => el && !el.hasAttribute('hidden');

  const openSearch = (q = '') => {
    if (searchOverlay) {
      helpOverlay?.setAttribute('hidden', '');
      searchOverlay.removeAttribute('hidden');
    } else if (input) {
      input.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
    if (!input) return;
    input.value = q;
    input.focus();
    input.select();
    runSearch(q);
  };
  const closeSearch = () => {
    searchOverlay?.setAttribute('hidden', '');
  };
  const openHelp = () => {
    closeSearch();
    if (helpOverlay) { helpOverlay.removeAttribute('hidden'); helpCloseBtn?.focus(); }
  };
  const closeHelp = () => helpOverlay?.setAttribute('hidden', '');

  openSearchBtn?.addEventListener('click', () => openSearch());
  closeSearchBtn?.addEventListener('click', closeSearch);
  helpCloseBtn?.addEventListener('click', closeHelp);
  helpOverlay?.addEventListener('click', (e) => { if (e.target === helpOverlay) closeHelp(); });
  searchOverlay?.addEventListener('click', (e) => { if (e.target === searchOverlay) closeSearch(); });

  // ── index + search ───────────────────────────────────────────────────────
  let index = null;
  let indexPromise = null;
  const loadIndex = () => {
    if (!indexPromise) {
      indexPromise = fetch('assets/search-index.json')
        .then((r) => r.json())
        .then((data) => { index = data; return data; })
        .catch(() => (index = []));
    }
    return indexPromise;
  };

  const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const highlight = (text, terms) => {
    let out = escapeHtml(text);
    for (const t of terms) {
      if (t.length < 2) continue;
      const re = new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
      out = out.replace(re, '<mark>$1</mark>');
    }
    return out;
  };

  let selected = -1;

  const runSearch = (raw) => {
    if (!results) return;
    const q = (raw || '').trim().toLowerCase();
    selected = -1;
    if (!index) {
      results.innerHTML = '<li class="empty">Loading index…</li>';
      loadIndex().then(() => runSearch(raw));
      return;
    }
    if (q.length < 2) {
      results.innerHTML = '<li class="empty">Type at least two characters. Try <mark>pin</mark>, <mark>MoE</mark>, <mark>VUGVA_HOT_CACHE_MB</mark>, <mark>zerocopy</mark>.</li>';
      return;
    }
    const terms = q.split(/\s+/).filter(Boolean);
    const scored = [];
    for (const chunk of index) {
      const hay = (chunk.heading + ' ' + chunk.title + ' ' + chunk.text).toLowerCase();
      let score = 0;
      for (const t of terms) {
        if (!hay.includes(t)) { score = -1; break; }
        score += 1;
        if (chunk.heading.toLowerCase().includes(t)) score += 6;
        if (chunk.title.toLowerCase().includes(t)) score += 2;
      }
      if (score > 0) scored.push({ chunk, score });
    }
    scored.sort((a, b) => b.score - a.score);
    if (!scored.length) {
      results.innerHTML = `<li class="empty">No matches for “${escapeHtml(raw.trim())}”.</li>`;
      return;
    }
    results.innerHTML = scored.slice(0, 12).map(({ chunk }) => {
      const href = chunk.page + '.html' + (chunk.anchor ? '#' + chunk.anchor : '');
      const text = chunk.text.length > 190 ? chunk.text.slice(0, 190) + '…' : chunk.text;
      return `<li role="option"><a href="${href}">
        <span class="r-page">${escapeHtml(chunk.title)}</span>
        <span class="r-head">${highlight(chunk.heading, terms)}</span>
        <span class="r-text">${highlight(text, terms)}</span>
      </a></li>`;
    }).join('');
    results.querySelectorAll('a').forEach((a) => a.addEventListener('click', closeSearch));
    select(0);
  };

  const options = () => Array.from(results?.querySelectorAll('li[role="option"]') || []);

  const select = (i) => {
    const opts = options();
    if (!opts.length) return;
    selected = (i + opts.length) % opts.length;
    opts.forEach((li, k) => li.setAttribute('aria-selected', k === selected ? 'true' : 'false'));
    opts[selected].scrollIntoView({ block: 'nearest' });
  };

  input?.addEventListener('input', (e) => runSearch(e.target.value));
  input?.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); select(selected + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); select(selected - 1); }
    else if (e.key === 'Enter') {
      const opts = options();
      const target = opts[selected >= 0 ? selected : 0];
      if (target) { e.preventDefault(); target.querySelector('a').click(); }
    }
  });

  // ── global keyboard shortcuts ────────────────────────────────────────────
  let gPressed = false;
  let gTimer = null;

  const typing = (el) => el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable);

  document.addEventListener('keydown', (e) => {
    const mod = e.ctrlKey || e.metaKey;

    if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearch(); return; }
    if (e.key === 'Escape') {
      if (isOpen(searchOverlay)) { closeSearch(); input?.blur(); }
      closeHelp();
      return;
    }
    if (typing(e.target)) return;

    if (e.key === '/') { e.preventDefault(); openSearch(); return; }
    if (e.key === '?') { e.preventDefault(); openHelp(); return; }

    if (gPressed) {
      gPressed = false;
      clearTimeout(gTimer);
      const dest = PAGES[e.key.toLowerCase()];
      if (dest) { e.preventDefault(); location.href = dest; }
      return;
    }
    if (e.key.toLowerCase() === 'g') {
      gPressed = true;
      gTimer = setTimeout(() => { gPressed = false; }, 1200);
      return;
    }
    if (e.key.toLowerCase() === 't') { e.preventDefault(); toggleTheme(); }

    // j/k scroll like vim
    if (e.key === 'j') window.scrollBy({ top: 220, behavior: 'smooth' });
    if (e.key === 'k') window.scrollBy({ top: -220, behavior: 'smooth' });
  });

  // ── deep links on the search page: ?q=term ───────────────────────────────
  if (location.pathname.endsWith('search.html')) {
    const q = new URLSearchParams(location.search).get('q') || '';
    openSearch(q);
  }
})();

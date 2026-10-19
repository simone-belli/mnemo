// Mnemo front end. Plain script, no build step.
// Sections: 1. helpers · 2. Markdown + maths · 3. modes · 4. recall · 5. audio and highlighting
//           6. feedback · 7. next steps · 8. voice input · 9. onboarding quiz · 10. debug · 11. start-up
(() => {
  'use strict';

  // ======================================================================
  // 1. Helpers
  // ======================================================================
  const $ = (id) => document.getElementById(id);
  const store = {
    get(k) { try { return localStorage.getItem('mnemo.' + k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem('mnemo.' + k, v); } catch { /* private mode */ } }
  };
  const DEBUG = new URLSearchParams(location.search).get('debug') === '1';
  const SESSION = (() => {
    try {
      let id = sessionStorage.getItem('mnemo.session');
      if (!id) { id = Math.random().toString(36).slice(2); sessionStorage.setItem('mnemo.session', id); }
      return id;
    } catch { return Math.random().toString(36).slice(2); }
  })();

  async function api(path, body) {
    const res = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    if (!res.ok) {
      let msg = res.status + ' ' + res.statusText;
      try { msg = (await res.json()).detail || msg; } catch { /* not JSON */ }
      throw new Error(msg);
    }
    return res.json();
  }

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  let statusTimer = 0;
  function status(msg, { error = false, timer = false } = {}) {
    clearInterval(statusTimer);
    const el = $('status');
    el.classList.toggle('error', error);
    el.textContent = msg || '';
    if (timer) {
      const t0 = Date.now();
      statusTimer = setInterval(() => { el.textContent = `${msg} ${Math.round((Date.now() - t0) / 1000)}s`; }, 1000);
    }
  }

  // ======================================================================
  // 2. Markdown + maths: maths is pulled out before Markdown parsing (so `_` and `*`
  //    inside formulas survive), then typeset with KaTeX.
  // ======================================================================
  const MATH_RE = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^\s$](?:[^$\n]*?[^\s$])?)\$/g;

  function renderMarkdown(src) {
    const maths = [];
    const text = src.replace(MATH_RE, (m, a, b, c, d) => {
      maths.push({ tex: a ?? b ?? c ?? d, display: a != null || b != null });
      return `@@M${maths.length - 1}@@`;
    });
    let html = window.marked ? window.marked.parse(text) :
      text.split(/\n{2,}/).map((p) => `<p>${esc(p)}</p>`).join('');
    html = html.replace(/@@M(\d+)@@/g, (m, i) => {
      const { tex, display } = maths[+i];
      if (!window.katex) return `<code>${esc(tex)}</code>`;
      return window.katex.renderToString(tex, { displayMode: display, throwOnError: false });
    });
    return html;
  }

  // ======================================================================
  // 3. Modes: Read · Listen · Read + Listen (remembered)
  // ======================================================================
  let mode = ['read', 'listen', 'both'].includes(store.get('mode')) ? store.get('mode') : 'both';

  function setMode(m) {
    mode = m;
    store.set('mode', m);
    for (const b of document.querySelectorAll('.modes button')) b.setAttribute('aria-checked', String(b.dataset.mode === m));
    $('card').dataset.mode = m;
    if (m === 'read') { $('written').open = true; stopAudio(); }
    else if (current) { $('written').open = false; ensureAudio(); }
  }
  document.querySelectorAll('.modes button').forEach((b) => b.addEventListener('click', () => {
    unlockAudio();
    setMode(b.dataset.mode);
  }));

  // ======================================================================
  // 4. Recall
  // ======================================================================
  let current = null;          // the recap on screen (the /api/recall payload)
  let busy = false;

  async function recall(query, nodeId = null) {
    if (busy || !query.trim()) return;
    busy = true;
    $('go').disabled = true;
    stopAudio();
    status('Recalling…', { timer: true });
    try {
      const r = await api('/api/recall', { query, session_id: SESSION, node_id: nodeId, audio: mode !== 'read' });
      current = r;
      showRecap(r);
      status(r.audio_error && mode !== 'read' ? r.audio_error : '');
      if (mode !== 'read') startAudio(true);
    } catch (e) {
      status('Could not recall: ' + e.message, { error: true });
    } finally {
      busy = false;
      $('go').disabled = false;
    }
  }

  function showRecap(r) {
    $('onboarding').hidden = true;
    const crumbs = r.node_path.map((n) => `<span>${esc(n)}</span>`).join('<span class="sep">›</span>');
    $('crumbs').innerHTML = `${crumbs} <span class="chip">${esc(r.mode)}</span>`;
    $('recap').innerHTML = renderMarkdown(r.text_md);
    $('written').open = mode === 'read';
    buildScript(r.audio_script || '');
    $('card').hidden = false;
    $('card').dataset.mode = mode;
    for (const b of document.querySelectorAll('#feedback button')) b.classList.remove('done', 'busy');
    $('lost').hidden = true;
    $('fb-result').textContent = '';
    showNext(r.next_steps || []);
    showDebug(r.debug);
  }

  $('ask').addEventListener('submit', (e) => {
    e.preventDefault();
    unlockAudio();
    recall($('q').value);
  });

  // ======================================================================
  // 5. Audio and synced highlighting (ported from Read Along).
  //    ElevenLabs gives a start time per character; each word of the script is a span,
  //    and on every animation frame the word under audio.currentTime is lit up.
  //    Without an ElevenLabs key, the browser's own voice reads the script and its
  //    word-boundary events drive the same highlight.
  // ======================================================================
  const audio = new Audio();
  audio.preload = 'auto';
  let words = [];              // [{el, start, end}] character offsets into the script
  let wordStarts = [];
  let lit = -1, raf = 0, speaking = null;

  // A tiny silent WAV played inside a tap, so iOS lets us start the real audio later.
  const SILENT_WAV = 'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQQAAACAgICA';
  let unlocked = false;
  function unlockAudio() {
    if (unlocked) return;
    unlocked = true;
    try {
      audio.src = SILENT_WAV;
      const p = audio.play();
      if (p) p.then(() => { if (audio.src === SILENT_WAV) audio.pause(); }, () => {});
    } catch { /* ignore */ }
  }

  function buildScript(text) {
    const box = $('script');
    box.textContent = '';
    words = []; wordStarts = []; lit = -1;
    let offset = 0;
    for (const para of text.split(/(\n\s*\n)/)) {
      if (/^\n\s*\n$/.test(para)) { offset += para.length; continue; }
      const p = document.createElement('p');
      for (const tok of para.split(/(\s+)/)) {
        if (!tok) continue;
        if (/^\s+$/.test(tok)) { p.appendChild(document.createTextNode(tok)); offset += tok.length; continue; }
        const span = document.createElement('span');
        span.className = 'w';
        span.textContent = tok;
        span.dataset.i = words.length;
        words.push({ el: span, start: offset, end: offset + tok.length });
        wordStarts.push(offset);
        p.appendChild(span);
        offset += tok.length;
      }
      box.appendChild(p);
    }
  }

  function wordAt(charIndex) {               // last word starting at or before charIndex
    let lo = 0, hi = wordStarts.length - 1, ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (wordStarts[mid] <= charIndex) { ans = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return ans;
  }

  function charAtTime(t) {                    // last character starting at or before t
    const s = current && current.timings;
    if (!s || !s.length) return -1;
    let lo = 0, hi = s.length - 1, ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (s[mid] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return ans;
  }

  function light(wi) {
    if (wi === lit) return;
    const from = Math.min(lit, wi), to = Math.max(lit, wi);
    for (let i = Math.max(0, from); i <= to && i < words.length; i++) words[i].el.classList.toggle('read', i < wi);
    if (lit >= 0 && words[lit]) words[lit].el.classList.remove('now');
    if (wi >= 0 && words[wi]) {
      words[wi].el.classList.add('now');
      const r = words[wi].el.getBoundingClientRect();
      if (r.bottom > innerHeight - 40 || r.top < 0) words[wi].el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
    lit = wi;
  }

  function clearLight() {
    for (const w of words) w.el.classList.remove('read', 'now');
    lit = -1;
  }

  function tick() {
    raf = 0;
    if (audio.paused) return;
    const k = charAtTime(audio.currentTime);
    if (k >= 0) light(wordAt(k));
    updateTime();
    raf = requestAnimationFrame(tick);
  }

  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  function updateTime() {
    const d = audio.duration;
    if (isFinite(d) && d > 0) {
      $('seek').value = Math.round((audio.currentTime / d) * 1000);
      $('time').textContent = fmt(d - audio.currentTime);
    }
  }

  function setPlaying(on) { $('play').classList.toggle('playing', on); $('play').setAttribute('aria-label', on ? 'Pause' : 'Play'); }
  audio.addEventListener('play', () => { setPlaying(true); if (!raf) raf = requestAnimationFrame(tick); });
  audio.addEventListener('pause', () => setPlaying(false));
  audio.addEventListener('ended', () => { setPlaying(false); light(words.length); });
  audio.addEventListener('loadedmetadata', updateTime);

  async function ensureAudio() {
    if (!current || current.audio_script) return true;
    status('Preparing audio…', { timer: true });
    try {
      Object.assign(current, await api('/api/audio', { interaction_id: current.interaction_id }));
      buildScript(current.audio_script);
      status(current.audio_error || '');
      return true;
    } catch (e) {
      status('No audio: ' + e.message, { error: true });
      return false;
    }
  }

  async function startAudio(fromStart) {
    if (!(await ensureAudio())) return;
    if (current.audio_url) {
      if (!audio.src.endsWith(current.audio_url)) { audio.src = current.audio_url; clearLight(); }
      else if (fromStart) { audio.currentTime = 0; clearLight(); }
      try { await audio.play(); } catch { setPlaying(false); status('Tap play to listen.'); }
    } else {
      speak(fromStart ? 0 : Math.max(0, lit >= 0 && words[lit] ? words[lit].start : 0));
    }
  }

  function stopAudio() {
    audio.pause();
    if (speaking) { speaking = null; speechSynthesis.cancel(); }
    setPlaying(false);
  }

  // Browser voice fallback (no ElevenLabs key). Word timing comes from `onboundary`.
  function speak(fromChar) {
    if (!('speechSynthesis' in window) || !current) return;
    speechSynthesis.cancel();
    const text = current.audio_script.slice(fromChar);
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1;
    u.onboundary = (e) => { if (speaking === u) light(wordAt(fromChar + e.charIndex)); };
    u.onend = () => { if (speaking === u) { speaking = null; setPlaying(false); light(words.length); } };
    speaking = u;
    setPlaying(true);
    speechSynthesis.speak(u);
  }

  $('play').addEventListener('click', () => {
    unlockAudio();
    if (current && !current.audio_url && current.audio_script) {
      if (speaking) stopAudio(); else speak(lit >= 0 && lit < words.length ? words[lit].start : 0);
      return;
    }
    if (audio.paused) startAudio(false); else audio.pause();
  });
  $('again').addEventListener('click', () => {      // asking for a repeat is an implicit signal
    unlockAudio();
    startAudio(true);
    sendFeedback('repeat');
  });
  $('seek').addEventListener('input', () => {
    if (isFinite(audio.duration)) audio.currentTime = ($('seek').value / 1000) * audio.duration;
    const k = charAtTime(audio.currentTime);
    if (k >= 0) light(wordAt(k));
  });
  $('script').addEventListener('click', (e) => {     // tap a word to jump there
    const w = e.target.closest('.w');
    if (!w || !current) return;
    const word = words[+w.dataset.i];
    if (current.audio_url && current.timings) {
      audio.currentTime = current.timings[word.start] || 0;
      light(+w.dataset.i);
      if (audio.paused) audio.play().catch(() => {});
    } else if (current.audio_script) {
      speak(word.start);
    }
  });

  // ======================================================================
  // 6. Feedback: Deeper · Simpler · Knew that · Lost me (and repeat)
  // ======================================================================
  const SIGNAL_LABEL = { deeper: 'Deeper', simpler: 'Simpler', knew_that: 'Knew that', lost_me: 'Lost me', repeat: 'Repeat' };

  async function sendFeedback(signal, note) {
    if (!current) return;
    const btn = document.querySelector(`#feedback [data-signal="${signal}"]`);
    if (btn) btn.classList.add('busy');
    try {
      const r = await api('/api/feedback', { interaction_id: current.interaction_id, signal, note: note || null });
      if (btn) { btn.classList.remove('busy'); btn.classList.add('done'); }
      let msg = signal === 'repeat' ? '' : `Noted: ${SIGNAL_LABEL[signal].toLowerCase()}.`;
      if (r.diagnosis) {
        const d = r.diagnosis;
        msg = d.kind === 'missed_word' ? 'Noted: just a missed word.'
          : d.kind === 'genuine_gap' ? 'Noted: this one is a step too far for now.'
          : `Gap noted: ${d.term} (${d.kind.replace('_', ' ')}). The next recap will cover it first.`;
      }
      $('fb-result').textContent = msg;
      appendDebug('feedback: ' + signal, { deltas: r.deltas, diagnosis: r.diagnosis, depth_preference: r.depth_preference });
    } catch (e) {
      if (btn) btn.classList.remove('busy');
      $('fb-result').textContent = 'Could not send feedback: ' + e.message;
    }
  }

  document.querySelectorAll('#feedback button').forEach((b) => b.addEventListener('click', () => {
    if (b.dataset.signal === 'lost_me') {
      $('lost').hidden = false;
      $('lost-note').focus();
    } else {
      sendFeedback(b.dataset.signal);
    }
  }));
  $('lost').addEventListener('submit', (e) => {
    e.preventDefault();
    $('lost').hidden = true;
    sendFeedback('lost_me', $('lost-note').value.trim());
    $('lost-note').value = '';
  });

  // ======================================================================
  // 7. Next steps: tapping a chip runs a recall on it
  // ======================================================================
  function showNext(steps) {
    const box = $('chips');
    box.textContent = '';
    for (const s of steps) {
      const b = document.createElement('button');
      b.type = 'button';
      const parent = s.path.length > 1 ? s.path[s.path.length - 2] : '';
      b.innerHTML = `${esc(s.name)}${parent ? `<small>${esc(parent)}</small>` : ''}`;
      b.addEventListener('click', () => {
        unlockAudio();
        $('q').value = s.name;
        recall(`${s.name} (${s.path.join(' › ')})`, s.node_id);
        scrollTo({ top: 0, behavior: 'smooth' });
      });
      box.appendChild(b);
    }
    $('next').hidden = steps.length === 0;
  }

  // ======================================================================
  // 8. Voice input (Web Speech API); falls back to the text box.
  //    While a recap is on screen, "deeper", "simpler", "knew that", "lost me" and
  //    "repeat" are feedback, not queries.
  // ======================================================================
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const COMMANDS = [
    [/^(go )?deeper\.?$/i, 'deeper'], [/^simpler\.?$/i, 'simpler'], [/^(i )?knew that\.?$/i, 'knew_that'],
    [/^(you )?lost me\.?$/i, 'lost_me'], [/^(repeat|again|say (that|it) again)\.?$/i, 'repeat']
  ];
  if (!Recognition) {
    $('mic').hidden = true;
  } else {
    let rec = null;
    $('mic').addEventListener('click', () => {
      unlockAudio();
      if (rec) { rec.stop(); return; }
      rec = new Recognition();
      rec.lang = 'en-GB';
      rec.interimResults = true;
      rec.onresult = (e) => {
        const text = Array.from(e.results).map((r) => r[0].transcript).join('');
        $('q').value = text;
        if (e.results[e.results.length - 1].isFinal) handleSpoken(text.trim());
      };
      rec.onend = () => { rec = null; $('mic').classList.remove('listening'); };
      rec.onerror = (e) => status(e.error === 'not-allowed' ? 'Microphone access was refused.' : '', { error: e.error === 'not-allowed' });
      $('mic').classList.add('listening');
      stopAudio();
      rec.start();
    });
  }

  function handleSpoken(text) {
    if (current) {
      for (const [re, signal] of COMMANDS) {
        if (!re.test(text)) continue;
        $('q').value = '';
        if (signal === 'repeat') { startAudio(true); sendFeedback('repeat'); }
        else sendFeedback(signal);
        return;
      }
    }
    recall(text);
  }

  // ======================================================================
  // 9. Onboarding: self-rating, then the adaptive quiz
  // ======================================================================
  let quizItem = null;

  function showOnboarding(roots) {
    $('card').hidden = true;
    $('next').hidden = true;
    $('onboarding').hidden = false;
    $('rate-step').hidden = false;
    $('quiz-step').hidden = true;
    const box = $('ratings');
    box.textContent = '';
    for (const r of roots) {
      const row = document.createElement('div');
      row.className = 'rating';
      row.innerHTML = `<label for="r-${esc(r.name)}">${esc(r.name)}</label><output>3</output>
        <input type="range" min="1" max="5" step="1" value="3" id="r-${esc(r.name)}" data-root="${esc(r.name)}">
        <small>${esc(r.description)}</small>`;
      const input = row.querySelector('input');
      input.addEventListener('input', () => { row.querySelector('output').textContent = input.value; });
      box.appendChild(row);
    }
  }

  $('rate-go').addEventListener('click', async () => {
    const ratings = {};
    for (const i of document.querySelectorAll('#ratings input')) ratings[i.dataset.root] = +i.value;
    try { await api('/api/quiz/rating', { ratings }); } catch (e) { status(e.message, { error: true }); }
    startQuiz();
  });
  $('rate-skip').addEventListener('click', startQuiz);
  $('quiz-stop').addEventListener('click', finishOnboarding);
  $('recalibrate').addEventListener('click', async () => {
    stopAudio();
    const st = await api('/api/state');
    showOnboarding(st.roots);
    scrollTo({ top: 0 });
  });

  async function startQuiz() {
    $('rate-step').hidden = true;
    $('quiz-step').hidden = false;
    $('quiz-q').textContent = '';
    status('Writing a question…', { timer: true });
    try { showItem(await api('/api/quiz/next')); status(''); }
    catch (e) { status('Quiz unavailable: ' + e.message, { error: true }); }
  }

  function showItem(item) {
    quizItem = item;
    $('quiz-bar').style.width = `${Math.min(100, (100 * item.progress.answered) / item.progress.target)}%`;
    $('quiz-topic').innerHTML = item.topic.map(esc).join('<span class="sep">›</span>') + ` <span class="chip">${esc(item.mode)}</span>`;
    $('quiz-q').textContent = item.question;
    $('quiz-result').textContent = '';
    $('quiz-next').hidden = true;
    const opts = $('quiz-options');
    opts.textContent = '';
    $('quiz-free').hidden = !!item.options;
    if (item.options) {
      item.options.forEach((o, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = o;
        b.addEventListener('click', () => answer(String(i), b));
        opts.appendChild(b);
      });
    } else {
      $('quiz-text').value = '';
      $('quiz-text').focus();
    }
  }

  let nextItem = null;
  async function answer(value, btn) {
    if (!quizItem) return;
    const item = quizItem;
    quizItem = null;
    for (const b of $('quiz-options').querySelectorAll('button')) b.disabled = true;
    status('Grading…', { timer: true });
    try {
      const r = await api('/api/quiz/answer', { item_id: item.item_id, answer: value });
      status('');
      if (btn) btn.classList.add(r.score >= 1 ? 'right' : 'wrong');
      if (item.options) {
        for (const b of $('quiz-options').querySelectorAll('button')) if (b.textContent === r.correct) b.classList.add('right');
      }
      $('quiz-result').textContent = r.score >= 1 ? 'Right.' : item.options ? r.rationale : `${Math.round(r.score * 100)}%: ${r.rationale}`;
      appendDebug('quiz answer', r.deltas);
      nextItem = r.next;
      if (nextItem) $('quiz-next').hidden = false;
      else { $('quiz-result').textContent += ' That was the last question. Mnemo is calibrated.'; store.set('onboarded', '1'); }
      if (nextItem) $('quiz-bar').style.width = `${Math.min(100, (100 * nextItem.progress.answered) / nextItem.progress.target)}%`;
    } catch (e) {
      quizItem = item;
      status('Could not grade: ' + e.message, { error: true });
    }
  }
  $('quiz-free').addEventListener('submit', (e) => { e.preventDefault(); answer($('quiz-text').value); });
  $('quiz-next').addEventListener('click', () => { if (nextItem) showItem(nextItem); });

  function finishOnboarding() {
    store.set('onboarded', '1');
    $('onboarding').hidden = true;
    status('');
    if (current) $('card').hidden = false;
    $('q').focus();
  }

  // ======================================================================
  // 10. Debug drawer (?debug=1): tag distribution, profile, prompts, mastery deltas
  // ======================================================================
  function showDebug(d) {
    if (!DEBUG || !d) return;
    $('debug').hidden = false;
    const rows = d.tag_dist.map((t) => `<tr><td>${t.p.toFixed(3)}</td><td>${esc(t.name)}</td><td>${esc(t.mode)}</td></tr>`).join('');
    $('debug-body').innerHTML = `
      <h4>Tag distribution</h4><table>${rows}</table>
      <h4>Tagging</h4><pre>${esc(JSON.stringify(d.tagging, null, 1))}</pre>
      <h4>Created nodes</h4><pre>${esc(JSON.stringify(d.created_nodes))}</pre>
      <h4>Profile sent to the model</h4><pre>${esc(JSON.stringify(d.profile, null, 1))}</pre>
      <h4>Prompt</h4><pre>${esc(d.user_prompt)}</pre>
      <h4>Mastery deltas</h4><pre id="debug-deltas">${esc(JSON.stringify(d.deltas, null, 1))}</pre>`;
  }

  function appendDebug(title, data) {
    if (!DEBUG) return;
    $('debug').hidden = false;
    $('debug-body').insertAdjacentHTML('beforeend', `<h4>${esc(title)}</h4><pre>${esc(JSON.stringify(data, null, 1))}</pre>`);
  }

  // ======================================================================
  // 11. Start-up
  // ======================================================================
  setMode(mode);
  (async () => {
    try {
      const st = await api('/api/state');
      const fresh = st.recalls === 0 && st.quiz_answered === 0 && !st.rated && !store.get('onboarded');
      if (fresh) showOnboarding(st.roots);
    } catch (e) {
      status('Mnemo’s server is not answering: ' + e.message, { error: true });
    }
  })();
  // test hook
  window.mnemo = { get current() { return current; }, audio, recall, sendFeedback };
})();

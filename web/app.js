'use strict';

/* ------------------------------------------------------------------ helpers */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
};
const fmtTime = (s) => {
  if (!isFinite(s)) return '0:00';
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

async function apiGet(path) {
  const res = await fetch(path);
  if (res.status === 401) { location.href = '/login'; throw new Error('Please log in'); }
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

const fileUrl = (jobId, name, download = false) =>
  `/api/jobs/${jobId}/files/${name}${download ? '?download=1' : ''}`;

/* ------------------------------------------------------------------ music theory (mirrors app/theory.py) */
const SHARPS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const FLATS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
const MAJOR_FIFTHS = { 0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5 };
const mod12 = (n) => ((n % 12) + 12) % 12;
const keyUsesFlats = (tonic, mode) => MAJOR_FIFTHS[mode === 'major' ? mod12(tonic) : mod12(tonic + 3)] < 0;
const pcName = (pc, flats) => (flats ? FLATS : SHARPS)[mod12(pc)];
const NASHVILLE = { 0: '1', 1: 'b2', 2: '2', 3: 'b3', 4: '3', 5: '4', 6: '#4', 7: '5', 8: 'b6', 9: '6', 10: 'b7', 11: '7' };

function chordLabel(c, shift, keyTonic, keyMode, nashville) {
  if (nashville) {
    let s = NASHVILLE[mod12(c.root - keyTonic)] + c.quality;
    if (c.bass != null) s += '/' + NASHVILLE[mod12(c.bass - keyTonic)];
    return s;
  }
  const flats = keyUsesFlats(keyTonic + shift, keyMode);
  let s = pcName(c.root + shift, flats) + c.quality;
  if (c.bass != null) s += '/' + pcName(c.bass + shift, flats);
  return s;
}

// Chord shapes that are easy in open position on guitar, used for capo tips.
const OPEN_SHAPES = new Set(['C', 'D', 'E', 'G', 'A', 'Am', 'Em', 'Dm', 'A7', 'B7', 'C7', 'D7', 'E7', 'G7',
  'Am7', 'Em7', 'Dm7', 'Cmaj7', 'Fmaj7', 'Dmaj7', 'Amaj7', 'Gmaj7', 'Dsus4', 'Asus4', 'Esus4', 'Dsus2', 'Asus2']);

function capoTip(segments, shift) {
  if (!segments.length) return '';
  const weight = (capo) => segments.reduce((acc, c) =>
    acc + (OPEN_SHAPES.has(pcName(c.root + shift - capo, false) + c.quality) ? 1 : 0), 0) / segments.length;
  let best = 0, bestScore = weight(0);
  for (let capo = 1; capo <= 7; capo++) {
    const sc = weight(capo);
    if (sc > bestScore + 0.15) { best = capo; bestScore = sc; }
  }
  if (best === 0) return null;
  const shapes = [...new Set(segments.map((c) => pcName(c.root + shift - best, false) + c.quality))].slice(0, 6);
  return { capo: best, text: `🎸 Easier on guitar: capo ${best}, play as ${shapes.join(', ')}` };
}

/* ------------------------------------------------------------------ router */
const view = $('#view');
let cleanups = [];
function onLeave(fn) { cleanups.push(fn); }

function route() {
  cleanups.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
  cleanups = [];
  view.replaceChildren();
  const m = location.hash.match(/^#\/song\/([a-z0-9]+)/i);
  if (m) showSong(m[1]); else showLibrary();
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', route);

async function loadSystem() {
  try {
    const info = await apiGet('/api/system');
    const badge = $('#device-badge');
    badge.hidden = false;
    if (info.device === 'cuda') { badge.textContent = `GPU: ${info.gpu || 'CUDA'}`; badge.classList.add('gpu'); }
    else if (info.device === 'mps') { badge.textContent = 'GPU: Apple'; badge.classList.add('gpu'); }
    else badge.textContent = 'CPU mode (slower)';
  } catch (e) { /* server not ready */ }
}

/* ------------------------------------------------------------------ library */
function showLibrary() {
  view.append($('#tpl-library').content.cloneNode(true));
  const form = $('#add-form'), urlInput = $('#url-input'), fileInput = $('#file-input');
  const drop = $('#drop-zone'), errBox = $('#add-error');

  async function submit(file) {
    errBox.hidden = true;
    const data = new FormData();
    if (file) data.append('file', file);
    else data.append('url', urlInput.value.trim());
    const btn = form.querySelector('button[type=submit]');
    btn.disabled = true;
    try {
      const res = await fetch('/api/jobs', { method: 'POST', body: data });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || 'Could not add the song');
      location.hash = `#/song/${body.id}`;
    } catch (e) {
      errBox.textContent = e.message;
      errBox.hidden = false;
    } finally { btn.disabled = false; }
  }

  form.addEventListener('submit', (e) => { e.preventDefault(); if (urlInput.value.trim()) submit(null); else urlInput.focus(); });
  fileInput.addEventListener('change', () => fileInput.files[0] && submit(fileInput.files[0]));
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault(); drop.classList.remove('over');
    if (e.dataTransfer.files[0]) submit(e.dataTransfer.files[0]);
  });

  let timer = null;
  async function refresh() {
    let jobs = [];
    try { jobs = await apiGet('/api/jobs'); } catch (e) { /* retry next tick */ }
    renderList(jobs);
    const busy = jobs.some((j) => j.status === 'queued' || j.status === 'running');
    timer = setTimeout(refresh, busy ? 1500 : 6000);
  }
  onLeave(() => clearTimeout(timer));
  refresh();
}

function renderList(jobs) {
  const list = $('#song-list');
  if (!list) return;
  $('#empty-list').hidden = jobs.length > 0;
  list.replaceChildren(...jobs.map((j) => {
    let status;
    if (j.status === 'done') status = el('span', { class: 'status done' }, '✓ Ready');
    else if (j.status === 'error') status = el('span', { class: 'status error' }, 'Failed');
    else if (j.status === 'queued') status = el('span', { class: 'status' }, 'Waiting…');
    else status = el('span', { class: 'status' },
      el('div', { class: 'mini-progress' }, el('div', { style: `width:${Math.round(j.progress * 100)}%` })));
    const del = el('button', {
      class: 'del-btn', title: 'Delete', 'aria-label': 'Delete song',
      onclick: async () => {
        if (!confirm(`Delete "${j.title}" and all its files?`)) return;
        const res = await fetch(`/api/jobs/${j.id}`, { method: 'DELETE' });
        if (!res.ok) alert((await res.json()).detail);
        else location.reload();
      },
    }, '🗑');
    if (j.status === 'running') del.disabled = true;
    return el('li', { class: 'song-item' },
      el('a', { class: 'title', href: `#/song/${j.id}` }, j.title || 'Untitled'), status, del);
  }));
}

/* ------------------------------------------------------------------ song page */
async function showSong(id) {
  view.append($('#tpl-song').content.cloneNode(true));
  let timer = null, built = false;
  onLeave(() => clearTimeout(timer));

  async function poll() {
    let job;
    try { job = await apiGet(`/api/jobs/${id}`); } catch (e) {
      $('#song-title').textContent = 'Song not found';
      return;
    }
    $('#song-title').textContent = job.title || 'Untitled';
    document.title = `${job.title} – Music App`;
    if (job.status === 'done') {
      $('#song-progress').hidden = true;
      if (!built) { built = true; buildSong(job); }
      return;
    }
    if (job.status === 'error') {
      $('#song-progress').hidden = true;
      const box = $('#song-error');
      box.hidden = false;
      box.textContent = `Something went wrong:\n${job.error}`;
      return;
    }
    $('#song-progress').hidden = false;
    $('#progress-msg').textContent = job.status === 'queued' ? 'Waiting for another song to finish…' : job.message;
    $('#progress-pct').textContent = `${Math.round(job.progress * 100)}%`;
    $('#progress-bar').style.width = `${job.progress * 100}%`;
    timer = setTimeout(poll, 1000);
  }
  poll();
}

function buildSong(job) {
  const r = job.result;
  const meta = $('#song-meta');
  const pills = [`Key: ${r.key}`, `${Math.round(r.tempo)} BPM`, r.time_signature, fmtTime(r.duration)];
  const found = r.instruments.filter((i) => i.present && i.stem !== 'vocals').map((i) => i.label.split(' (')[0]);
  if (r.instruments.find((i) => i.stem === 'vocals' && i.present)) found.unshift('Vocals');
  pills.push(`Heard: ${found.join(', ') || '—'}`);
  meta.replaceChildren(...pills.map((p) => el('span', { class: 'pill' }, p)));
  $('#song-body').hidden = false;

  const mixer = new Mixer(job);
  onLeave(() => mixer.destroy());
  const chords = new ChordView(job, mixer);
  const sheet = new SheetView(job, mixer);
  onLeave(() => sheet.destroy());

  $$('.tab').forEach((tab) => tab.addEventListener('click', () => {
    $$('.tab').forEach((t) => t.classList.toggle('active', t === tab));
    $$('.tab-panel').forEach((p) => { p.hidden = p.dataset.panel !== tab.dataset.tab; });
    if (tab.dataset.tab === 'sheet') sheet.open();
    if (tab.dataset.tab === 'chords') chords.scrollToNow();
  }));

  // downloads
  const dl = (name, label, main = false) =>
    el('a', { href: fileUrl(job.id, name, true), class: main ? 'main' : '', download: '' }, label);
  $('#minus-downloads').append(
    dl(r.files.minus_mp3, '⬇ Minus track (MP3)', true),
    dl(r.files.minus_wav, '⬇ Minus track (WAV)'),
    dl(r.files.vocals, '⬇ Vocals only (MP3)'),
    dl(r.files.original, '⬇ Original (MP3)'),
  );
  $('#sheet-downloads').append(
    dl(r.files.musicxml, '⬇ MusicXML (MuseScore, Guitar Pro…)'),
    dl(r.files.midi, '⬇ MIDI (all instruments)'),
  );
}

/* ------------------------------------------------------------------ stem mixer + transport */
const STEM_ORDER = ['vocals', 'guitar', 'piano', 'other', 'bass', 'drums'];

class Mixer {
  constructor(job) {
    this.job = job;
    this.listeners = [];
    this.loopA = null; this.loopB = null;
    const r = job.result;
    this.stems = STEM_ORDER.map((key) => r.instruments.find((i) => i.stem === key)).filter(Boolean).map((inst) => {
      const audio = new Audio(fileUrl(job.id, inst.file));
      audio.preload = 'auto';
      audio.preservesPitch = true;
      return { ...inst, audio, volume: 1, mute: inst.stem === 'vocals', solo: false };
    });
    // the finished mp3s (the minus track uses the cleaner vocal model than the stems), played in sync with the stems
    this.files = [['minus', r.files.minus_mp3, 'Minus track'], ['vocals_only', r.files.vocals, 'Vocals only'],
      ['original', r.files.original, 'Original']].map(([key, file, label]) => ({
      key, label,
      audio: Object.assign(new Audio(fileUrl(job.id, file)), { preload: key === 'minus' ? 'auto' : 'metadata', preservesPitch: true }),
    }));
    this.all = [...this.stems, ...this.files];
    this.source = null; // key of the file being heard, or null for the stem mix
    this.duration = r.duration;
    for (const s of this.all) s.audio.addEventListener('ended', () => { if (s === this.master) this.pause(); });
    this.buildUI();
    this.applyGains();
    this.raf = requestAnimationFrame(() => this.tick());
    this.lastSync = 0;
    this.keyHandler = (e) => {
      if (e.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement.tagName)) {
        e.preventDefault(); this.toggle();
      }
    };
    document.addEventListener('keydown', this.keyHandler);
  }

  buildUI() {
    const mixer = $('#mixer');
    for (const s of this.stems) {
      const playBtn = el('button', { class: 'play', title: 'Play only this part', 'aria-label': `Play only ${s.label}` }, '▶');
      const muteBtn = el('button', { class: 'mute', title: 'Mute' }, 'M');
      const soloBtn = el('button', { class: 'solo', title: 'Solo' }, 'S');
      const vol = el('input', { type: 'range', min: 0, max: 1, step: 0.01, value: 1, 'aria-label': `${s.label} volume` });
      playBtn.onclick = () => this.playOnly(s);
      muteBtn.onclick = () => { s.mute = !s.mute; this.clearPreset(); this.applyGains(); };
      soloBtn.onclick = () => { s.solo = !s.solo; this.clearPreset(); this.applyGains(); };
      vol.oninput = () => { s.volume = Number(vol.value); if (this.source) this.clearPreset(); this.applyGains(); };
      s.ui = { playBtn, muteBtn, soloBtn, vol };
      const note = s.present ? `${s.level_db} dB` : 'barely present';
      mixer.append(el('div', { class: `stem-row${s.present ? '' : ' absent'}` },
        el('span', { class: 'name' }, s.label, el('small', {}, note)), playBtn, muteBtn, soloBtn, vol,
        el('a', { href: fileUrl(this.job.id, s.file, true), download: '', title: 'Download this stem (FLAC)' }, '⬇')));
    }
    const presetOf = { minus: 'minus', vocals_only: 'vocals', original: 'all' };
    for (const f of this.files) {
      f.btn = el('button', { class: f.key === 'minus' ? 'primary' : '' }, `▶ ${f.label}`);
      f.btn.onclick = () => {
        if (this.playing && this.source === f.key) return this.pause();
        this.preset(presetOf[f.key]);
        if (!this.playing) this.play();
      };
    }
    $('#minus-play').append(...this.files.map((f) => f.btn));
    $$('[data-preset]').forEach((b) => b.addEventListener('click', () => this.preset(b.dataset.preset)));
    this.preset('minus');

    const play = $('#tp-play'), seek = $('#tp-seek');
    play.onclick = () => this.toggle();
    seek.oninput = () => this.seek((seek.value / 1000) * this.duration);
    $('#tp-dur').textContent = fmtTime(this.duration);
    $('#tp-speed').onchange = (e) => this.setSpeed(Number(e.target.value));
    $('#tp-loop').onclick = () => this.loopClick();
  }

  preset(name) {
    for (const s of this.stems) {
      s.solo = false;
      s.mute = name === 'minus' ? s.stem === 'vocals' : name === 'vocals' ? s.stem !== 'vocals' : false;
    }
    this.setSource({ minus: 'minus', vocals: 'vocals_only', all: 'original' }[name] || null);
    this.applyGains();
    this.markPreset(name);
  }
  markPreset(name) { $$('[data-preset]').forEach((b) => b.classList.toggle('on', b.dataset.preset === name)); }
  clearPreset() { this.setSource(null); this.markPreset(null); } // back to the stem mix

  // Only what is heard plays: a muted element kept running still gets re-seeked to stay in sync,
  // and each re-seek is an audible skip in whatever it is synced to (and a wasted download).
  setSource(src) {
    if (src === this.source) return;
    const t = this.time, wasPlaying = this.playing;
    this.source = src;
    for (const s of this.all) if (!this.active.includes(s)) s.audio.pause();
    for (const s of this.active) s.audio.currentTime = t;
    if (wasPlaying) this.play();
  }
  get active() { return this.source ? [this.files.find((f) => f.key === this.source)] : this.stems; }
  get master() { return this.active.find((s) => s.present !== false) || this.active[0]; }

  playOnly(s) {
    const already = this.playing && !this.source && this.stems.every((x) => x.solo === (x === s));
    if (already) return this.pause();
    for (const x of this.stems) x.solo = x === s;
    this.clearPreset();
    this.applyGains();
    if (!this.playing) this.play();
  }

  applyGains() {
    const anySolo = this.stems.some((s) => s.solo);
    for (const s of this.stems) {
      const audible = !this.source && (anySolo ? s.solo : !s.mute);
      s.audio.volume = s.volume;
      s.audio.muted = !audible; // iPhones ignore .volume set by a page, but respect .muted
      s.ui.muteBtn.classList.toggle('on', s.mute);
      s.ui.soloBtn.classList.toggle('on', s.solo);
    }
    for (const f of this.files) f.audio.muted = f.key !== this.source;
    this.updatePlayButtons();
  }

  updatePlayButtons() {
    const solos = this.stems.filter((s) => s.solo);
    for (const s of this.stems) {
      s.ui.playBtn.textContent = this.playing && !this.source && solos.length === 1 && solos[0] === s ? '❚❚' : '▶';
    }
    for (const f of this.files) f.btn.textContent = `${this.playing && this.source === f.key ? '❚❚' : '▶'} ${f.label}`;
  }

  get time() { return this.master.audio.currentTime; }
  get playing() { return !this.master.audio.paused; }

  async play() {
    this.onPlay && this.onPlay();
    const t = this.time;
    for (const s of this.active) s.audio.currentTime = t;
    await Promise.all(this.active.map((s) => s.audio.play().catch(() => {})));
    $('#tp-play').textContent = '❚❚';
    this.updatePlayButtons();
  }
  pause() {
    for (const s of this.all) s.audio.pause();
    $('#tp-play').textContent = '▶';
    this.updatePlayButtons();
  }
  toggle() { this.playing ? this.pause() : this.play(); }
  seek(t) {
    t = Math.max(0, Math.min(this.duration, t));
    for (const s of this.active) s.audio.currentTime = t;
  }
  setSpeed(rate) {
    for (const s of this.all) { s.audio.playbackRate = rate; s.audio.preservesPitch = true; }
    this.speedListeners?.forEach((fn) => fn(rate));
  }
  onSpeed(fn) { (this.speedListeners ||= []).push(fn); }

  loopClick() {
    const btn = $('#tp-loop');
    if (this.loopA === null) { this.loopA = this.time; btn.textContent = `A ${fmtTime(this.loopA)} → set B`; btn.classList.add('on'); }
    else if (this.loopB === null) {
      if (this.time <= this.loopA + 0.5) return;
      this.loopB = this.time; btn.textContent = `Loop ${fmtTime(this.loopA)}–${fmtTime(this.loopB)} ✕`;
      this.seek(this.loopA);
    } else { this.loopA = this.loopB = null; btn.textContent = 'Loop A-B'; btn.classList.remove('on'); }
  }

  onTime(fn) { this.listeners.push(fn); }

  tick() {
    const t = this.time;
    if (this.loopB !== null && t >= this.loopB) this.seek(this.loopA);
    // keep all stems locked to the master clock
    const now = performance.now();
    if (this.playing && now - this.lastSync > 500) {
      this.lastSync = now;
      for (const s of this.active) {
        if (s !== this.master && Math.abs(s.audio.currentTime - t) > 0.04) s.audio.currentTime = t;
      }
    }
    const seek = $('#tp-seek');
    if (seek && document.activeElement !== seek) seek.value = String(Math.round((t / this.duration) * 1000));
    const timeEl = $('#tp-time');
    if (timeEl) timeEl.textContent = fmtTime(t);
    this.listeners.forEach((fn) => fn(t));
    this.raf = requestAnimationFrame(() => this.tick());
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    document.removeEventListener('keydown', this.keyHandler);
    for (const s of this.all) { s.audio.pause(); s.audio.src = ''; }
  }
}

/* ------------------------------------------------------------------ chord chart */
class ChordView {
  constructor(job, mixer) {
    this.r = job.result;
    this.mixer = mixer;
    this.shift = 0;
    this.capo = 0; // with a capo the chart shows the shapes to play; the song still sounds in the same key
    this.nashville = false;
    this.nowBar = -1;
    $('#tr-down').onclick = () => { this.shift = Math.max(-11, this.shift - 1); this.render(); };
    $('#tr-up').onclick = () => { this.shift = Math.min(11, this.shift + 1); this.render(); };
    $('#capo').onchange = (e) => { this.capo = Number(e.target.value); this.render(); };
    $('#show-nashville').onchange = (e) => { this.nashville = e.target.checked; this.render(); };
    this.render();
    mixer.onTime((t) => this.follow(t));
  }

  label(c) { return chordLabel(c, this.shift - this.capo, this.r.key_tonic, this.r.key_mode, this.nashville); }

  render() {
    const r = this.r;
    const name = (pc) => `${pcName(pc, keyUsesFlats(pc, r.key_mode))} ${r.key_mode}`;
    const tonic = r.key_tonic + this.shift;
    $('#tr-value').textContent = this.shift > 0 ? `+${this.shift}` : String(this.shift);
    $('#capo').value = String(this.capo);
    $('#tr-key').textContent = this.capo ? `${name(tonic)} (playing ${name(tonic - this.capo)} shapes)` : name(tonic);
    const tip = this.nashville || this.capo ? null : capoTip(r.chords.segments, this.shift);
    $('#capo-tip').replaceChildren(...(tip ? [tip.text, ' ', el('button', { class: 'chip', onclick: () => { this.capo = tip.capo; this.render(); } }, `Use capo ${tip.capo}`)] : []));

    const seen = new Map();
    for (const c of r.chords.segments) { const l = this.label(c); if (!seen.has(l)) seen.set(l, c); }
    $('#chord-summary').replaceChildren(...[...seen.keys()].map((l) => el('span', {}, l)));

    // drop silent bars at the very start/end
    const bars = r.chords.bars;
    let first = bars.findIndex((b) => b.chords.length);
    let last = bars.length - 1 - [...bars].reverse().findIndex((b) => b.chords.length);
    if (first < 0) { first = 0; last = bars.length - 1; }
    this.barEls = [];
    const chart = $('#chord-chart');
    chart.replaceChildren(...bars.slice(first, last + 1).map((b) => {
      const div = el('div', { class: 'bar', onclick: () => { this.mixer.seek(b.time); } },
        el('span', { class: 'num' }, String(b.bar)),
        ...b.chords.map((c) => el('span', { class: `ch slot${c.beat}${c.continued ? ' cont' : ''}` }, this.label(c))));
      this.barEls.push({ el: div, time: b.time });
      return div;
    }));
    this.nowBar = -1;
  }

  follow(t) {
    // current bar highlight
    let idx = -1;
    for (let i = 0; i < this.barEls.length; i++) { if (this.barEls[i].time <= t + 0.05) idx = i; else break; }
    if (idx !== this.nowBar) {
      if (this.barEls[this.nowBar]) this.barEls[this.nowBar].el.classList.remove('now');
      if (this.barEls[idx]) {
        this.barEls[idx].el.classList.add('now');
        if (this.mixer.playing && !$('[data-panel=chords]').hidden) {
          this.barEls[idx].el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
      }
      this.nowBar = idx;
    }
    // current chord in the transport bar
    const segs = this.r.chords.segments;
    let cur = null;
    for (const s of segs) { if (s.start <= t + 0.05) cur = s; else break; }
    const box = $('#tp-current-chord');
    if (box) box.textContent = cur ? this.label(cur) : '';
  }

  scrollToNow() { if (this.barEls[this.nowBar]) this.barEls[this.nowBar].el.scrollIntoView({ block: 'nearest' }); }
}

/* ------------------------------------------------------------------ sheet music (alphaTab) */
class SheetView {
  constructor(job, mixer) {
    this.job = job;
    this.mixer = mixer;
    this.api = null;
  }

  open() {
    if (this.api) return;
    if (!window.alphaTab) { $('#at-status').textContent = 'The sheet music viewer failed to load.'; return; }
    const at = window.alphaTab;
    const wrap = $('#at-wrap');
    this.api = new at.AlphaTabApi($('#alphatab'), {
      core: {
        file: fileUrl(this.job.id, this.job.result.files.musicxml),
        fontDirectory: '/static/vendor/alphatab/font/',
        tracks: 'all',
      },
      display: { layoutMode: at.LayoutMode.Page, scale: 1 },
      notation: { elements: { scoreTitle: true, scoreWordsAndMusic: false, effectTempo: true } },
      player: {
        playerMode: at.PlayerMode.EnabledSynthesizer,
        soundFont: '/static/vendor/alphatab/soundfont/sonivox.sf3',
        scrollElement: wrap,
        enableCursor: true,
        enableAnimatedBeatCursor: true,
      },
    });
    const api = this.api;
    const status = $('#at-status');
    api.scoreLoaded.on((score) => {
      this.tracks = score.tracks;
      this.selected = new Set(score.tracks.map((t) => t.index));
      this.buildToggles();
    });
    api.renderStarted.on(() => { status.hidden = false; status.textContent = 'Drawing sheet music…'; });
    api.renderFinished.on(() => { status.hidden = true; });
    api.error.on((e) => { status.hidden = false; status.textContent = `Could not show the sheet music: ${e.message || e}`; });
    api.playerReady.on(() => { $('#at-play').disabled = false; });
    api.playerStateChanged.on((e) => { $('#at-play').textContent = e.state === 1 ? '❚❚ Pause score' : '▶ Play score'; });

    $('#at-play').onclick = () => { this.mixer.pause(); api.playPause(); };
    this.mixer.onPlay = () => { if (api.playerState === 1) api.pause(); };
    this.mixer.onSpeed((rate) => { api.playbackSpeed = rate; });
    $('#at-zoom').onchange = (e) => { api.settings.display.scale = Number(e.target.value); api.updateSettings(); api.render(); };
    $('#at-layout').onchange = (e) => {
      api.settings.display.layoutMode = at.LayoutMode[e.target.value];
      api.updateSettings(); api.render();
    };
    $('#at-print').onclick = () => api.print();
  }

  buildToggles() {
    const box = $('#track-toggles');
    box.replaceChildren(...this.tracks.map((t) => {
      const chip = el('button', { class: 'chip on', title: 'Show and play this part' }, t.name);
      chip.onclick = () => {
        if (this.selected.has(t.index)) {
          if (this.selected.size === 1) return; // keep at least one
          this.selected.delete(t.index);
        } else this.selected.add(t.index);
        chip.classList.toggle('on', this.selected.has(t.index));
        this.api.changeTrackMute([t], !this.selected.has(t.index)); // hidden parts are silent too
        this.api.renderTracks(this.tracks.filter((x) => this.selected.has(x.index)));
      };
      return chip;
    }));
  }

  destroy() { if (this.api) { this.api.destroy(); this.api = null; } }
}

/* ------------------------------------------------------------------ start */
loadSystem();
route();

"use strict";

const byId = (id) => document.getElementById(id);
const picker = byId("case-select");
const status = byId("status");
const retry = byId("retry");
const playButton = byId("play");
const stopButton = byId("stop");
const seek = byId("seek");
const overlapButton = byId("jump-overlap");
const soloButton = byId("solo");
const otherLevel = byId("other-level");
const otherValue = byId("other-value");
const trackArea = byId("track-area");
const mixLane = byId("mix-lane");
const cursorArea = trackArea.querySelector(".cursor-area");
const laneTemplate = byId("track-lane");
const embeddedManifest = byId("embedded-manifest");
const AudioContextClass = window.AudioContext || window.webkitAudioContext;
const OfflineAudioContextClass = window.OfflineAudioContext || window.webkitOfflineAudioContext;
const seekStep = Number(seek.step);

let manifestUrl;
let modelName;
let cases = [];
let currentCase = null;
let request;
let audioContext;
let decodingContext;
let buffers = [];
let duration = 0;
let position = 0;
let headroom = 1;
let waveformScale = 1;
let playback = null;
let playRequest = 0;
let loadingPlayback = false;
let animationFrame = 0;
let waveformFrame = 0;
let resumeAfterScrub = false;
let activeScrub = null;
let mode = "mix";
let focus = 0;
let solo = false;
let lanes = [mixLane];
let canvases = [mixLane.querySelector(".waveform")];
let trackButtons = [];

function setStatus(message, error = false) {
  status.textContent = message;
  status.classList.toggle("error", error);
  retry.hidden = !error;
}

async function checkedFetch(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response;
}

function caseName(item) {
  return item.label || item.pair_label || item.id;
}

function setCaseOptions() {
  const groups = new Map();
  for (const item of cases) {
    const name = item.pair_label || "样本";
    if (!groups.has(name)) {
      const group = document.createElement("optgroup");
      group.label = name;
      groups.set(name, group);
    }
    groups.get(name).append(new Option(caseName(item), item.id));
  }
  picker.replaceChildren(...groups.values());
}

function describeCase(item) {
  const details = byId("case-details");
  details.replaceChildren();
  if (!item) {
    details.hidden = true;
    return;
  }

  const sweepLabels = { level: "干扰电平", overlap: "语音重叠", recording: "录音试听" };
  const score = item.scores?.mean_sisdri;
  const relativeLevels = item.source_levels_db;
  const sourceScores = item.evaluation?.per_source;
  const threeSourceScores = Array.isArray(sourceScores) && sourceScores.length === 3 &&
    sourceScores.every((source) => Number.isFinite(source?.sisdri));
  const db = (value) => `${value > 0 ? "+" : ""}${value.toFixed(2)} dB`;
  const fields = [
    ["语料", item.pair_label],
    ["模型", item.model || modelName],
    ["分离轨", Array.isArray(item.files?.tracks) ? `${item.files.tracks.length} 条` : null],
    ["设置", sweepLabels[item.sweep]],
    ["干扰声相对电平", Number.isFinite(item.interferer_db) ? `${item.interferer_db} dB` : null],
    ["各声源设定相对电平", Array.isArray(relativeLevels) && relativeLevels.length === 3 &&
        relativeLevels.every(Number.isFinite)
      ? `${relativeLevels.join(" / ")} dB` : null],
    ["重叠段时长", Number.isFinite(item.overlap_seconds) ? `${item.overlap_seconds.toFixed(2)} 秒` : null],
    ["平均 SI-SDR 改善", Number.isFinite(score) ? db(score) : null],
    ["分离轨 1 / 2 / 3 的 SI-SDR 改善", threeSourceScores
      ? sourceScores.map((source) => db(source.sisdri)).join(" / ") : null],
  ];
  for (const [name, value] of fields) {
    if (value == null || value === "") continue;
    const group = document.createElement("div");
    const title = document.createElement("dt");
    const description = document.createElement("dd");
    title.textContent = name;
    description.textContent = value;
    group.append(title, description);
    details.append(group);
  }
  details.hidden = !details.childElementCount;
}

function getCaseFromUrl() {
  const id = new URL(location.href).searchParams.get("case");
  return cases.find((item) => item.id === id) || cases[0];
}

async function loadManifest() {
  picker.disabled = true;
  picker.replaceChildren(new Option("读取实验清单…"));
  describeCase(null);
  setStatus("正在读取实验清单…");

  if (location.protocol === "file:" && !embeddedManifest) {
    setStatus("请在项目目录运行 python -m http.server 8080 --bind 127.0.0.1，再打开 http://127.0.0.1:8080/cafe_demo/。", true);
    return;
  }
  if (!AudioContextClass || !OfflineAudioContextClass) {
    setStatus("当前浏览器无法解码和同步播放。请使用新版 Chrome、Edge 或 Safari。", true);
    return;
  }

  try {
    let manifest;
    if (embeddedManifest) {
      manifest = JSON.parse(embeddedManifest.textContent);
    } else {
      const requested = new URL(location.href).searchParams.get("manifest");
      const url = new URL(requested || "../runs/cafe-demo/manifest.json", location.href);
      const response = await checkedFetch(url);
      manifest = await response.json();
      manifestUrl = new URL(response.url || url);
    }
    if (!Array.isArray(manifest.cases) || manifest.cases.length === 0 ||
        manifest.cases.some((item) => !item || typeof item.id !== "string" || !item.id)) {
      throw new Error("The manifest has no valid cases");
    }
    modelName = typeof manifest.model === "string" ? manifest.model.split("/").at(-1) : null;
    cases = manifest.cases;
    setCaseOptions();
    picker.disabled = false;
    await loadCase(getCaseFromUrl());
  } catch (error) {
    console.error("Could not load café manifest", error);
    setStatus(embeddedManifest
      ? "演示数据无法读取。请重新打开完整的 HTML 文件。"
      : "实验清单无法读取。请先生成实验音频，或检查地址中的 manifest 路径。", true);
  }
}

function setTrackCount(count) {
  if (count === trackButtons.length) return;
  lanes.slice(1).forEach((lane) => lane.remove());
  const tracks = Array.from({ length: count }, (_, index) => {
    const lane = laneTemplate.content.firstElementChild.cloneNode(true);
    const button = lane.querySelector(".track-button");
    button.dataset.track = String(index);
    button.querySelector(".lane-name").textContent = `分离轨 ${index + 1}`;
    trackArea.insertBefore(lane, cursorArea);
    return { lane, button };
  });
  lanes = [mixLane, ...tracks.map((track) => track.lane)];
  canvases = lanes.map((lane) => lane.querySelector(".waveform"));
  trackButtons = tracks.map((track) => track.button);
  byId("other-label").textContent = count === 2 ? "另一轨音量" : "其他音轨音量";
  byId("monitor-hint").textContent = count === 2
    ? "点选任一分离轨作为主轨，再调节另一轨的音量。"
    : "点选任一分离轨作为主轨。音量滑块同时调节其余两轨。";
}

function updateControls() {
  const ready = buffers.length > 0;
  playButton.disabled = !ready || loadingPlayback;
  playButton.textContent = playback ? "暂停" : ready && position >= duration ? "重播" : "播放";
  stopButton.disabled = !ready || (!loadingPlayback && !playback && position === 0);
  byId("modes").disabled = !ready;
  seek.disabled = !ready;
  overlapButton.hidden = !(ready && overlapRange());
  soloButton.disabled = !ready;
  otherLevel.disabled = !ready || solo;
  for (const button of trackButtons) button.disabled = !ready;
  trackArea.dataset.ready = String(ready);
}

function overlapRange() {
  const start = currentCase?.overlap_start_seconds;
  const length = currentCase?.overlap_seconds;
  if (!Number.isFinite(start) || start < 0 || start >= duration ||
      !Number.isFinite(length) || length <= 0) return null;
  return { start, end: Math.min(duration, start + length) };
}

function setMonitoring(nextMode, nextFocus = focus, nextSolo = solo) {
  const lanesChanged = nextMode !== mode || nextFocus !== focus;
  mode = nextMode;
  focus = nextFocus;
  solo = nextSolo;
  document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  soloButton.setAttribute("aria-pressed", String(solo));
  soloButton.textContent = solo ? "取消独听" : "独听主轨";
  byId("mix-status").textContent = mode === "mix" ? "当前试听" : "";
  lanes[0].dataset.active = String(mode === "mix");
  for (const [index, button] of trackButtons.entries()) {
    const isFocus = index === focus;
    button.setAttribute("aria-pressed", String(isFocus));
    lanes[index + 1].dataset.active = String(mode === "separated" && isFocus);
    button.querySelector(".lane-status").textContent =
      isFocus ? "主轨" : mode === "separated" && (solo || Number(otherLevel.value) === 0) ? "静音" : "";
  }
  updateControls();
  if (playback) {
    const now = audioContext.currentTime;
    monitorGains().forEach((gain, index) => {
      playback.gains[index].gain.setTargetAtTime(gain, now, 0.012);
    });
  }
  if (lanesChanged) scheduleWaveforms();
}

async function loadCase(item) {
  request?.abort();
  request = new AbortController();
  const { signal } = request;
  currentCase = item;
  picker.value = item.id;
  describeCase(item);
  stopAt(0);
  activeScrub = null;
  resumeAfterScrub = false;
  buffers = [];
  duration = 0;
  seek.max = "1";
  updateTime(0);
  setMonitoring("mix", 0, false);
  scheduleWaveforms();
  setStatus("正在读取并对齐音频…");

  try {
    const { mixture, tracks } = item.files || {};
    if (typeof mixture !== "string" || !mixture || !Array.isArray(tracks) ||
        ![2, 3].includes(tracks.length) || tracks.some((file) => typeof file !== "string" || !file)) {
      throw new Error("The case must list a mix and two or three separated tracks");
    }
    setTrackCount(tracks.length);
    setMonitoring("mix", 0, false);
    decodingContext ??= new OfflineAudioContextClass(1, 1, 48000);
    const decoded = await Promise.all([mixture, ...tracks].map(async (file) => {
      const response = await checkedFetch(embeddedManifest ? file : new URL(file, manifestUrl), signal);
      const bytes = await response.arrayBuffer();
      if (signal.aborted) throw new DOMException("Loading aborted", "AbortError");
      return decodingContext.decodeAudioData(bytes);
    }));
    if (signal.aborted) return;
    buffers = decoded;
    duration = buffers[0].duration;
    const peaks = buffers.map((buffer) => {
      let peak = 0;
      for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
        for (const sample of buffer.getChannelData(channel)) peak = Math.max(peak, Math.abs(sample));
      }
      return peak;
    });
    // Use shared headroom so switching between the mix and the estimates preserves gain.
    const combinedPeak = peaks.slice(1).reduce((sum, peak) => sum + peak, 0);
    headroom = Math.min(1, 0.92 / Math.max(peaks[0], combinedPeak));
    waveformScale = Math.max(0.01, ...peaks);
    seek.max = String(Math.ceil(duration / seekStep) * seekStep);
    updateTime(0);
    updateControls();
    scheduleWaveforms();
    setStatus("音频已就绪。");
  } catch (error) {
    if (signal.aborted) return;
    console.error("Could not load case", item.id, error);
    setStatus("这组音频无法载入。请检查清单中的 WAV 路径，或切换到另一组样本。", true);
  }
}

function monitorGains() {
  const background = solo ? 0 : Number(otherLevel.value) / 100;
  return buffers.map((_, index) => {
    if (mode === "mix") return index === 0 ? headroom : 0;
    if (index === 0) return 0;
    return (index === focus + 1 ? 1 : background) * headroom;
  });
}

function currentPosition() {
  if (!playback) return position;
  const played = Math.max(0, audioContext.currentTime - playback.startedAt);
  return Math.min(duration, playback.offset + played);
}

function stopAt(time) {
  playRequest++;
  loadingPlayback = false;
  cancelAnimationFrame(animationFrame);
  if (playback) {
    for (const source of playback.sources) {
      source.stop();
      source.disconnect();
    }
    for (const gain of playback.gains) gain.disconnect();
    playback = null;
  }
  position = time;
  updateTime(position);
  updateControls();
}

function startSources(offset) {
  const startedAt = audioContext.currentTime + 0.025;
  const endingAt = startedAt + duration - offset;
  const sources = [];
  const gains = monitorGains().map((level, index) => {
    const gain = audioContext.createGain();
    gain.gain.value = level;
    gain.connect(audioContext.destination);
    const buffer = buffers[index];
    if (offset < buffer.duration) {
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(gain);
      source.start(startedAt, offset);
      source.stop(endingAt);
      sources.push(source);
    }
    return gain;
  });
  playback = { startedAt, offset, sources, gains };
  updateControls();
  tick();
}

async function startPlaying() {
  if (!buffers.length || loadingPlayback) return;
  const ticket = ++playRequest;
  loadingPlayback = true;
  updateControls();
  try {
    audioContext ??= new AudioContextClass();
    if (audioContext.state !== "running") await audioContext.resume();
    if (ticket !== playRequest) return;
    loadingPlayback = false;
    startSources(position >= duration - 0.005 ? 0 : position);
    setStatus("正在播放。");
  } catch (error) {
    if (ticket !== playRequest) return;
    console.error("Could not start Web Audio", error);
    stopAt(position);
    setStatus("无法播放音频。请检查浏览器的声音权限后再点播放。", true);
  }
}

function tick() {
  if (!playback) return;
  const time = currentPosition();
  if (time >= duration) {
    stopAt(duration);
    setStatus("播放结束。");
    return;
  }
  updateTime(time);
  animationFrame = requestAnimationFrame(tick);
}

function jumpTo(time) {
  if (!buffers.length) return;
  const nextPosition = Math.max(0, Math.min(duration, time));
  const wasPlaying = Boolean(playback);
  stopAt(nextPosition);
  if (wasPlaying && nextPosition < duration) startSources(nextPosition);
  if (wasPlaying && nextPosition >= duration) setStatus("播放结束。");
}

function formattedTime(seconds, decimals = true) {
  const units = Math.floor(Math.max(0, seconds) * (decimals ? 10 : 1) + 1e-6);
  const totalSeconds = decimals ? Math.floor(units / 10) : units;
  const minutes = Math.floor(totalSeconds / 60);
  const rest = String(totalSeconds % 60).padStart(2, "0");
  return decimals ? `${minutes}:${rest}.${units % 10}` : `${minutes}:${rest}`;
}

function updateTime(time) {
  byId("elapsed").textContent = formattedTime(time);
  byId("duration").textContent = formattedTime(duration);
  seek.value = time === duration && duration > 0 ? seek.max : String(time);
  const accessible = `${time.toFixed(1)} 秒，共 ${duration.toFixed(1)} 秒`;
  if (seek.getAttribute("aria-valuetext") !== accessible) {
    seek.setAttribute("aria-valuetext", accessible);
  }
  trackArea.style.setProperty("--progress", `${duration ? 100 * time / duration : 0}%`);
}

function drawWaveforms() {
  const palette = getComputedStyle(trackArea);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const ticks = byId("ticks");
  const count = ticks.clientWidth > 560 ? 5 : 3;
  const overlap = overlapRange();
  ticks.replaceChildren(...Array.from({ length: duration ? count : 1 }, (_, index) => {
    const tick = document.createElement("span");
    const time = duration * index / (count - 1);
    tick.textContent = formattedTime(time, duration < 8 && duration > 0);
    return tick;
  }));

  for (const [index, canvas] of canvases.entries()) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const drawing = canvas.getContext("2d");
    drawing.scale(dpr, dpr);
    if (overlap) {
      drawing.fillStyle = palette.getPropertyValue("--selected");
      drawing.fillRect(width * overlap.start / duration, 0, width * (overlap.end - overlap.start) / duration, height);
    }
    const center = height / 2;
    drawing.strokeStyle = palette.getPropertyValue("--wave-axis");
    drawing.beginPath();
    drawing.moveTo(0, center + 0.5);
    drawing.lineTo(width, center + 0.5);
    drawing.stroke();

    const buffer = buffers[index];
    if (!buffer) continue;
    const samples = buffer.getChannelData(0);
    const step = duration * buffer.sampleRate / Math.max(1, width);
    const amplitude = (center - 4) / waveformScale;
    const active = lanes[index].dataset.active === "true";
    drawing.strokeStyle = palette.getPropertyValue(active ? "--rust" : "--wave");
    drawing.beginPath();
    for (let x = 0; x < width; x++) {
      const from = Math.floor(x * step);
      const to = Math.min(samples.length, Math.max(from + 1, Math.floor((x + 1) * step)));
      if (from >= to) break;
      let min = 0;
      let max = 0;
      for (let i = from; i < to; i++) {
        min = Math.min(min, samples[i]);
        max = Math.max(max, samples[i]);
      }
      drawing.moveTo(x + 0.5, center - max * amplitude);
      drawing.lineTo(x + 0.5, center - min * amplitude);
    }
    drawing.stroke();
  }
}

function scheduleWaveforms() {
  if (waveformFrame) return;
  waveformFrame = requestAnimationFrame(() => {
    waveformFrame = 0;
    drawWaveforms();
  });
}

picker.addEventListener("change", () => {
  const item = cases.find((entry) => entry.id === picker.value);
  if (!item) return;
  if (location.protocol !== "file:") {
    const url = new URL(location.href);
    url.searchParams.set("case", item.id);
    history.replaceState(null, "", url);
  }
  loadCase(item);
});

playButton.addEventListener("click", () => {
  if (playback) {
    stopAt(currentPosition());
    setStatus("已暂停。");
  } else {
    startPlaying();
  }
});
stopButton.addEventListener("click", () => {
  resumeAfterScrub = false;
  stopAt(0);
  setStatus("已停止。");
});

byId("modes").addEventListener("change", (event) => setMonitoring(event.target.value));
soloButton.addEventListener("click", () => setMonitoring("separated", focus, !solo));
otherLevel.addEventListener("input", () => {
  const percentage = `${otherLevel.value}%`;
  otherValue.value = percentage;
  otherLevel.setAttribute("aria-valuetext", percentage);
  setMonitoring("separated");
});

seek.addEventListener("pointerdown", (event) => {
  if (!event.isPrimary || event.button !== 0 || seek.disabled) return;
  activeScrub = event.pointerId;
  resumeAfterScrub = Boolean(playback);
  if (playback) stopAt(currentPosition());
});
seek.addEventListener("input", () => jumpTo(Number(seek.value)));
overlapButton.addEventListener("click", () => {
  const overlap = overlapRange();
  if (!overlap) return;
  jumpTo(overlap.start);
  setStatus(`重叠段从 ${overlap.start.toFixed(1)} 秒开始。`);
});
function finishScrub(event) {
  if (activeScrub !== event.pointerId) return;
  activeScrub = null;
  const shouldResume = resumeAfterScrub && position < duration;
  resumeAfterScrub = false;
  if (shouldResume) startPlaying();
  else if (position >= duration && duration > 0) setStatus("已定位到末尾。");
}
window.addEventListener("pointerup", finishScrub);
window.addEventListener("pointercancel", finishScrub);

trackArea.addEventListener("click", (event) => {
  const button = event.target.closest(".track-button");
  if (button && !button.disabled) {
    setMonitoring("separated", Number(button.dataset.track));
  } else if (event.target.matches(".waveform")) {
    const { left, width } = event.target.getBoundingClientRect();
    if (width) jumpTo((event.clientX - left) / width * duration);
  }
});
retry.addEventListener("click", () => currentCase ? loadCase(currentCase) : loadManifest());
if (window.ResizeObserver) {
  new ResizeObserver(scheduleWaveforms).observe(trackArea);
} else {
  window.addEventListener("resize", scheduleWaveforms);
}
setTrackCount(2);
setMonitoring("mix");
scheduleWaveforms();
loadManifest();

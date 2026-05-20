(async()=>{
  if (window.__browserBrainOpen) return;
  window.__browserBrainOpen = true;

  if (!window.documentPictureInPicture) {
    alert('Doc PiP not supported');
    window.__browserBrainOpen = false;
    return;
  }

  if (documentPictureInPicture.window) {
    documentPictureInPicture.window.close();
    window.__browserBrainOpen = false;
    return;
  }

  const CHANNEL = 'browser-brain';
  const bc = new BroadcastChannel(CHANNEL);

  const pip = await documentPictureInPicture.requestWindow({ width: 480, height: 640 });
  const d = pip.document;

  d.body.innerHTML = `
  <style>
    html,body{margin:0;width:100%;height:100%;overflow:hidden;background:radial-gradient(circle at top,#1f2937,#0b1020 60%);color:#fff;font-family:Arial,system-ui,sans-serif}
    body{display:flex;align-items:center;justify-content:center}
    .wrap{width:100%;height:100%;box-sizing:border-box;padding:12px;display:flex;align-items:center;justify-content:center}
    .card{width:min(460px,100%);height:100%;background:rgba(15,23,42,.94);border:1px solid rgba(255,255,255,.08);border-radius:18px;box-shadow:0 18px 50px rgba(0,0,0,.35);backdrop-filter:blur(10px);display:flex;flex-direction:column;overflow:hidden}
    .top{padding:12px 12px 10px;border-bottom:1px solid rgba(255,255,255,.08);background:rgba(17,24,39,.75)}
    .title{font-size:12px;font-weight:700;letter-spacing:1.6px;opacity:.9;margin-bottom:8px}
    .status{font-size:11px;opacity:.7;margin-top:6px;min-height:14px}
    textarea{width:100%;box-sizing:border-box;min-height:88px;resize:vertical;border:1px solid #334155;border-radius:12px;background:#0f172a;color:#fff;padding:10px;font-size:14px;outline:none}
    .bar{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    button{background:#334155;color:#fff;border:none;border-radius:10px;padding:8px 12px;cursor:pointer;font-size:13px}
    button.primary{background:#4f46e5}
    button.good{background:#10b981}
    button:disabled{opacity:.4;cursor:not-allowed}
    .out{flex:1;min-height:0;overflow:auto;padding:12px;white-space:pre-wrap;line-height:1.45;font-size:13px;background:linear-gradient(180deg,rgba(15,23,42,.92),rgba(2,6,23,.95))}
    .msg{border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:10px 12px;margin-bottom:10px;background:rgba(15,23,42,.82)}
    .msg.user{background:rgba(17,24,39,.95)}
    .msg.assistant{background:rgba(8,47,73,.45)}
    .msg.guest{background:rgba(22,101,52,.18)}
    .label{font-size:11px;opacity:.7;margin-bottom:6px;letter-spacing:.6px;text-transform:uppercase}
    .hint{font-size:11px;opacity:.65;margin-top:8px;line-height:1.35}
    .minirow{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
    .pill{font-size:11px;padding:4px 8px;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.06)}
    .stream{opacity:.92}
  </style>

  <div class="wrap">
    <div class="card">
      <div class="top">
        <div class="title">BROWSER BRAIN</div>
        <textarea id="input" placeholder="Ask the brain... Example: summarize this page, classify this data, or tell me the next step."></textarea>
        <div class="bar">
          <button class="primary" id="run">Run</button>
          <button class="good" id="page">Analyze guest page</button>
          <button id="sum">Summarize</button>
          <button id="cls">Classify</button>
          <button id="clear">Clear</button>
          <button id="close">Close</button>
        </div>
        <div class="minirow">
          <span class="pill" id="mode">idle</span>
          <span class="pill" id="model">model: loading...</span>
        </div>
        <div class="status" id="status">Ready</div>
        <div class="hint">Use the guest-tab sender later to push extracted tab data into this brain through <code>${CHANNEL}</code>.</div>
      </div>
      <div class="out" id="out">Brain is ready to load a model.</div>
    </div>
  </div>`;

  const out = d.getElementById('out');
  const input = d.getElementById('input');
  const status = d.getElementById('status');
  const mode = d.getElementById('mode');
  const modelTag = d.getElementById('model');

  const logs = [];
  const maxLogs = 40;
  let pipePromise = null;
  let pipe = null;
  let tokenizer = null;

  const device = navigator.gpu ? 'webgpu' : 'wasm';
  const dtype = navigator.gpu ? 'q4' : 'q8';

  // Smaller instruct model for faster load and snappier replies.
  // Models like SmolLM2-135M-Instruct and SmolLM2-360M-Instruct are available in the
  // Hugging Face transformers.js model list filtered by text-generation + transformers.js.
  const modelId = 'HuggingFaceTB/SmolLM2-360M-Instruct';

  function esc(v) {
    return String(v)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');
  }

  function render() {
    out.innerHTML = logs.map(m => `
      <div class="msg ${m.role}">
        <div class="label">${esc(m.role)}</div>
        <div class="${m.streaming ? 'stream' : ''}">${esc(m.text)}</div>
      </div>
    `).join('');
    out.scrollTop = out.scrollHeight;
  }

  function push(role, text, streaming = false) {
    logs.push({ role, text, streaming });
    while (logs.length > maxLogs) logs.shift();
    render();
  }

  function updateLastAssistant(text, streaming = true) {
    for (let i = logs.length - 1; i >= 0; i--) {
      if (logs[i].role === 'assistant') {
        logs[i].text = text;
        logs[i].streaming = streaming;
        render();
        return;
      }
    }
  }

  function setStatus(text) { status.textContent = text; }
  function setMode(text) { mode.textContent = text; }

  async function getPipe() {
    if (pipe) return pipe;
    if (!pipePromise) {
      pipePromise = (async () => {
        setStatus('Loading Transformers.js...');
        const tf = await import('https://esm.sh/@huggingface/transformers');
        const { pipeline } = tf;
        tokenizer = null;
        setStatus('Loading model...');
        modelTag.textContent = `model: ${modelId}`;
        const p = await pipeline('text-generation', modelId, { device, dtype });
        tokenizer = p.tokenizer;
        return p;
      })();
    }
    pipe = await pipePromise;
    return pipe;
  }

  function buildPrompt(userText, extraContext = '') {
    const ctx = extraContext ? `\n\nCONTEXT:\n${extraContext}` : '';
    return `
<|system|>
You are a small browser brain running locally in the user's browser.
Your job is to help with actions on extracted page data.
Return concise, practical answers.
If the user asks for a plan, return a step-by-step plan.
If the user asks to classify or extract, return structured output.
If the data is messy, clean it mentally and respond clearly.
</s>
<|user|>
${userText}${ctx}
</s>
<|assistant|>
`.trim();
  }

  async function runBrain({ text, extraContext = '', label = 'assistant' }) {
    const task = text.trim();
    if (!task) return;

    push('user', task);
    setMode('thinking');
    setStatus('Running...');

    const assistantIndex = logs.push({ role: 'assistant', text: '', streaming: true }) - 1;
    render();

    try {
      const p = await getPipe();
      const prompt = buildPrompt(task, extraContext);
      let streamed = '';

      const { TextStreamer } = await import('https://esm.sh/@huggingface/transformers');
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        callback_function: (chunk) => {
          streamed += chunk;
          logs[assistantIndex].text = streamed;
          logs[assistantIndex].streaming = true;
          render();
        }
      });

      await p(prompt, {
        max_new_tokens: 128,
        do_sample: false,
        streamer
      });

      logs[assistantIndex].text = streamed.trim() || '(no response)';
      logs[assistantIndex].streaming = false;
      render();
      setStatus('Ready');
    } catch (err) {
      console.error(err);
      logs[assistantIndex].text = err && err.message ? err.message : String(err);
      logs[assistantIndex].streaming = false;
      render();
      setStatus('Failed');
    } finally {
      setMode('idle');
    }
  }

  async function summarize() {
    await runBrain({ text: input.value, extraContext: 'Summarize the provided text in a short useful form.' });
  }

  async function classify() {
    await runBrain({ text: input.value, extraContext: 'Classify the provided text and explain the category briefly.' });
  }

  async function analyzeGuestPayload(payload) {
    const title = payload?.title || '';
    const url = payload?.url || '';
    const text = payload?.text || payload?.data || '';
    const extra = [
      title ? `Page title: ${title}` : '',
      url ? `URL: ${url}` : '',
      text ? `Extracted page data:\n${text.slice(0, 6000)}` : ''
    ].filter(Boolean).join('\n\n');

    push('guest', `Received page snapshot from guest tab:\n${title || '(no title)'}\n${url || ''}`);
    await runBrain({
      text: 'Analyze the supplied page snapshot and tell me the best next action I should take on it.',
      extraContext: extra,
      label: 'assistant'
    });
  }

  d.getElementById('run').onclick = () => runBrain({ text: input.value });
  d.getElementById('sum').onclick = summarize;
  d.getElementById('cls').onclick = classify;
  d.getElementById('clear').onclick = () => { logs.length = 0; render(); setStatus('Cleared'); };
  d.getElementById('close').onclick = () => { pip.close(); };
  d.getElementById('page').onclick = () => {
    runBrain({ text: input.value || 'Analyze the current guest page data and tell me what to do next.', extraContext: 'If guest page data arrives later, use it as primary context.' });
  };

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      runBrain({ text: input.value });
    }
  });

  bc.onmessage = async (ev) => {
    const msg = ev.data || {};
    if (msg.type === 'guest:page' || msg.type === 'page' || msg.type === 'data') {
      await analyzeGuestPayload(msg);
    }
    if (msg.type === 'guest:text') {
      push('guest', msg.text || '');
      await runBrain({ text: msg.text || '', label: 'assistant' });
    }
  };

  push('assistant', 'Browser brain opened. Type a task, or send extracted page data from another tab.');
  setStatus('Ready');
  setMode('idle');
  window.__browserBrainOpen = true;

  pip.addEventListener('pagehide', () => {
    window.__browserBrainOpen = false;
    try { bc.close(); } catch {}
    pipe = null;
    pipePromise = null;
  });
})()

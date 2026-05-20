(async()=>{

if(window.__brain__) return;
window.__brain__ = true;

if(!window.documentPictureInPicture){
  alert("Document PiP not supported");
  return;
}

if(documentPictureInPicture.window){
  documentPictureInPicture.window.close();
  return;
}

const w =
  await documentPictureInPicture.requestWindow({
    width:420,
    height:420
  });

const d = w.document;

d.body.innerHTML = `
<style>

html,body{
  margin:0;
  width:100%;
  height:100%;
  overflow:hidden;
  background:#111827;
  color:white;
  font-family:Arial,system-ui,sans-serif;
}

body{
  display:flex;
  flex-direction:column;
}

.top{
  padding:12px;
  background:#0f172a;
  border-bottom:1px solid #334155;
}

.title{
  font-size:12px;
  font-weight:700;
  letter-spacing:1.5px;
  opacity:.85;
  margin-bottom:10px;
}

textarea{
  width:100%;
  min-height:90px;
  resize:none;
  box-sizing:border-box;
  background:#111827;
  color:white;
  border:1px solid #334155;
  border-radius:10px;
  padding:10px;
  font-size:14px;
}

.bar{
  display:flex;
  gap:8px;
  margin-top:10px;
}

button{
  background:#334155;
  color:white;
  border:none;
  border-radius:10px;
  padding:8px 12px;
  cursor:pointer;
  font-size:13px;
}

button.primary{
  background:#4f46e5;
}

.out{
  flex:1;
  overflow:auto;
  padding:12px;
  white-space:pre-wrap;
  line-height:1.45;
  font-size:13px;
}

.status{
  font-size:11px;
  opacity:.7;
  margin-top:8px;
}

.card{
  border:1px solid #334155;
  border-radius:12px;
  padding:10px;
  background:#0f172a;
  margin-bottom:10px;
}

.label{
  font-size:11px;
  opacity:.7;
  margin-bottom:6px;
}

</style>

<div class="top">

  <div class="title">
    TRANSFORMERS.JS BRAIN
  </div>

  <textarea
    id="input"
    placeholder="Type task text here..."
  ></textarea>

  <div class="bar">
    <button class="primary" id="run">
      Run
    </button>

    <button id="sum">
      Summarize
    </button>

    <button id="cls">
      Classify
    </button>

    <button id="clear">
      Clear
    </button>

    <button id="close">
      Close
    </button>
  </div>

  <div class="status" id="status">
    Ready
  </div>

</div>

<div class="out" id="out">
  Browser Brain Ready
</div>
`;

const out =
  d.getElementById("out");

const status =
  d.getElementById("status");

const input =
  d.getElementById("input");

function log(label,text){

  out.innerHTML =
    `
      <div class="card">
        <div class="label">${label}</div>
        <div>${text}</div>
      </div>
    ` + out.innerHTML;
}

function setStatus(text){
  status.textContent = text;
}

const transformers =
  await import(
    "https://esm.sh/@huggingface/transformers"
  );

const {
  pipeline
} = transformers;

const device =
  navigator.gpu
    ? "webgpu"
    : "wasm";

const dtype =
  navigator.gpu
    ? "q4"
    : "q8";

setStatus(
  "Loading models..."
);

const sentimentPipe =
  await pipeline(
    "sentiment-analysis",
    "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
    {
      device,
      dtype
    }
  );

const summarizePipe =
  await pipeline(
    "summarization",
    "Xenova/distilbart-cnn-6-6",
    {
      device,
      dtype
    }
  );

setStatus(
  "Brain Ready"
);

async function classify(){

  const text =
    input.value.trim();

  if(!text) return;

  setStatus(
    "Classifying..."
  );

  try{

    const result =
      await sentimentPipe(text);

    log(
      "Classification",
      JSON.stringify(
        result,
        null,
        2
      )
    );

  }catch(e){

    log(
      "Error",
      e.message || String(e)
    );
  }

  setStatus(
    "Ready"
  );
}

async function summarize(){

  const text =
    input.value.trim();

  if(!text) return;

  setStatus(
    "Summarizing..."
  );

  try{

    const result =
      await summarizePipe(
        text.slice(0,3000),
        {
          max_length:80,
          min_length:20
        }
      );

    log(
      "Summary",
      result[0].summary_text
    );

  }catch(e){

    log(
      "Error",
      e.message || String(e)
    );
  }

  setStatus(
    "Ready"
  );
}

async function smartRun(){

  const text =
    input.value.trim();

  if(!text) return;

  if(
    text.length > 300
  ){
    await summarize();
  }else{
    await classify();
  }
}

d.getElementById(
  "run"
).onclick = smartRun;

d.getElementById(
  "sum"
).onclick = summarize;

d.getElementById(
  "cls"
).onclick = classify;

d.getElementById(
  "clear"
).onclick = ()=>{

  out.innerHTML =
    "Cleared";

  input.value = "";
};

d.getElementById(
  "close"
).onclick = ()=>{

  w.close();
};

input.addEventListener(
  "keydown",
  e=>{

    if(
      e.key === "Enter" &&
      (
        e.ctrlKey ||
        e.metaKey
      )
    ){
      e.preventDefault();
      smartRun();
    }
  }
);

})();
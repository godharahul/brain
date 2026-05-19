(async()=>{

if(window.__brain__) return;
window.__brain__ = true;

const bc = new BroadcastChannel("brain");

const pip = await documentPictureInPicture.requestWindow({
  width: 420,
  height: 650
});

const doc = pip.document;

doc.body.innerHTML = `
<style>
body{
  margin:0;
  background:#0f172a;
  color:white;
  font-family:Arial;
}

#top{
  padding:10px;
  background:#111827;
  display:flex;
  gap:10px;
}

#prompt{
  flex:1;
  background:#1e293b;
  color:white;
  border:none;
  padding:10px;
  border-radius:8px;
}

button{
  background:#4f46e5;
  color:white;
  border:none;
  padding:10px;
  border-radius:8px;
}

#chat{
  padding:10px;
  height:calc(100vh - 80px);
  overflow:auto;
  white-space:pre-wrap;
}
</style>

<div id="top">
  <input id="prompt" placeholder="Ask brain..." />
  <button id="send">Send</button>
</div>

<div id="chat"></div>
`;

const chat = doc.getElementById("chat");

function log(text){
  chat.innerHTML += `<div>${text}</div><hr>`;
  chat.scrollTop = chat.scrollHeight;
}

log("Loading WebLLM...");

const webllm =
  await import("https://esm.run/@mlc-ai/web-llm");

const engine =
  await webllm.CreateMLCEngine(
    "Llama-3.2-1B-Instruct"
  );

log("Brain Ready");

async function ask(prompt){

  log("YOU: " + prompt);

  const reply =
    await engine.chat.completions.create({
      messages:[
        {
          role:"system",
          content:
            "You are a browser brain assistant."
        },
        {
          role:"user",
          content:prompt
        }
      ]
    });

  const text =
    reply.choices[0].message.content;

  log("BRAIN: " + text);
}

doc.getElementById("send").onclick = ()=>{

  const p =
    doc.getElementById("prompt");

  ask(p.value);

  p.value = "";
};

bc.onmessage = async(e)=>{

  const msg = e.data;

  if(msg.type === "page"){

    log("PAGE RECEIVED");

    await ask(
      "Analyze this page:\n\n" +
      msg.text.slice(0,4000)
    );
  }
};

})();
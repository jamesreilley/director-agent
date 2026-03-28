import { useState, useRef, useEffect } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// SYSTEM PROMPT
// ─────────────────────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are a Director of Photography and Prompt Architect for an AI film production pipeline.

Your output — a START FRAME and END FRAME — will be sent to ComfyUI for production rendering. These prompts must be PHYSICALLY INTERPOLATABLE when used as the first and last frames of a video generation model.

You receive:
1. VISUAL BIBLE — the universe rulebook governing every material, surface, character, and environment.
2. FRAME 1 DESCRIPTION — the director's complete instruction for the start frame.
3. FRAME 2 DESCRIPTION — the director's complete instruction for the end frame.
4. DIRECTOR FEEDBACK (optional) — if present, apply this feedback. Keep everything not criticised. Only change what the feedback addresses.

━━━ VIDEO INTERPOLATION — NON-NEGOTIABLE CONSTRAINTS ━━━

CAMERA LOCK — Camera position, height, angle, and focal length IDENTICAL in both frames unless movement is explicitly described.
LIGHTING FREEZE — Lighting frozen at a single moment. Key light direction, shadow positions, light source states identical in both frames.
SPATIAL PLAUSIBILITY — Frame 2 subject position must be physically reachable from Frame 1 within one continuous shot.
IMPLIED MOTION — Frame 1 must compositionally suggest the motion that Frame 2 resolves.
LOCKS — Costume, hair, environment, set dressing identical in both frames. Only subject position, expression, gesture, action state may change.

━━━ VISUAL BIBLE — THREE BINDING LAYERS ━━━

LAYER 1 — POSITIVE MATERIAL LANGUAGE: Extract every approved material and texture. Only these may appear in any prompt.
LAYER 2 — NEGATIVE MATERIAL CONSTRAINT: Extract every prohibited material. Apply as hard negative prompt exclusions AND avoid in positive prompts.
LAYER 3 — REINTERPRETATION RULE: Every object named that would normally use a prohibited material must be translated through the bible's approved material system. No exceptions.

━━━ PROMPT CONSTRUCTION ━━━

For each frame: framing → subject position and action state → environment through bible material language → lighting, palette, lens from bible verbatim → style keywords. 120–160 words per prompt. Write for ComfyUI/Stable Diffusion — dense, comma-separated descriptors work well alongside natural language.

━━━ MANDATORY PRE-OUTPUT CHECKLIST ━━━

□ Camera identical in both frames?
□ Lighting frozen?
□ Frame 2 position spatially reachable from Frame 1?
□ Frame 1 implies the motion Frame 2 resolves?
□ Costume, hair, environment locked?
□ Every element translated through bible's material system?
□ All prohibited materials in negative prompts?
□ Hero elements exact?
□ Director feedback applied if provided?

━━━ OUTPUT ━━━
Valid JSON only. No markdown. No preamble.

{
  "shotSummary": "one sentence: complete motion arc Frame 1 to Frame 2",
  "sceneSlug": "kebab-case-max-5-words",
  "sharedContext": "camera, lighting, environment, costume — locked across both frames in the bible's material language",
  "motionArc": "what physically moves, how far, in what direction",
  "feedbackApplied": "what changed from previous version, or null if first generation",
  "startFrame": {
    "compositionNote": "subject position and action state at Frame 1",
    "prompt": "full bible-compliant ComfyUI prompt 120-160 words",
    "negativePrompt": "all prohibited materials and treatments from the bible"
  },
  "endFrame": {
    "compositionNote": "subject position and action state at Frame 2",
    "prompt": "full bible-compliant ComfyUI prompt 120-160 words",
    "negativePrompt": "all prohibited materials and treatments from the bible"
  },
  "auditResult": {
    "cameraLock": "confirmed | [issue]",
    "lightingFreeze": "confirmed | [issue]",
    "spatialPlausibility": "confirmed | [issue]",
    "impliedMotion": "confirmed — [description]",
    "materialCompliance": "confirmed | [violations corrected]",
    "heroElementLock": "confirmed | not applicable | [issue]",
    "reinterpretationApplied": "confirmed — [key reinterpretations]"
  }
}`;

// ─────────────────────────────────────────────────────────────────────────────
// STANDARD COMFYUI WORKFLOW
// This is injected with the generated prompts and sent to the ComfyUI API.
// The user can override this with their own workflow JSON in Settings.
// ─────────────────────────────────────────────────────────────────────────────
// Flux.1 dev workflow — tuned for flux1-dev-fp8.safetensors
// CFG must be 1.0, scheduler must be "simple", guidance via FluxGuidance node at 3.5
function buildComfyWorkflow({ positivePrompt, negativePrompt, modelName, width, height, steps, seed }) {
  return {
    "1": {
      "inputs": { "ckpt_name": modelName || "FLUX1/flux1-dev-fp8.safetensors" },
      "class_type": "CheckpointLoaderSimple"
    },
    "2": {
      "inputs": { "width": width || 1024, "height": height || 576, "batch_size": 1 },
      "class_type": "EmptyLatentImage"
    },
    "3": {
      "inputs": { "text": positivePrompt, "clip": ["1", 1] },
      "class_type": "CLIPTextEncode"
    },
    "4": {
      "inputs": { "text": negativePrompt || "blurry, deformed, low quality, watermark", "clip": ["1", 1] },
      "class_type": "CLIPTextEncode"
    },
    "5": {
      "inputs": { "guidance": 3.5, "conditioning": ["3", 0] },
      "class_type": "FluxGuidance"
    },
    "6": {
      "inputs": {
        "seed": seed || Math.floor(Math.random() * 999999999),
        "steps": steps || 20,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "model": ["1", 0],
        "positive": ["5", 0],
        "negative": ["4", 0],
        "latent_image": ["2", 0]
      },
      "class_type": "KSampler"
    },
    "7": {
      "inputs": { "samples": ["6", 0], "vae": ["1", 2] },
      "class_type": "VAEDecode"
    },
    "8": {
      "inputs": { "filename_prefix": "director-agent", "images": ["7", 0] },
      "class_type": "SaveImage"
    }
  };
}

// Inject prompts into a custom workflow JSON
// Finds CLIPTextEncode nodes and injects positive/negative prompts
function injectIntoCustomWorkflow(workflowJson, positivePrompt, negativePrompt) {
  const workflow = JSON.parse(JSON.stringify(workflowJson)); // deep clone
  const nodes = Object.entries(workflow);

  // Find positive and negative CLIPTextEncode nodes
  // Heuristic: positive is the one connected to KSampler positive input
  let positiveNodeId = null;
  let negativeNodeId = null;

  // Find KSampler node
  for (const [id, node] of nodes) {
    if (node.class_type === "KSampler" || node.class_type === "KSamplerAdvanced") {
      if (node.inputs?.positive) positiveNodeId = node.inputs.positive[0];
      if (node.inputs?.negative) negativeNodeId = node.inputs.negative[0];
      break;
    }
  }

  // Inject into found nodes
  if (positiveNodeId && workflow[positiveNodeId]) {
    workflow[positiveNodeId].inputs.text = positivePrompt;
  }
  if (negativeNodeId && workflow[negativeNodeId]) {
    workflow[negativeNodeId].inputs.text = negativePrompt;
  }

  return workflow;
}

// ─────────────────────────────────────────────────────────────────────────────
// COMFYUI API
// ─────────────────────────────────────────────────────────────────────────────
async function comfySubmit(serverUrl, workflow) {
  const res = await fetch(`${serverUrl}/prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: workflow, client_id: "director-agent" }),
  });
  if (!res.ok) throw new Error(`ComfyUI submit ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return data.prompt_id;
}

async function comfyPoll(serverUrl, promptId, onProgress) {
  for (let i = 0; i < 120; i++) {
    await new Promise(r => setTimeout(r, 2500));
    const res = await fetch(`${serverUrl}/history/${promptId}`);
    if (!res.ok) continue;
    const data = await res.json();
    const entry = data[promptId];
    if (!entry) continue;

    // Check for completion
    if (entry.status?.status_str === "success" || entry.outputs) {
      // Find the SaveImage output
      for (const nodeId of Object.keys(entry.outputs || {})) {
        const nodeOut = entry.outputs[nodeId];
        if (nodeOut.images?.length) {
          const img = nodeOut.images[0];
          return `${serverUrl}/view?filename=${encodeURIComponent(img.filename)}&subfolder=${encodeURIComponent(img.subfolder || "")}&type=${img.type || "output"}`;
        }
      }
    }
    if (entry.status?.status_str === "error") {
      throw new Error(`ComfyUI render failed: ${JSON.stringify(entry.status)}`);
    }
    if (onProgress) onProgress(i);
  }
  throw new Error("ComfyUI: timeout waiting for render");
}

async function comfyRenderFrame(serverUrl, workflow, positivePrompt, negativePrompt, customWorkflow, comfySettings) {
  let wf;
  if (customWorkflow) {
    try {
      const parsed = JSON.parse(customWorkflow);
      wf = injectIntoCustomWorkflow(parsed, positivePrompt, negativePrompt);
    } catch(e) {
      throw new Error("Invalid custom workflow JSON: " + e.message);
    }
  } else {
    wf = buildComfyWorkflow({
      positivePrompt,
      negativePrompt,
      ...comfySettings,
      seed: Math.floor(Math.random() * 999999999),
    });
  }
  const promptId = await comfySubmit(serverUrl, wf);
  return comfyPoll(serverUrl, promptId);
}

// ─────────────────────────────────────────────────────────────────────────────
// PREVIEW RENDERERS (NanoBanana / Flux) — for quick iteration
// ─────────────────────────────────────────────────────────────────────────────
const PREVIEW_PROVIDERS = {
  gemini: {
    label: "Google AI Studio", icon: "✦",
    models: [
      { id: "gemini-2.0-flash-preview-image-generation", label: "Gemini 2.0 Flash", desc: "Fast, high quality" },
    ],
  },
  nanobanana: {
    label: "NanoBanana", icon: "🍌",
    models: [
      { id: "nano-banana-2",   label: "Nano Banana 2",   desc: "Gemini 3.1 Flash · fast" },
      { id: "nano-banana-pro", label: "Nano Banana Pro", desc: "Gemini 3 Pro · premium"  },
    ],
  },
  fal: {
    label: "Flux", icon: "⚡",
    models: [
      { id: "fal-ai/flux/dev",     label: "Flux Dev",     desc: "Best fidelity" },
      { id: "fal-ai/flux/schnell", label: "Flux Schnell", desc: "Fastest"        },
    ],
  },
};

const ASPECT_RATIOS = [
  { id: "landscape_16_9", label: "16:9", width: 1024, height: 576,  nb: "16:9" },
  { id: "landscape_21_9", label: "21:9", width: 1280, height: 544,  nb: "21:9" },
  { id: "landscape_4_3",  label: "4:3",  width: 1024, height: 768,  nb: "4:3"  },
];

async function nbGenerate(apiKey, model, prompt, negativePrompt, aspectRatio, refUrls) {
  const body = {
    prompt: negativePrompt ? `${prompt} --no ${negativePrompt}` : prompt,
    selectedModel: model,
    aspect_ratio: ASPECT_RATIOS.find(r => r.id === aspectRatio)?.nb || "16:9",
  };
  if (refUrls?.length) body.referenceImageUrls = refUrls;
  const res = await fetch("https://www.nananobanana.com/api/v1/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`NanoBanana ${res.status}: ${await res.text()}`);
  const init = await res.json();
  const taskId = init?.data?.id || init?.id;
  if (!taskId) throw new Error("NanoBanana: no task ID");
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 3000));
    const poll = await fetch(`https://www.nananobanana.com/api/v1/generate/${taskId}`, {
      headers: { "Authorization": `Bearer ${apiKey}` },
    });
    const pd = await poll.json();
    const status = pd?.data?.processingStatus || pd?.status;
    if (status === "completed" || status === "success") {
      const urls = pd?.data?.outputImageUrls || pd?.outputImageUrls;
      if (urls?.length) return urls[0];
      throw new Error("NanoBanana: no image URL");
    }
    if (status === "failed" || status === "error") throw new Error("NanoBanana: generation failed");
  }
  throw new Error("NanoBanana: timeout");
}

async function falText2Img(apiKey, model, prompt, negativePrompt, aspectRatio) {
  const res = await fetch(`https://fal.run/${model}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Key ${apiKey}` },
    body: JSON.stringify({
      prompt, negative_prompt: negativePrompt || "blurry, deformed, watermark",
      image_size: aspectRatio,
      num_inference_steps: model.includes("schnell") ? 4 : 28,
      guidance_scale: 3.5, num_images: 1, enable_safety_checker: false,
    }),
  });
  if (!res.ok) throw new Error(`fal.ai ${res.status}: ${await res.text()}`);
  const d = await res.json();
  return d.images?.[0]?.url || d.image?.url;
}

async function falUpload(apiKey, file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("https://fal.run/storage/upload", {
    method: "POST", headers: { "Authorization": `Key ${apiKey}` }, body: fd,
  });
  if (!res.ok) throw new Error(`Upload: ${res.status}`);
  return (await res.json()).url;
}

// ─────────────────────────────────────────────────────────────────────────────
// WEAVY
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// GOOGLE AI STUDIO — Gemini image generation
// ─────────────────────────────────────────────────────────────────────────────
async function geminiGenerateImage(apiKey, prompt, aspectRatio) {
  const aspectMap = {
    landscape_16_9: "16:9",
    landscape_21_9: "21:9",
    landscape_4_3:  "4:3",
  };
  const ar = aspectMap[aspectRatio] || "16:9";
  const fullPrompt = `${prompt} --aspect_ratio ${ar}`;

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-preview-image-generation:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: fullPrompt }] }],
        generationConfig: { responseModalities: ["TEXT", "IMAGE"] },
      }),
    }
  );
  if (!res.ok) throw new Error(`Gemini ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const parts = data.candidates?.[0]?.content?.parts || [];
  for (const part of parts) {
    if (part.inlineData?.mimeType?.startsWith("image/")) {
      return `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
    }
  }
  throw new Error("Gemini: no image in response");
}

async function weavyUpsertApp(envUrl, apiKey, uid, name) {
  const res = await fetch(`${envUrl}/api/apps`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
    body: JSON.stringify({ uid, name, type: "chat" }),
  });
  if (res.status === 409) {
    const g = await fetch(`${envUrl}/api/apps/${uid}`, { headers: { "Authorization": `Bearer ${apiKey}` } });
    return g.json();
  }
  if (!res.ok) throw new Error(`Weavy upsert ${res.status}`);
  return res.json();
}

async function weavyPostMessage(envUrl, apiKey, appUid, text) {
  const res = await fetch(`${envUrl}/api/apps/${appUid}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`Weavy post ${res.status}`);
  return res.json();
}

async function weavyGetMessages(envUrl, apiKey, appUid) {
  const res = await fetch(`${envUrl}/api/apps/${appUid}/messages?order_by=created_at+desc&take=20`, {
    headers: { "Authorization": `Bearer ${apiKey}` },
  });
  if (!res.ok) throw new Error(`Weavy get ${res.status}`);
  const data = await res.json();
  return (data.data || []).reverse();
}

// ─────────────────────────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────────────────────────
const css = `
  @keyframes spin    { to { transform: rotate(360deg); } }
  @keyframes fadeIn  { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
  @keyframes pulse   { 0%,100%{opacity:.3} 50%{opacity:.85} }
  @keyframes slideIn { from { transform:translateX(100%); opacity:0; } to { transform:translateX(0); opacity:1; } }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:#09090a; }
  textarea, input, select { box-sizing:border-box; }
  ::-webkit-scrollbar { width:4px; }
  ::-webkit-scrollbar-thumb { background:rgba(200,160,80,.13); border-radius:2px; }
  .ta::placeholder { color:rgba(232,224,212,.15); font-style:italic; font-size:11px; line-height:1.65; }
  .ta:focus { border-color:rgba(200,160,80,.3) !important; outline:none; }
  .drop-active { border-color:rgba(200,160,80,.55) !important; background:rgba(200,160,80,.06) !important; }
`;

// ─────────────────────────────────────────────────────────────────────────────
// ATOMS
// ─────────────────────────────────────────────────────────────────────────────
const Spin = ({ size=12, color="rgba(200,160,80,.18)", top="#c8a050" }) => (
  <span style={{ display:"inline-block", width:size, height:size, border:`2px solid ${color}`, borderTopColor:top, borderRadius:"50%", animation:"spin .8s linear infinite", flexShrink:0 }} />
);

function CopyBtn({ text }) {
  const [ok, setOk] = useState(false);
  return (
    <button onClick={() => { navigator.clipboard.writeText(text); setOk(true); setTimeout(()=>setOk(false),2000); }}
      style={{ background:ok?"rgba(80,160,100,.1)":"rgba(200,160,80,.06)", border:`1px solid ${ok?"rgba(80,160,100,.25)":"rgba(200,160,80,.15)"}`, borderRadius:4, padding:"3px 9px", cursor:"pointer", fontSize:10, letterSpacing:".1em", textTransform:"uppercase", color:ok?"#7dc493":"rgba(200,160,80,.55)", fontFamily:"sans-serif", transition:"all .2s" }}>
      {ok?"✓":"COPY"}
    </button>
  );
}

function FieldLabel({ main, sub }) {
  return (
    <div style={{ marginBottom:8 }}>
      <div style={{ fontSize:10, letterSpacing:".14em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.65)" }}>{main}</div>
      {sub && <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:3, lineHeight:1.45 }}>{sub}</div>}
    </div>
  );
}

function Divider() { return <div style={{ height:1, background:"rgba(255,255,255,.05)", margin:"2px 0" }} />; }

function SLabel({ children }) {
  return <div style={{ fontSize:10, letterSpacing:".13em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.55)", marginBottom:4 }}>{children}</div>;
}

function SecretInput({ value, onChange, placeholder }) {
  const [show, setShow] = useState(false);
  return (
    <div style={{ position:"relative" }}>
      <input type={show?"text":"password"} value={value} onChange={e=>onChange(e.target.value)} placeholder={placeholder}
        style={{ width:"100%", background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"9px 34px 9px 11px", fontFamily:"monospace", outline:"none" }}
        onFocus={e=>e.target.style.borderColor="rgba(200,160,80,.28)"}
        onBlur={e=>e.target.style.borderColor="rgba(255,255,255,.07)"} />
      <button onClick={()=>setShow(!show)} style={{ position:"absolute", right:9, top:"50%", transform:"translateY(-50%)", background:"none", border:"none", color:"rgba(232,224,212,.28)", cursor:"pointer", fontSize:11 }}>{show?"🙈":"👁"}</button>
    </div>
  );
}

function TextInput({ value, onChange, placeholder, mono }) {
  return (
    <input value={value} onChange={e=>onChange(e.target.value)} placeholder={placeholder}
      style={{ width:"100%", background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"9px 11px", fontFamily:mono?"monospace":"sans-serif", outline:"none" }}
      onFocus={e=>e.target.style.borderColor="rgba(200,160,80,.28)"}
      onBlur={e=>e.target.style.borderColor="rgba(255,255,255,.07)"} />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// AUTO-RESIZE TEXTAREA
// ─────────────────────────────────────────────────────────────────────────────
function AutoTA({ value, onChange, placeholder, minHeight=80, fontSize=12 }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) { ref.current.style.height="auto"; ref.current.style.height=ref.current.scrollHeight+"px"; }
  }, [value]);
  return (
    <textarea ref={ref} className="ta" value={value} onChange={e=>onChange(e.target.value)} placeholder={placeholder}
      style={{ width:"100%", minHeight, background:"rgba(255,255,255,.02)", border:"1px solid rgba(255,255,255,.07)", borderRadius:9, color:"rgba(232,224,212,.85)", fontSize, padding:"12px 13px", fontFamily:"Georgia,serif", resize:"none", lineHeight:1.72, transition:"border-color .2s", overflow:"hidden", display:"block" }} />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FRAME INPUT
// ─────────────────────────────────────────────────────────────────────────────
function FrameInput({ number, label, value, onChange, placeholder }) {
  const ref = useRef(null);
  const rgb = number===1 ? "80,160,220" : "210,120,55";
  useEffect(() => {
    if (ref.current) { ref.current.style.height="auto"; ref.current.style.height=ref.current.scrollHeight+"px"; }
  }, [value]);
  return (
    <div style={{ borderRadius:9, border:`1px solid ${value?`rgba(${rgb},.28)`:"rgba(255,255,255,.07)"}`, overflow:"hidden", transition:"border-color .2s", background:"rgba(255,255,255,.015)" }}>
      <div style={{ padding:"9px 13px", background:`rgba(${rgb},.07)`, borderBottom:`1px solid rgba(${rgb},.14)`, display:"flex", alignItems:"center", gap:9 }}>
        <div style={{ width:22, height:22, borderRadius:5, background:`rgba(${rgb},.15)`, border:`1px solid rgba(${rgb},.32)`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <span style={{ fontSize:11, color:`rgba(${rgb},1)`, fontFamily:"sans-serif", fontWeight:700 }}>{number}</span>
        </div>
        <span style={{ fontSize:10, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", color:`rgba(${rgb},1)`, fontWeight:700 }}>{label}</span>
      </div>
      <textarea ref={ref} className="ta" value={value} onChange={e=>onChange(e.target.value)} placeholder={placeholder}
        style={{ width:"100%", minHeight:100, background:"transparent", border:"none", borderRadius:0, color:"#e8e0d4", fontSize:12, padding:"12px 13px", fontFamily:"Georgia,serif", resize:"none", lineHeight:1.72, overflow:"hidden", display:"block", outline:"none" }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// REFERENCE ZONE
// ─────────────────────────────────────────────────────────────────────────────
function RefZone({ refs, onAdd, onRemove }) {
  const inputRef = useRef();
  const [dragOver, setDragOver] = useState(false);
  function handleFiles(files) { Array.from(files).forEach(f => { if (f.type.startsWith("image/")) onAdd(f); }); }
  return (
    <div className={dragOver?"drop-active":""} onDragOver={e=>{e.preventDefault();setDragOver(true);}} onDragLeave={()=>setDragOver(false)} onDrop={e=>{e.preventDefault();setDragOver(false);handleFiles(e.dataTransfer.files);}}
      style={{ borderRadius:9, border:`1px dashed ${refs.length?"rgba(200,160,80,.28)":"rgba(255,255,255,.08)"}`, background:refs.length?"rgba(200,160,80,.04)":"rgba(255,255,255,.01)", padding:"12px", transition:"all .2s", minHeight:68 }}>
      {refs.length===0?(
        <div onClick={()=>inputRef.current?.click()} style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:5, textAlign:"center", cursor:"pointer" }}>
          <div style={{ fontSize:20, opacity:.35 }}>📎</div>
          <div style={{ fontSize:10, fontFamily:"sans-serif", color:"rgba(232,224,212,.35)", fontWeight:700, textTransform:"uppercase", letterSpacing:".08em" }}>Drop references here</div>
          <div style={{ fontSize:9, color:"rgba(232,224,212,.18)", fontFamily:"sans-serif", fontStyle:"italic" }}>Character, style, mood, environment — anything relevant</div>
          <div style={{ fontSize:9, color:"rgba(200,160,80,.28)", fontFamily:"sans-serif", marginTop:2 }}>drop or click to browse</div>
        </div>
      ):(
        <div>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:8 }}>
            {refs.map((r,i)=>(
              <div key={i} style={{ position:"relative", flexShrink:0 }}>
                <img src={r.preview} alt="" style={{ width:50, height:50, objectFit:"cover", borderRadius:6, border:`1px solid ${r.falUrl?"rgba(80,160,100,.35)":"rgba(200,160,80,.2)"}`, display:"block" }} />
                {r.uploading&&<div style={{ position:"absolute", inset:0, background:"rgba(9,9,10,.6)", borderRadius:6, display:"flex", alignItems:"center", justifyContent:"center" }}><Spin/></div>}
                {r.falUrl&&<div style={{ position:"absolute", bottom:2, right:2, width:11, height:11, borderRadius:"50%", background:"#5cb87a", border:"1.5px solid #09090a", display:"flex", alignItems:"center", justifyContent:"center" }}><span style={{ fontSize:6, color:"#fff" }}>✓</span></div>}
                <button onClick={()=>onRemove(i)} style={{ position:"absolute", top:-4, right:-4, width:15, height:15, borderRadius:"50%", background:"rgba(0,0,0,.8)", border:"1px solid rgba(255,255,255,.15)", color:"rgba(255,255,255,.7)", cursor:"pointer", fontSize:8, display:"flex", alignItems:"center", justifyContent:"center" }}>×</button>
              </div>
            ))}
            <button onClick={()=>inputRef.current?.click()} style={{ width:50, height:50, borderRadius:6, border:"1px dashed rgba(200,160,80,.22)", background:"rgba(200,160,80,.05)", cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", color:"rgba(200,160,80,.38)", fontSize:18, flexShrink:0 }}>+</button>
          </div>
          <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic" }}>{refs.length} reference{refs.length>1?"s":""} — used for preview consistency</div>
        </div>
      )}
      <input ref={inputRef} type="file" accept="image/*" multiple style={{ display:"none" }} onChange={e=>{handleFiles(e.target.files);e.target.value="";}} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FRAME OUTPUT PANEL
// ─────────────────────────────────────────────────────────────────────────────
function FramePanel({ number, label, frame, imageUrl, loading, loadingMsg, error, renderMode }) {
  const [promptOpen, setPromptOpen] = useState(false);
  const rgb = number===1 ? "80,160,220" : "210,120,55";
  const accent      = `rgba(${rgb},.85)`;
  const accentFaint = `rgba(${rgb},.09)`;
  const accentLine  = `rgba(${rgb},.18)`;

  return (
    <div style={{ flex:1, minWidth:0, animation:"fadeIn .4s ease both", animationDelay:number===1?"0s":".1s" }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:9 }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={{ width:22, height:22, borderRadius:5, background:accentFaint, border:`1px solid ${accentLine}`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
            <span style={{ fontSize:10, color:accent, fontFamily:"sans-serif", fontWeight:700 }}>{number}</span>
          </div>
          <span style={{ fontSize:11, color:accent, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".12em", textTransform:"uppercase" }}>{label}</span>
          {renderMode && <span style={{ fontSize:9, padding:"2px 7px", borderRadius:10, background:"rgba(255,255,255,.05)", color:"rgba(232,224,212,.3)", fontFamily:"sans-serif" }}>{renderMode}</span>}
        </div>
        <div style={{ display:"flex", gap:6 }}>
          {frame?.prompt && <CopyBtn text={frame.prompt} />}
          {imageUrl && <a href={imageUrl} download={`frame-${number}.jpg`} target="_blank" rel="noreferrer" style={{ fontSize:10, padding:"3px 9px", borderRadius:4, background:"rgba(80,130,200,.08)", border:"1px solid rgba(80,130,200,.18)", color:"rgba(130,180,240,.6)", fontFamily:"sans-serif", textDecoration:"none", letterSpacing:".08em", textTransform:"uppercase" }}>↓ Save</a>}
        </div>
      </div>

      {frame?.compositionNote && <div style={{ fontSize:10, color:"rgba(232,224,212,.3)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:8, lineHeight:1.45 }}>{frame.compositionNote}</div>}

      <div style={{ background:"#0d0d0e", borderRadius:9, border:"1px solid rgba(255,255,255,.07)", overflow:"hidden", marginBottom:9, aspectRatio:"16/9", display:"flex", alignItems:"center", justifyContent:"center" }}>
        {loading && !imageUrl && (
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:9 }}>
            <Spin/>
            <span style={{ fontSize:9, color:accent, letterSpacing:".14em", textTransform:"uppercase", fontFamily:"sans-serif", animation:"pulse 1.6s ease infinite" }}>{loadingMsg||"Rendering…"}</span>
          </div>
        )}
        {error && !imageUrl && <div style={{ padding:20, textAlign:"center" }}><div style={{ fontSize:18, marginBottom:6 }}>⚠</div><div style={{ fontSize:11, color:"rgba(220,100,100,.6)", fontFamily:"sans-serif" }}>{error}</div></div>}
        {!loading && !error && !imageUrl && <div style={{ fontSize:11, color:"rgba(255,255,255,.07)", fontFamily:"sans-serif", fontStyle:"italic" }}>Frame {number}</div>}
        {imageUrl && <img src={imageUrl} alt={label} style={{ width:"100%", height:"100%", objectFit:"cover", display:"block" }} />}
      </div>

      {frame?.prompt && (
        <>
          <button onClick={()=>setPromptOpen(!promptOpen)} style={{ width:"100%", background:accentFaint, border:`1px solid ${accentLine}`, borderRadius:promptOpen?"7px 7px 0 0":"7px", padding:"7px 11px", cursor:"pointer", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <span style={{ fontSize:10, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:accent }}>ComfyUI Prompt</span>
            <span style={{ fontSize:10, color:accent, opacity:.5 }}>{promptOpen?"−":"+"}</span>
          </button>
          {promptOpen && (
            <div style={{ background:"rgba(255,255,255,.018)", border:`1px solid ${accentLine}`, borderTop:"none", borderRadius:"0 0 7px 7px", padding:"11px 12px" }}>
              <div style={{ fontSize:11, lineHeight:1.72, color:"rgba(232,224,212,.55)", fontFamily:"sans-serif", marginBottom:9 }}>{frame.prompt}</div>
              {frame.negativePrompt && (
                <div style={{ paddingTop:8, borderTop:"1px solid rgba(255,255,255,.05)" }}>
                  <div style={{ fontSize:9, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(220,100,100,.38)", marginBottom:4 }}>Negative</div>
                  <div style={{ fontSize:10, color:"rgba(220,150,150,.36)", fontFamily:"sans-serif", lineHeight:1.5 }}>{frame.negativePrompt}</div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MOTION ARC + AUDIT
// ─────────────────────────────────────────────────────────────────────────────
function MotionArc({ text }) {
  if (!text) return null;
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10, padding:"9px 13px", background:"rgba(255,255,255,.02)", border:"1px solid rgba(255,255,255,.055)", borderRadius:7, marginBottom:16 }}>
      <span style={{ fontSize:14, flexShrink:0 }}>↗</span>
      <div>
        <div style={{ fontSize:9, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.45)", marginBottom:2 }}>Motion Arc</div>
        <div style={{ fontSize:11, color:"rgba(232,224,212,.5)", fontFamily:"sans-serif", fontStyle:"italic", lineHeight:1.4 }}>{text}</div>
      </div>
    </div>
  );
}

function AuditPanel({ audit }) {
  const [open, setOpen] = useState(false);
  if (!audit) return null;
  const entries = [
    ["cameraLock","Camera Lock"],["lightingFreeze","Lighting Freeze"],["spatialPlausibility","Spatial Plausibility"],
    ["impliedMotion","Implied Motion"],["materialCompliance","Material Compliance"],
    ["heroElementLock","Hero Element"],["reinterpretationApplied","Reinterpretation"],
  ];
  const issues = entries.filter(([k])=>{ const v=audit[k]||""; return v&&v!=="confirmed"&&v!=="not applicable"&&!v.startsWith("confirmed"); });
  const allOk = issues.length===0;
  return (
    <div>
      <button onClick={()=>setOpen(!open)} style={{ display:"flex", alignItems:"center", gap:7, background:"none", border:"none", cursor:"pointer", padding:0 }}>
        <span style={{ width:7, height:7, borderRadius:"50%", background:allOk?"#5cb87a":"#c8a050", flexShrink:0 }} />
        <span style={{ fontSize:10, fontFamily:"sans-serif", letterSpacing:".1em", textTransform:"uppercase", color:allOk?"rgba(80,180,120,.7)":"rgba(200,160,80,.7)" }}>
          {allOk?"All checks passed":`${issues.length} audit note${issues.length>1?"s":""}`}
        </span>
        <span style={{ fontSize:9, color:"rgba(232,224,212,.2)", fontFamily:"sans-serif" }}>{open?"▲":"▼"}</span>
      </button>
      {open && (
        <div style={{ marginTop:9, padding:"11px 13px", background:"rgba(255,255,255,.02)", border:"1px solid rgba(255,255,255,.06)", borderRadius:8, display:"grid", gap:7 }}>
          {entries.map(([k,l])=>{
            const val=audit[k]||"—";
            const ok=val==="confirmed"||val==="not applicable"||val.startsWith("confirmed");
            return (
              <div key={k} style={{ display:"flex", alignItems:"flex-start", gap:7 }}>
                <span style={{ fontSize:11, color:ok?"rgba(80,180,120,.65)":"rgba(200,160,80,.7)", flexShrink:0, marginTop:.5 }}>{ok?"✓":"⚠"}</span>
                <div>
                  <span style={{ fontSize:10, fontFamily:"sans-serif", fontWeight:700, color:"rgba(232,224,212,.45)", textTransform:"uppercase", letterSpacing:".07em" }}>{l} </span>
                  <span style={{ fontSize:10, fontFamily:"sans-serif", color:ok?"rgba(232,224,212,.32)":"rgba(232,224,212,.55)", fontStyle:"italic" }}>{val}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// WEAVY PANEL
// ─────────────────────────────────────────────────────────────────────────────
function WeavyPanel({ shot, threadUrl, messages, onCheckFeedback, onClose, checkingFeedback, feedbackFound }) {
  return (
    <div style={{ position:"absolute", inset:0, background:"#0f0f11", borderLeft:"1px solid rgba(255,255,255,.07)", display:"flex", flexDirection:"column", animation:"slideIn .25s ease", zIndex:10 }}>
      <div style={{ padding:"14px 18px", borderBottom:"1px solid rgba(255,255,255,.07)", display:"flex", justifyContent:"space-between", alignItems:"center", flexShrink:0 }}>
        <div>
          <div style={{ fontSize:11, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".1em", textTransform:"uppercase", color:"rgba(80,180,120,.8)" }}>💬 Weavy Review Thread</div>
          {shot?.sceneSlug && <div style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"monospace", marginTop:3 }}>{shot.sceneSlug}</div>}
        </div>
        <div style={{ display:"flex", gap:8, alignItems:"center" }}>
          {threadUrl && <a href={threadUrl} target="_blank" rel="noreferrer" style={{ fontSize:10, padding:"5px 12px", borderRadius:5, background:"rgba(80,180,120,.1)", border:"1px solid rgba(80,180,120,.25)", color:"rgba(80,180,120,.8)", fontFamily:"sans-serif", textDecoration:"none", letterSpacing:".08em", textTransform:"uppercase" }}>Open ↗</a>}
          <button onClick={onClose} style={{ background:"none", border:"none", color:"rgba(232,224,212,.3)", cursor:"pointer", fontSize:18 }}>×</button>
        </div>
      </div>
      <div style={{ flex:1, overflowY:"auto", padding:"16px 18px", display:"flex", flexDirection:"column", gap:12 }}>
        {messages.length===0
          ? <div style={{ textAlign:"center", padding:"40px 0", color:"rgba(232,224,212,.2)", fontSize:12, fontFamily:"sans-serif", fontStyle:"italic" }}>No messages yet.</div>
          : messages.map((msg,i)=>{
              const isAgent = msg.text?.startsWith("🎬");
              return (
                <div key={i} style={{ padding:"11px 14px", background:isAgent?"rgba(200,160,80,.05)":"rgba(80,130,200,.05)", border:`1px solid ${isAgent?"rgba(200,160,80,.12)":"rgba(80,130,200,.12)"}`, borderRadius:8 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                    <span style={{ fontSize:9, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".1em", textTransform:"uppercase", color:isAgent?"rgba(200,160,80,.55)":"rgba(130,180,240,.55)" }}>{isAgent?"Director Agent":msg.created_by?.display_name||"Director"}</span>
                    <span style={{ fontSize:9, color:"rgba(232,224,212,.2)", fontFamily:"sans-serif" }}>{msg.created_at?new Date(msg.created_at).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}):""}</span>
                  </div>
                  <div style={{ fontSize:11, color:"rgba(232,224,212,.65)", fontFamily:"sans-serif", lineHeight:1.65, whiteSpace:"pre-wrap" }}>{msg.plain||msg.text||""}</div>
                </div>
              );
            })
        }
      </div>
      <div style={{ padding:"14px 18px", borderTop:"1px solid rgba(255,255,255,.07)", flexShrink:0 }}>
        {feedbackFound && <div style={{ marginBottom:10, padding:"9px 12px", background:"rgba(80,180,120,.07)", border:"1px solid rgba(80,180,120,.18)", borderRadius:7, fontSize:11, color:"rgba(80,180,120,.8)", fontFamily:"sans-serif" }}>✓ Feedback found — regenerating. Close this panel to see new frames.</div>}
        <button onClick={onCheckFeedback} disabled={checkingFeedback} style={{ width:"100%", padding:"11px", borderRadius:7, border:"1px solid rgba(80,180,120,.35)", background:"rgba(80,180,120,.1)", color:checkingFeedback?"rgba(80,180,120,.4)":"rgba(80,180,120,.85)", fontSize:11, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", fontWeight:700, cursor:checkingFeedback?"not-allowed":"pointer", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
          {checkingFeedback?<><Spin color="rgba(80,180,120,.3)" top="rgba(80,180,120,1)"/>Checking for feedback…</>:"↺ Check for Director Feedback"}
        </button>
        <p style={{ marginTop:7, fontSize:9, color:"rgba(232,224,212,.2)", fontFamily:"sans-serif", textAlign:"center", fontStyle:"italic" }}>Reads latest director reply and regenerates both frames with feedback applied</p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// LOG STRIP
// ─────────────────────────────────────────────────────────────────────────────
function LogStrip({ log, onSelect }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position:"fixed", bottom:0, left:0, right:0, zIndex:50 }}>
      <div style={{ display:"flex", justifyContent:"center" }}>
        <button onClick={()=>setOpen(!open)} style={{ background:"#17171a", border:"1px solid rgba(255,255,255,.08)", borderBottom:"none", borderRadius:"8px 8px 0 0", padding:"5px 18px", cursor:"pointer", display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:9, letterSpacing:".15em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(232,224,212,.3)" }}>Shot Log</span>
          <span style={{ fontSize:10, padding:"1px 6px", background:"rgba(200,160,80,.1)", borderRadius:10, color:"rgba(200,160,80,.6)", fontFamily:"sans-serif" }}>{log.length}</span>
          <span style={{ fontSize:9, color:"rgba(232,224,212,.2)" }}>{open?"▼":"▲"}</span>
        </button>
      </div>
      {open && (
        <div style={{ background:"#131315", borderTop:"1px solid rgba(255,255,255,.07)", maxHeight:180, overflowX:"auto", overflowY:"hidden" }}>
          {log.length===0?<div style={{ padding:"18px", textAlign:"center", fontSize:11, color:"rgba(232,224,212,.18)", fontFamily:"sans-serif", fontStyle:"italic" }}>No shots yet</div>
            :<div style={{ display:"flex", padding:"12px 16px", minWidth:"max-content", gap:10 }}>
              {log.map((s,i)=>(
                <button key={i} onClick={()=>{setOpen(false);onSelect(s);}} style={{ background:"rgba(255,255,255,.03)", border:"1px solid rgba(255,255,255,.07)", borderRadius:7, padding:"8px 12px", cursor:"pointer", textAlign:"left", minWidth:175 }}>
                  <div style={{ fontSize:9, color:"rgba(200,160,80,.45)", fontFamily:"sans-serif", marginBottom:4 }}>#{i+1} · {s.sceneSlug||"—"}{s._version>1?` · v${s._version}`:""}</div>
                  <div style={{ fontSize:11, color:"rgba(232,224,212,.5)", fontFamily:"sans-serif", fontStyle:"italic", lineHeight:1.35 }}>{(s.shotSummary||"").slice(0,65)}{(s.shotSummary||"").length>65?"…":""}</div>
                </button>
              ))}
            </div>
          }
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SETTINGS PANEL
// ─────────────────────────────────────────────────────────────────────────────
function Settings({ s, set, onClose }) {
  const upd = (k,v) => set(p=>({...p,[k]:v}));
  const [jsonError, setJsonError] = useState(null);

  function validateJson(val) {
    if (!val.trim()) { setJsonError(null); return; }
    try { JSON.parse(val); setJsonError(null); } catch(e) { setJsonError(e.message); }
  }

  return (
    <div style={{ position:"fixed", inset:0, zIndex:1000 }}>
      <div onClick={onClose} style={{ position:"absolute", inset:0, background:"rgba(0,0,0,.65)", backdropFilter:"blur(5px)" }} />
      <div style={{ position:"absolute", right:0, top:0, bottom:0, width:460, background:"#111113", borderLeft:"1px solid rgba(255,255,255,.08)", overflowY:"auto", display:"flex", flexDirection:"column" }}>
        <div style={{ padding:"18px 22px 14px", borderBottom:"1px solid rgba(255,255,255,.07)", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <span style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".1em", textTransform:"uppercase", color:"rgba(232,224,212,.6)" }}>Settings</span>
          <button onClick={onClose} style={{ background:"none", border:"none", color:"rgba(232,224,212,.35)", cursor:"pointer", fontSize:18 }}>×</button>
        </div>
        <div style={{ padding:"18px 22px", display:"grid", gap:24, flex:1 }}>

          {/* ── CLAUDE ── */}
          <div style={{ padding:"14px 16px", background:"rgba(200,160,80,.04)", border:"1px solid rgba(200,160,80,.12)", borderRadius:10 }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:12 }}>
              <span style={{ fontSize:16 }}>🧠</span>
              <div>
                <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:"rgba(200,160,80,.85)", letterSpacing:".06em" }}>Claude API</div>
                <div style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:2 }}>Prompt architect — writes your bible-locked prompts</div>
              </div>
            </div>
            <SLabel>API Key</SLabel>
            <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:7 }}>console.anthropic.com → API Keys</div>
            <SecretInput value={s.claudeKey} onChange={v=>upd("claudeKey",v)} placeholder="sk-ant-…" />
          </div>

          {/* ── COMFYUI — primary renderer ── */}
          <div style={{ padding:"16px 18px", background:"rgba(130,80,200,.05)", border:"1px solid rgba(130,80,200,.2)", borderRadius:10 }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:16 }}>
              <span style={{ fontSize:18 }}>🎨</span>
              <div>
                <div style={{ fontSize:13, fontFamily:"sans-serif", fontWeight:700, color:"rgba(180,130,255,.85)", letterSpacing:".06em" }}>ComfyUI</div>
                <div style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:2 }}>Primary production renderer</div>
              </div>
            </div>

            <div style={{ display:"grid", gap:12 }}>
              <div>
                <SLabel>Server URL</SLabel>
                <TextInput value={s.comfyUrl} onChange={v=>upd("comfyUrl",v)} placeholder="http://127.0.0.1:8188" />
                <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:5 }}>ComfyUI Desktop default: http://127.0.0.1:8188</div>
              </div>

              <div>
                <SLabel>Model / Checkpoint</SLabel>
                <TextInput value={s.comfyModel} onChange={v=>upd("comfyModel",v)} placeholder="FLUX1/flux1-dev-fp8.safetensors" />
                <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:4 }}>Include subfolder — e.g. FLUX1/flux1-dev-fp8.safetensors</div>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
                <div>
                  <SLabel>Steps</SLabel>
                  <input type="number" min={1} max={50} value={s.comfySteps} onChange={e=>upd("comfySteps",Number(e.target.value))}
                    style={{ width:"100%", background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"9px 11px", outline:"none" }} />
                </div>
                <div>
                  <SLabel>Guidance</SLabel>
                  <input type="number" min={1} max={10} step={0.5} value={s.comfyGuidance||3.5} onChange={e=>upd("comfyGuidance",Number(e.target.value))}
                    style={{ width:"100%", background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"9px 11px", outline:"none" }} />
                </div>
              </div>
              <div style={{ padding:"9px 11px", background:"rgba(130,80,200,.08)", border:"1px solid rgba(130,80,200,.2)", borderRadius:7, fontSize:10, color:"rgba(180,130,255,.75)", fontFamily:"sans-serif", lineHeight:1.6 }}>
                🎨 Flux.1 settings locked: CFG 1.0 · Sampler euler · Scheduler simple · FluxGuidance node
              </div>

              <div>
                <SLabel>Output Dimensions</SLabel>
                <div style={{ display:"flex", gap:7 }}>
                  {ASPECT_RATIOS.map(r=>(
                    <button key={r.id} onClick={()=>{ upd("ratio",r.id); upd("comfyWidth",r.width); upd("comfyHeight",r.height); }}
                      style={{ flex:1, padding:"8px", borderRadius:6, border:`1px solid ${s.ratio===r.id?"rgba(130,80,200,.45)":"rgba(255,255,255,.06)"}`, background:s.ratio===r.id?"rgba(130,80,200,.12)":"rgba(255,255,255,.02)", cursor:"pointer", fontSize:12, fontFamily:"sans-serif", color:s.ratio===r.id?"rgba(180,130,255,.9)":"rgba(232,224,212,.42)", fontWeight:s.ratio===r.id?700:400 }}>
                      {r.label}
                    </button>
                  ))}
                </div>
                <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:5 }}>{s.comfyWidth} × {s.comfyHeight}px</div>
              </div>

              <div>
                <SLabel>Custom Workflow JSON <span style={{ color:"rgba(232,224,212,.3)", fontWeight:400, textTransform:"none", letterSpacing:0 }}>— optional</span></SLabel>
                <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:7 }}>
                  Paste your own ComfyUI workflow. The agent will auto-inject prompts into the KSampler's positive/negative nodes. Leave blank to use the standard SD workflow.
                </div>
                <textarea
                  value={s.comfyWorkflow}
                  onChange={e=>{ upd("comfyWorkflow",e.target.value); validateJson(e.target.value); }}
                  placeholder={'{\n  "1": { "class_type": "CheckpointLoaderSimple", ... },\n  ...\n}'}
                  style={{ width:"100%", minHeight:120, background:"rgba(255,255,255,.04)", border:`1px solid ${jsonError?"rgba(200,80,80,.4)":"rgba(255,255,255,.07)"}`, borderRadius:6, color:"#e8e0d4", fontSize:11, padding:"9px 11px", fontFamily:"monospace", outline:"none", resize:"vertical", lineHeight:1.5 }} />
                {jsonError && <div style={{ fontSize:10, color:"rgba(220,100,100,.7)", fontFamily:"sans-serif", marginTop:4 }}>⚠ {jsonError}</div>}
                {s.comfyWorkflow && !jsonError && <div style={{ fontSize:10, color:"rgba(80,180,120,.6)", fontFamily:"sans-serif", marginTop:4 }}>✓ Valid workflow JSON</div>}
              </div>
            </div>
          </div>

          {/* ── PREVIEW renderer ── */}
          <div style={{ padding:"16px 18px", background:"rgba(255,255,255,.02)", border:"1px solid rgba(255,255,255,.07)", borderRadius:10 }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:14 }}>
              <span style={{ fontSize:16 }}>👁</span>
              <div>
                <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:"rgba(232,224,212,.65)", letterSpacing:".06em" }}>Quick Preview</div>
                <div style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:2 }}>Fast iteration before sending to ComfyUI</div>
              </div>
            </div>
            <div style={{ display:"grid", gap:10 }}>
              <div>
                <SLabel>Provider</SLabel>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:7, marginTop:7 }}>
                  {Object.entries(PREVIEW_PROVIDERS).map(([k,p])=>(
                    <button key={k} onClick={()=>upd("previewProvider",k)}
                      style={{ padding:"9px 11px", borderRadius:7, border:`1px solid ${s.previewProvider===k?"rgba(200,160,80,.38)":"rgba(255,255,255,.07)"}`, background:s.previewProvider===k?"rgba(200,160,80,.08)":"rgba(255,255,255,.02)", cursor:"pointer", textAlign:"left" }}>
                      <div style={{ fontSize:12, fontFamily:"sans-serif", color:s.previewProvider===k?"#c8a050":"rgba(232,224,212,.55)", fontWeight:700 }}>{p.icon} {p.label}</div>
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <SLabel>Model</SLabel>
                <div style={{ display:"grid", gap:6, marginTop:6 }}>
                  {PREVIEW_PROVIDERS[s.previewProvider].models.map(m=>{
                    const active = s.previewProvider==="nanobanana"?s.nbModel===m.id:s.falModel===m.id;
                    return (
                      <button key={m.id} onClick={()=>upd(s.previewProvider==="nanobanana"?"nbModel":"falModel",m.id)}
                        style={{ padding:"8px 11px", borderRadius:7, border:`1px solid ${active?"rgba(200,160,80,.35)":"rgba(255,255,255,.06)"}`, background:active?"rgba(200,160,80,.08)":"rgba(255,255,255,.02)", cursor:"pointer", textAlign:"left", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                        <div>
                          <div style={{ fontSize:11, fontFamily:"sans-serif", color:active?"#c8a050":"rgba(232,224,212,.58)", fontWeight:700 }}>{m.label}</div>
                          <div style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif", marginTop:1 }}>{m.desc}</div>
                        </div>
                        {active&&<span style={{ color:"#c8a050", fontSize:12 }}>✓</span>}
                      </button>
                    );
                  })}
                </div>
              </div>
              {s.previewProvider === "gemini" && (
                <div>
                  <SLabel>Google AI Studio API Key</SLabel>
                  <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:6 }}>aistudio.google.com → Get API Key</div>
                  <SecretInput value={s.geminiKey} onChange={v=>upd("geminiKey",v)} placeholder="AIza…" />
                </div>
              )}
              {s.previewProvider === "nanobanana" && (
                <div>
                  <SLabel>NanoBanana API Key</SLabel>
                  <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:6 }}>nananobanana.com → Settings → API Keys</div>
                  <SecretInput value={s.nbKey} onChange={v=>upd("nbKey",v)} placeholder="nb_…" />
                </div>
              )}
              {s.previewProvider === "fal" && (
                <div>
                  <SLabel>fal.ai API Key</SLabel>
                  <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:6 }}>fal.ai/dashboard/keys</div>
                  <SecretInput value={s.falKey} onChange={v=>upd("falKey",v)} placeholder="fal_…" />
                </div>
              )}
            </div>
          </div>

          {/* ── WEAVY ── */}
          <div style={{ padding:"16px 18px", background:"rgba(80,180,120,.04)", border:"1px solid rgba(80,180,120,.14)", borderRadius:10 }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:14 }}>
              <span style={{ fontSize:16 }}>💬</span>
              <div>
                <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:"rgba(80,180,120,.85)", letterSpacing:".06em" }}>Weavy Review</div>
                <div style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:2 }}>Shot threads + feedback-to-regenerate</div>
              </div>
            </div>
            <div style={{ display:"grid", gap:10 }}>
              <div>
                <SLabel>Environment URL</SLabel>
                <TextInput value={s.weavyUrl} onChange={v=>upd("weavyUrl",v)} placeholder="https://your-env.weavy.io" />
              </div>
              <div>
                <SLabel>API Key</SLabel>
                <SecretInput value={s.weavyKey} onChange={v=>upd("weavyKey",v)} placeholder="wys_…" />
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [showSettings, setShowSettings] = useState(false);

  const defaultSettings = {
    claudeKey:     "",
    comfyUrl:      "",
    comfyModel:    "FLUX1/flux1-dev-fp8.safetensors",
    comfySteps:    20,
    comfyGuidance: 3.5,
    comfyWidth:    1024,
    comfyHeight:   576,
    comfyWorkflow: "",
    ratio:         "landscape_16_9",
    previewProvider: "gemini",
    nbModel:         "nano-banana-2",
    falModel:        "fal-ai/flux/dev",
    geminiKey:       "",
    nbKey:           "",
    falKey:          "",
    weavyUrl: "",
    weavyKey: "",
  };

  const [settings, setSettings] = useState(() => {
    try {
      const saved = localStorage.getItem("da_settings");
      return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
    } catch(e) { return defaultSettings; }
  });

  // Persist settings on every change
  useEffect(() => {
    try { localStorage.setItem("da_settings", JSON.stringify(settings)); } catch(e) {}
  }, [settings]);

  const [bible,  setBible]  = useState(() => { try { return localStorage.getItem("da_bible")  || ""; } catch(e) { return ""; } });
  const [frame1, setFrame1] = useState(() => { try { return localStorage.getItem("da_frame1") || ""; } catch(e) { return ""; } });
  const [frame2, setFrame2] = useState(() => { try { return localStorage.getItem("da_frame2") || ""; } catch(e) { return ""; } });

  // Persist bible and frames on change
  useEffect(() => { try { localStorage.setItem("da_bible",  bible);  } catch(e) {} }, [bible]);
  useEffect(() => { try { localStorage.setItem("da_frame1", frame1); } catch(e) {} }, [frame1]);
  useEffect(() => { try { localStorage.setItem("da_frame2", frame2); } catch(e) {} }, [frame2]);
  const [refs,   setRefs]   = useState([]);

  const [shot,         setShot]         = useState(null);
  const [startImg,     setStartImg]     = useState(null);
  const [endImg,       setEndImg]       = useState(null);
  const [startLoading, setStartLoading] = useState(false);
  const [endLoading,   setEndLoading]   = useState(false);
  const [startMsg,     setStartMsg]     = useState("");
  const [endMsg,       setEndMsg]       = useState("");
  const [startErr,     setStartErr]     = useState(null);
  const [endErr,       setEndErr]       = useState(null);
  const [genBusy,      setGenBusy]      = useState(false);
  const [renderBusy,   setRenderBusy]   = useState(false);
  const [activeRenderMode, setActiveRenderMode] = useState(null); // "comfy" | "preview"
  const [genError,     setGenError]     = useState(null);
  const [log,          setLog]          = useState([]);
  const [version,      setVersion]      = useState(1);

  // Weavy
  const [weavyAppUid,      setWeavyAppUid]      = useState(null);
  const [weavyMessages,    setWeavyMessages]    = useState([]);
  const [weavyPosting,     setWeavyPosting]     = useState(false);
  const [weavyPostStatus,  setWeavyPostStatus]  = useState("idle");
  const [showWeavyPanel,   setShowWeavyPanel]   = useState(false);
  const [checkingFeedback, setCheckingFeedback] = useState(false);
  const [feedbackFound,    setFeedbackFound]    = useState(false);
  const [threadUrl,        setThreadUrl]        = useState(null);

  const {
    claudeKey,
    comfyUrl, comfyModel, comfySampler, comfySteps, comfyCfg,
    comfyWidth, comfyHeight, comfyWorkflow,
    ratio, previewProvider, nbModel, falModel, geminiKey, nbKey, falKey,
    weavyUrl, weavyKey,
  } = settings;

  const comfyConfigured  = !!comfyUrl.trim();
  const weavyConfigured  = !!(weavyUrl.trim() && weavyKey.trim());
  const previewKey       = previewProvider === "gemini" ? geminiKey : previewProvider === "nanobanana" ? nbKey : falKey;
  const previewAvailable = !!previewKey.trim();
  const busy             = genBusy || renderBusy;

  // ── Refs ──────────────────────────────────────────────────────────────────
  function handleAddRef(file) {
    const idx = refs.length;
    setRefs(p => [...p, { file, preview:URL.createObjectURL(file), falUrl:null, uploading:false }]);
    if (falKey.trim()) uploadRef(file, idx);
  }
  async function uploadRef(file, index) {
    setRefs(p => p.map((r,i)=>i===index?{...r,uploading:true}:r));
    try {
      const url = await falUpload(falKey, file);
      setRefs(p => p.map((r,i)=>i===index?{...r,falUrl:url,uploading:false}:r));
    } catch(e) { setRefs(p => p.map((r,i)=>i===index?{...r,uploading:false}:r)); }
  }
  function handleRemoveRef(index) { setRefs(p=>p.filter((_,i)=>i!==index)); }

  // ── Build prompts via Claude ───────────────────────────────────────────────
  async function buildPrompts(feedback=null) {
    const refContext = refs.length ? `${refs.length} reference image${refs.length>1?"s":""} provided.` : "No reference images.";
    const userMsg = `VISUAL BIBLE:\n${bible}\n\n---\n\nFRAME 1 — START FRAME:\n${frame1}\n\n---\n\nFRAME 2 — END FRAME:\n${frame2}\n\n---\n\nREFERENCES: ${refContext}${feedback?`\n\n---\n\nDIRECTOR FEEDBACK TO APPLY:\n${feedback}`:""}\n\nSHOT LOG:\n${log.length?log.map((s,i)=>`#${i+1}: ${s.shotSummary}`).join("\n"):"No previous shots."}`;
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method:"POST", headers:{"Content-Type":"application/json", "x-api-key": claudeKey, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true"},
      body: JSON.stringify({ model:"claude-opus-4-6", max_tokens:1400, system:SYSTEM_PROMPT, messages:[{role:"user",content:userMsg}] }),
    });
    const data = await res.json();
    const txt = data.content?.map(b=>b.text||"").join("")||"";
    return JSON.parse(txt.replace(/```json|```/g,"").trim());
  }

  // ── Render via ComfyUI ────────────────────────────────────────────────────
  async function renderWithComfy(parsed) {
    setActiveRenderMode("comfy");
    setStartLoading(true); setEndLoading(true);
    setStartMsg("Queuing in ComfyUI…"); setEndMsg("Queuing in ComfyUI…");
    setStartErr(null); setEndErr(null); setStartImg(null); setEndImg(null);

    const comfySettings = { modelName:comfyModel, width:comfyWidth, height:comfyHeight, steps:comfySteps };
    const customWf = comfyWorkflow.trim() || null;

    const renderFrame = async (frame, setImg, setLoad, setMsg, setErr) => {
      try {
        setMsg("Rendering in ComfyUI…");
        const url = await comfyRenderFrame(comfyUrl, null, frame.prompt, frame.negativePrompt, customWf, comfySettings);
        setImg(url); setMsg(""); return url;
      } catch(e) {
        setErr(e.message.includes("fetch")||e.message.includes("Failed")?"Cannot reach ComfyUI server — check URL":e.message.slice(0,80));
        return null;
      } finally { setLoad(false); }
    };

    // Run both frames, start first then end (ComfyUI queue usually serial)
    const s1 = await renderFrame(parsed.startFrame, setStartImg, setStartLoading, setStartMsg, setStartErr);
    const s2 = await renderFrame(parsed.endFrame,   setEndImg,   setEndLoading,   setEndMsg,   setEndErr);
    return { s1, s2 };
  }

  // ── Render via preview provider ───────────────────────────────────────────
  async function renderWithPreview(parsed) {
    setActiveRenderMode("preview");
    setStartLoading(true); setEndLoading(true);
    setStartMsg("Preview rendering…"); setEndMsg("Preview rendering…");
    setStartErr(null); setEndErr(null); setStartImg(null); setEndImg(null);

    let uploadedRefs = [...refs];
    if (falKey.trim()) {
      uploadedRefs = await Promise.all(refs.map(async(r,i)=>{
        if (r.file&&!r.falUrl) { try { const url=await falUpload(falKey,r.file); setRefs(p=>p.map((x,idx)=>idx===i?{...x,falUrl:url}:x)); return {...r,falUrl:url}; } catch(e){return r;} }
        return r;
      }));
    }
    const refUrls = uploadedRefs.map(r=>r.falUrl).filter(Boolean);

    const renderFrame = async(frame, setImg, setLoad, setMsg, setErr) => {
      try {
        let url;
        if (previewProvider==="gemini") {
          url = await geminiGenerateImage(geminiKey, frame.prompt, ratio);
        } else if (previewProvider==="nanobanana") {
          url = await nbGenerate(nbKey, nbModel, frame.prompt, frame.negativePrompt, ratio, refUrls.length?refUrls:undefined);
        } else {
          url = await falText2Img(falKey, falModel, frame.prompt, frame.negativePrompt, ratio);
        }
        setImg(url); setMsg(""); return url;
      } catch(e) {
        setErr(e.message.includes("401")?"Invalid API key":e.message.includes("402")?"Insufficient credits":"Preview failed");
        return null;
      } finally { setLoad(false); }
    };

    const [s1,s2] = await Promise.all([
      renderFrame(parsed.startFrame, setStartImg, setStartLoading, setStartMsg, setStartErr),
      renderFrame(parsed.endFrame,   setEndImg,   setEndLoading,   setEndMsg,   setEndErr),
    ]);
    return { s1, s2 };
  }

  // ── Post to Weavy ─────────────────────────────────────────────────────────
  async function postToWeavy(parsed, s1, s2, ver) {
    if (!weavyConfigured) return;
    setWeavyPosting(true);
    try {
      const uid = `da-${parsed.sceneSlug||"shot"}`;
      setWeavyAppUid(uid);
      await weavyUpsertApp(weavyUrl, weavyKey, uid, `Shot: ${(parsed.shotSummary||"").slice(0,60)}`);
      setThreadUrl(`${weavyUrl}/messenger/${uid}`);
      const verStr = ver>1?` — v${ver}`:"";
      const fbStr = parsed.feedbackApplied&&parsed.feedbackApplied!=="null"?`\n\n📝 Feedback applied: ${parsed.feedbackApplied}`:"";
      await weavyPostMessage(weavyUrl, weavyKey, uid,
        `🎬 **${parsed.shotSummary}**${verStr}${fbStr}\n\n↗ Motion arc: ${parsed.motionArc||"—"}\n\n▶ Frame 1 — Start:\n${s1||"(render failed)"}\n\n⏹ Frame 2 — End:\n${s2||"(render failed)"}\n\n---\n_Reply with feedback to regenerate, or "approved" to mark ready._`
      );
      const msgs = await weavyGetMessages(weavyUrl, weavyKey, uid);
      setWeavyMessages(msgs);
      setWeavyPostStatus("ok");
      setShowWeavyPanel(true);
    } catch(e) { console.error("Weavy post failed:",e); setWeavyPostStatus("error"); }
    setWeavyPosting(false);
  }

  // ── Main generate ─────────────────────────────────────────────────────────
  async function handleGenerate(renderMode) {
    if (!canGenerate) return;
    const ver = 1;
    setVersion(ver);
    setGenBusy(true); setGenError(null); setShot(null);
    setWeavyPostStatus("idle"); setFeedbackFound(false);
    try {
      const parsed = await buildPrompts();
      setShot(parsed);
      setLog(p=>[...p,{...parsed,_version:ver}].slice(-20));
      setRenderBusy(true);
      const { s1, s2 } = renderMode==="comfy" ? await renderWithComfy(parsed) : await renderWithPreview(parsed);
      setRenderBusy(false);
      await postToWeavy(parsed, s1, s2, ver);
    } catch(e) {
      console.error("Generation error full:", e);
      const msg = e.message?.includes("401") ? "Invalid Claude API key — check Settings"
        : e.message?.includes("403") ? "Claude API key forbidden — check console.anthropic.com"
        : e.message?.includes("429") ? "Rate limit — wait a moment and retry"
        : e.message?.includes("Failed to fetch") || e.message?.includes("NetworkError") ? "Network error — check your internet and API key"
        : `Generation failed: ${e.message || "unknown error"}`;
      setGenError(msg);
      setRenderBusy(false);
    }
    setGenBusy(false);
  }

  // ── Check feedback + regenerate ───────────────────────────────────────────
  async function handleCheckFeedback() {
    if (!weavyConfigured||!weavyAppUid) return;
    setCheckingFeedback(true); setFeedbackFound(false);
    try {
      const msgs = await weavyGetMessages(weavyUrl, weavyKey, weavyAppUid);
      setWeavyMessages(msgs);
      const directorMsgs = msgs.filter(m=>!m.text?.startsWith("🎬")&&m.plain?.trim());
      const latest = directorMsgs[directorMsgs.length-1];
      if (!latest) { setCheckingFeedback(false); return; }
      const feedback = latest.plain||latest.text||"";
      if (feedback.toLowerCase().includes("approved")||feedback.toLowerCase().includes("approve")) {
        await weavyPostMessage(weavyUrl, weavyKey, weavyAppUid, "✅ Shot approved. Ready for ComfyUI production render.");
        const refreshed = await weavyGetMessages(weavyUrl, weavyKey, weavyAppUid);
        setWeavyMessages(refreshed); setCheckingFeedback(false); return;
      }
      setFeedbackFound(true);
      const newVer = version+1;
      setVersion(newVer);
      setShowWeavyPanel(false);
      setGenBusy(true); setRenderBusy(true);
      const parsed = await buildPrompts(feedback);
      setShot(parsed);
      setLog(p=>[...p,{...parsed,_version:newVer}].slice(-20));
      const { s1, s2 } = comfyConfigured ? await renderWithComfy(parsed) : await renderWithPreview(parsed);
      setRenderBusy(false); setGenBusy(false);
      await postToWeavy(parsed, s1, s2, newVer);
    } catch(e) { console.error("Feedback check failed:",e); }
    setCheckingFeedback(false);
  }

  // ── Re-render ─────────────────────────────────────────────────────────────
  async function handleRerender(mode) {
    if (!shot) return;
    setRenderBusy(true);
    const { s1, s2 } = mode==="comfy" ? await renderWithComfy(shot) : await renderWithPreview(shot);
    setRenderBusy(false);
    await postToWeavy(shot, s1, s2, version);
  }

  const canGenerate = !busy && bible.trim().length>40 && frame1.trim().length>20 && frame2.trim().length>20;

  return (
    <div style={{ minHeight:"100vh", background:"#09090a", color:"#e8e0d4", fontFamily:"Georgia,'Times New Roman',serif" }}>
      <style>{css}</style>
      <div style={{ position:"fixed", inset:0, opacity:.03, pointerEvents:"none", zIndex:200, backgroundImage:`url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`, backgroundSize:"120px" }} />
      <div style={{ position:"fixed", inset:0, pointerEvents:"none", zIndex:1, background:"radial-gradient(ellipse 80% 35% at 50% 0%, rgba(180,120,40,.05) 0%, transparent 55%)" }} />

      {/* ── Nav ── */}
      <nav style={{ height:50, borderBottom:"1px solid rgba(255,255,255,.065)", display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 22px", position:"sticky", top:0, background:"rgba(9,9,10,.96)", backdropFilter:"blur(8px)", zIndex:100 }}>
        <div style={{ display:"flex", alignItems:"baseline", gap:10 }}>
          <span style={{ fontSize:14, fontWeight:400, letterSpacing:".08em" }}>DIRECTOR AGENT</span>
          <span style={{ fontSize:9, letterSpacing:".16em", color:"rgba(200,160,80,.38)", textTransform:"uppercase", fontFamily:"sans-serif" }}>Start · End Frame</span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          {/* Status dots */}
          <div style={{ display:"flex", gap:10, alignItems:"center" }}>
            <div style={{ display:"flex", alignItems:"center", gap:5 }}>
              <span style={{ width:6, height:6, borderRadius:"50%", background:comfyConfigured?"rgba(180,130,255,.8)":"rgba(255,255,255,.12)" }} />
              <span style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif" }}>{comfyConfigured?"ComfyUI":"No ComfyUI"}</span>
            </div>
            {previewAvailable && (
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <span style={{ width:6, height:6, borderRadius:"50%", background:"rgba(200,160,80,.6)" }} />
                <span style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif" }}>Preview</span>
              </div>
            )}
            {weavyConfigured && (
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <span style={{ width:6, height:6, borderRadius:"50%", background:weavyPostStatus==="ok"?"#5cb87a":weavyPosting?"#c8a050":"rgba(80,180,120,.4)" }} />
                <span style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif" }}>Weavy</span>
              </div>
            )}
          </div>
          {weavyAppUid && weavyConfigured && (
            <button onClick={()=>setShowWeavyPanel(!showWeavyPanel)}
              style={{ background:showWeavyPanel?"rgba(80,180,120,.15)":"rgba(80,180,120,.07)", border:"1px solid rgba(80,180,120,.28)", borderRadius:6, padding:"5px 13px", cursor:"pointer", fontSize:10, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(80,180,120,.85)" }}>
              💬 {showWeavyPanel?"Hide":"Review"}
            </button>
          )}
          <button onClick={()=>setShowSettings(true)} style={{ background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.08)", borderRadius:6, padding:"5px 13px", cursor:"pointer", fontSize:10, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(232,224,212,.45)" }}>⚙ Settings</button>
        </div>
      </nav>

      {/* ── Two-column ── */}
      <div style={{ display:"grid", gridTemplateColumns:"420px 1fr", minHeight:"calc(100vh - 50px)", position:"relative", zIndex:2 }}>

        {/* LEFT */}
        <div style={{ borderRight:"1px solid rgba(255,255,255,.055)", padding:"22px 20px 120px", overflowY:"auto", maxHeight:"calc(100vh - 50px)", position:"sticky", top:50, display:"flex", flexDirection:"column", gap:18 }}>

          <div>
            <FieldLabel main="Visual Bible" sub="Permanent universe rulebook — materials, constraints, reinterpretation rules, characters, lighting, colour, style keywords" />
            <AutoTA value={bible} onChange={setBible} placeholder="Your complete visual universe — materials, what cannot exist, reinterpretation rules, characters, lighting, colour palette, tone, style keywords…" minHeight={200} fontSize={12} />
          </div>

          <Divider />

          <div style={{ display:"grid", gap:12 }}>
            <FrameInput number={1} label="Frame 1 — Start" value={frame1} onChange={setFrame1} placeholder="Framing & camera, hero element with exact specs, emotional tone, material reminder, subject position and action state at the START…" />
            <FrameInput number={2} label="Frame 2 — End"   value={frame2} onChange={setFrame2} placeholder="Same framing & camera (or state if it moves), same hero element details, emotional resolution, subject position and action state at the END…" />
          </div>

          <Divider />

          <div>
            <FieldLabel main="References" sub="Optional — character, style, mood, environment. Used for preview render consistency." />
            <RefZone refs={refs} onAdd={handleAddRef} onRemove={handleRemoveRef} />
          </div>

          {/* Generate buttons */}
          <div style={{ display:"grid", gap:9 }}>
            {/* ComfyUI — primary */}
            <button onClick={()=>handleGenerate("comfy")} disabled={!canGenerate||!comfyConfigured}
              style={{ width:"100%", padding:"14px", borderRadius:8, border:`1px solid ${canGenerate&&comfyConfigured?"rgba(130,80,200,.5)":"rgba(255,255,255,.06)"}`, background:canGenerate&&comfyConfigured?"rgba(130,80,200,.15)":"rgba(255,255,255,.018)", color:canGenerate&&comfyConfigured?"rgba(180,130,255,.95)":"rgba(232,224,212,.18)", fontSize:11, letterSpacing:".16em", textTransform:"uppercase", fontFamily:"sans-serif", fontWeight:700, cursor:canGenerate&&comfyConfigured?"pointer":"not-allowed", transition:"all .25s", display:"flex", alignItems:"center", justifyContent:"center", gap:10 }}>
              {genBusy&&activeRenderMode==="comfy"?<><Spin color="rgba(130,80,200,.3)" top="rgba(180,130,255,1)"/>Rendering in ComfyUI…</>:"🎨 Generate via ComfyUI"}
            </button>
            {!comfyConfigured && <p style={{ fontSize:9, color:"rgba(180,130,255,.4)", fontFamily:"sans-serif", textAlign:"center", marginTop:-4, fontStyle:"italic" }}>Add your ComfyUI server URL in Settings</p>}

            {/* Preview — secondary */}
            <button onClick={()=>handleGenerate("preview")} disabled={!canGenerate||!previewAvailable}
              style={{ width:"100%", padding:"11px", borderRadius:8, border:`1px solid ${canGenerate&&previewAvailable?"rgba(200,160,80,.35)":"rgba(255,255,255,.05)"}`, background:canGenerate&&previewAvailable?"rgba(200,160,80,.08)":"rgba(255,255,255,.012)", color:canGenerate&&previewAvailable?"rgba(200,160,80,.8)":"rgba(232,224,212,.15)", fontSize:10, letterSpacing:".14em", textTransform:"uppercase", fontFamily:"sans-serif", fontWeight:700, cursor:canGenerate&&previewAvailable?"pointer":"not-allowed", transition:"all .25s", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
              {genBusy&&activeRenderMode==="preview"?<><Spin/>Quick preview rendering…</>:"👁 Quick Preview"}
            </button>
            {!previewAvailable && canGenerate && <p style={{ fontSize:9, color:"rgba(200,160,80,.3)", fontFamily:"sans-serif", textAlign:"center", marginTop:-4, fontStyle:"italic" }}>Add NanoBanana or fal.ai key in Settings for preview</p>}

            {!canGenerate && !busy && (
              <p style={{ fontSize:9, color:"rgba(232,224,212,.18)", fontFamily:"sans-serif", textAlign:"center", fontStyle:"italic" }}>
                {!bible.trim()?"Add your Visual Bible to continue":!frame1.trim()?"Describe Frame 1 to continue":!frame2.trim()?"Describe Frame 2 to continue":""}
              </p>
            )}
            {genError && <div style={{ padding:"9px 12px", background:"rgba(180,60,60,.08)", border:"1px solid rgba(180,60,60,.16)", borderRadius:6, fontSize:11, color:"#e08080", fontFamily:"sans-serif" }}>{genError}</div>}
          </div>
        </div>

        {/* RIGHT */}
        <div style={{ position:"relative", overflow:"hidden" }}>
          <div style={{ padding:"22px 24px 120px", overflowY:"auto", maxHeight:"calc(100vh - 50px)", opacity:showWeavyPanel?.3:1, transition:"opacity .2s", pointerEvents:showWeavyPanel?"none":"auto" }}>
            {!shot&&!busy&&(
              <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", height:"80%", gap:14, opacity:.2 }}>
                <div style={{ fontSize:44 }}>🎬</div>
                <div style={{ fontSize:13, color:"rgba(232,224,212,.6)", fontFamily:"sans-serif", fontStyle:"italic" }}>Frame 1 and Frame 2 will appear here</div>
                <div style={{ display:"flex", gap:16, alignItems:"center" }}>
                  {comfyConfigured && <span style={{ fontSize:11, color:"rgba(180,130,255,.5)", fontFamily:"sans-serif" }}>🎨 ComfyUI connected</span>}
                  {weavyConfigured && <span style={{ fontSize:11, color:"rgba(80,180,120,.4)", fontFamily:"sans-serif" }}>💬 Weavy connected</span>}
                </div>
              </div>
            )}

            {shot && (
              <div style={{ animation:"fadeIn .5s ease both" }}>
                <div style={{ marginBottom:14, paddingBottom:14, borderBottom:"1px solid rgba(255,255,255,.055)" }}>
                  <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", gap:12, flexWrap:"wrap" }}>
                    <div>
                      <div style={{ fontSize:14, color:"#e8e0d4", lineHeight:1.45, marginBottom:4 }}>{shot.shotSummary}</div>
                      <div style={{ fontSize:11, color:"rgba(232,224,212,.32)", fontFamily:"sans-serif", fontStyle:"italic", lineHeight:1.5 }}>{shot.sharedContext}</div>
                    </div>
                    <div style={{ display:"flex", gap:7, flexShrink:0 }}>
                      {version>1 && <span style={{ fontSize:10, padding:"3px 9px", background:"rgba(200,160,80,.1)", border:"1px solid rgba(200,160,80,.22)", borderRadius:20, color:"rgba(200,160,80,.7)", fontFamily:"sans-serif" }}>v{version}</span>}
                      {activeRenderMode && <span style={{ fontSize:10, padding:"3px 9px", background:activeRenderMode==="comfy"?"rgba(130,80,200,.12)":"rgba(200,160,80,.08)", border:`1px solid ${activeRenderMode==="comfy"?"rgba(130,80,200,.3)":"rgba(200,160,80,.2)"}`, borderRadius:20, color:activeRenderMode==="comfy"?"rgba(180,130,255,.8)":"rgba(200,160,80,.6)", fontFamily:"sans-serif" }}>{activeRenderMode==="comfy"?"🎨 ComfyUI":"👁 Preview"}</span>}
                    </div>
                  </div>
                  {shot.feedbackApplied&&shot.feedbackApplied!=="null"&&<div style={{ marginTop:9, padding:"8px 12px", background:"rgba(80,180,120,.06)", border:"1px solid rgba(80,180,120,.15)", borderRadius:6, fontSize:11, color:"rgba(80,180,120,.75)", fontFamily:"sans-serif" }}>📝 {shot.feedbackApplied}</div>}
                </div>

                <MotionArc text={shot.motionArc} />

                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:16 }}>
                  <FramePanel number={1} label="Start Frame" frame={shot.startFrame} imageUrl={startImg} loading={startLoading} loadingMsg={startMsg} error={startErr} renderMode={activeRenderMode==="comfy"?"ComfyUI":activeRenderMode==="preview"?PREVIEW_PROVIDERS[settings.previewProvider].label:null} />
                  <FramePanel number={2} label="End Frame"   frame={shot.endFrame}   imageUrl={endImg}   loading={endLoading}   loadingMsg={endMsg}   error={endErr}   renderMode={activeRenderMode==="comfy"?"ComfyUI":activeRenderMode==="preview"?PREVIEW_PROVIDERS[settings.previewProvider].label:null} />
                </div>

                <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:10 }}>
                  <AuditPanel audit={shot.auditResult} />
                  <div style={{ display:"flex", gap:8, alignItems:"center", flexWrap:"wrap" }}>
                    {!busy && comfyConfigured && <button onClick={()=>handleRerender("comfy")} style={{ background:"rgba(130,80,200,.1)", border:"1px solid rgba(130,80,200,.28)", borderRadius:6, color:"rgba(180,130,255,.75)", padding:"6px 12px", fontSize:10, letterSpacing:".1em", textTransform:"uppercase", cursor:"pointer", fontFamily:"sans-serif" }}>🎨 Re-render Comfy</button>}
                    {!busy && previewAvailable && <button onClick={()=>handleRerender("preview")} style={{ background:"transparent", border:"1px solid rgba(200,160,80,.2)", borderRadius:6, color:"rgba(200,160,80,.55)", padding:"6px 12px", fontSize:10, letterSpacing:".1em", textTransform:"uppercase", cursor:"pointer", fontFamily:"sans-serif" }}>👁 Re-preview</button>}
                    {weavyPostStatus==="ok"&&!showWeavyPanel&&<button onClick={()=>setShowWeavyPanel(true)} style={{ background:"rgba(80,180,120,.1)", border:"1px solid rgba(80,180,120,.25)", borderRadius:6, color:"rgba(80,180,120,.8)", padding:"6px 12px", fontSize:10, letterSpacing:".1em", textTransform:"uppercase", cursor:"pointer", fontFamily:"sans-serif" }}>💬 Review Thread</button>}
                    {weavyPostStatus==="error"&&<span style={{ fontSize:10, color:"rgba(220,100,100,.55)", fontFamily:"sans-serif" }}>Weavy post failed</span>}
                  </div>
                </div>
              </div>
            )}
          </div>

          {showWeavyPanel && <WeavyPanel shot={shot} threadUrl={threadUrl} messages={weavyMessages} onCheckFeedback={handleCheckFeedback} onClose={()=>setShowWeavyPanel(false)} checkingFeedback={checkingFeedback} feedbackFound={feedbackFound} />}
        </div>
      </div>

      <LogStrip log={log} onSelect={s=>setShot(s)} />
      {showSettings && <Settings s={settings} set={setSettings} onClose={()=>setShowSettings(false)} />}
    </div>
  );
}

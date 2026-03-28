import { useState, useRef, useEffect } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// SYSTEM PROMPT
// ─────────────────────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are a Director of Photography and Prompt Architect for an AI film production pipeline.

Your output — a START FRAME and END FRAME — will be fed directly into a video generation model as the first and last frames of a shot. The model generates everything in between. Your two frames must be PHYSICALLY INTERPOLATABLE.

You receive:
1. VISUAL BIBLE — the universe rulebook governing every material, surface, character, and environment.
2. SHOT ASSETS — locked identity references. ASSET ENFORCEMENT RULE: Asset descriptions override any conflicting description in the frame inputs. Do not improvise or alter them.
   - ENVIRONMENT SHEET: exact setting — reproduce faithfully
   - CHARACTER SHEETS: exact identity per character — appearance, costume, expression, distinctive features locked
   - OBJECT SHEETS: exact description of hero props — scale, appearance, behaviour locked
3. CAMERA LOCK — focal length, framing, height, angle, movement. Immutable across BOTH frames.
4. FRAME 1 DESCRIPTION — complete instruction for the start frame.
5. FRAME 2 DESCRIPTION — complete instruction for the end frame.
6. DIRECTOR FEEDBACK (optional) — apply this feedback only. Keep everything not criticised.

━━━ VIDEO INTERPOLATION CONSTRAINTS ━━━

CAMERA LOCK — identical position, height, angle, focal length in both frames unless movement explicitly described.
LIGHTING FREEZE — same key light direction, shadow positions, light source states in both frames.
SPATIAL PLAUSIBILITY — Frame 2 subject position reachable from Frame 1 in one continuous shot.
IMPLIED MOTION — Frame 1 compositionally suggests the motion Frame 2 resolves.
LOCKS — costume, hair, environment, set dressing identical. Only subject position, expression, gesture, action state may change.

━━━ VISUAL BIBLE — THREE BINDING LAYERS ━━━

LAYER 1 — POSITIVE MATERIAL LANGUAGE: Only approved materials may appear in any prompt.
LAYER 2 — NEGATIVE MATERIAL CONSTRAINT: All prohibited materials go in negative prompts AND are avoided in positive prompts.
LAYER 3 — REINTERPRETATION RULE: Every object that would normally use a prohibited material must be translated through the bible's approved material system. No exceptions.

━━━ PROMPT CONSTRUCTION ━━━

For each frame write in this order:
1. CAMERA — state lens, framing, and subject scale: "subject occupies [X]% of frame height, positioned [left/centre/right] third, approximately [X] metres from camera"
2. SUBJECT — position, action state, expression for this specific frame
3. ENVIRONMENT — through bible material language only
4. LIGHTING — from bible verbatim, identical in both frames
5. STYLE KEYWORDS — from bible

For Frame 2 open with: "IDENTICAL CAMERA TO FRAME 1 — [repeat exact camera description from Frame 1]" then describe only what changes.

Each prompt 130-160 words.

━━━ MANDATORY CHECKLIST ━━━

Check all before outputting:
- Camera identical in both frames?
- Lighting frozen — same shadow positions?
- Frame 2 position spatially reachable from Frame 1?
- Frame 1 implies the motion Frame 2 resolves?
- Costume, hair, environment locked?
- Every element translated through bible material system?
- All prohibited materials in negative prompts?
- Hero elements exact per asset sheets?
- Frame 2 opens with IDENTICAL CAMERA TO FRAME 1?
- Subject scale described as percentage of frame height?
- Director feedback applied if provided?

Rewrite any non-compliant prompt before outputting.

━━━ OUTPUT — valid JSON only, no markdown, no preamble ━━━

{
  "shotSummary": "one sentence: complete motion arc Frame 1 to Frame 2",
  "sceneSlug": "kebab-case-max-5-words",
  "sharedContext": "camera, lighting, environment, costume locked across both frames in bible material language",
  "motionArc": "what physically moves, how far, in what direction",
  "feedbackApplied": "what changed from previous version, or null",
  "startFrame": {
    "compositionNote": "subject position and action state at Frame 1",
    "prompt": "full bible-compliant prompt 130-160 words",
    "negativePrompt": "all prohibited materials and treatments from the bible"
  },
  "endFrame": {
    "compositionNote": "subject position and action state at Frame 2",
    "prompt": "full bible-compliant prompt 130-160 words",
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
// COMFYUI — Flux.1 dev fp8 workflow
// ─────────────────────────────────────────────────────────────────────────────
function buildComfyWorkflow(positivePrompt, negativePrompt, modelName, width, height, steps, seed) {
  return {
    "1": { inputs: { ckpt_name: modelName || "FLUX1/flux1-dev-fp8.safetensors" }, class_type: "CheckpointLoaderSimple" },
    "2": { inputs: { width: width || 1024, height: height || 576, batch_size: 1 }, class_type: "EmptyLatentImage" },
    "3": { inputs: { text: positivePrompt, clip: ["1", 1] }, class_type: "CLIPTextEncode" },
    "4": { inputs: { text: negativePrompt || "blurry, deformed, low quality", clip: ["1", 1] }, class_type: "CLIPTextEncode" },
    "5": { inputs: { guidance: 3.5, conditioning: ["3", 0] }, class_type: "FluxGuidance" },
    "6": {
      inputs: {
        seed: seed || Math.floor(Math.random() * 999999999),
        steps: steps || 20, cfg: 1.0, sampler_name: "euler", scheduler: "simple", denoise: 1.0,
        model: ["1", 0], positive: ["5", 0], negative: ["4", 0], latent_image: ["2", 0],
      },
      class_type: "KSampler",
    },
    "7": { inputs: { samples: ["6", 0], vae: ["1", 2] }, class_type: "VAEDecode" },
    "8": { inputs: { filename_prefix: "director-agent", images: ["7", 0] }, class_type: "SaveImage" },
  };
}

function injectIntoCustomWorkflow(wfJson, pos, neg) {
  const wf = JSON.parse(JSON.stringify(wfJson));
  let posId = null;
  let negId = null;
  for (const [, node] of Object.entries(wf)) {
    if (node.class_type === "KSampler" || node.class_type === "KSamplerAdvanced") {
      posId = node.inputs && node.inputs.positive && node.inputs.positive[0];
      negId = node.inputs && node.inputs.negative && node.inputs.negative[0];
      break;
    }
  }
  if (posId && wf[posId]) wf[posId].inputs.text = pos;
  if (negId && wf[negId]) wf[negId].inputs.text = neg;
  return wf;
}

async function comfySubmit(serverUrl, workflow) {
  const res = await fetch(serverUrl + "/prompt", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: workflow, client_id: "director-agent" }),
  });
  if (!res.ok) throw new Error("ComfyUI " + res.status + ": " + (await res.text()));
  return (await res.json()).prompt_id;
}

async function comfyPoll(serverUrl, promptId) {
  for (let i = 0; i < 120; i++) {
    await new Promise(r => setTimeout(r, 2500));
    const res = await fetch(serverUrl + "/history/" + promptId);
    if (!res.ok) continue;
    const data = await res.json();
    const entry = data[promptId];
    if (!entry) continue;
    if (entry.status && entry.status.status_str === "success" || entry.outputs) {
      for (const nodeOut of Object.values(entry.outputs || {})) {
        if (nodeOut.images && nodeOut.images.length) {
          const img = nodeOut.images[0];
          return serverUrl + "/view?filename=" + encodeURIComponent(img.filename) + "&subfolder=" + encodeURIComponent(img.subfolder || "") + "&type=" + (img.type || "output");
        }
      }
    }
    if (entry.status && entry.status.status_str === "error") throw new Error("ComfyUI render failed");
  }
  throw new Error("ComfyUI: timeout");
}

// ─────────────────────────────────────────────────────────────────────────────
// PREVIEW PROVIDERS
// ─────────────────────────────────────────────────────────────────────────────
const PREVIEW_PROVIDERS = {
  gemini: {
    label: "Google AI Studio", icon: "✦",
    models: [{ id: "gemini-3.1-flash-image-preview", label: "Gemini 3.1 Flash", desc: "Fast, high quality" }],
  },
  nanobanana: {
    label: "NanoBanana", icon: "🍌",
    models: [
      { id: "nano-banana-2", label: "Nano Banana 2", desc: "4K" },
      { id: "nano-banana-pro", label: "Nano Banana Pro", desc: "Premium" },
    ],
  },
  fal: {
    label: "Flux", icon: "⚡",
    models: [
      { id: "fal-ai/flux/dev", label: "Flux Dev", desc: "Best fidelity" },
      { id: "fal-ai/flux/schnell", label: "Flux Schnell", desc: "Fastest" },
    ],
  },
};

const ASPECT_RATIOS = [
  { id: "landscape_16_9", label: "16:9", nb: "16:9", width: 1024, height: 576 },
  { id: "landscape_21_9", label: "21:9", nb: "21:9", width: 1280, height: 544 },
  { id: "landscape_4_3",  label: "4:3",  nb: "4:3",  width: 1024, height: 768 },
];

const LENS_OPTIONS = [
  { id: "24mm",  label: "24mm",  desc: "Ultra wide · expansive, environmental" },
  { id: "35mm",  label: "35mm",  desc: "Wide · natural, cinematic" },
  { id: "50mm",  label: "50mm",  desc: "Normal · closest to human eye" },
  { id: "85mm",  label: "85mm",  desc: "Portrait · flattering, compressed" },
  { id: "135mm", label: "135mm", desc: "Telephoto · compressed, isolating" },
];

const FRAMING_OPTIONS = [
  { id: "extreme_wide",  label: "Extreme Wide",  desc: "Full environment, subject tiny" },
  { id: "wide",          label: "Wide",          desc: "Subject + full environment" },
  { id: "medium",        label: "Medium",        desc: "Waist up" },
  { id: "close_up",      label: "Close-Up",      desc: "Face and shoulders" },
  { id: "extreme_close", label: "Extreme Close", desc: "Detail, expression" },
];

// ─────────────────────────────────────────────────────────────────────────────
// IMAGE GENERATION APIs
// ─────────────────────────────────────────────────────────────────────────────
async function geminiGenerateImage(apiKey, prompt, refBase64, refMimeType, seed) {
  const parts = [{ text: prompt }];
  if (refBase64 && refMimeType) {
    parts.push({ inlineData: { mimeType: refMimeType, data: refBase64 } });
  }
  const body = {
    contents: [{ parts }],
    generationConfig: {
      responseModalities: ["TEXT", "IMAGE"],
      seed: seed || Math.floor(Math.random() * 999999999),
    },
  };
  const res = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent?key=" + apiKey,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
  );
  if (!res.ok) throw new Error("Gemini " + res.status + ": " + (await res.text()));
  const data = await res.json();
  const resParts = (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) || [];
  for (const part of resParts) {
    if (part.inlineData && part.inlineData.mimeType && part.inlineData.mimeType.startsWith("image/")) {
      return {
        url: "data:" + part.inlineData.mimeType + ";base64," + part.inlineData.data,
        base64: part.inlineData.data,
        mimeType: part.inlineData.mimeType,
      };
    }
  }
  throw new Error("Gemini: no image in response");
}

async function nbGenerate(apiKey, model, prompt, negativePrompt, aspectRatio, refUrls) {
  const body = {
    prompt: negativePrompt ? prompt + " --no " + negativePrompt : prompt,
    selectedModel: model,
    aspect_ratio: (ASPECT_RATIOS.find(r => r.id === aspectRatio) || {}).nb || "16:9",
  };
  if (refUrls && refUrls.length) body.referenceImageUrls = refUrls;
  const res = await fetch("https://www.nananobanana.com/api/v1/generate", {
    method: "POST", headers: { "Content-Type": "application/json", "Authorization": "Bearer " + apiKey },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("NanaBanana " + res.status);
  const init = await res.json();
  const taskId = (init.data && init.data.id) || init.id;
  if (!taskId) throw new Error("NanoBanana: no task ID");
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 3000));
    const poll = await fetch("https://www.nananobanana.com/api/v1/generate/" + taskId, {
      headers: { "Authorization": "Bearer " + apiKey },
    });
    const pd = await poll.json();
    const status = (pd.data && pd.data.processingStatus) || pd.status;
    if (status === "completed" || status === "success") {
      const urls = (pd.data && pd.data.outputImageUrls) || pd.outputImageUrls;
      if (urls && urls.length) return urls[0];
      throw new Error("NanoBanana: no image URL");
    }
    if (status === "failed" || status === "error") throw new Error("NanoBanana: failed");
  }
  throw new Error("NanoBanana: timeout");
}

async function falText2Img(apiKey, model, prompt, negativePrompt, aspectRatio) {
  const res = await fetch("https://fal.run/" + model, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": "Key " + apiKey },
    body: JSON.stringify({
      prompt, negative_prompt: negativePrompt || "blurry, deformed, watermark",
      image_size: aspectRatio,
      num_inference_steps: model.includes("schnell") ? 4 : 28,
      guidance_scale: 3.5, num_images: 1, enable_safety_checker: false,
    }),
  });
  if (!res.ok) throw new Error("fal.ai " + res.status);
  const d = await res.json();
  return (d.images && d.images[0] && d.images[0].url) || (d.image && d.image.url);
}

async function falUpload(apiKey, file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("https://fal.run/storage/upload", {
    method: "POST", headers: { "Authorization": "Key " + apiKey }, body: fd,
  });
  if (!res.ok) throw new Error("Upload: " + res.status);
  return (await res.json()).url;
}

// ─────────────────────────────────────────────────────────────────────────────
// CROP LOCK
// ─────────────────────────────────────────────────────────────────────────────
async function cropToCentre(dataUrl, tw, th) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = tw; canvas.height = th;
      const ctx = canvas.getContext("2d");
      const sx = Math.max(0, (img.width - tw) / 2);
      const sy = Math.max(0, (img.height - th) / 2);
      ctx.drawImage(img, sx, sy, Math.min(img.width, tw), Math.min(img.height, th), 0, 0, tw, th);
      resolve(canvas.toDataURL("image/jpeg", 0.95));
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

async function applyCropLock(url1, url2) {
  if (!url1 || !url2) return { url1, url2 };
  if (!url1.startsWith("data:") || !url2.startsWith("data:")) return { url1, url2 };
  try {
    const getDims = url => new Promise(res => {
      const img = new Image();
      img.onload = () => res({ w: img.width, h: img.height });
      img.src = url;
    });
    const [d1, d2] = await Promise.all([getDims(url1), getDims(url2)]);
    const w = Math.min(d1.w, d2.w);
    const h = Math.min(d1.h, d2.h);
    const [c1, c2] = await Promise.all([cropToCentre(url1, w, h), cropToCentre(url2, w, h)]);
    return { url1: c1, url2: c2 };
  } catch (e) {
    console.warn("Crop lock failed:", e);
    return { url1, url2 };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// WEAVY
// ─────────────────────────────────────────────────────────────────────────────
async function weavyUpsertApp(envUrl, apiKey, uid, name) {
  const res = await fetch(envUrl + "/api/apps", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": "Bearer " + apiKey },
    body: JSON.stringify({ uid, name, type: "chat" }),
  });
  if (res.status === 409) {
    return (await fetch(envUrl + "/api/apps/" + uid, { headers: { "Authorization": "Bearer " + apiKey } })).json();
  }
  if (!res.ok) throw new Error("Weavy " + res.status);
  return res.json();
}

async function weavyPostMsg(envUrl, apiKey, appUid, text) {
  await fetch(envUrl + "/api/apps/" + appUid + "/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": "Bearer " + apiKey },
    body: JSON.stringify({ text }),
  });
}

async function weavyGetMsgs(envUrl, apiKey, appUid) {
  const res = await fetch(envUrl + "/api/apps/" + appUid + "/messages?order_by=created_at+desc&take=20", {
    headers: { "Authorization": "Bearer " + apiKey },
  });
  if (!res.ok) throw new Error("Weavy get " + res.status);
  return ((await res.json()).data || []).reverse();
}

// ─────────────────────────────────────────────────────────────────────────────
// CSS
// ─────────────────────────────────────────────────────────────────────────────
const css = `
  @keyframes spin    { to { transform: rotate(360deg); } }
  @keyframes fadeIn  { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
  @keyframes pulse   { 0%,100%{opacity:.4} 50%{opacity:1} }
  @keyframes shimmer { 0%{left:-100%} 100%{left:200%} }
  @keyframes slideIn { from{transform:translateX(100%);opacity:0} to{transform:translateX(0);opacity:1} }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:#09090a; }
  textarea,input,select { box-sizing:border-box; }
  ::-webkit-scrollbar { width:4px; }
  ::-webkit-scrollbar-thumb { background:rgba(200,160,80,.13); border-radius:2px; }
  .ta::placeholder { color:rgba(232,224,212,.15); font-style:italic; font-size:11px; line-height:1.65; }
  .ta:focus { border-color:rgba(200,160,80,.3) !important; outline:none; }
`;

// ─────────────────────────────────────────────────────────────────────────────
// ATOMS
// ─────────────────────────────────────────────────────────────────────────────
function Spin({ size }) {
  const s = size || 12;
  return (
    <span style={{ display:"inline-block", width:s, height:s, border:"2px solid rgba(200,160,80,.18)", borderTopColor:"#c8a050", borderRadius:"50%", animation:"spin .8s linear infinite", flexShrink:0 }} />
  );
}

function Toggle({ on, onToggle, activeColor }) {
  const bg = on ? (activeColor || "#c8a050") : "rgba(255,255,255,.12)";
  return (
    <button onClick={onToggle} style={{ width:38, height:22, borderRadius:11, background:bg, border:"none", cursor:"pointer", position:"relative", transition:"background .2s", flexShrink:0 }}>
      <span style={{ position:"absolute", top:3, left:on?18:3, width:16, height:16, borderRadius:"50%", background:"#fff", transition:"left .2s", display:"block" }} />
    </button>
  );
}

function CopyBtn({ text }) {
  const [ok, setOk] = useState(false);
  return (
    <button onClick={() => { navigator.clipboard.writeText(text); setOk(true); setTimeout(() => setOk(false), 2000); }}
      style={{ background:ok?"rgba(80,160,100,.1)":"rgba(200,160,80,.06)", border:"1px solid " + (ok?"rgba(80,160,100,.25)":"rgba(200,160,80,.15)"), borderRadius:4, padding:"3px 9px", cursor:"pointer", fontSize:10, letterSpacing:".1em", textTransform:"uppercase", color:ok?"#7dc493":"rgba(200,160,80,.55)", fontFamily:"sans-serif", transition:"all .2s" }}>
      {ok ? "✓" : "COPY"}
    </button>
  );
}

function FieldLabel({ main, sub }) {
  return (
    <div style={{ marginBottom:8 }}>
      <div style={{ fontSize:11, letterSpacing:".14em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.7)" }}>{main}</div>
      {sub && <div style={{ fontSize:10, color:"rgba(232,224,212,.4)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:3, lineHeight:1.45 }}>{sub}</div>}
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
      <input type={show?"text":"password"} value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        style={{ width:"100%", background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"9px 34px 9px 11px", fontFamily:"monospace", outline:"none" }}
        onFocus={e => { e.target.style.borderColor = "rgba(200,160,80,.28)"; }}
        onBlur={e => { e.target.style.borderColor = "rgba(255,255,255,.07)"; }} />
      <button onClick={() => setShow(!show)} style={{ position:"absolute", right:9, top:"50%", transform:"translateY(-50%)", background:"none", border:"none", color:"rgba(232,224,212,.28)", cursor:"pointer", fontSize:11 }}>{show ? "🙈" : "👁"}</button>
    </div>
  );
}

function TextInput({ value, onChange, placeholder }) {
  return (
    <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
      style={{ width:"100%", background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"9px 11px", fontFamily:"sans-serif", outline:"none" }}
      onFocus={e => { e.target.style.borderColor = "rgba(200,160,80,.28)"; }}
      onBlur={e => { e.target.style.borderColor = "rgba(255,255,255,.07)"; }} />
  );
}

function AutoTA({ value, onChange, placeholder, minHeight, fontSize }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) { ref.current.style.height = "auto"; ref.current.style.height = ref.current.scrollHeight + "px"; }
  }, [value]);
  return (
    <textarea ref={ref} className="ta" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
      style={{ width:"100%", minHeight:minHeight||80, background:"rgba(255,255,255,.02)", border:"1px solid rgba(255,255,255,.07)", borderRadius:9, color:"rgba(232,224,212,.9)", fontSize:fontSize||12, padding:"12px 13px", fontFamily:"Georgia,serif", resize:"none", lineHeight:1.72, transition:"border-color .2s", overflow:"hidden", display:"block" }} />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FRAME INPUT
// ─────────────────────────────────────────────────────────────────────────────
function FrameInput({ number, label, value, onChange, placeholder }) {
  const ref = useRef(null);
  const rgb = number === 1 ? "80,160,220" : "210,120,55";
  useEffect(() => {
    if (ref.current) { ref.current.style.height = "auto"; ref.current.style.height = ref.current.scrollHeight + "px"; }
  }, [value]);
  return (
    <div style={{ borderRadius:9, border:"1px solid " + (value ? "rgba(" + rgb + ",.28)" : "rgba(255,255,255,.07)"), overflow:"hidden", transition:"border-color .2s", background:"rgba(255,255,255,.015)" }}>
      <div style={{ padding:"9px 13px", background:"rgba(" + rgb + ",.07)", borderBottom:"1px solid rgba(" + rgb + ",.14)", display:"flex", alignItems:"center", gap:9 }}>
        <div style={{ width:22, height:22, borderRadius:5, background:"rgba(" + rgb + ",.15)", border:"1px solid rgba(" + rgb + ",.32)", display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <span style={{ fontSize:11, color:"rgba(" + rgb + ",1)", fontFamily:"sans-serif", fontWeight:700 }}>{number}</span>
        </div>
        <span style={{ fontSize:11, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(" + rgb + ",1)", fontWeight:700 }}>{label}</span>
      </div>
      <textarea ref={ref} className="ta" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        style={{ width:"100%", minHeight:100, background:"transparent", border:"none", borderRadius:0, color:"#e8e0d4", fontSize:13, padding:"12px 13px", fontFamily:"Georgia,serif", resize:"none", lineHeight:1.72, overflow:"hidden", display:"block", outline:"none" }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SHOT ASSETS
// ─────────────────────────────────────────────────────────────────────────────
function AssetField({ label, value, onChange, placeholder, multiline }) {
  const ref = useRef(null);
  useEffect(() => {
    if (multiline && ref.current) { ref.current.style.height = "auto"; ref.current.style.height = ref.current.scrollHeight + "px"; }
  }, [value, multiline]);
  const baseStyle = { width:"100%", background:"rgba(255,255,255,.03)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"8px 11px", fontFamily:"Georgia,serif", outline:"none" };
  return (
    <div>
      <div style={{ fontSize:10, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.5)", marginBottom:5 }}>{label}</div>
      {multiline
        ? <textarea ref={ref} value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
            style={{ ...baseStyle, resize:"none", lineHeight:1.6, minHeight:60, overflow:"hidden", display:"block" }}
            onFocus={e => { e.target.style.borderColor = "rgba(200,160,80,.28)"; }}
            onBlur={e => { e.target.style.borderColor = "rgba(255,255,255,.07)"; }} />
        : <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} style={baseStyle}
            onFocus={e => { e.target.style.borderColor = "rgba(200,160,80,.28)"; }}
            onBlur={e => { e.target.style.borderColor = "rgba(255,255,255,.07)"; }} />
      }
    </div>
  );
}

function AssetImageUpload({ assetKey, assetImages, setAssetImages, label }) {
  const inputRef = useRef();
  const img = assetImages[assetKey];
  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = e => {
      const dataUrl = e.target.result;
      const base64 = dataUrl.split(",")[1];
      setAssetImages(p => ({ ...p, [assetKey]: { preview: dataUrl, base64, mimeType: file.type } }));
    };
    reader.readAsDataURL(file);
  }
  return (
    <div>
      <div style={{ fontSize:10, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.5)", marginBottom:6 }}>{label} Reference Image</div>
      {img ? (
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <img src={img.preview} alt="" style={{ width:64, height:64, objectFit:"cover", borderRadius:7, border:"1px solid rgba(200,160,80,.3)", display:"block" }} />
          <div>
            <div style={{ fontSize:11, color:"rgba(80,180,120,.7)", fontFamily:"sans-serif", marginBottom:4 }}>✓ Reference uploaded</div>
            <button onClick={() => setAssetImages(p => { const n = { ...p }; delete n[assetKey]; return n; })}
              style={{ fontSize:11, color:"rgba(220,100,100,.5)", background:"none", border:"none", cursor:"pointer", fontFamily:"sans-serif", padding:0 }}>Remove</button>
          </div>
        </div>
      ) : (
        <div onClick={() => inputRef.current && inputRef.current.click()}
          style={{ borderRadius:7, border:"1px dashed rgba(200,160,80,.2)", background:"rgba(200,160,80,.03)", padding:"10px 13px", cursor:"pointer", display:"flex", alignItems:"center", gap:9 }}>
          <span style={{ fontSize:18, opacity:.4 }}>📎</span>
          <div style={{ fontSize:12, color:"rgba(232,224,212,.4)", fontFamily:"sans-serif", fontStyle:"italic" }}>Upload reference image</div>
        </div>
      )}
      <input ref={inputRef} type="file" accept="image/*" style={{ display:"none" }} onChange={e => { handleFile(e.target.files && e.target.files[0]); e.target.value = ""; }} />
    </div>
  );
}

function ShotAssets({ environment, setEnvironment, characters, setCharacters, objects, setObjects, assetTab, setAssetTab, assetImages, setAssetImages }) {
  const tabs = [
    { id:"environment", label:"Environment", icon:"🌿" },
    { id:"characters",  label:"Characters",  icon:"👤" },
    { id:"objects",     label:"Objects",      icon:"📦" },
  ];
  const updEnv  = (k, v) => setEnvironment(p => ({ ...p, [k]: v }));
  const updChar = (i, k, v) => setCharacters(p => p.map((c, idx) => idx === i ? { ...c, [k]: v } : c));
  const updObj  = (i, k, v) => setObjects(p => p.map((o, idx) => idx === i ? { ...o, [k]: v } : o));
  const addChar = () => setCharacters(p => [...p, { name:"", role:"", appearance:"", costume:"", expression:"", distinctive:"", notes:"" }]);
  const addObj  = () => setObjects(p => [...p, { name:"", description:"", scale:"", behaviour:"", notes:"" }]);
  const rmChar  = i => setCharacters(p => p.filter((_, idx) => idx !== i));
  const rmObj   = i => setObjects(p => p.filter((_, idx) => idx !== i));
  const totalAssets = ((environment.name || environment.setting) ? 1 : 0) + characters.filter(c => c.name || c.appearance).length + objects.filter(o => o.name || o.description).length;

  return (
    <div style={{ borderRadius:9, border:"1px solid rgba(200,160,80,.2)", background:"rgba(200,160,80,.02)" }}>
      <div style={{ padding:"10px 14px", background:"rgba(200,160,80,.07)", borderBottom:"1px solid rgba(200,160,80,.15)", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:14 }}>🗂</span>
          <div>
            <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:"rgba(200,160,80,.9)" }}>Shot Assets</div>
            <div style={{ fontSize:10, fontFamily:"sans-serif", color:"rgba(232,224,212,.4)", fontStyle:"italic", marginTop:1 }}>Identity references — locked across every frame</div>
          </div>
        </div>
        {totalAssets > 0 && <span style={{ fontSize:10, padding:"2px 8px", background:"rgba(200,160,80,.15)", borderRadius:10, color:"rgba(200,160,80,.8)", fontFamily:"sans-serif" }}>{totalAssets} asset{totalAssets > 1 ? "s" : ""}</span>}
      </div>
      <div style={{ display:"flex", borderBottom:"1px solid rgba(255,255,255,.07)" }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setAssetTab(t.id)}
            style={{ flex:1, padding:"9px 6px", background:assetTab===t.id?"rgba(200,160,80,.09)":"transparent", border:"none", borderBottom:assetTab===t.id?"2px solid #c8a050":"2px solid transparent", cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", gap:5, transition:"all .15s" }}>
            <span style={{ fontSize:13 }}>{t.icon}</span>
            <span style={{ fontSize:11, fontFamily:"sans-serif", fontWeight:700, color:assetTab===t.id?"#c8a050":"rgba(232,224,212,.45)" }}>{t.label}</span>
          </button>
        ))}
      </div>
      <div style={{ padding:"14px", display:"grid", gap:14 }}>
        {assetTab === "environment" && (
          <>
            <AssetField label="Location Name" value={environment.name} onChange={v => updEnv("name", v)} placeholder="e.g. The Woolwich Garden" />
            <AssetField label="Setting Description" value={environment.setting} onChange={v => updEnv("setting", v)} placeholder="Describe the environment — no material language (bible handles that)" multiline />
            <AssetField label="Time of Day / Lighting Moment" value={environment.time} onChange={v => updEnv("time", v)} placeholder="e.g. Late afternoon, golden hour" />
            <AssetField label="Emotional Atmosphere" value={environment.mood} onChange={v => updEnv("mood", v)} placeholder="e.g. Warm, intimate, slightly melancholic" />
            <AssetField label="Additional Notes" value={environment.notes} onChange={v => updEnv("notes", v)} placeholder="Anything else Claude must know" multiline />
            <AssetImageUpload assetKey="environment" assetImages={assetImages} setAssetImages={setAssetImages} label="Environment" />
          </>
        )}
        {assetTab === "characters" && (
          <>
            {characters.length === 0 && <div style={{ textAlign:"center", padding:"12px 0", color:"rgba(232,224,212,.3)", fontSize:12, fontFamily:"sans-serif", fontStyle:"italic" }}>No characters yet</div>}
            {characters.map((c, i) => (
              <div key={i} style={{ borderRadius:8, border:"1px solid rgba(255,255,255,.07)", overflow:"hidden" }}>
                <div style={{ padding:"8px 12px", background:"rgba(80,160,220,.06)", borderBottom:"1px solid rgba(80,160,220,.12)", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                  <span style={{ fontSize:11, fontFamily:"sans-serif", fontWeight:700, color:"rgba(80,160,220,.8)" }}>{c.name || ("Character " + (i+1))}</span>
                  <button onClick={() => rmChar(i)} style={{ fontSize:11, color:"rgba(220,100,100,.5)", background:"none", border:"none", cursor:"pointer", fontFamily:"sans-serif" }}>Remove</button>
                </div>
                <div style={{ padding:"12px", display:"grid", gap:11 }}>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
                    <AssetField label="Name" value={c.name} onChange={v => updChar(i, "name", v)} placeholder="Character name" />
                    <AssetField label="Role" value={c.role} onChange={v => updChar(i, "role", v)} placeholder="e.g. Protagonist, child" />
                  </div>
                  <AssetField label="Appearance" value={c.appearance} onChange={v => updChar(i, "appearance", v)} placeholder="Physical description — what the character looks like" multiline />
                  <AssetField label="Costume" value={c.costume} onChange={v => updChar(i, "costume", v)} placeholder="What they wear — colours, silhouette, key details" multiline />
                  <AssetField label="Expression Range" value={c.expression} onChange={v => updChar(i, "expression", v)} placeholder="e.g. Wide smile, bead-like dark eyes, soft features" />
                  <AssetField label="Distinctive Features" value={c.distinctive} onChange={v => updChar(i, "distinctive", v)} placeholder="Anything unique that must appear in every frame" />
                  <AssetField label="Additional Notes" value={c.notes} onChange={v => updChar(i, "notes", v)} placeholder="Anything else Claude must lock" multiline />
                  <AssetImageUpload assetKey={"char_" + i} assetImages={assetImages} setAssetImages={setAssetImages} label="Character" />
                </div>
              </div>
            ))}
            <button onClick={addChar} style={{ width:"100%", padding:"10px", borderRadius:7, border:"1px dashed rgba(80,160,220,.3)", background:"rgba(80,160,220,.05)", color:"rgba(80,160,220,.7)", fontSize:11, fontFamily:"sans-serif", fontWeight:700, cursor:"pointer", letterSpacing:".1em", textTransform:"uppercase" }}>
              + Add Character
            </button>
          </>
        )}
        {assetTab === "objects" && (
          <>
            {objects.length === 0 && <div style={{ textAlign:"center", padding:"12px 0", color:"rgba(232,224,212,.3)", fontSize:12, fontFamily:"sans-serif", fontStyle:"italic" }}>No objects yet</div>}
            {objects.map((o, i) => (
              <div key={i} style={{ borderRadius:8, border:"1px solid rgba(255,255,255,.07)", overflow:"hidden" }}>
                <div style={{ padding:"8px 12px", background:"rgba(210,120,55,.06)", borderBottom:"1px solid rgba(210,120,55,.12)", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                  <span style={{ fontSize:11, fontFamily:"sans-serif", fontWeight:700, color:"rgba(210,120,55,.8)" }}>{o.name || ("Object " + (i+1))}</span>
                  <button onClick={() => rmObj(i)} style={{ fontSize:11, color:"rgba(220,100,100,.5)", background:"none", border:"none", cursor:"pointer", fontFamily:"sans-serif" }}>Remove</button>
                </div>
                <div style={{ padding:"12px", display:"grid", gap:11 }}>
                  <AssetField label="Name" value={o.name} onChange={v => updObj(i, "name", v)} placeholder="Object name" />
                  <AssetField label="Description" value={o.description} onChange={v => updObj(i, "description", v)} placeholder="What it looks like — no material language" multiline />
                  <AssetField label="Scale Relative to Character" value={o.scale} onChange={v => updObj(i, "scale", v)} placeholder="e.g. Swing seat at knee height when standing" />
                  <AssetField label="Behaviour in Motion" value={o.behaviour} onChange={v => updObj(i, "behaviour", v)} placeholder="How it moves, hangs, or responds in the shot" />
                  <AssetField label="Additional Notes" value={o.notes} onChange={v => updObj(i, "notes", v)} placeholder="Anything else Claude must lock" multiline />
                  <AssetImageUpload assetKey={"obj_" + i} assetImages={assetImages} setAssetImages={setAssetImages} label="Object" />
                </div>
              </div>
            ))}
            <button onClick={addObj} style={{ width:"100%", padding:"10px", borderRadius:7, border:"1px dashed rgba(210,120,55,.3)", background:"rgba(210,120,55,.05)", color:"rgba(210,120,55,.7)", fontSize:11, fontFamily:"sans-serif", fontWeight:700, cursor:"pointer", letterSpacing:".1em", textTransform:"uppercase" }}>
              + Add Object
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CAMERA CONTROL
// ─────────────────────────────────────────────────────────────────────────────
function CameraControl({ settings, setSettings }) {
  const upd = (k, v) => setSettings(p => ({ ...p, [k]: v }));
  const on = settings.useCameraLock;
  const lens = LENS_OPTIONS.find(l => l.id === settings.cameraLens);
  const framing = FRAMING_OPTIONS.find(f => f.id === settings.cameraFraming);
  const composed = [framing ? framing.label : "", lens ? lens.label + " lens" : "", settings.cameraFreeText || "", "camera locked across both frames"].filter(Boolean).join(" · ");

  return (
    <div style={{ borderRadius:9, border:"1px solid " + (on ? "rgba(200,160,80,.3)" : "rgba(255,255,255,.08)"), overflow:"hidden", transition:"border-color .2s" }}>
      <div style={{ padding:"10px 13px", background:"rgba(200,160,80," + (on ? ".08" : ".03") + ")", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <div style={{ display:"flex", alignItems:"center", gap:9 }}>
          <span style={{ fontSize:15 }}>🎥</span>
          <div>
            <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:on ? "rgba(200,160,80,.95)" : "rgba(232,224,212,.45)" }}>Camera Control</div>
            <div style={{ fontSize:10, fontFamily:"sans-serif", color:"rgba(232,224,212,.4)", fontStyle:"italic", marginTop:1 }}>{on ? "Camera locked identically across both frames" : "Camera may vary between frames"}</div>
          </div>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:11, fontFamily:"sans-serif", color:"rgba(232,224,212,.35)" }}>{on ? "Locked" : "Free"}</span>
          <Toggle on={on} onToggle={() => upd("useCameraLock", !on)} />
        </div>
      </div>
      {on && (
        <div style={{ padding:"13px", background:"rgba(255,255,255,.015)", display:"grid", gap:13, borderTop:"1px solid rgba(200,160,80,.12)" }}>
          <div>
            <div style={{ fontSize:11, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.55)", marginBottom:8 }}>Lens</div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(5,1fr)", gap:5 }}>
              {LENS_OPTIONS.map(l => {
                const active = settings.cameraLens === l.id;
                return (
                  <button key={l.id} onClick={() => upd("cameraLens", active ? "" : l.id)} title={l.desc}
                    style={{ padding:"7px 4px", borderRadius:6, border:"1px solid " + (active ? "rgba(200,160,80,.45)" : "rgba(255,255,255,.07)"), background:active ? "rgba(200,160,80,.12)" : "rgba(255,255,255,.02)", cursor:"pointer", textAlign:"center", transition:"all .15s" }}>
                    <div style={{ fontSize:11, fontFamily:"sans-serif", fontWeight:700, color:active ? "#c8a050" : "rgba(232,224,212,.55)" }}>{l.label}</div>
                  </button>
                );
              })}
            </div>
            {lens && <div style={{ fontSize:11, color:"rgba(200,160,80,.45)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:5 }}>{lens.desc}</div>}
          </div>
          <div>
            <div style={{ fontSize:11, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.55)", marginBottom:8 }}>Shot Framing</div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(5,1fr)", gap:5 }}>
              {FRAMING_OPTIONS.map(f => {
                const active = settings.cameraFraming === f.id;
                return (
                  <button key={f.id} onClick={() => upd("cameraFraming", active ? "" : f.id)} title={f.desc}
                    style={{ padding:"7px 4px", borderRadius:6, border:"1px solid " + (active ? "rgba(200,160,80,.45)" : "rgba(255,255,255,.07)"), background:active ? "rgba(200,160,80,.12)" : "rgba(255,255,255,.02)", cursor:"pointer", textAlign:"center", transition:"all .15s" }}>
                    <div style={{ fontSize:10, fontFamily:"sans-serif", fontWeight:700, color:active ? "#c8a050" : "rgba(232,224,212,.55)", lineHeight:1.3 }}>{f.label}</div>
                  </button>
                );
              })}
            </div>
            {framing && <div style={{ fontSize:11, color:"rgba(200,160,80,.45)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:5 }}>{framing.desc}</div>}
          </div>
          <div>
            <div style={{ fontSize:11, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.55)", marginBottom:6 }}>Additional Camera Notes</div>
            <input value={settings.cameraFreeText || ""} onChange={e => upd("cameraFreeText", e.target.value)}
              placeholder="e.g. low angle, slight dutch tilt, slow push in…"
              style={{ width:"100%", background:"rgba(255,255,255,.03)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"8px 11px", fontFamily:"Georgia,serif", outline:"none" }}
              onFocus={e => { e.target.style.borderColor = "rgba(200,160,80,.28)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(255,255,255,.07)"; }} />
          </div>
          {(settings.cameraLens || settings.cameraFraming || settings.cameraFreeText) && (
            <div style={{ padding:"8px 11px", background:"rgba(200,160,80,.04)", border:"1px solid rgba(200,160,80,.12)", borderRadius:6 }}>
              <div style={{ fontSize:10, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.45)", marginBottom:4 }}>Sent to Claude as</div>
              <div style={{ fontSize:12, color:"rgba(232,224,212,.6)", fontFamily:"sans-serif", fontStyle:"italic", lineHeight:1.5 }}>{composed}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CONSISTENCY CONTROLS
// ─────────────────────────────────────────────────────────────────────────────
function ConsistencyControls({ settings, setSettings, seed, comfyConfigured }) {
  const upd = (k, v) => setSettings(p => ({ ...p, [k]: v }));
  return (
    <div>
      <FieldLabel main="Consistency Controls" sub="Improve visual coherence between Frame 1 and Frame 2" />
      <div style={{ display:"grid", gap:10 }}>
        <CameraControl settings={settings} setSettings={setSettings} />
        <div style={{ padding:"10px 13px", borderRadius:8, border:"1px solid " + (settings.useFrame1Ref ? "rgba(80,160,220,.25)" : "rgba(255,255,255,.08)"), background:"rgba(80,160,220," + (settings.useFrame1Ref ? ".05" : ".01") + ")", display:"flex", alignItems:"center", justifyContent:"space-between", transition:"all .2s" }}>
          <div style={{ display:"flex", alignItems:"center", gap:9 }}>
            <span style={{ fontSize:14 }}>🔗</span>
            <div>
              <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:settings.useFrame1Ref ? "rgba(80,160,220,.9)" : "rgba(232,224,212,.4)" }}>Frame 1 to Frame 2 Reference</div>
              <div style={{ fontSize:10, fontFamily:"sans-serif", color:"rgba(232,224,212,.38)", fontStyle:"italic", marginTop:1 }}>Passes Frame 1 as visual anchor when generating Frame 2</div>
            </div>
          </div>
          <Toggle on={settings.useFrame1Ref} onToggle={() => upd("useFrame1Ref", !settings.useFrame1Ref)} activeColor="rgba(80,160,220,.8)" />
        </div>
        <div style={{ borderRadius:8, border:"1px solid " + (settings.useSharedSeed ? "rgba(130,80,200,.25)" : "rgba(255,255,255,.08)"), overflow:"hidden", transition:"border-color .2s" }}>
          <div style={{ padding:"10px 13px", background:"rgba(130,80,200," + (settings.useSharedSeed ? ".06" : ".01") + ")", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
            <div style={{ display:"flex", alignItems:"center", gap:9 }}>
              <span style={{ fontSize:14 }}>🎲</span>
              <div>
                <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:settings.useSharedSeed ? "rgba(180,130,255,.9)" : "rgba(232,224,212,.4)" }}>Shared Seed</div>
                <div style={{ fontSize:10, fontFamily:"sans-serif", color:"rgba(232,224,212,.38)", fontStyle:"italic", marginTop:1 }}>Same seed initialises both frames for visual coherence</div>
              </div>
            </div>
            <Toggle on={settings.useSharedSeed} onToggle={() => upd("useSharedSeed", !settings.useSharedSeed)} activeColor="rgba(130,80,200,.8)" />
          </div>
          {settings.useSharedSeed && (
            <div style={{ padding:"10px 13px", borderTop:"1px solid rgba(130,80,200,.15)", background:"rgba(255,255,255,.02)", display:"flex", alignItems:"center", gap:10 }}>
              <div style={{ fontSize:12, fontFamily:"sans-serif", color:"rgba(232,224,212,.45)", flexShrink:0 }}>Seed</div>
              <input type="number" value={settings.manualSeed} onChange={e => upd("manualSeed", e.target.value)}
                placeholder={seed ? String(seed) : "Auto-generated each shot"}
                style={{ flex:1, background:"rgba(255,255,255,.04)", border:"1px solid rgba(130,80,200,.2)", borderRadius:5, color:"#e8e0d4", fontSize:12, padding:"6px 10px", fontFamily:"monospace", outline:"none" }} />
              {seed && <div style={{ fontSize:11, color:"rgba(180,130,255,.5)", fontFamily:"monospace", flexShrink:0 }}>Last: {seed}</div>}
              <button onClick={() => upd("manualSeed", "")}
                style={{ fontSize:11, color:"rgba(232,224,212,.3)", background:"none", border:"none", cursor:"pointer", fontFamily:"sans-serif", flexShrink:0 }}>Reset</button>
            </div>
          )}
        </div>
        <div style={{ padding:"10px 13px", borderRadius:8, border:"1px solid " + (settings.useCropLock ? "rgba(80,180,120,.25)" : "rgba(255,255,255,.08)"), background:"rgba(80,180,120," + (settings.useCropLock ? ".05" : ".01") + ")", display:"flex", alignItems:"center", justifyContent:"space-between", transition:"all .2s" }}>
          <div style={{ display:"flex", alignItems:"center", gap:9 }}>
            <span style={{ fontSize:14 }}>✂️</span>
            <div>
              <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:settings.useCropLock ? "rgba(80,180,120,.9)" : "rgba(232,224,212,.4)" }}>Crop Lock</div>
              <div style={{ fontSize:10, fontFamily:"sans-serif", color:"rgba(232,224,212,.38)", fontStyle:"italic", marginTop:1 }}>Crops both frames to identical dimensions from centre</div>
            </div>
          </div>
          <Toggle on={settings.useCropLock} onToggle={() => upd("useCropLock", !settings.useCropLock)} activeColor="rgba(80,180,120,.85)" />
        </div>
        <div style={{ padding:"10px 13px", borderRadius:8, border:"1px solid " + (settings.useControlNet ? "rgba(130,80,200,.35)" : "rgba(255,255,255,.06)"), background:"rgba(130,80,200," + (settings.useControlNet ? ".07" : ".01") + ")", display:"flex", alignItems:"center", justifyContent:"space-between", transition:"all .2s", opacity:comfyConfigured ? 1 : 0.45 }}>
          <div style={{ display:"flex", alignItems:"center", gap:9 }}>
            <span style={{ fontSize:14 }}>🧭</span>
            <div>
              <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:settings.useControlNet ? "rgba(180,130,255,.9)" : "rgba(232,224,212,.4)" }}>
                ControlNet Depth
                <span style={{ fontSize:10, padding:"2px 6px", background:"rgba(130,80,200,.15)", borderRadius:10, color:"rgba(180,130,255,.6)", fontWeight:400, marginLeft:6 }}>ComfyUI only</span>
              </div>
              <div style={{ fontSize:10, fontFamily:"sans-serif", color:"rgba(232,224,212,.38)", fontStyle:"italic", marginTop:1 }}>Uses Frame 1 depth map to lock Frame 2 spatial composition</div>
            </div>
          </div>
          <Toggle on={settings.useControlNet} onToggle={() => { if (comfyConfigured) upd("useControlNet", !settings.useControlNet); }} activeColor="rgba(130,80,200,.85)" />
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// REF ZONE
// ─────────────────────────────────────────────────────────────────────────────
function RefZone({ refs, onAdd, onRemove }) {
  const inputRef = useRef();
  const [dragOver, setDragOver] = useState(false);
  function handleFiles(files) { Array.from(files).forEach(f => { if (f.type.startsWith("image/")) onAdd(f); }); }
  return (
    <div onDragOver={e => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
      style={{ borderRadius:9, border:"1px dashed " + (refs.length || dragOver ? "rgba(200,160,80,.28)" : "rgba(255,255,255,.08)"), background:refs.length ? "rgba(200,160,80,.04)" : "rgba(255,255,255,.01)", padding:"12px", transition:"all .2s", minHeight:68 }}>
      {refs.length === 0 ? (
        <div onClick={() => inputRef.current && inputRef.current.click()} style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:5, textAlign:"center", cursor:"pointer" }}>
          <div style={{ fontSize:20, opacity:.35 }}>📎</div>
          <div style={{ fontSize:11, fontFamily:"sans-serif", color:"rgba(232,224,212,.4)", fontWeight:700, textTransform:"uppercase", letterSpacing:".08em" }}>Drop references here</div>
          <div style={{ fontSize:10, color:"rgba(200,160,80,.3)", fontFamily:"sans-serif", marginTop:2 }}>drop or click to browse</div>
        </div>
      ) : (
        <div>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:8 }}>
            {refs.map((r, i) => (
              <div key={i} style={{ position:"relative", flexShrink:0 }}>
                <img src={r.preview} alt="" style={{ width:50, height:50, objectFit:"cover", borderRadius:6, border:"1px solid " + (r.falUrl ? "rgba(80,160,100,.35)" : "rgba(200,160,80,.2)"), display:"block" }} />
                {r.uploading && <div style={{ position:"absolute", inset:0, background:"rgba(9,9,10,.6)", borderRadius:6, display:"flex", alignItems:"center", justifyContent:"center" }}><Spin /></div>}
                <button onClick={() => onRemove(i)} style={{ position:"absolute", top:-4, right:-4, width:15, height:15, borderRadius:"50%", background:"rgba(0,0,0,.8)", border:"1px solid rgba(255,255,255,.15)", color:"rgba(255,255,255,.7)", cursor:"pointer", fontSize:8, display:"flex", alignItems:"center", justifyContent:"center" }}>×</button>
              </div>
            ))}
            <button onClick={() => inputRef.current && inputRef.current.click()} style={{ width:50, height:50, borderRadius:6, border:"1px dashed rgba(200,160,80,.22)", background:"rgba(200,160,80,.05)", cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", color:"rgba(200,160,80,.38)", fontSize:18, flexShrink:0 }}>+</button>
          </div>
          <div style={{ fontSize:10, color:"rgba(232,224,212,.3)", fontFamily:"sans-serif", fontStyle:"italic" }}>{refs.length} reference{refs.length > 1 ? "s" : ""}</div>
        </div>
      )}
      <input ref={inputRef} type="file" accept="image/*" multiple style={{ display:"none" }} onChange={e => { handleFiles(e.target.files); e.target.value = ""; }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FRAME PANEL (output)
// ─────────────────────────────────────────────────────────────────────────────
function FramePanel({ number, label, frame, imageUrl, loading, loadingMsg, error }) {
  const [open, setOpen] = useState(false);
  const rgb = number === 1 ? "80,160,220" : "210,120,55";
  const accent = "rgba(" + rgb + ",.85)";
  const accentFaint = "rgba(" + rgb + ",.09)";
  const accentLine = "rgba(" + rgb + ",.18)";
  return (
    <div style={{ flex:1, minWidth:0, animation:"fadeIn .4s ease both", animationDelay:number===1?"0s":".1s" }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:9 }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={{ width:22, height:22, borderRadius:5, background:accentFaint, border:"1px solid " + accentLine, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
            <span style={{ fontSize:11, color:accent, fontFamily:"sans-serif", fontWeight:700 }}>{number}</span>
          </div>
          <span style={{ fontSize:12, color:accent, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".12em", textTransform:"uppercase" }}>{label}</span>
        </div>
        <div style={{ display:"flex", gap:6 }}>
          {frame && frame.prompt && <CopyBtn text={frame.prompt} />}
          {imageUrl && <a href={imageUrl} download={"frame-" + number + ".jpg"} target="_blank" rel="noreferrer" style={{ fontSize:11, padding:"3px 9px", borderRadius:4, background:"rgba(80,130,200,.08)", border:"1px solid rgba(80,130,200,.18)", color:"rgba(130,180,240,.6)", fontFamily:"sans-serif", textDecoration:"none", letterSpacing:".08em", textTransform:"uppercase" }}>↓ Save</a>}
        </div>
      </div>
      {frame && frame.compositionNote && <div style={{ fontSize:11, color:"rgba(232,224,212,.5)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:8, lineHeight:1.45 }}>{frame.compositionNote}</div>}
      <div style={{ background:"#0d0d0e", borderRadius:9, border:"1px solid rgba(255,255,255,.07)", overflow:"hidden", marginBottom:9, aspectRatio:"16/9", display:"flex", alignItems:"center", justifyContent:"center" }}>
        {loading && !imageUrl && (
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:9 }}>
            <Spin size={16} />
            <span style={{ fontSize:10, color:accent, letterSpacing:".14em", textTransform:"uppercase", fontFamily:"sans-serif", animation:"pulse 1.6s ease infinite" }}>{loadingMsg || "Rendering…"}</span>
          </div>
        )}
        {error && !imageUrl && (
          <div style={{ padding:20, textAlign:"center" }}>
            <div style={{ fontSize:18, marginBottom:6 }}>⚠</div>
            <div style={{ fontSize:12, color:"rgba(220,100,100,.6)", fontFamily:"sans-serif" }}>{error}</div>
          </div>
        )}
        {!loading && !error && !imageUrl && <div style={{ fontSize:12, color:"rgba(255,255,255,.07)", fontFamily:"sans-serif", fontStyle:"italic" }}>Frame {number}</div>}
        {imageUrl && <img src={imageUrl} alt={label} style={{ width:"100%", height:"100%", objectFit:"cover", display:"block" }} />}
      </div>
      {frame && frame.prompt && (
        <>
          <button onClick={() => setOpen(!open)} style={{ width:"100%", background:accentFaint, border:"1px solid " + accentLine, borderRadius:open?"7px 7px 0 0":"7px", padding:"7px 11px", cursor:"pointer", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <span style={{ fontSize:11, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:accent }}>Generated Prompt</span>
            <span style={{ fontSize:11, color:accent, opacity:.5 }}>{open ? "−" : "+"}</span>
          </button>
          {open && (
            <div style={{ background:"rgba(255,255,255,.018)", border:"1px solid " + accentLine, borderTop:"none", borderRadius:"0 0 7px 7px", padding:"11px 12px" }}>
              <div style={{ fontSize:12, lineHeight:1.72, color:"rgba(232,224,212,.6)", fontFamily:"sans-serif", marginBottom:9 }}>{frame.prompt}</div>
              {frame.negativePrompt && <div style={{ paddingTop:8, borderTop:"1px solid rgba(255,255,255,.05)", fontSize:11, color:"rgba(220,150,150,.4)", fontFamily:"sans-serif", lineHeight:1.5 }}>– {frame.negativePrompt}</div>}
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
        <div style={{ fontSize:10, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(200,160,80,.45)", marginBottom:2 }}>Motion Arc</div>
        <div style={{ fontSize:12, color:"rgba(232,224,212,.55)", fontFamily:"sans-serif", fontStyle:"italic", lineHeight:1.4 }}>{text}</div>
      </div>
    </div>
  );
}

function AuditPanel({ audit }) {
  const [open, setOpen] = useState(false);
  if (!audit) return null;
  const entries = [["cameraLock","Camera Lock"],["lightingFreeze","Lighting Freeze"],["spatialPlausibility","Spatial Plausibility"],["impliedMotion","Implied Motion"],["materialCompliance","Material Compliance"],["heroElementLock","Hero Element"],["reinterpretationApplied","Reinterpretation"]];
  const issues = entries.filter(([k]) => { const v = audit[k] || ""; return v && v !== "confirmed" && v !== "not applicable" && !v.startsWith("confirmed"); });
  const allOk = issues.length === 0;
  return (
    <div>
      <button onClick={() => setOpen(!open)} style={{ display:"flex", alignItems:"center", gap:7, background:"none", border:"none", cursor:"pointer", padding:0 }}>
        <span style={{ width:7, height:7, borderRadius:"50%", background:allOk?"#5cb87a":"#c8a050", flexShrink:0 }} />
        <span style={{ fontSize:11, fontFamily:"sans-serif", letterSpacing:".1em", textTransform:"uppercase", color:allOk?"rgba(80,180,120,.7)":"rgba(200,160,80,.7)" }}>{allOk ? "All checks passed" : issues.length + " audit note" + (issues.length>1?"s":"")}</span>
        <span style={{ fontSize:10, color:"rgba(232,224,212,.2)", fontFamily:"sans-serif" }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div style={{ marginTop:9, padding:"11px 13px", background:"rgba(255,255,255,.02)", border:"1px solid rgba(255,255,255,.06)", borderRadius:8, display:"grid", gap:7 }}>
          {entries.map(([k, l]) => {
            const val = audit[k] || "—";
            const ok = val === "confirmed" || val === "not applicable" || val.startsWith("confirmed");
            return (
              <div key={k} style={{ display:"flex", alignItems:"flex-start", gap:7 }}>
                <span style={{ fontSize:11, color:ok?"rgba(80,180,120,.65)":"rgba(200,160,80,.7)", flexShrink:0, marginTop:.5 }}>{ok ? "✓" : "⚠"}</span>
                <div>
                  <span style={{ fontSize:11, fontFamily:"sans-serif", fontWeight:700, color:"rgba(232,224,212,.5)", textTransform:"uppercase", letterSpacing:".07em" }}>{l} </span>
                  <span style={{ fontSize:11, fontFamily:"sans-serif", color:ok?"rgba(232,224,212,.38)":"rgba(232,224,212,.6)", fontStyle:"italic" }}>{val}</span>
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
          <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".1em", textTransform:"uppercase", color:"rgba(80,180,120,.8)" }}>💬 Weavy Review Thread</div>
          {shot && shot.sceneSlug && <div style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"monospace", marginTop:3 }}>{shot.sceneSlug}</div>}
        </div>
        <div style={{ display:"flex", gap:8 }}>
          {threadUrl && <a href={threadUrl} target="_blank" rel="noreferrer" style={{ fontSize:11, padding:"5px 12px", borderRadius:5, background:"rgba(80,180,120,.1)", border:"1px solid rgba(80,180,120,.25)", color:"rgba(80,180,120,.8)", fontFamily:"sans-serif", textDecoration:"none", letterSpacing:".08em", textTransform:"uppercase" }}>Open ↗</a>}
          <button onClick={onClose} style={{ background:"none", border:"none", color:"rgba(232,224,212,.3)", cursor:"pointer", fontSize:18 }}>×</button>
        </div>
      </div>
      <div style={{ flex:1, overflowY:"auto", padding:"16px 18px", display:"flex", flexDirection:"column", gap:12 }}>
        {messages.length === 0
          ? <div style={{ textAlign:"center", padding:"40px 0", color:"rgba(232,224,212,.2)", fontSize:12, fontFamily:"sans-serif", fontStyle:"italic" }}>No messages yet.</div>
          : messages.map((msg, i) => {
            const isAgent = msg.text && msg.text.startsWith("🎬");
            return (
              <div key={i} style={{ padding:"11px 14px", background:isAgent?"rgba(200,160,80,.05)":"rgba(80,130,200,.05)", border:"1px solid " + (isAgent?"rgba(200,160,80,.12)":"rgba(80,130,200,.12)"), borderRadius:8 }}>
                <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                  <span style={{ fontSize:10, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".1em", textTransform:"uppercase", color:isAgent?"rgba(200,160,80,.55)":"rgba(130,180,240,.55)" }}>{isAgent ? "Director Agent" : ((msg.created_by && msg.created_by.display_name) || "Director")}</span>
                  <span style={{ fontSize:10, color:"rgba(232,224,212,.2)", fontFamily:"sans-serif" }}>{msg.created_at ? new Date(msg.created_at).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"}) : ""}</span>
                </div>
                <div style={{ fontSize:12, color:"rgba(232,224,212,.65)", fontFamily:"sans-serif", lineHeight:1.65, whiteSpace:"pre-wrap" }}>{msg.plain || msg.text || ""}</div>
              </div>
            );
          })
        }
      </div>
      <div style={{ padding:"14px 18px", borderTop:"1px solid rgba(255,255,255,.07)", flexShrink:0 }}>
        {feedbackFound && <div style={{ marginBottom:10, padding:"9px 12px", background:"rgba(80,180,120,.07)", border:"1px solid rgba(80,180,120,.18)", borderRadius:7, fontSize:12, color:"rgba(80,180,120,.8)", fontFamily:"sans-serif" }}>✓ Feedback found — regenerating. Close panel to see new frames.</div>}
        <button onClick={onCheckFeedback} disabled={checkingFeedback} style={{ width:"100%", padding:"11px", borderRadius:7, border:"1px solid rgba(80,180,120,.35)", background:"rgba(80,180,120,.1)", color:checkingFeedback?"rgba(80,180,120,.4)":"rgba(80,180,120,.85)", fontSize:12, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", fontWeight:700, cursor:checkingFeedback?"not-allowed":"pointer", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
          {checkingFeedback ? <><Spin />Checking…</> : "↺ Check for Director Feedback"}
        </button>
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
        <button onClick={() => setOpen(!open)} style={{ background:"#17171a", border:"1px solid rgba(255,255,255,.08)", borderBottom:"none", borderRadius:"8px 8px 0 0", padding:"5px 18px", cursor:"pointer", display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:10, letterSpacing:".15em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(232,224,212,.3)" }}>Shot Log</span>
          <span style={{ fontSize:11, padding:"1px 6px", background:"rgba(200,160,80,.1)", borderRadius:10, color:"rgba(200,160,80,.6)", fontFamily:"sans-serif" }}>{log.length}</span>
          <span style={{ fontSize:10, color:"rgba(232,224,212,.2)" }}>{open ? "▼" : "▲"}</span>
        </button>
      </div>
      {open && (
        <div style={{ background:"#131315", borderTop:"1px solid rgba(255,255,255,.07)", maxHeight:180, overflowX:"auto", overflowY:"hidden" }}>
          {log.length === 0
            ? <div style={{ padding:"18px", textAlign:"center", fontSize:12, color:"rgba(232,224,212,.18)", fontFamily:"sans-serif", fontStyle:"italic" }}>No shots yet</div>
            : (
              <div style={{ display:"flex", padding:"12px 16px", minWidth:"max-content", gap:10 }}>
                {log.map((s, i) => (
                  <button key={i} onClick={() => { setOpen(false); onSelect(s); }} style={{ background:"rgba(255,255,255,.03)", border:"1px solid rgba(255,255,255,.07)", borderRadius:7, padding:"8px 12px", cursor:"pointer", textAlign:"left", minWidth:175 }}>
                    <div style={{ fontSize:10, color:"rgba(200,160,80,.45)", fontFamily:"sans-serif", marginBottom:4 }}>{"#" + (i+1) + " · " + (s.sceneSlug || "—")}</div>
                    <div style={{ fontSize:12, color:"rgba(232,224,212,.5)", fontFamily:"sans-serif", fontStyle:"italic", lineHeight:1.35 }}>{(s.shotSummary || "").slice(0, 65)}{(s.shotSummary || "").length > 65 ? "…" : ""}</div>
                  </button>
                ))}
              </div>
            )
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
  const upd = (k, v) => set(p => ({ ...p, [k]: v }));
  const [jsonError, setJsonError] = useState(null);
  return (
    <div style={{ position:"fixed", inset:0, zIndex:1000 }}>
      <div onClick={onClose} style={{ position:"absolute", inset:0, background:"rgba(0,0,0,.65)", backdropFilter:"blur(5px)" }} />
      <div style={{ position:"absolute", right:0, top:0, bottom:0, width:440, background:"#111113", borderLeft:"1px solid rgba(255,255,255,.08)", overflowY:"auto", display:"flex", flexDirection:"column" }}>
        <div style={{ padding:"18px 22px 14px", borderBottom:"1px solid rgba(255,255,255,.07)", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <span style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".1em", textTransform:"uppercase", color:"rgba(232,224,212,.6)" }}>Settings</span>
          <button onClick={onClose} style={{ background:"none", border:"none", color:"rgba(232,224,212,.35)", cursor:"pointer", fontSize:18 }}>×</button>
        </div>
        <div style={{ padding:"18px 22px", display:"grid", gap:20, flex:1 }}>
          <div style={{ padding:"14px 16px", background:"rgba(200,160,80,.04)", border:"1px solid rgba(200,160,80,.12)", borderRadius:10 }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:12 }}>
              <span style={{ fontSize:16 }}>🧠</span>
              <div>
                <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:"rgba(200,160,80,.85)", letterSpacing:".06em" }}>Claude API</div>
                <div style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:2 }}>Prompt architect — writes your bible-locked prompts</div>
              </div>
            </div>
            <SLabel>API Key</SLabel>
            <div style={{ fontSize:10, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:7 }}>console.anthropic.com → API Keys</div>
            <SecretInput value={s.claudeKey} onChange={v => upd("claudeKey", v)} placeholder="sk-ant-…" />
          </div>
          <div style={{ padding:"14px 16px", background:"rgba(130,80,200,.05)", border:"1px solid rgba(130,80,200,.2)", borderRadius:10 }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:14 }}>
              <span style={{ fontSize:18 }}>🎨</span>
              <div>
                <div style={{ fontSize:13, fontFamily:"sans-serif", fontWeight:700, color:"rgba(180,130,255,.85)", letterSpacing:".06em" }}>ComfyUI</div>
                <div style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:2 }}>Primary production renderer</div>
              </div>
            </div>
            <div style={{ display:"grid", gap:12 }}>
              <div>
                <SLabel>Server URL</SLabel>
                <TextInput value={s.comfyUrl} onChange={v => upd("comfyUrl", v)} placeholder="http://127.0.0.1:8188" />
                <div style={{ fontSize:10, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:4 }}>ComfyUI Desktop default: http://127.0.0.1:8188</div>
              </div>
              <div>
                <SLabel>Model / Checkpoint</SLabel>
                <TextInput value={s.comfyModel} onChange={v => upd("comfyModel", v)} placeholder="FLUX1/flux1-dev-fp8.safetensors" />
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
                <div><SLabel>Steps</SLabel><input type="number" min={1} max={50} value={s.comfySteps} onChange={e => upd("comfySteps", Number(e.target.value))} style={{ width:"100%", background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"9px 11px", outline:"none" }} /></div>
                <div><SLabel>Guidance</SLabel><input type="number" min={1} max={10} step={0.5} value={s.comfyGuidance || 3.5} onChange={e => upd("comfyGuidance", Number(e.target.value))} style={{ width:"100%", background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"9px 11px", outline:"none" }} /></div>
              </div>
              <div>
                <SLabel>Output Dimensions</SLabel>
                <div style={{ display:"flex", gap:7, marginTop:6 }}>
                  {ASPECT_RATIOS.map(r => (
                    <button key={r.id} onClick={() => { upd("ratio", r.id); upd("comfyWidth", r.width); upd("comfyHeight", r.height); }}
                      style={{ flex:1, padding:"8px", borderRadius:6, border:"1px solid " + (s.ratio===r.id?"rgba(130,80,200,.45)":"rgba(255,255,255,.06)"), background:s.ratio===r.id?"rgba(130,80,200,.12)":"rgba(255,255,255,.02)", cursor:"pointer", fontSize:12, fontFamily:"sans-serif", color:s.ratio===r.id?"rgba(180,130,255,.9)":"rgba(232,224,212,.42)", fontWeight:s.ratio===r.id?700:400 }}>
                      {r.label}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ padding:"9px 11px", background:"rgba(130,80,200,.08)", border:"1px solid rgba(130,80,200,.2)", borderRadius:7, fontSize:11, color:"rgba(180,130,255,.75)", fontFamily:"sans-serif", lineHeight:1.6 }}>
                Flux.1 locked: CFG 1.0 · euler · simple scheduler · FluxGuidance node
              </div>
              <div>
                <SLabel>Custom Workflow JSON — optional</SLabel>
                <textarea value={s.comfyWorkflow} onChange={e => { upd("comfyWorkflow", e.target.value); try { JSON.parse(e.target.value); setJsonError(null); } catch (err) { setJsonError(err.message); } }}
                  placeholder="Paste your ComfyUI workflow JSON here"
                  style={{ width:"100%", minHeight:90, background:"rgba(255,255,255,.04)", border:"1px solid " + (jsonError?"rgba(200,80,80,.4)":"rgba(255,255,255,.07)"), borderRadius:6, color:"#e8e0d4", fontSize:11, padding:"9px 11px", fontFamily:"monospace", outline:"none", resize:"vertical", lineHeight:1.5, marginTop:6 }} />
                {jsonError && <div style={{ fontSize:11, color:"rgba(220,100,100,.7)", fontFamily:"sans-serif", marginTop:4 }}>{"⚠ " + jsonError}</div>}
                {s.comfyWorkflow && !jsonError && <div style={{ fontSize:11, color:"rgba(80,180,120,.6)", fontFamily:"sans-serif", marginTop:4 }}>✓ Valid workflow JSON</div>}
              </div>
            </div>
          </div>
          <div style={{ padding:"14px 16px", background:"rgba(255,255,255,.02)", border:"1px solid rgba(255,255,255,.07)", borderRadius:10 }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:14 }}>
              <span style={{ fontSize:16 }}>👁</span>
              <div>
                <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:"rgba(232,224,212,.65)", letterSpacing:".06em" }}>Quick Preview</div>
                <div style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:2 }}>Fast iteration before sending to ComfyUI</div>
              </div>
            </div>
            <div style={{ display:"grid", gap:10 }}>
              <div>
                <SLabel>Provider</SLabel>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:7, marginTop:6 }}>
                  {Object.entries(PREVIEW_PROVIDERS).map(([k, p]) => (
                    <button key={k} onClick={() => upd("previewProvider", k)}
                      style={{ padding:"9px 8px", borderRadius:7, border:"1px solid " + (s.previewProvider===k?"rgba(200,160,80,.38)":"rgba(255,255,255,.07)"), background:s.previewProvider===k?"rgba(200,160,80,.08)":"rgba(255,255,255,.02)", cursor:"pointer", textAlign:"center" }}>
                      <div style={{ fontSize:13, fontFamily:"sans-serif", color:s.previewProvider===k?"#c8a050":"rgba(232,224,212,.55)", fontWeight:700 }}>{p.icon}</div>
                      <div style={{ fontSize:10, fontFamily:"sans-serif", color:s.previewProvider===k?"#c8a050":"rgba(232,224,212,.4)", marginTop:2 }}>{p.label}</div>
                    </button>
                  ))}
                </div>
              </div>
              {s.previewProvider !== "gemini" && (
                <div>
                  <SLabel>Model</SLabel>
                  <div style={{ display:"grid", gap:6, marginTop:6 }}>
                    {PREVIEW_PROVIDERS[s.previewProvider].models.map(m => {
                      const active = s.previewProvider === "nanobanana" ? s.nbModel === m.id : s.falModel === m.id;
                      return (
                        <button key={m.id} onClick={() => upd(s.previewProvider === "nanobanana" ? "nbModel" : "falModel", m.id)}
                          style={{ padding:"9px 11px", borderRadius:7, border:"1px solid " + (active?"rgba(200,160,80,.35)":"rgba(255,255,255,.06)"), background:active?"rgba(200,160,80,.08)":"rgba(255,255,255,.02)", cursor:"pointer", textAlign:"left", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                          <div>
                            <div style={{ fontSize:12, fontFamily:"sans-serif", color:active?"#c8a050":"rgba(232,224,212,.58)", fontWeight:700 }}>{m.label}</div>
                            <div style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif", marginTop:1 }}>{m.desc}</div>
                          </div>
                          {active && <span style={{ color:"#c8a050", fontSize:12 }}>✓</span>}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              <div>
                {s.previewProvider === "gemini" && (
                  <><SLabel>Google AI Studio API Key</SLabel>
                  <div style={{ fontSize:10, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:7 }}>aistudio.google.com → Get API Key</div>
                  <SecretInput value={s.geminiKey} onChange={v => upd("geminiKey", v)} placeholder="AIza…" /></>
                )}
                {s.previewProvider === "nanobanana" && (
                  <><SLabel>NanoBanana API Key</SLabel>
                  <SecretInput value={s.nbKey} onChange={v => upd("nbKey", v)} placeholder="nb_…" /></>
                )}
                {s.previewProvider === "fal" && (
                  <><SLabel>fal.ai API Key</SLabel>
                  <SecretInput value={s.falKey} onChange={v => upd("falKey", v)} placeholder="fal_…" /></>
                )}
              </div>
              {s.previewProvider !== "fal" && (
                <div>
                  <SLabel>fal.ai Key — for reference image uploads</SLabel>
                  <SecretInput value={s.falKey} onChange={v => upd("falKey", v)} placeholder="fal_…" />
                </div>
              )}
            </div>
          </div>
          <div style={{ padding:"14px 16px", background:"rgba(80,180,120,.04)", border:"1px solid rgba(80,180,120,.14)", borderRadius:10 }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:14 }}>
              <span style={{ fontSize:16 }}>💬</span>
              <div>
                <div style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, color:"rgba(80,180,120,.85)", letterSpacing:".06em" }}>Weavy Review</div>
                <div style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif", fontStyle:"italic", marginTop:2 }}>Shot threads + feedback-to-regenerate</div>
              </div>
            </div>
            <div style={{ display:"grid", gap:10 }}>
              <div><SLabel>Environment URL</SLabel><TextInput value={s.weavyUrl} onChange={v => upd("weavyUrl", v)} placeholder="https://your-env.weavy.io" /></div>
              <div><SLabel>API Key</SLabel><SecretInput value={s.weavyKey} onChange={v => upd("weavyKey", v)} placeholder="wys_…" /></div>
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
    claudeKey:"", comfyUrl:"", comfyModel:"FLUX1/flux1-dev-fp8.safetensors",
    comfySteps:20, comfyGuidance:3.5, comfyWidth:1024, comfyHeight:576,
    comfyWorkflow:"", ratio:"landscape_16_9",
    previewProvider:"gemini", nbModel:"nano-banana-2", falModel:"fal-ai/flux/dev",
    geminiKey:"", nbKey:"", falKey:"", weavyUrl:"", weavyKey:"",
    useFrame1Ref:true, useSharedSeed:true, manualSeed:"",
    useCameraLock:true, cameraLens:"35mm", cameraFraming:"wide", cameraFreeText:"",
    useCropLock:true, useControlNet:false,
  };

  const [settings, setSettings] = useState(() => {
    try { const s = localStorage.getItem("da_settings"); return s ? { ...defaultSettings, ...JSON.parse(s) } : defaultSettings; }
    catch (e) { return defaultSettings; }
  });
  useEffect(() => { try { localStorage.setItem("da_settings", JSON.stringify(settings)); } catch (e) {} }, [settings]);

  const [bible,  setBible]  = useState(() => { try { return localStorage.getItem("da_bible")  || ""; } catch (e) { return ""; } });
  const [frame1, setFrame1] = useState(() => { try { return localStorage.getItem("da_frame1") || ""; } catch (e) { return ""; } });
  const [frame2, setFrame2] = useState(() => { try { return localStorage.getItem("da_frame2") || ""; } catch (e) { return ""; } });
  useEffect(() => { try { localStorage.setItem("da_bible",  bible);  } catch (e) {} }, [bible]);
  useEffect(() => { try { localStorage.setItem("da_frame1", frame1); } catch (e) {} }, [frame1]);
  useEffect(() => { try { localStorage.setItem("da_frame2", frame2); } catch (e) {} }, [frame2]);

  const defEnv  = { name:"", setting:"", time:"", mood:"", notes:"" };
  const defChar = { name:"", role:"", appearance:"", costume:"", expression:"", distinctive:"", notes:"" };
  const defObj  = { name:"", description:"", scale:"", behaviour:"", notes:"" };

  const [environment,  setEnvironment]  = useState(() => { try { const s = localStorage.getItem("da_environment"); return s ? JSON.parse(s) : defEnv; } catch (e) { return defEnv; } });
  const [characters,   setCharacters]   = useState(() => { try { const s = localStorage.getItem("da_characters");  return s ? JSON.parse(s) : [];     } catch (e) { return [];     } });
  const [objects,      setObjects]      = useState(() => { try { const s = localStorage.getItem("da_objects");     return s ? JSON.parse(s) : [];     } catch (e) { return [];     } });
  const [assetTab,     setAssetTab]     = useState("environment");
  const [assetImages,  setAssetImages]  = useState({});

  useEffect(() => { try { localStorage.setItem("da_environment", JSON.stringify(environment)); } catch (e) {} }, [environment]);
  useEffect(() => { try { localStorage.setItem("da_characters",  JSON.stringify(characters));  } catch (e) {} }, [characters]);
  useEffect(() => { try { localStorage.setItem("da_objects",     JSON.stringify(objects));     } catch (e) {} }, [objects]);

  const [refs,   setRefs]   = useState([]);
  const [shot,   setShot]   = useState(null);
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
  const [activeMode,   setActiveMode]   = useState(null);
  const [genError,     setGenError]     = useState(null);
  const [log,          setLog]          = useState([]);
  const [seed,         setSeed]         = useState(null);
  const [weavyAppUid,      setWeavyAppUid]      = useState(null);
  const [weavyMessages,    setWeavyMessages]    = useState([]);
  const [weavyPosting,     setWeavyPosting]     = useState(false);
  const [weavyStatus,      setWeavyStatus]      = useState("idle");
  const [showWeavyPanel,   setShowWeavyPanel]   = useState(false);
  const [checkingFeedback, setCheckingFeedback] = useState(false);
  const [feedbackFound,    setFeedbackFound]    = useState(false);
  const [threadUrl,        setThreadUrl]        = useState(null);

  const { claudeKey, comfyUrl, comfyModel, comfySteps, comfyGuidance, comfyWidth, comfyHeight, comfyWorkflow,
          ratio, previewProvider, nbModel, falModel, geminiKey, nbKey, falKey, weavyUrl, weavyKey,
          useFrame1Ref, useSharedSeed, manualSeed, useCameraLock, cameraLens, cameraFraming, cameraFreeText,
          useCropLock } = settings;

  const comfyConfigured  = !!comfyUrl.trim();
  const weavyConfigured  = !!(weavyUrl.trim() && weavyKey.trim());
  const previewKey       = previewProvider === "gemini" ? geminiKey : previewProvider === "nanobanana" ? nbKey : falKey;
  const previewAvailable = !!previewKey.trim();
  const busy             = genBusy || renderBusy;
  const providerLabel    = PREVIEW_PROVIDERS[previewProvider] ? PREVIEW_PROVIDERS[previewProvider].label : "preview";

  function handleAddRef(file) {
    const idx = refs.length;
    setRefs(p => [...p, { file, preview:URL.createObjectURL(file), falUrl:null, uploading:false }]);
    if (falKey.trim()) uploadRef(file, idx);
  }
  async function uploadRef(file, index) {
    setRefs(p => p.map((r, i) => i===index ? { ...r, uploading:true } : r));
    try {
      const url = await falUpload(falKey, file);
      setRefs(p => p.map((r, i) => i===index ? { ...r, falUrl:url, uploading:false } : r));
    } catch (e) { setRefs(p => p.map((r, i) => i===index ? { ...r, uploading:false } : r)); }
  }
  function handleRemoveRef(index) { setRefs(p => p.filter((_, i) => i !== index)); }

  async function buildPrompts(feedback) {
    if (!claudeKey.trim()) throw new Error("No Claude API key — add it in Settings");

    const assetLines = [];
    if (environment.name || environment.setting) {
      assetLines.push("ENVIRONMENT SHEET:");
      if (environment.name)    assetLines.push("  Name: " + environment.name);
      if (environment.setting) assetLines.push("  Setting: " + environment.setting);
      if (environment.time)    assetLines.push("  Time: " + environment.time);
      if (environment.mood)    assetLines.push("  Mood: " + environment.mood);
      if (environment.notes)   assetLines.push("  Notes: " + environment.notes);
    }
    characters.forEach((c, i) => {
      if (!c.name && !c.appearance) return;
      assetLines.push("\nCHARACTER SHEET " + (i+1) + ": " + (c.name || "Unnamed"));
      if (c.role)        assetLines.push("  Role: " + c.role);
      if (c.appearance)  assetLines.push("  Appearance: " + c.appearance);
      if (c.costume)     assetLines.push("  Costume: " + c.costume);
      if (c.expression)  assetLines.push("  Expression: " + c.expression);
      if (c.distinctive) assetLines.push("  Distinctive: " + c.distinctive);
      if (c.notes)       assetLines.push("  Notes: " + c.notes);
    });
    objects.forEach((o, i) => {
      if (!o.name && !o.description) return;
      assetLines.push("\nOBJECT SHEET " + (i+1) + ": " + (o.name || "Unnamed"));
      if (o.description) assetLines.push("  Description: " + o.description);
      if (o.scale)       assetLines.push("  Scale: " + o.scale);
      if (o.behaviour)   assetLines.push("  Behaviour: " + o.behaviour);
      if (o.notes)       assetLines.push("  Notes: " + o.notes);
    });
    const assetBrief = assetLines.join("\n");

    const lens = LENS_OPTIONS.find(l => l.id === cameraLens);
    const framing = FRAMING_OPTIONS.find(f => f.id === cameraFraming);
    const cameraString = [framing ? framing.label : "", lens ? lens.label + " lens" : "", cameraFreeText || "", "camera locked across both frames"].filter(Boolean).join(" · ");

    const shotSeed = useSharedSeed ? (manualSeed.trim() ? parseInt(manualSeed.trim(), 10) : Math.floor(Math.random() * 999999999)) : Math.floor(Math.random() * 999999999);
    setSeed(shotSeed);

    const refCount = refs.length;
    const userMsg = [
      "VISUAL BIBLE:",
      bible,
      assetBrief.trim() ? "\n---\n\nSHOT ASSETS:\n" + assetBrief : "",
      useCameraLock ? "\n---\n\nCAMERA LOCK (apply identically to both frames):\n" + cameraString : "",
      "\n---\n\nFRAME 1 — START FRAME:",
      frame1,
      "\n---\n\nFRAME 2 — END FRAME:",
      frame2,
      "\n---\n\nSHARED SEED: " + shotSeed,
      "\nREFERENCES: " + (refCount > 0 ? refCount + " reference image" + (refCount>1?"s":"") + " provided." : "No reference images."),
      feedback ? "\n---\n\nDIRECTOR FEEDBACK:\n" + feedback : "",
      "\n\nSHOT LOG:\n" + (log.length ? log.map((s, i) => "#" + (i+1) + ": " + s.shotSummary).join("\n") : "No previous shots."),
    ].join("\n");

    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": claudeKey,
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
      },
      body: JSON.stringify({ model:"claude-opus-4-6", max_tokens:8000, system:SYSTEM_PROMPT, messages:[{ role:"user", content:userMsg }] }),
    });

    if (!res.ok) {
      const errText = await res.text();
      if (res.status === 401) throw new Error("Invalid Claude API key — check Settings");
      if (res.status === 429) throw new Error("Claude rate limit — wait a moment and retry");
      throw new Error("Claude API error " + res.status + ": " + errText.slice(0, 80));
    }

    const data = await res.json();
    const txt = (data.content || []).map(b => b.text || "").join("");
    const clean = txt.replace(/```json|```/g, "").trim();
    try {
      return JSON.parse(clean);
    } catch (parseErr) {
      const opens  = (clean.match(/{/g) || []).length;
      const closes = (clean.match(/}/g) || []).length;
      const needed = Math.max(0, opens - closes);
      try { return JSON.parse(clean + "}".repeat(needed)); }
      catch (e2) { throw new Error("Response truncated — try shorter frame descriptions."); }
    }
  }

  async function renderFrames(parsed, mode) {
    setActiveMode(mode);
    setStartLoading(true); setEndLoading(true);
    setStartErr(null); setEndErr(null); setStartImg(null); setEndImg(null);
    setStartMsg(mode === "comfy" ? "Queued in ComfyUI…" : "Rendering…");
    setEndMsg(mode === "comfy" ? "Queued in ComfyUI…" : "Rendering…");

    let uploadedRefs = [...refs];
    if (falKey.trim()) {
      uploadedRefs = await Promise.all(refs.map(async (r, i) => {
        if (r.file && !r.falUrl) {
          try { const url = await falUpload(falKey, r.file); setRefs(p => p.map((x, idx) => idx===i ? { ...x, falUrl:url } : x)); return { ...r, falUrl:url }; }
          catch (e) { return r; }
        }
        return r;
      }));
    }
    const refUrls = uploadedRefs.map(r => r.falUrl).filter(Boolean);
    const activeSeed = useSharedSeed ? seed : Math.floor(Math.random() * 999999999);

    async function renderFrame(frame, setImg, setLoad, setMsg, setErr, refResult) {
      try {
        let url;
        if (mode === "comfy") {
          const customWf = comfyWorkflow.trim() || null;
          const wf = customWf
            ? injectIntoCustomWorkflow(JSON.parse(customWf), frame.prompt, frame.negativePrompt)
            : buildComfyWorkflow(frame.prompt, frame.negativePrompt, comfyModel, comfyWidth, comfyHeight, comfySteps, Math.floor(Math.random() * 999999999));
          setMsg("Rendering in ComfyUI…");
          const promptId = await comfySubmit(comfyUrl, wf);
          url = await comfyPoll(comfyUrl, promptId);
        } else if (previewProvider === "gemini") {
          const refB64   = refResult ? refResult.base64   : null;
          const refMime  = refResult ? refResult.mimeType : null;
          const result = await geminiGenerateImage(geminiKey, frame.prompt, refB64, refMime, activeSeed);
          frame._geminiResult = result;
          url = result.url;
        } else if (previewProvider === "nanobanana") {
          url = await nbGenerate(nbKey, nbModel, frame.prompt, frame.negativePrompt, ratio, refUrls.length ? refUrls : undefined);
        } else {
          url = await falText2Img(falKey, falModel, frame.prompt, frame.negativePrompt, ratio);
        }
        setImg(url); setMsg(""); return url;
      } catch (e) {
        const msg = e.message.includes("401") ? "Invalid API key" : e.message.includes("402") ? "Insufficient credits" : e.message.slice(0, 80);
        setErr(msg); return null;
      } finally { setLoad(false); }
    }

    let s1, s2;
    if (mode === "comfy") {
      s1 = await renderFrame(parsed.startFrame, setStartImg, setStartLoading, setStartMsg, setStartErr, null);
      s2 = await renderFrame(parsed.endFrame,   setEndImg,   setEndLoading,   setEndMsg,   setEndErr,   null);
    } else if (previewProvider === "gemini") {
      s1 = await renderFrame(parsed.startFrame, setStartImg, setStartLoading, setStartMsg, setStartErr, null);
      if (useFrame1Ref && parsed.startFrame._geminiResult) {
        const anchor = "CAMERA ANCHOR — maintain identical camera to Frame 1: " + parsed.startFrame.compositionNote + ". Subject scale, distance from camera, and framing must match exactly.";
        parsed.endFrame.prompt = anchor + "\n\n" + parsed.endFrame.prompt;
      }
      s2 = await renderFrame(parsed.endFrame, setEndImg, setEndLoading, setEndMsg, setEndErr, useFrame1Ref ? parsed.startFrame._geminiResult : null);
    } else {
      const [r1, r2] = await Promise.all([
        renderFrame(parsed.startFrame, setStartImg, setStartLoading, setStartMsg, setStartErr, null),
        renderFrame(parsed.endFrame,   setEndImg,   setEndLoading,   setEndMsg,   setEndErr,   null),
      ]);
      s1 = r1; s2 = r2;
    }

    if (useCropLock && s1 && s2 && s1.startsWith("data:") && s2.startsWith("data:")) {
      try {
        const cropped = await applyCropLock(s1, s2);
        s1 = cropped.url1; s2 = cropped.url2;
        setStartImg(s1); setEndImg(s2);
      } catch (e) { console.warn("Crop lock failed:", e); }
    }

    return { s1, s2 };
  }

  async function postToWeavy(parsed, s1, s2, ver) {
    if (!weavyConfigured) return;
    setWeavyPosting(true);
    try {
      const uid = "da-" + (parsed.sceneSlug || String(Date.now()));
      setWeavyAppUid(uid);
      await weavyUpsertApp(weavyUrl, weavyKey, uid, "Shot: " + (parsed.shotSummary || "").slice(0, 60));
      setThreadUrl(weavyUrl + "/messenger/" + uid);
      const verStr = ver > 1 ? " — v" + ver : "";
      const fbStr = parsed.feedbackApplied && parsed.feedbackApplied !== "null" ? "\n\n📝 Feedback: " + parsed.feedbackApplied : "";
      await weavyPostMsg(weavyUrl, weavyKey, uid, "🎬 **" + parsed.shotSummary + "**" + verStr + fbStr + "\n\n↗ " + (parsed.motionArc || "—") + "\n\nFrame 1:\n" + (s1 || "(failed)") + "\n\nFrame 2:\n" + (s2 || "(failed)") + "\n\n---\n_Reply with feedback or 'approved'._");
      setWeavyMessages(await weavyGetMsgs(weavyUrl, weavyKey, uid));
      setWeavyStatus("ok");
      setShowWeavyPanel(true);
    } catch (e) { console.error("Weavy:", e); setWeavyStatus("error"); }
    setWeavyPosting(false);
  }

  const [version, setVersion] = useState(1);

  async function handleGenerate(mode) {
    if (!canGenerate) return;
    const ver = 1;
    setVersion(ver);
    setGenBusy(true); setGenError(null); setShot(null);
    setWeavyStatus("idle"); setFeedbackFound(false);
    try {
      const parsed = await buildPrompts(null);
      setShot(parsed);
      setLog(p => [...p, { ...parsed, _version:ver }].slice(-20));
      setRenderBusy(true);
      const { s1, s2 } = await renderFrames(parsed, mode);
      setRenderBusy(false);
      await postToWeavy(parsed, s1, s2, ver);
    } catch (e) {
      console.error("Generate:", e);
      setGenError(e.message || "Generation failed — check your API keys in Settings");
      setRenderBusy(false);
    }
    setGenBusy(false);
  }

  async function handleCheckFeedback() {
    if (!weavyConfigured || !weavyAppUid) return;
    setCheckingFeedback(true); setFeedbackFound(false);
    try {
      const msgs = await weavyGetMsgs(weavyUrl, weavyKey, weavyAppUid);
      setWeavyMessages(msgs);
      const dirMsgs = msgs.filter(m => !(m.text && m.text.startsWith("🎬")) && m.plain && m.plain.trim());
      const latest = dirMsgs[dirMsgs.length - 1];
      if (!latest) { setCheckingFeedback(false); return; }
      const fb = latest.plain || latest.text || "";
      if (fb.toLowerCase().includes("approved")) {
        await weavyPostMsg(weavyUrl, weavyKey, weavyAppUid, "✅ Shot approved. Ready for production render.");
        setWeavyMessages(await weavyGetMsgs(weavyUrl, weavyKey, weavyAppUid));
        setCheckingFeedback(false); return;
      }
      setFeedbackFound(true);
      const newVer = version + 1;
      setVersion(newVer);
      setShowWeavyPanel(false);
      setGenBusy(true); setRenderBusy(true);
      const parsed = await buildPrompts(fb);
      setShot(parsed);
      setLog(p => [...p, { ...parsed, _version:newVer }].slice(-20));
      const mode = comfyConfigured ? "comfy" : "preview";
      const { s1, s2 } = await renderFrames(parsed, mode);
      setRenderBusy(false); setGenBusy(false);
      await postToWeavy(parsed, s1, s2, newVer);
    } catch (e) { console.error("Feedback:", e); }
    setCheckingFeedback(false);
  }

  async function handleRerender(mode) {
    if (!shot) return;
    setRenderBusy(true);
    const { s1, s2 } = await renderFrames(shot, mode);
    setRenderBusy(false);
    await postToWeavy(shot, s1, s2, version);
  }

  const canGenerate = !busy && bible.trim().length > 40 && frame1.trim().length > 20 && frame2.trim().length > 20 && !!claudeKey.trim();

  return (
    <div style={{ minHeight:"100vh", background:"#09090a", color:"#e8e0d4", fontFamily:"Georgia,'Times New Roman',serif" }}>
      <style>{css}</style>
      <nav style={{ height:52, borderBottom:"1px solid rgba(255,255,255,.065)", display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 22px", position:"sticky", top:0, background:"rgba(9,9,10,.96)", backdropFilter:"blur(8px)", zIndex:100 }}>
        <div style={{ display:"flex", alignItems:"baseline", gap:10 }}>
          <span style={{ fontSize:14, fontWeight:400, letterSpacing:".08em" }}>DIRECTOR AGENT</span>
          <span style={{ fontSize:10, letterSpacing:".16em", color:"rgba(200,160,80,.38)", textTransform:"uppercase", fontFamily:"sans-serif" }}>Start · End Frame</span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <div style={{ display:"flex", gap:10, alignItems:"center" }}>
            <div style={{ display:"flex", alignItems:"center", gap:5 }}>
              <span style={{ width:6, height:6, borderRadius:"50%", background:claudeKey.trim() ? "#5cb87a" : "rgba(200,80,80,.5)" }} />
              <span style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif" }}>{claudeKey.trim() ? "Claude" : "No Claude key"}</span>
            </div>
            <div style={{ display:"flex", alignItems:"center", gap:5 }}>
              <span style={{ width:6, height:6, borderRadius:"50%", background:previewAvailable ? "rgba(200,160,80,.6)" : "rgba(255,255,255,.12)" }} />
              <span style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif" }}>{previewAvailable ? providerLabel : "No preview key"}</span>
            </div>
            {comfyConfigured && (
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <span style={{ width:6, height:6, borderRadius:"50%", background:"rgba(180,130,255,.7)" }} />
                <span style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif" }}>ComfyUI</span>
              </div>
            )}
            {weavyConfigured && (
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <span style={{ width:6, height:6, borderRadius:"50%", background:weavyStatus==="ok"?"#5cb87a":weavyPosting?"#c8a050":"rgba(80,180,120,.4)" }} />
                <span style={{ fontSize:11, color:"rgba(232,224,212,.35)", fontFamily:"sans-serif" }}>Weavy</span>
              </div>
            )}
          </div>
          {weavyAppUid && weavyConfigured && (
            <button onClick={() => setShowWeavyPanel(!showWeavyPanel)}
              style={{ background:showWeavyPanel?"rgba(80,180,120,.15)":"rgba(80,180,120,.07)", border:"1px solid rgba(80,180,120,.28)", borderRadius:6, padding:"5px 13px", cursor:"pointer", fontSize:11, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(80,180,120,.85)" }}>
              {"💬 " + (showWeavyPanel ? "Hide" : "Review")}
            </button>
          )}
          <button onClick={() => setShowSettings(true)} style={{ background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.08)", borderRadius:6, padding:"5px 13px", cursor:"pointer", fontSize:11, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(232,224,212,.5)" }}>⚙ Settings</button>
        </div>
      </nav>
      <div style={{ display:"grid", gridTemplateColumns:"430px 1fr", minHeight:"calc(100vh - 52px)", position:"relative", zIndex:2 }}>
        <div style={{ borderRight:"1px solid rgba(255,255,255,.055)", padding:"22px 20px 120px", overflowY:"auto", maxHeight:"calc(100vh - 52px)", position:"sticky", top:52, display:"flex", flexDirection:"column", gap:18 }}>
          <div>
            <FieldLabel main="Visual Bible" sub="Permanent universe rulebook — materials, constraints, reinterpretation rules, characters, lighting, colour, style keywords" />
            <AutoTA value={bible} onChange={setBible} placeholder="Your complete visual universe — approved materials, prohibited materials, reinterpretation rules, characters, lighting, colour, tone, style keywords…" minHeight={200} fontSize={12} />
          </div>
          <Divider />
          <ShotAssets environment={environment} setEnvironment={setEnvironment} characters={characters} setCharacters={setCharacters} objects={objects} setObjects={setObjects} assetTab={assetTab} setAssetTab={setAssetTab} assetImages={assetImages} setAssetImages={setAssetImages} />
          <Divider />
          <div style={{ display:"grid", gap:12 }}>
            <FrameInput number={1} label="Frame 1 — Start" value={frame1} onChange={setFrame1} placeholder="Framing, camera, hero element, emotional tone, subject position and action state at the START…" />
            <FrameInput number={2} label="Frame 2 — End" value={frame2} onChange={setFrame2} placeholder="Same camera (or state if it moves), same hero element, emotional resolution, subject position and action state at the END…" />
          </div>
          <Divider />
          <ConsistencyControls settings={settings} setSettings={setSettings} seed={seed} comfyConfigured={comfyConfigured} />
          <Divider />
          <div>
            <FieldLabel main="References" sub="Optional — character, style, mood, environment. Used for preview render consistency." />
            <RefZone refs={refs} onAdd={handleAddRef} onRemove={handleRemoveRef} />
          </div>
          <div style={{ display:"grid", gap:9 }}>
            <button onClick={() => handleGenerate("comfy")} disabled={!canGenerate || !comfyConfigured}
              style={{ width:"100%", padding:"14px", borderRadius:8, border:"1px solid " + (canGenerate&&comfyConfigured?"rgba(130,80,200,.5)":"rgba(255,255,255,.06)"), background:canGenerate&&comfyConfigured?"rgba(130,80,200,.15)":"rgba(255,255,255,.018)", color:canGenerate&&comfyConfigured?"rgba(180,130,255,.95)":"rgba(232,224,212,.18)", fontSize:12, letterSpacing:".16em", textTransform:"uppercase", fontFamily:"sans-serif", fontWeight:700, cursor:canGenerate&&comfyConfigured?"pointer":"not-allowed", transition:"all .25s", display:"flex", alignItems:"center", justifyContent:"center", gap:10 }}>
              {busy && activeMode==="comfy" ? <><Spin />Rendering in ComfyUI…</> : "🎨 Generate via ComfyUI"}
            </button>
            {!comfyConfigured && <p style={{ fontSize:10, color:"rgba(180,130,255,.4)", fontFamily:"sans-serif", textAlign:"center", marginTop:-4, fontStyle:"italic" }}>Add ComfyUI server URL in Settings</p>}
            <button onClick={() => handleGenerate("preview")} disabled={!canGenerate || !previewAvailable}
              style={{ width:"100%", padding:"12px", borderRadius:8, border:"1px solid " + (canGenerate&&previewAvailable?"rgba(200,160,80,.38)":"rgba(255,255,255,.05)"), background:canGenerate&&previewAvailable?"rgba(200,160,80,.1)":"rgba(255,255,255,.012)", color:canGenerate&&previewAvailable?"rgba(200,160,80,.85)":"rgba(232,224,212,.15)", fontSize:12, letterSpacing:".14em", textTransform:"uppercase", fontFamily:"sans-serif", fontWeight:700, cursor:canGenerate&&previewAvailable?"pointer":"not-allowed", transition:"all .25s", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
              {busy && activeMode==="preview" ? <><Spin />Generating preview…</> : "👁 Quick Preview — " + providerLabel}
            </button>
            {busy && (
              <div style={{ borderRadius:8, border:"1px solid rgba(200,160,80,.25)", background:"rgba(200,160,80,.07)", overflow:"hidden" }}>
                <div style={{ height:3, background:"rgba(200,160,80,.1)", position:"relative", overflow:"hidden" }}>
                  <div style={{ position:"absolute", top:0, left:"-100%", width:"60%", height:"100%", background:"linear-gradient(90deg, transparent, rgba(200,160,80,.7), transparent)", animation:"shimmer 1.3s ease infinite" }} />
                </div>
                <div style={{ padding:"11px 13px", display:"flex", alignItems:"center", gap:10 }}>
                  <Spin size={14} />
                  <div>
                    <div style={{ fontSize:12, color:"#c8a050", fontFamily:"sans-serif", fontWeight:700 }}>{genBusy ? "Claude is writing your prompts…" : "Rendering via " + (activeMode==="comfy"?"ComfyUI":providerLabel) + "…"}</div>
                    <div style={{ fontSize:11, color:"rgba(232,224,212,.4)", fontFamily:"sans-serif", marginTop:3 }}>{genBusy ? "Applying Visual Bible · Compliance check · Building frame pair" : activeMode==="comfy" ? "Sent to ComfyUI — polling for result…" : "Generating start + end frames"}</div>
                  </div>
                </div>
              </div>
            )}
            {!canGenerate && !busy && (
              <p style={{ fontSize:11, color:"rgba(232,224,212,.25)", fontFamily:"sans-serif", textAlign:"center", fontStyle:"italic" }}>
                {!claudeKey.trim() ? "Add your Claude API key in Settings" : !bible.trim() ? "Add your Visual Bible to continue" : !frame1.trim() ? "Describe Frame 1 to continue" : !frame2.trim() ? "Describe Frame 2 to continue" : !previewAvailable ? "Add a preview API key in Settings" : ""}
              </p>
            )}
            {genError && <div style={{ padding:"10px 12px", background:"rgba(180,60,60,.09)", border:"1px solid rgba(180,60,60,.2)", borderRadius:7, fontSize:12, color:"#e08080", fontFamily:"sans-serif", lineHeight:1.5 }}>{genError}</div>}
          </div>
        </div>
        <div style={{ position:"relative", overflow:"hidden" }}>
          <div style={{ padding:"22px 24px 120px", overflowY:"auto", maxHeight:"calc(100vh - 52px)", opacity:showWeavyPanel?0.3:1, transition:"opacity .2s", pointerEvents:showWeavyPanel?"none":"auto" }}>
            {!shot && !busy && (
              <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", height:"80%", gap:14, opacity:.2 }}>
                <div style={{ fontSize:44 }}>🎬</div>
                <div style={{ fontSize:14, color:"rgba(232,224,212,.6)", fontFamily:"sans-serif", fontStyle:"italic" }}>Frame 1 and Frame 2 will appear here</div>
              </div>
            )}
            {shot && (
              <div style={{ animation:"fadeIn .5s ease both" }}>
                <div style={{ marginBottom:14, paddingBottom:14, borderBottom:"1px solid rgba(255,255,255,.055)" }}>
                  <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", gap:12, flexWrap:"wrap" }}>
                    <div>
                      <div style={{ fontSize:15, color:"#e8e0d4", lineHeight:1.45, marginBottom:4 }}>{shot.shotSummary}</div>
                      <div style={{ fontSize:12, color:"rgba(232,224,212,.45)", fontFamily:"sans-serif", fontStyle:"italic", lineHeight:1.5 }}>{shot.sharedContext}</div>
                    </div>
                    {version > 1 && <span style={{ fontSize:11, padding:"3px 9px", background:"rgba(200,160,80,.1)", border:"1px solid rgba(200,160,80,.22)", borderRadius:20, color:"rgba(200,160,80,.7)", fontFamily:"sans-serif", flexShrink:0 }}>{"v" + version}</span>}
                  </div>
                  {shot.feedbackApplied && shot.feedbackApplied !== "null" && (
                    <div style={{ marginTop:9, padding:"8px 12px", background:"rgba(80,180,120,.06)", border:"1px solid rgba(80,180,120,.15)", borderRadius:6, fontSize:12, color:"rgba(80,180,120,.75)", fontFamily:"sans-serif" }}>{"📝 " + shot.feedbackApplied}</div>
                  )}
                </div>
                <MotionArc text={shot.motionArc} />
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:16 }}>
                  <FramePanel number={1} label="Start Frame" frame={shot.startFrame} imageUrl={startImg} loading={startLoading} loadingMsg={startMsg} error={startErr} />
                  <FramePanel number={2} label="End Frame"   frame={shot.endFrame}   imageUrl={endImg}   loading={endLoading}   loadingMsg={endMsg}   error={endErr} />
                </div>
                <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:10 }}>
                  <AuditPanel audit={shot.auditResult} />
                  <div style={{ display:"flex", gap:8, alignItems:"center", flexWrap:"wrap" }}>
                    {!busy && comfyConfigured && <button onClick={() => handleRerender("comfy")} style={{ background:"rgba(130,80,200,.1)", border:"1px solid rgba(130,80,200,.28)", borderRadius:6, color:"rgba(180,130,255,.75)", padding:"6px 12px", fontSize:11, letterSpacing:".1em", textTransform:"uppercase", cursor:"pointer", fontFamily:"sans-serif" }}>🎨 Re-render Comfy</button>}
                    {!busy && previewAvailable && <button onClick={() => handleRerender("preview")} style={{ background:"transparent", border:"1px solid rgba(200,160,80,.2)", borderRadius:6, color:"rgba(200,160,80,.55)", padding:"6px 12px", fontSize:11, letterSpacing:".1em", textTransform:"uppercase", cursor:"pointer", fontFamily:"sans-serif" }}>👁 Re-preview</button>}
                    {weavyStatus==="ok" && !showWeavyPanel && <button onClick={() => setShowWeavyPanel(true)} style={{ background:"rgba(80,180,120,.1)", border:"1px solid rgba(80,180,120,.25)", borderRadius:6, color:"rgba(80,180,120,.8)", padding:"6px 12px", fontSize:11, letterSpacing:".1em", textTransform:"uppercase", cursor:"pointer", fontFamily:"sans-serif" }}>💬 Review Thread</button>}
                    {weavyStatus==="error" && <span style={{ fontSize:11, color:"rgba(220,100,100,.55)", fontFamily:"sans-serif" }}>Weavy post failed</span>}
                  </div>
                </div>
              </div>
            )}
          </div>
          {showWeavyPanel && (
            <WeavyPanel shot={shot} threadUrl={threadUrl} messages={weavyMessages} onCheckFeedback={handleCheckFeedback} onClose={() => setShowWeavyPanel(false)} checkingFeedback={checkingFeedback} feedbackFound={feedbackFound} />
          )}
        </div>
      </div>
      <LogStrip log={log} onSelect={s => setShot(s)} />
      {showSettings && <Settings s={settings} set={setSettings} onClose={() => setShowSettings(false)} />}
    </div>
  );
}

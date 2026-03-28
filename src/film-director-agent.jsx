import { useState, useRef, useEffect } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// SYSTEM PROMPT
// ─────────────────────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are a Director of Photography and Prompt Architect for an AI film production pipeline.

Your output — a START FRAME and END FRAME — will be fed directly into a video generation model (Runway, Kling, or similar) as the first and last frames of a shot. The model generates everything in between. Your two frames must be PHYSICALLY INTERPOLATABLE.

You receive two inputs:
1. VISUAL BIBLE — the universe rulebook governing every material, surface, character, and environment.
2. FRAME 1 DESCRIPTION — the director's complete instruction for the start frame.
3. FRAME 2 DESCRIPTION — the director's complete instruction for the end frame.

Each frame description contains the director's full intent: framing, camera position, hero elements, emotional tone, camera movement, material reminders, AND the specific subject position and action state for that frame. Both descriptions will share the same framing, camera, and hero context — only the subject position, expression, and action state will differ between them.

Your job is to:
- Extract the shared context from both descriptions (framing, camera, hero elements, tone, material reminders)
- Extract what is specific to each frame (subject position, action state, expression)
- Build two fully bible-compliant prompts — one for each frame

━━━ VIDEO INTERPOLATION — NON-NEGOTIABLE CONSTRAINTS ━━━

CAMERA LOCK
Camera position, height, angle, and focal length are IDENTICAL in both frames unless camera movement is explicitly described. If movement is described, Frame 1 = camera open position, Frame 2 = camera landing position.

LIGHTING FREEZE
Lighting is frozen at a single moment. Key light direction, shadow positions, and light source states are identical in both frames.

SPATIAL PLAUSIBILITY
The subject's Frame 2 position must be physically reachable from their Frame 1 position within one continuous shot. Scale and distance from camera remain consistent.

IMPLIED MOTION
Frame 1 must compositionally suggest the motion that Frame 2 resolves. These are two points on the same continuous arc.

LOCKS
Costume, hair, environment, and set dressing are identical in both frames. Only subject position, expression, gesture, and action state may change.

━━━ VISUAL BIBLE — THREE BINDING LAYERS ━━━

LAYER 1 — POSITIVE MATERIAL LANGUAGE
Extract every approved material and texture. Only these materials may appear in any prompt.

LAYER 2 — NEGATIVE MATERIAL CONSTRAINT
Extract every prohibited material. Apply as hard negative prompt exclusions AND avoid in positive prompts.

LAYER 3 — REINTERPRETATION RULE
Every object named that would normally use a prohibited material must be translated through the bible's approved material system. No exceptions.

━━━ PROMPT CONSTRUCTION ━━━

For each frame:
- Open with framing and camera position
- Describe the subject's exact position and action state for that frame
- Describe the environment through the bible's material language
- Embed lighting, palette, and lens from the bible verbatim
- End with the full style keyword string from the bible
- 120–160 words per prompt

━━━ MANDATORY PRE-OUTPUT CHECKLIST ━━━

□ Camera identical in both frames (or explicitly progressed if movement described)?
□ Lighting frozen — same shadow positions and light states?
□ Frame 2 subject position spatially reachable from Frame 1?
□ Frame 1 implies the motion Frame 2 resolves?
□ Costume, hair, environment locked across both?
□ Every named element translated through the bible's material system?
□ All prohibited materials listed in negative prompts?
□ Hero elements reproduced with exact precision in both frames?
□ Both prompts 120–160 words?

Rewrite any non-compliant prompt before outputting.

━━━ OUTPUT ━━━
Valid JSON only. No markdown. No preamble.

{
  "shotSummary": "one sentence: the complete motion arc from Frame 1 to Frame 2",
  "sceneSlug": "kebab-case-max-5-words",
  "sharedContext": "camera, lighting state, environment, and costume locked across both frames — described through the bible's material language",
  "motionArc": "what physically moves, how far, in what direction",
  "startFrame": {
    "compositionNote": "subject position and action state at Frame 1",
    "prompt": "full bible-compliant prompt 120-160 words",
    "negativePrompt": "all prohibited materials and treatments from the bible"
  },
  "endFrame": {
    "compositionNote": "subject position and action state at Frame 2",
    "prompt": "full bible-compliant prompt 120-160 words",
    "negativePrompt": "all prohibited materials and treatments from the bible"
  },
  "auditResult": {
    "cameraLock": "confirmed | [issue]",
    "lightingFreeze": "confirmed | [issue]",
    "spatialPlausibility": "confirmed | [issue]",
    "impliedMotion": "confirmed — [motion arc description]",
    "materialCompliance": "confirmed | [violations found and corrected]",
    "heroElementLock": "confirmed | not applicable | [issue]",
    "reinterpretationApplied": "confirmed — [key reinterpretations made]"
  }
}`;

// ─────────────────────────────────────────────────────────────────────────────
// PLACEHOLDERS — guide the user toward director-level input
// ─────────────────────────────────────────────────────────────────────────────
const BIBLE_PLACEHOLDER = `Your complete visual universe — write everything here:

WORLD MATERIAL LANGUAGE
What is everything made of? Characters, props, furniture, architecture, environments. Every approved material, texture, and construction method.

WHAT CANNOT EXIST
Every prohibited material. These become hard constraints on every prompt.

REINTERPRETATION RULE
How must objects built from prohibited materials be translated? E.g. a window becomes... a road becomes...

CHARACTERS
Skin, hair, eyes, clothing — how people are constructed in this world.

LIGHTING & COLOUR
Lighting quality, temperature, shadow behaviour, colour palette.

TONE
The emotional register. What the world must never feel like.

STYLE KEYWORDS
Global image generation tags appended to every prompt.`;

const FRAME1_PLACEHOLDER = `Describe Frame 1 completely — include everything:

FRAMING & CAMERA — wide/medium/close, angle, symmetry, composition, movement (or "camera locked")
HERO ELEMENT — any specific product, prop, or garment to preserve exactly (every detail)
EMOTIONAL TONE — the feeling this shot carries
MATERIAL REMINDER — any shot-specific reinforcement of bible rules
SUBJECT — where they are in frame, body posture, action state, expression at the START of the shot`;

const FRAME2_PLACEHOLDER = `Describe Frame 2 completely — include everything:

FRAMING & CAMERA — same as Frame 1 unless camera moves (state if it does and where it lands)
HERO ELEMENT — same hero element details as Frame 1 (must match exactly)
EMOTIONAL TONE — same tone, or how it shifts at the shot's resolution
MATERIAL REMINDER — same material rules apply
SUBJECT — where they are in frame, body posture, action state, expression at the END of the shot`;

// ─────────────────────────────────────────────────────────────────────────────
// API HELPERS
// ─────────────────────────────────────────────────────────────────────────────
const ASPECT_RATIOS = [
  { id: "landscape_16_9", label: "16:9", nb: "16:9" },
  { id: "landscape_21_9", label: "21:9", nb: "21:9" },
  { id: "landscape_4_3",  label: "4:3",  nb: "4:3"  },
];

const PROVIDERS = {
  nanobanana: {
    label: "NanoBanana", icon: "🍌",
    models: [
      { id: "nano-banana-2",   label: "Nano Banana 2",   desc: "Gemini 3.1 Flash · 4K" },
      { id: "nano-banana-pro", label: "Nano Banana Pro", desc: "Gemini 3 Pro · premium" },
      { id: "nano-banana",     label: "Nano Banana",     desc: "Gemini 2.5 Flash" },
    ],
  },
  fal: {
    label: "Flux", icon: "⚡",
    models: [
      { id: "fal-ai/flux/dev",     label: "Flux Dev",     desc: "Best fidelity" },
      { id: "fal-ai/flux/schnell", label: "Flux Schnell", desc: "Fastest" },
    ],
  },
};

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
      prompt,
      negative_prompt: negativePrompt || "photo-realistic, CGI, plastic, metal, glass, wood",
      image_size: aspectRatio,
      num_inference_steps: model.includes("schnell") ? 4 : 28,
      guidance_scale: 3.5, num_images: 1, enable_safety_checker: false,
    }),
  });
  if (!res.ok) throw new Error(`fal.ai ${res.status}: ${await res.text()}`);
  const d = await res.json();
  return d.images?.[0]?.url || d.image?.url;
}

async function falPuLID(apiKey, faceUrl, prompt, negativePrompt, aspectRatio) {
  const sizeMap = {
    landscape_16_9: { width: 1024, height: 576 },
    landscape_21_9: { width: 1280, height: 544 },
    landscape_4_3:  { width: 1024, height: 768 },
  };
  const sz = sizeMap[aspectRatio] || sizeMap.landscape_16_9;
  const res = await fetch("https://fal.run/fal-ai/pulid", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Key ${apiKey}` },
    body: JSON.stringify({ face_image_url: faceUrl, prompt, negative_prompt: negativePrompt, ...sz, num_inference_steps: 20, guidance_scale: 7, num_images: 1 }),
  });
  if (!res.ok) throw new Error(`PuLID ${res.status}: ${await res.text()}`);
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

async function weavyUpsertApp(weavyUrl, apiKey, uid, name) {
  const res = await fetch(`${weavyUrl}/api/apps`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
    body: JSON.stringify({ uid, name, type: "chat" }),
  });
  if (res.status === 409) {
    const g = await fetch(`${weavyUrl}/api/apps/${uid}`, { headers: { "Authorization": `Bearer ${apiKey}` } });
    return g.json();
  }
  return res.json();
}

async function weavyPost(weavyUrl, apiKey, appUid, text) {
  await fetch(`${weavyUrl}/api/apps/${appUid}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
    body: JSON.stringify({ text }),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────────────────────────
const css = `
  @keyframes spin   { to { transform: rotate(360deg); } }
  @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
  @keyframes pulse  { 0%,100%{opacity:.3} 50%{opacity:.85} }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:#09090a; }
  textarea, input { box-sizing:border-box; }
  ::-webkit-scrollbar { width:4px; }
  ::-webkit-scrollbar-thumb { background:rgba(200,160,80,.13); border-radius:2px; }
  .ta::placeholder { color:rgba(232,224,212,.15); font-style:italic; font-size:11px; line-height:1.65; }
  .ta:focus { border-color:rgba(200,160,80,.3) !important; outline:none; }
  .drop-active { border-color:rgba(200,160,80,.55) !important; background:rgba(200,160,80,.06) !important; }
`;

// ─────────────────────────────────────────────────────────────────────────────
// ATOMS
// ─────────────────────────────────────────────────────────────────────────────
const Spin = () => (
  <span style={{ display:"inline-block", width:12, height:12, border:"2px solid rgba(200,160,80,.18)", borderTopColor:"#c8a050", borderRadius:"50%", animation:"spin .8s linear infinite", flexShrink:0 }} />
);

function CopyBtn({ text }) {
  const [ok, setOk] = useState(false);
  return (
    <button onClick={() => { navigator.clipboard.writeText(text); setOk(true); setTimeout(() => setOk(false), 2000); }}
      style={{ background:ok?"rgba(80,160,100,.1)":"rgba(200,160,80,.06)", border:`1px solid ${ok?"rgba(80,160,100,.25)":"rgba(200,160,80,.15)"}`, borderRadius:4, padding:"3px 9px", cursor:"pointer", fontSize:10, letterSpacing:".1em", textTransform:"uppercase", color:ok?"#7dc493":"rgba(200,160,80,.55)", fontFamily:"sans-serif", transition:"all .2s" }}>
      {ok ? "✓" : "COPY"}
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

function Divider() {
  return <div style={{ height:1, background:"rgba(255,255,255,.05)", margin:"2px 0" }} />;
}

// ─────────────────────────────────────────────────────────────────────────────
// AUTO-RESIZE TEXTAREA — bible
// ─────────────────────────────────────────────────────────────────────────────
function AutoTA({ value, onChange, placeholder, minHeight = 80, fontSize = 12 }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) { ref.current.style.height = "auto"; ref.current.style.height = ref.current.scrollHeight + "px"; }
  }, [value]);
  return (
    <textarea ref={ref} className="ta" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
      style={{ width:"100%", minHeight, background:"rgba(255,255,255,.02)", border:"1px solid rgba(255,255,255,.07)", borderRadius:9, color:"rgba(232,224,212,.85)", fontSize, padding:"12px 13px", fontFamily:"Georgia,serif", resize:"none", lineHeight:1.72, transition:"border-color .2s", overflow:"hidden", display:"block" }} />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FRAME INPUT — numbered, colour-coded, context baked into placeholder
// ─────────────────────────────────────────────────────────────────────────────
function FrameInput({ number, label, accent, value, onChange, placeholder }) {
  const ref = useRef(null);
  const rgb = number === 1 ? "80,160,220" : "210,120,55";

  useEffect(() => {
    if (ref.current) { ref.current.style.height = "auto"; ref.current.style.height = ref.current.scrollHeight + "px"; }
  }, [value]);

  return (
    <div style={{ borderRadius:9, border:`1px solid ${value ? `rgba(${rgb},.28)` : "rgba(255,255,255,.07)"}`, overflow:"hidden", transition:"border-color .2s", background:"rgba(255,255,255,.015)" }}>
      {/* Coloured label bar */}
      <div style={{ padding:"9px 13px", background:`rgba(${rgb},.07)`, borderBottom:`1px solid rgba(${rgb},.14)`, display:"flex", alignItems:"center", gap:9 }}>
        <div style={{ width:22, height:22, borderRadius:5, background:`rgba(${rgb},.15)`, border:`1px solid rgba(${rgb},.32)`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <span style={{ fontSize:11, color:`rgba(${rgb},1)`, fontFamily:"sans-serif", fontWeight:700 }}>{number}</span>
        </div>
        <span style={{ fontSize:10, letterSpacing:".12em", textTransform:"uppercase", fontFamily:"sans-serif", color:`rgba(${rgb},1)`, fontWeight:700 }}>{label}</span>
      </div>
      {/* Input area */}
      <textarea
        ref={ref}
        className="ta"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        style={{ width:"100%", minHeight:100, background:"transparent", border:"none", borderRadius:0, color:"#e8e0d4", fontSize:12, padding:"12px 13px", fontFamily:"Georgia,serif", resize:"none", lineHeight:1.72, overflow:"hidden", display:"block", outline:"none" }}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// REFERENCE ZONE
// ─────────────────────────────────────────────────────────────────────────────
function RefZone({ refs, onAdd, onRemove }) {
  const inputRef = useRef();
  const [dragOver, setDragOver] = useState(false);

  function handleFiles(files) {
    Array.from(files).forEach(file => { if (file.type.startsWith("image/")) onAdd(file); });
  }

  return (
    <div
      className={dragOver ? "drop-active" : ""}
      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
      style={{ borderRadius:9, border:`1px dashed ${refs.length ? "rgba(200,160,80,.28)" : "rgba(255,255,255,.08)"}`, background:refs.length ? "rgba(200,160,80,.04)" : "rgba(255,255,255,.01)", padding:"12px", transition:"all .2s", minHeight:68 }}>
      {refs.length === 0 ? (
        <div onClick={() => inputRef.current?.click()} style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:5, textAlign:"center", cursor:"pointer" }}>
          <div style={{ fontSize:20, opacity:.35 }}>📎</div>
          <div style={{ fontSize:10, fontFamily:"sans-serif", color:"rgba(232,224,212,.35)", fontWeight:700, textTransform:"uppercase", letterSpacing:".08em" }}>Drop references here</div>
          <div style={{ fontSize:9, color:"rgba(232,224,212,.18)", fontFamily:"sans-serif", fontStyle:"italic" }}>Character, style, mood, environment — anything relevant. Multiple accepted.</div>
          <div style={{ fontSize:9, color:"rgba(200,160,80,.28)", fontFamily:"sans-serif", marginTop:2 }}>drop or click to browse</div>
        </div>
      ) : (
        <div>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:8 }}>
            {refs.map((ref, i) => (
              <div key={i} style={{ position:"relative", flexShrink:0 }}>
                <img src={ref.preview} alt="" style={{ width:50, height:50, objectFit:"cover", borderRadius:6, border:`1px solid ${ref.falUrl ? "rgba(80,160,100,.35)" : "rgba(200,160,80,.2)"}`, display:"block" }} />
                {ref.uploading && (
                  <div style={{ position:"absolute", inset:0, background:"rgba(9,9,10,.6)", borderRadius:6, display:"flex", alignItems:"center", justifyContent:"center" }}><Spin /></div>
                )}
                {ref.falUrl && (
                  <div style={{ position:"absolute", bottom:2, right:2, width:11, height:11, borderRadius:"50%", background:"#5cb87a", border:"1.5px solid #09090a", display:"flex", alignItems:"center", justifyContent:"center" }}>
                    <span style={{ fontSize:6, color:"#fff" }}>✓</span>
                  </div>
                )}
                <button onClick={() => onRemove(i)} style={{ position:"absolute", top:-4, right:-4, width:15, height:15, borderRadius:"50%", background:"rgba(0,0,0,.8)", border:"1px solid rgba(255,255,255,.15)", color:"rgba(255,255,255,.7)", cursor:"pointer", fontSize:8, display:"flex", alignItems:"center", justifyContent:"center" }}>×</button>
              </div>
            ))}
            <button onClick={() => inputRef.current?.click()} style={{ width:50, height:50, borderRadius:6, border:"1px dashed rgba(200,160,80,.22)", background:"rgba(200,160,80,.05)", cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", color:"rgba(200,160,80,.38)", fontSize:18, flexShrink:0 }}>+</button>
          </div>
          <div style={{ fontSize:9, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic" }}>
            {refs.length} reference{refs.length > 1 ? "s" : ""} — used for consistency across both frames
          </div>
        </div>
      )}
      <input ref={inputRef} type="file" accept="image/*" multiple style={{ display:"none" }}
        onChange={e => { handleFiles(e.target.files); e.target.value = ""; }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FRAME OUTPUT PANEL
// ─────────────────────────────────────────────────────────────────────────────
function FramePanel({ number, label, frame, imageUrl, loading, error }) {
  const [promptOpen, setPromptOpen] = useState(false);
  const rgb = number === 1 ? "80,160,220" : "210,120,55";
  const accent      = `rgba(${rgb},.85)`;
  const accentFaint = `rgba(${rgb},.09)`;
  const accentLine  = `rgba(${rgb},.18)`;

  return (
    <div style={{ flex:1, minWidth:0, animation:"fadeIn .4s ease both", animationDelay: number === 1 ? "0s" : ".1s" }}>
      {/* Label row */}
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:9 }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={{ width:22, height:22, borderRadius:5, background:accentFaint, border:`1px solid ${accentLine}`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
            <span style={{ fontSize:10, color:accent, fontFamily:"sans-serif", fontWeight:700 }}>{number}</span>
          </div>
          <span style={{ fontSize:11, color:accent, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".12em", textTransform:"uppercase" }}>{label}</span>
        </div>
        <div style={{ display:"flex", gap:6, alignItems:"center" }}>
          {frame?.prompt && <CopyBtn text={frame.prompt} />}
          {imageUrl && (
            <a href={imageUrl} download={`frame-${number}-${label.toLowerCase().replace(" ","-")}.jpg`} target="_blank" rel="noreferrer"
              style={{ fontSize:10, padding:"3px 9px", borderRadius:4, background:"rgba(80,130,200,.08)", border:"1px solid rgba(80,130,200,.18)", color:"rgba(130,180,240,.6)", fontFamily:"sans-serif", textDecoration:"none", letterSpacing:".08em", textTransform:"uppercase" }}>
              ↓ Save
            </a>
          )}
        </div>
      </div>

      {/* Composition note */}
      {frame?.compositionNote && (
        <div style={{ fontSize:10, color:"rgba(232,224,212,.3)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:8, lineHeight:1.45, paddingLeft:2 }}>
          {frame.compositionNote}
        </div>
      )}

      {/* Image */}
      <div style={{ background:"#0d0d0e", borderRadius:9, border:"1px solid rgba(255,255,255,.07)", overflow:"hidden", marginBottom:9, aspectRatio:"16/9", display:"flex", alignItems:"center", justifyContent:"center" }}>
        {loading && !imageUrl && (
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:9 }}>
            <Spin />
            <span style={{ fontSize:9, color:accent, letterSpacing:".14em", textTransform:"uppercase", fontFamily:"sans-serif", animation:"pulse 1.6s ease infinite" }}>Rendering…</span>
          </div>
        )}
        {error && !imageUrl && (
          <div style={{ padding:20, textAlign:"center" }}>
            <div style={{ fontSize:18, marginBottom:6 }}>⚠</div>
            <div style={{ fontSize:11, color:"rgba(220,100,100,.6)", fontFamily:"sans-serif" }}>{error}</div>
          </div>
        )}
        {!loading && !error && !imageUrl && (
          <div style={{ fontSize:11, color:"rgba(255,255,255,.07)", fontFamily:"sans-serif", fontStyle:"italic" }}>Frame {number}</div>
        )}
        {imageUrl && <img src={imageUrl} alt={label} style={{ width:"100%", height:"100%", objectFit:"cover", display:"block" }} />}
      </div>

      {/* Prompt accordion */}
      {frame?.prompt && (
        <>
          <button onClick={() => setPromptOpen(!promptOpen)}
            style={{ width:"100%", background:accentFaint, border:`1px solid ${accentLine}`, borderRadius:promptOpen ? "7px 7px 0 0" : "7px", padding:"7px 11px", cursor:"pointer", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <span style={{ fontSize:10, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:accent }}>Generated Prompt</span>
            <span style={{ fontSize:10, color:accent, opacity:.5 }}>{promptOpen ? "−" : "+"}</span>
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
// MOTION ARC
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

// ─────────────────────────────────────────────────────────────────────────────
// AUDIT PANEL
// ─────────────────────────────────────────────────────────────────────────────
function AuditPanel({ audit }) {
  const [open, setOpen] = useState(false);
  if (!audit) return null;
  const entries = [
    ["cameraLock",             "Camera Lock"],
    ["lightingFreeze",         "Lighting Freeze"],
    ["spatialPlausibility",    "Spatial Plausibility"],
    ["impliedMotion",          "Implied Motion"],
    ["materialCompliance",     "Material Compliance"],
    ["heroElementLock",        "Hero Element"],
    ["reinterpretationApplied","Reinterpretation"],
  ];
  const issues = entries.filter(([k]) => { const v = audit[k]||""; return v && v!=="confirmed" && v!=="not applicable" && !v.startsWith("confirmed"); });
  const allOk = issues.length === 0;

  return (
    <div>
      <button onClick={() => setOpen(!open)} style={{ display:"flex", alignItems:"center", gap:7, background:"none", border:"none", cursor:"pointer", padding:0 }}>
        <span style={{ width:7, height:7, borderRadius:"50%", background:allOk?"#5cb87a":"#c8a050", flexShrink:0 }} />
        <span style={{ fontSize:10, fontFamily:"sans-serif", letterSpacing:".1em", textTransform:"uppercase", color:allOk?"rgba(80,180,120,.7)":"rgba(200,160,80,.7)" }}>
          {allOk ? "All checks passed" : `${issues.length} audit note${issues.length>1?"s":""}`}
        </span>
        <span style={{ fontSize:9, color:"rgba(232,224,212,.2)", fontFamily:"sans-serif" }}>{open?"▲":"▼"}</span>
      </button>
      {open && (
        <div style={{ marginTop:9, padding:"11px 13px", background:"rgba(255,255,255,.02)", border:"1px solid rgba(255,255,255,.06)", borderRadius:8, display:"grid", gap:7 }}>
          {entries.map(([k,l]) => {
            const val = audit[k]||"—";
            const ok = val==="confirmed"||val==="not applicable"||val.startsWith("confirmed");
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
// LOG STRIP
// ─────────────────────────────────────────────────────────────────────────────
function LogStrip({ log, onSelect }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position:"fixed", bottom:0, left:0, right:0, zIndex:50 }}>
      <div style={{ display:"flex", justifyContent:"center" }}>
        <button onClick={() => setOpen(!open)} style={{ background:"#17171a", border:"1px solid rgba(255,255,255,.08)", borderBottom:"none", borderRadius:"8px 8px 0 0", padding:"5px 18px", cursor:"pointer", display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:9, letterSpacing:".15em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(232,224,212,.3)" }}>Shot Log</span>
          <span style={{ fontSize:10, padding:"1px 6px", background:"rgba(200,160,80,.1)", borderRadius:10, color:"rgba(200,160,80,.6)", fontFamily:"sans-serif" }}>{log.length}</span>
          <span style={{ fontSize:9, color:"rgba(232,224,212,.2)" }}>{open?"▼":"▲"}</span>
        </button>
      </div>
      {open && (
        <div style={{ background:"#131315", borderTop:"1px solid rgba(255,255,255,.07)", maxHeight:180, overflowX:"auto", overflowY:"hidden" }}>
          {log.length === 0
            ? <div style={{ padding:"18px", textAlign:"center", fontSize:11, color:"rgba(232,224,212,.18)", fontFamily:"sans-serif", fontStyle:"italic" }}>No shots yet</div>
            : (
              <div style={{ display:"flex", padding:"12px 16px", minWidth:"max-content", gap:10 }}>
                {log.map((s,i) => (
                  <button key={i} onClick={() => { setOpen(false); onSelect(s); }}
                    style={{ background:"rgba(255,255,255,.03)", border:"1px solid rgba(255,255,255,.07)", borderRadius:7, padding:"8px 12px", cursor:"pointer", textAlign:"left", minWidth:175 }}>
                    <div style={{ fontSize:9, color:"rgba(200,160,80,.45)", fontFamily:"sans-serif", marginBottom:4 }}>#{i+1} · {s.sceneSlug||"—"}</div>
                    <div style={{ fontSize:11, color:"rgba(232,224,212,.5)", fontFamily:"sans-serif", fontStyle:"italic", lineHeight:1.35 }}>
                      {(s.shotSummary||"").slice(0,65)}{(s.shotSummary||"").length>65?"…":""}
                    </div>
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
// SETTINGS
// ─────────────────────────────────────────────────────────────────────────────
function Settings({ s, set, onClose }) {
  const upd = (k,v) => set(p => ({...p,[k]:v}));
  return (
    <div style={{ position:"fixed", inset:0, zIndex:1000 }}>
      <div onClick={onClose} style={{ position:"absolute", inset:0, background:"rgba(0,0,0,.65)", backdropFilter:"blur(5px)" }} />
      <div style={{ position:"absolute", right:0, top:0, bottom:0, width:400, background:"#111113", borderLeft:"1px solid rgba(255,255,255,.08)", overflowY:"auto", display:"flex", flexDirection:"column" }}>
        <div style={{ padding:"18px 22px 14px", borderBottom:"1px solid rgba(255,255,255,.07)", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <span style={{ fontSize:12, fontFamily:"sans-serif", fontWeight:700, letterSpacing:".1em", textTransform:"uppercase", color:"rgba(232,224,212,.6)" }}>Settings</span>
          <button onClick={onClose} style={{ background:"none", border:"none", color:"rgba(232,224,212,.35)", cursor:"pointer", fontSize:18 }}>×</button>
        </div>
        <div style={{ padding:"18px 22px", display:"grid", gap:22, flex:1 }}>
          <div>
            <SLabel>Renderer</SLabel>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:7, marginTop:7 }}>
              {Object.entries(PROVIDERS).map(([k,p]) => (
                <button key={k} onClick={() => upd("provider",k)}
                  style={{ padding:"9px 11px", borderRadius:7, border:`1px solid ${s.provider===k?"rgba(200,160,80,.38)":"rgba(255,255,255,.07)"}`, background:s.provider===k?"rgba(200,160,80,.08)":"rgba(255,255,255,.02)", cursor:"pointer", textAlign:"left" }}>
                  <div style={{ fontSize:12, fontFamily:"sans-serif", color:s.provider===k?"#c8a050":"rgba(232,224,212,.55)", fontWeight:700 }}>{p.icon} {p.label}</div>
                </button>
              ))}
            </div>
          </div>
          <div>
            <SLabel>{PROVIDERS[s.provider].label} Model</SLabel>
            <div style={{ display:"grid", gap:6, marginTop:7 }}>
              {PROVIDERS[s.provider].models.map(m => {
                const active = s.provider==="nanobanana" ? s.nbModel===m.id : s.falModel===m.id;
                return (
                  <button key={m.id} onClick={() => upd(s.provider==="nanobanana"?"nbModel":"falModel",m.id)}
                    style={{ padding:"9px 11px", borderRadius:7, border:`1px solid ${active?"rgba(200,160,80,.35)":"rgba(255,255,255,.06)"}`, background:active?"rgba(200,160,80,.08)":"rgba(255,255,255,.02)", cursor:"pointer", textAlign:"left", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                    <div>
                      <div style={{ fontSize:12, fontFamily:"sans-serif", color:active?"#c8a050":"rgba(232,224,212,.58)", fontWeight:700 }}>{m.label}</div>
                      <div style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif", marginTop:2 }}>{m.desc}</div>
                    </div>
                    {active && <span style={{ color:"#c8a050", fontSize:12 }}>✓</span>}
                  </button>
                );
              })}
            </div>
          </div>
          <div>
            <SLabel>Aspect Ratio</SLabel>
            <div style={{ display:"flex", gap:7, marginTop:7 }}>
              {ASPECT_RATIOS.map(r => (
                <button key={r.id} onClick={() => upd("ratio",r.id)}
                  style={{ flex:1, padding:"8px", borderRadius:6, border:`1px solid ${s.ratio===r.id?"rgba(200,160,80,.36)":"rgba(255,255,255,.06)"}`, background:s.ratio===r.id?"rgba(200,160,80,.08)":"rgba(255,255,255,.02)", cursor:"pointer", fontSize:12, fontFamily:"sans-serif", color:s.ratio===r.id?"#c8a050":"rgba(232,224,212,.42)", fontWeight:s.ratio===r.id?700:400 }}>
                  {r.label}
                </button>
              ))}
            </div>
          </div>
          {[
            ["NanoBanana API Key","nananobanana.com → Settings → API Keys","nbKey","nb_…"],
            ["fal.ai API Key","fal.ai/dashboard/keys — required for reference uploads","falKey","fal_…"],
          ].map(([label,hint,key,ph]) => (
            <div key={key}>
              <SLabel>{label}</SLabel>
              <div style={{ fontSize:10, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:7 }}>{hint}</div>
              <SecretInput value={s[key]} onChange={v => upd(key,v)} placeholder={ph} />
            </div>
          ))}
          <div>
            <SLabel>Weavy</SLabel>
            <div style={{ fontSize:10, color:"rgba(232,224,212,.22)", fontFamily:"sans-serif", fontStyle:"italic", marginBottom:8 }}>Creates a review thread per shot for team feedback</div>
            <div style={{ display:"grid", gap:7 }}>
              {[["weavyUrl","https://your-env.weavy.io",false],["weavyKey","wys_…",true]].map(([k,ph,mono]) => (
                <input key={k} value={s[k]} onChange={e=>upd(k,e.target.value)} placeholder={ph}
                  style={{ width:"100%", background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.07)", borderRadius:6, color:"#e8e0d4", fontSize:12, padding:"9px 11px", fontFamily:mono?"monospace":"sans-serif", outline:"none" }}
                  onFocus={e=>e.target.style.borderColor="rgba(200,160,80,.28)"}
                  onBlur={e=>e.target.style.borderColor="rgba(255,255,255,.07)"} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

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
      <button onClick={() => setShow(!show)} style={{ position:"absolute", right:9, top:"50%", transform:"translateY(-50%)", background:"none", border:"none", color:"rgba(232,224,212,.28)", cursor:"pointer", fontSize:11 }}>{show?"🙈":"👁"}</button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    provider:"nanobanana", nbModel:"nano-banana-2", falModel:"fal-ai/flux/dev",
    nbKey:"", falKey:"", weavyUrl:"", weavyKey:"", ratio:"landscape_16_9",
  });

  const [bible,  setBible]  = useState("");
  const [frame1, setFrame1] = useState("");
  const [frame2, setFrame2] = useState("");
  const [refs,   setRefs]   = useState([]);

  const [shot,         setShot]         = useState(null);
  const [startImg,     setStartImg]     = useState(null);
  const [endImg,       setEndImg]       = useState(null);
  const [startLoading, setStartLoading] = useState(false);
  const [endLoading,   setEndLoading]   = useState(false);
  const [startErr,     setStartErr]     = useState(null);
  const [endErr,       setEndErr]       = useState(null);
  const [genBusy,      setGenBusy]      = useState(false);
  const [renderBusy,   setRenderBusy]   = useState(false);
  const [genError,     setGenError]     = useState(null);
  const [weavyStatus,  setWeavyStatus]  = useState("idle");
  const [log,          setLog]          = useState([]);

  const { provider, nbModel, falModel, nbKey, falKey, weavyUrl, weavyKey, ratio } = settings;
  const activeKey = provider === "nanobanana" ? nbKey : falKey;

  function handleAddRef(file) {
    const idx = refs.length;
    setRefs(p => [...p, { file, preview:URL.createObjectURL(file), falUrl:null, uploading:false }]);
    if (falKey.trim()) uploadRef(file, idx);
  }

  async function uploadRef(file, index) {
    setRefs(p => p.map((r,i) => i===index ? {...r,uploading:true} : r));
    try {
      const url = await falUpload(falKey, file);
      setRefs(p => p.map((r,i) => i===index ? {...r,falUrl:url,uploading:false} : r));
    } catch(e) {
      setRefs(p => p.map((r,i) => i===index ? {...r,uploading:false} : r));
    }
  }

  function handleRemoveRef(index) {
    setRefs(p => p.filter((_,i) => i !== index));
  }

  const canGenerate = !genBusy && !renderBusy
    && bible.trim().length > 40
    && frame1.trim().length > 20
    && frame2.trim().length > 20;

  async function handleGenerate() {
    if (!canGenerate) return;
    setGenBusy(true); setGenError(null); setShot(null);
    setStartImg(null); setEndImg(null); setStartErr(null); setEndErr(null); setWeavyStatus("idle");

    const refContext = refs.length
      ? `${refs.length} reference image${refs.length>1?"s":""} provided — use for identity, style, and mood consistency across both frames.`
      : "No reference images.";

    const userMsg = `VISUAL BIBLE:
${bible}

---

FRAME 1 — START FRAME:
${frame1}

---

FRAME 2 — END FRAME:
${frame2}

---

REFERENCES: ${refContext}

SHOT LOG (continuity):
${log.length ? log.map((s,i)=>`#${i+1}: ${s.shotSummary}`).join("\n") : "No previous shots."}`;

    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          model:"claude-sonnet-4-20250514",
          max_tokens:1400,
          system:SYSTEM_PROMPT,
          messages:[{role:"user",content:userMsg}],
        }),
      });
      const data = await res.json();
      const txt = data.content?.map(b=>b.text||"").join("")||"";
      const parsed = JSON.parse(txt.replace(/```json|```/g,"").trim());
      setShot(parsed);
      setLog(p => [...p,parsed].slice(-20));
      if (activeKey.trim()) await doRender(parsed);
    } catch(e) {
      setGenError("Generation failed — check your API key and try again."); console.error(e);
    }
    setGenBusy(false);
  }

  async function doRender(parsedShot) {
    if (!activeKey.trim()) return;
    const s = parsedShot || shot;
    if (!s) return;
    setRenderBusy(true);
    setStartLoading(true); setEndLoading(true);
    setStartErr(null); setEndErr(null); setStartImg(null); setEndImg(null);

    let uploadedRefs = [...refs];
    if (falKey.trim()) {
      uploadedRefs = await Promise.all(refs.map(async (r,i) => {
        if (r.file && !r.falUrl) {
          try {
            const url = await falUpload(falKey, r.file);
            setRefs(p => p.map((x,idx) => idx===i?{...x,falUrl:url}:x));
            return {...r,falUrl:url};
          } catch(e) { return r; }
        }
        return r;
      }));
    }

    const refUrls = uploadedRefs.map(r=>r.falUrl).filter(Boolean);
    const faceUrl = refUrls[0]||null;

    const renderFrame = async (frame, setImg, setLoad, setErr) => {
      try {
        let url;
        if (provider==="nanobanana") {
          url = await nbGenerate(nbKey, nbModel, frame.prompt, frame.negativePrompt, ratio, refUrls.length?refUrls:undefined);
        } else {
          url = faceUrl
            ? await falPuLID(falKey, faceUrl, frame.prompt, frame.negativePrompt, ratio)
            : await falText2Img(falKey, falModel, frame.prompt, frame.negativePrompt, ratio);
        }
        setImg(url); return url;
      } catch(e) {
        const msg = e.message.includes("401")?"Invalid API key":e.message.includes("402")?"Insufficient credits":e.message.includes("timeout")?"Render timed out":"Render failed";
        setErr(msg); return null;
      } finally { setLoad(false); }
    };

    const [s1,s2] = await Promise.all([
      renderFrame(s.startFrame, setStartImg, setStartLoading, setStartErr),
      renderFrame(s.endFrame,   setEndImg,   setEndLoading,   setEndErr),
    ]);

    setRenderBusy(false);

    if (weavyUrl.trim() && weavyKey.trim()) {
      setWeavyStatus("posting");
      try {
        const uid = `da-${s.sceneSlug||Date.now()}`;
        await weavyUpsertApp(weavyUrl, weavyKey, uid, `Shot: ${(s.shotSummary||"").slice(0,60)}`);
        await weavyPost(weavyUrl, weavyKey, uid,
          `🎬 **${s.shotSummary}**\n\n${s.sharedContext}\n\nMotion arc: ${s.motionArc||"—"}\n\n▶ Frame 1 (Start): ${s1||"(render failed)"}\n⏹ Frame 2 (End): ${s2||"(render failed)"}\n\nReview and reply with feedback or approve for video generation handoff.`
        );
        setWeavyStatus("ok");
      } catch(e) { setWeavyStatus("error"); }
    }
  }

  const busy = genBusy || renderBusy;

  return (
    <div style={{ minHeight:"100vh", background:"#09090a", color:"#e8e0d4", fontFamily:"Georgia,'Times New Roman',serif" }}>
      <style>{css}</style>
      <div style={{ position:"fixed", inset:0, opacity:.03, pointerEvents:"none", zIndex:200, backgroundImage:`url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`, backgroundSize:"120px" }} />
      <div style={{ position:"fixed", inset:0, pointerEvents:"none", zIndex:1, background:"radial-gradient(ellipse 80% 35% at 50% 0%, rgba(180,120,40,.05) 0%, transparent 55%)" }} />

      {/* Nav */}
      <nav style={{ height:50, borderBottom:"1px solid rgba(255,255,255,.065)", display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 22px", position:"sticky", top:0, background:"rgba(9,9,10,.96)", backdropFilter:"blur(8px)", zIndex:100 }}>
        <div style={{ display:"flex", alignItems:"baseline", gap:10 }}>
          <span style={{ fontSize:14, fontWeight:400, letterSpacing:".08em" }}>DIRECTOR AGENT</span>
          <span style={{ fontSize:9, letterSpacing:".16em", color:"rgba(200,160,80,.38)", textTransform:"uppercase", fontFamily:"sans-serif" }}>Start · End Frame</span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:14 }}>
          <div style={{ display:"flex", gap:12, alignItems:"center" }}>
            <div style={{ display:"flex", alignItems:"center", gap:5 }}>
              <span style={{ width:6, height:6, borderRadius:"50%", background:activeKey.trim()?"#5cb87a":"rgba(255,255,255,.12)" }} />
              <span style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif" }}>{activeKey.trim()?PROVIDERS[provider].label:"No key"}</span>
            </div>
            {weavyUrl.trim() && (
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <span style={{ width:6, height:6, borderRadius:"50%", background:weavyStatus==="ok"?"#5cb87a":weavyStatus==="error"?"rgba(200,80,80,.55)":weavyStatus==="posting"?"#c8a050":"rgba(255,255,255,.12)" }} />
                <span style={{ fontSize:10, color:"rgba(232,224,212,.28)", fontFamily:"sans-serif" }}>Weavy</span>
              </div>
            )}
          </div>
          <button onClick={() => setShowSettings(true)}
            style={{ background:"rgba(255,255,255,.04)", border:"1px solid rgba(255,255,255,.08)", borderRadius:6, padding:"5px 13px", cursor:"pointer", fontSize:10, letterSpacing:".1em", textTransform:"uppercase", fontFamily:"sans-serif", color:"rgba(232,224,212,.45)" }}>
            ⚙ Settings
          </button>
        </div>
      </nav>

      {/* Two-column */}
      <div style={{ display:"grid", gridTemplateColumns:"420px 1fr", minHeight:"calc(100vh - 50px)", position:"relative", zIndex:2 }}>

        {/* LEFT PANEL */}
        <div style={{ borderRight:"1px solid rgba(255,255,255,.055)", padding:"22px 20px 120px", overflowY:"auto", maxHeight:"calc(100vh - 50px)", position:"sticky", top:50, display:"flex", flexDirection:"column", gap:18 }}>

          {/* Visual Bible */}
          <div>
            <FieldLabel
              main="Visual Bible"
              sub="Permanent universe rulebook — materials, constraints, reinterpretation rules, characters, lighting, colour, style keywords" />
            <AutoTA value={bible} onChange={setBible} placeholder={BIBLE_PLACEHOLDER} minHeight={200} fontSize={12} />
          </div>

          <Divider />

          {/* Frame inputs */}
          <div style={{ display:"grid", gap:12 }}>
            <FrameInput
              number={1}
              label="Frame 1 — Start"
              value={frame1}
              onChange={setFrame1}
              placeholder={FRAME1_PLACEHOLDER} />
            <FrameInput
              number={2}
              label="Frame 2 — End"
              value={frame2}
              onChange={setFrame2}
              placeholder={FRAME2_PLACEHOLDER} />
          </div>

          <Divider />

          {/* References */}
          <div>
            <FieldLabel main="References" sub="Optional — character, style, mood, environment. Used for consistency across both frames." />
            <RefZone refs={refs} onAdd={handleAddRef} onRemove={handleRemoveRef} />
          </div>

          {/* Generate */}
          <div>
            <button onClick={handleGenerate} disabled={!canGenerate}
              style={{ width:"100%", padding:"14px", borderRadius:8, border:`1px solid ${canGenerate?"rgba(200,160,80,.42)":"rgba(255,255,255,.06)"}`, background:canGenerate?"rgba(200,160,80,.13)":"rgba(255,255,255,.018)", color:canGenerate?"#c8a050":"rgba(232,224,212,.18)", fontSize:11, letterSpacing:".18em", textTransform:"uppercase", fontFamily:"sans-serif", fontWeight:700, cursor:canGenerate?"pointer":"not-allowed", transition:"all .25s", display:"flex", alignItems:"center", justifyContent:"center", gap:10 }}>
              {genBusy?<><Spin/>Writing prompts…</>:renderBusy?<><Spin/>Rendering frames…</>:"Generate Start and End"}
            </button>
            {!canGenerate && !busy && (
              <p style={{ fontSize:9, color:"rgba(232,224,212,.18)", fontFamily:"sans-serif", textAlign:"center", marginTop:7, fontStyle:"italic" }}>
                {!bible.trim()?"Add your Visual Bible to continue"
                  :!frame1.trim()?"Describe Frame 1 to continue"
                  :!frame2.trim()?"Describe Frame 2 to continue"
                  :!activeKey.trim()?"Add API key in Settings to render images":""}
              </p>
            )}
            {genError && (
              <div style={{ marginTop:9, padding:"9px 12px", background:"rgba(180,60,60,.08)", border:"1px solid rgba(180,60,60,.16)", borderRadius:6, fontSize:11, color:"#e08080", fontFamily:"sans-serif" }}>{genError}</div>
            )}
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div style={{ padding:"22px 24px 120px", overflowY:"auto", maxHeight:"calc(100vh - 50px)" }}>
          {!shot && !busy && (
            <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", height:"80%", gap:12, opacity:.2 }}>
              <div style={{ fontSize:44 }}>🎬</div>
              <div style={{ fontSize:13, color:"rgba(232,224,212,.6)", fontFamily:"sans-serif", fontStyle:"italic" }}>Frame 1 and Frame 2 will appear here</div>
            </div>
          )}

          {shot && (
            <div style={{ animation:"fadeIn .5s ease both" }}>
              <div style={{ marginBottom:16, paddingBottom:14, borderBottom:"1px solid rgba(255,255,255,.055)" }}>
                <div style={{ fontSize:14, color:"#e8e0d4", lineHeight:1.45, marginBottom:5 }}>{shot.shotSummary}</div>
                <div style={{ fontSize:11, color:"rgba(232,224,212,.32)", fontFamily:"sans-serif", fontStyle:"italic", lineHeight:1.5 }}>{shot.sharedContext}</div>
              </div>

              <MotionArc text={shot.motionArc} />

              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:16 }}>
                <FramePanel number={1} label="Start Frame" frame={shot.startFrame} imageUrl={startImg} loading={startLoading} error={startErr} />
                <FramePanel number={2} label="End Frame"   frame={shot.endFrame}   imageUrl={endImg}   loading={endLoading}   error={endErr} />
              </div>

              <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:10 }}>
                <AuditPanel audit={shot.auditResult} />
                <div style={{ display:"flex", gap:8, alignItems:"center" }}>
                  {activeKey.trim() && !busy && (
                    <button onClick={() => doRender()} style={{ background:"transparent", border:"1px solid rgba(80,130,200,.2)", borderRadius:6, color:"rgba(130,180,240,.5)", padding:"6px 13px", fontSize:10, letterSpacing:".12em", textTransform:"uppercase", cursor:"pointer", fontFamily:"sans-serif" }}>↺ Re-render</button>
                  )}
                  {weavyStatus==="ok"    && <span style={{ fontSize:10, color:"rgba(80,180,120,.65)", fontFamily:"sans-serif" }}>✓ Posted to Weavy</span>}
                  {weavyStatus==="error" && <span style={{ fontSize:10, color:"rgba(220,100,100,.55)", fontFamily:"sans-serif" }}>Weavy post failed</span>}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <LogStrip log={log} onSelect={s => setShot(s)} />
      {showSettings && <Settings s={settings} set={setSettings} onClose={() => setShowSettings(false)} />}
    </div>
  );
}

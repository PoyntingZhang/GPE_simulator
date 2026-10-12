"use strict";

let workflowState = null, monitorData = null, selectedView = "experiment", activePrompt = null;
let statePending = false, monitorPending = false, lastRefresh = 0, wasBusy = false, lastFolder = null;
let lastOperationError = "";
const profileFields = ["run-target", "run-output", "run-cupy", "run-python", "run-modules", "run-account", "run-partition", "run-walltime", "ssh-host", "ssh-user", "ssh-port", "remote-parent"];
const profileKey = "gpe-workflow-profile-v1";

function workflowNotice(message, error=false) {
  $("#workflow-notice").textContent = message;
  $("#workflow-notice").className = "notice" + (error ? " error" : "");
  $("#workflow-notice").hidden = false;
}

function showView(name) {
  selectedView = name;
  document.querySelectorAll(".workflow-view").forEach(view => view.hidden = view.id !== "view-" + name);
  document.querySelectorAll("[data-view]").forEach(button => {
    button.classList.toggle("active", button.dataset.view === name);
    if (button.dataset.view === name) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (name === "launch" && config) $("#prepare-summary").textContent = `${config.stages.length} stages · ${config.simulation.N_grids}³ grid · ${config.stages.reduce((n,s)=>n+s.evolution.steps,0).toLocaleString()} steps. Prepare copies this draft; later edits need a new run.`;
  if (name === "monitor") readMonitor();
}
document.querySelectorAll("[data-view]").forEach(button => button.onclick = () => showView(button.dataset.view));
$("#go-monitor").onclick = () => showView("monitor");

function targetChanged() {
  const local = $("#run-target").value === "local";
  $("#connection-card").hidden = local;
  document.querySelectorAll(".hpc-setting").forEach(node => node.hidden = local);
  $("#environment-hint").textContent = local ? "On Colab, use the existing environment if dependencies are installed. Set up environment creates a separate environment for this run when needed." : "Set up environment installs this run's dependencies on the HPC. Skip it when an appropriate environment already exists.";
}
try {
  const profile = JSON.parse(localStorage.getItem(profileKey) || "{}");
  profileFields.forEach(id => { if (profile[id] !== undefined) $("#"+id).value = profile[id]; });
} catch (_) { /* Profiles are optional; credentials never go here. */ }
profileFields.forEach(id => $("#"+id).addEventListener("change", () => {
  try { localStorage.setItem(profileKey, JSON.stringify(Object.fromEntries(profileFields.map(key => [key,$("#"+key).value])))); } catch (_) {}
  targetChanged();
}));
targetChanged();

async function action(path, body={}) {
  try {
    $("#workflow-notice").hidden = true;
    const result = await api(path, body);
    if (result.errors) throw new Error(result.errors.map(item=>item.message).join(" "));
    await pollState();
  } catch (error) { workflowNotice(error.message, true); }
}
$("#prepare-run").onclick = async () => {
  if (!config) return;
  const checked = await api("/api/render", {config});
  if (checked.errors?.length) { workflowNotice("Resolve the Experiment validation issues before preparing a run.", true); showView("experiment"); return; }
  await action("/api/run/prepare", {config:clone(config), name:$("#run-name").value.trim(), notes:$("#run-notes").value,
    output:$("#run-output").value.trim(), cupy_package:$("#run-cupy").value,
    launch:{target:$("#run-target").value, account:$("#run-account").value, partition:$("#run-partition").value,
      walltime:$("#run-walltime").value, modules:$("#run-modules").value, python:$("#run-python").value}});
};
$("#open-run").onclick = () => action("/api/run/open", {folder:$("#open-run-path").value.trim()});
$("#ssh-connect").onclick = () => action("/api/ssh/connect", {host:$("#ssh-host").value.trim(), username:$("#ssh-user").value.trim(), port:Number($("#ssh-port").value)});
$("#ssh-disconnect").onclick = () => action("/api/ssh/disconnect");
$("#ssh-prompt").onsubmit = async event => {
  event.preventDefault();
  if (!activePrompt) return;
  const answer = $("#ssh-answer").value, id = activePrompt;
  $("#ssh-answer").value = "";
  $("#ssh-prompt").hidden = true;
  await action("/api/ssh/answer", {id, answer});
};
$("#upload-run").onclick = () => action("/api/run/upload", {parent:$("#remote-parent").value.trim()});
$("#setup-run").onclick = () => action("/api/run/setup");
$("#submit-run").onclick = () => action("/api/run/submit");
$("#cancel-run").onclick = () => {
  if (confirm(`Cancel ${workflowState?.run?.name || "this run"}? This stops its active job.`)) action("/api/run/cancel");
};
$("#download-run").onclick = async () => {
  try {
    const response = await fetch("api/run/download", {headers:{"X-GPE-Session":window.GPE_SESSION}});
    if (!response.ok) throw new Error((await response.json()).error);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a"); link.href=url; link.download=workflowState.run.name+".tar.gz"; link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  } catch (error) { workflowNotice(error.message,true); }
};

async function pollState() {
  if (statePending) return;
  statePending = true;
  try {
    workflowState = await api("/api/workflow");
    const {operation:op, connection:connection, run, folder} = workflowState;
    $("#job-badge").textContent = run.name ? `${run.name} · ${run.status}` : "No run prepared";
    $("#launch-status").textContent = run.name ? `${run.status}${run.job_id ? " · Job " + run.job_id : ""}` : "Prepare a run to begin.";
    $("#prepared-details").textContent = folder ? `Prepared folder: ${folder}\n${run.remote_path ? "Remote folder: " + run.remote_path : ""}` : "";
    $("#operation-name").textContent = (op.busy ? "Working: " : "") + (op.name || "Activity");
    $("#operation-output").textContent = op.output || (op.busy ? "Working…" : "No operation yet.");
    $("#operation-error").textContent = op.error || "";
    if (op.error && op.error !== lastOperationError) workflowNotice(op.error, true);
    lastOperationError = op.error || "";
    $("#connection-status").textContent = connection.connected ? `Connected · ${connection.target}` : connection.connecting ? "Waiting for SSH…" : "Disconnected";
    $("#ssh-output").textContent = connection.output;
    $("#ssh-output").hidden = !connection.output;
    if (connection.prompt && connection.prompt.id !== activePrompt) {
      activePrompt = connection.prompt.id;
      $("#ssh-prompt-text").textContent = connection.prompt.text;
      $("#ssh-prompt").hidden = false;
      $("#ssh-answer").value = "";
      $("#ssh-answer").focus();
    } else if (!connection.prompt) { activePrompt = null; $("#ssh-prompt").hidden = true; }
    ["prepare-run", "open-run", "ssh-connect", "upload-run", "setup-run", "submit-run", "refresh-monitor", "cancel-run"].forEach(id => $("#"+id).disabled = op.busy);
    $("#ssh-connect").disabled ||= connection.connecting || connection.connected;
    $("#download-run").disabled = !folder || op.busy;
    $("#upload-run").disabled ||= !folder || !connection.connected || run.target !== "slurm" || !!run.remote_path;
    $("#setup-run").disabled ||= !folder || !!run.job_id || !!run.pid || (run.target === "slurm" && (!connection.connected || !run.remote_path));
    $("#submit-run").disabled ||= !folder || !!run.job_id || !!run.pid || (run.target === "slurm" && (!connection.connected || !run.remote_path));
    $("#submit-run").textContent = run.target === "slurm" ? "Submit GPU job" : "Launch simulation";
    $("#cancel-run").disabled ||= !(run.job_id || run.pid) || /completed|failed|cancelled|stopped/i.test(run.status);
    if (folder && folder !== lastFolder) {
      lastFolder = folder;
      $("#run-target").value = run.target;
      targetChanged();
      $("#monitor-folder").value = "";
      $("#monitor-stage").value = "";
      $("#monitor-spectrum").value = "";
      workflowNotice(`Opened ${run.name}. Launch uses this prepared configuration.`);
      if (selectedView === "monitor") readMonitor();
    }
    if (wasBusy && !op.busy && selectedView === "monitor") readMonitor();
    wasBusy = op.busy;
  } catch (error) { $("#job-badge").textContent = "Workspace connection unavailable"; }
  finally { statePending=false; }
}

function options(select, values, first, selectedValue) {
  select.replaceChildren();
  if (first !== null) select.add(new Option(first,""));
  values.forEach(value=>select.add(new Option(value,value)));
  select.value = values.includes(selectedValue) ? selectedValue : first !== null ? "" : values[0] || "";
}
async function readMonitor() {
  if (monitorPending) return;
  if (!workflowState?.folder && !$("#monitor-folder").value.trim()) return;
  monitorPending = true;
  try {
    const stageValue = $("#monitor-stage").value, spectrumValue = $("#monitor-spectrum").value;
    monitorData = await api("/api/monitor", {folder:$("#monitor-folder").value.trim(), stage:stageValue,
      spectrum:spectrumValue, image:$("#monitor-image").value, image_limit:Number($("#image-limit").value)});
    options($("#monitor-stage"),monitorData.stages,"Latest active stage",stageValue);
    options($("#monitor-spectrum"),monitorData.spectra,"Latest readable sample",spectrumValue);
    options($("#monitor-image"),monitorData.images,null,monitorData.image?.kind);
    $("#monitor-message").textContent = `${monitorData.stage ? "Showing " + monitorData.stage + ". " : ""}${monitorData.warnings.join(" ")}`;
    $("#simulation-log").textContent = monitorData.log || "No log available yet.";
    drawMonitor();
  } catch (error) { $("#monitor-message").textContent = error.message; }
  finally { monitorPending=false; }
}
async function refreshMonitor() {
  lastRefresh=Date.now();
  if (workflowState?.folder && !$("#monitor-folder").value.trim()) {
    if (!workflowState.operation.busy) await action("/api/run/refresh");
  }
  await readMonitor();
}
$("#refresh-monitor").onclick = refreshMonitor;
$("#open-monitor").onclick = () => { $("#monitor-stage").value=""; $("#monitor-spectrum").value=""; readMonitor(); };
["monitor-stage","monitor-spectrum","monitor-image","image-limit"].forEach(id => $("#"+id).onchange=readMonitor);
["spectrum-log","image-log","k-max","scalar-quantity"].forEach(id=>$("#"+id).onchange=drawMonitor);

function plot(canvas, points, {log=false, xlabel="", ylabel="", color="#2e6652"}={}) {
  const ctx=canvas.getContext("2d"), w=canvas.width, h=canvas.height, box={l:76,r:24,t:26,b:55};
  ctx.clearRect(0,0,w,h); ctx.fillStyle="#fffefa"; ctx.fillRect(0,0,w,h);
  const valid=points.filter(p=>p[0] !== null && p[1] !== null && Number.isFinite(p[0]) && Number.isFinite(p[1]) && (!log || (p[0]>0 && p[1]>0)));
  ctx.font="13px sans-serif"; ctx.fillStyle="#737b72";
  if (!valid.length) {ctx.fillText("No saved data for this view yet.",box.l,h/2);return;}
  const transformed=valid.map(([x,y,complete])=>[log?Math.log10(x):x,log?Math.log10(y):y,complete]);
  let xmin=Math.min(...transformed.map(p=>p[0])),xmax=Math.max(...transformed.map(p=>p[0]));
  let ymin=Math.min(...transformed.map(p=>p[1])),ymax=Math.max(...transformed.map(p=>p[1]));
  if (xmin===xmax) {xmin-=.5;xmax+=.5;} if(ymin===ymax){const d=Math.max(Math.abs(ymin)*.05,.01);ymin-=d;ymax+=d;}
  const x=x=>box.l+(x-xmin)/(xmax-xmin)*(w-box.l-box.r),y=y=>h-box.b-(y-ymin)/(ymax-ymin)*(h-box.t-box.b);
  for(let i=0;i<=5;i++) {
    const a=xmin+(xmax-xmin)*i/5,b=ymin+(ymax-ymin)*i/5;
    ctx.strokeStyle="#e5e6df";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(a),box.t);ctx.lineTo(x(a),h-box.b);ctx.moveTo(box.l,y(b));ctx.lineTo(w-box.r,y(b));ctx.stroke();
    ctx.fillStyle="#737b72";ctx.textAlign="center";ctx.fillText(log?Math.pow(10,a).toExponential(1):a.toPrecision(3),x(a),h-box.b+22);
    ctx.textAlign="right";ctx.fillText(log?Math.pow(10,b).toExponential(1):b.toPrecision(3),box.l-10,y(b)+4);
  }
  ctx.lineWidth=2;
  for(let i=1;i<transformed.length;i++) {const a=transformed[i-1],b=transformed[i];ctx.strokeStyle=b[2]===false?"#a3a3a3":color;ctx.beginPath();ctx.moveTo(x(a[0]),y(a[1]));ctx.lineTo(x(b[0]),y(b[1]));ctx.stroke();}
  if(transformed.length===1){ctx.fillStyle=color;ctx.beginPath();ctx.arc(x(transformed[0][0]),y(transformed[0][1]),3,0,Math.PI*2);ctx.fill();}
  ctx.fillStyle="#24372f";ctx.textAlign="center";ctx.fillText(xlabel,(box.l+w-box.r)/2,h-10);ctx.save();ctx.translate(16,h/2);ctx.rotate(-Math.PI/2);ctx.fillText(ylabel,0,0);ctx.restore();
}

function drawImage(data) {
  const canvas=$("#image-canvas"),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#737b72";ctx.font="13px sans-serif";
  if (!data) {ctx.fillText("No saved image for this view yet.",75,h/2);return;}
  const log=$("#image-log").checked && !data.kind.startsWith("phase");
  const values=data.values.flat().filter(v=>v!==null && Number.isFinite(v) && (!log || v>0)).map(v=>log?Math.log10(v):v);
  if(!values.length)return;
  const lo=Math.min(...values),hi=Math.max(...values),nx=data.x.length,ny=data.y.length;
  const off=document.createElement("canvas");off.width=nx;off.height=ny;const pixels=off.getContext("2d").createImageData(nx,ny);
  const stops=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  function rgb(t){t=Math.max(0,Math.min(1,t));const index=Math.min(3,Math.floor(t*4)),f=t*4-index;return stops[index].map((v,i)=>Math.round(v*(1-f)+stops[index+1][i]*f));}
  for(let ix=0;ix<nx;ix++)for(let iy=0;iy<ny;iy++){
    const v=data.values[ix][iy],index=4*((ny-1-iy)*nx+ix);
    if(v===null || (log && v<=0)){pixels.data.set([235,235,230,255],index);continue;}
    const t=hi===lo?.5:((log?Math.log10(v):v)-lo)/(hi-lo);pixels.data.set([...rgb(t),255],index);
  }
  off.getContext("2d").putImageData(pixels,0,0);ctx.imageSmoothingEnabled=false;
  const side=Math.min(w-150,h-85),left=(w-side)/2-12,top=20;ctx.drawImage(off,left,top,side,side);
  for(let i=0;i<side;i++){ctx.fillStyle=`rgb(${rgb(1-i/side).join(",")})`;ctx.fillRect(left+side+18,top+i,12,1);}
  ctx.fillStyle="#737b72";ctx.textAlign="left";ctx.fillText((log?Math.pow(10,hi):hi).toExponential(2),left+side+35,top+8);ctx.fillText((log?Math.pow(10,lo):lo).toExponential(2),left+side+35,top+side);
  ctx.textAlign="center";ctx.fillText(`${data.axes[0]}: ${data.x[0].toPrecision(3)} … ${data.x.at(-1).toPrecision(3)}`,left+side/2,top+side+30);
  ctx.save();ctx.translate(left-20,top+side/2);ctx.rotate(-Math.PI/2);ctx.fillText(`${data.axes[1]}: ${data.y[0].toPrecision(3)} … ${data.y.at(-1).toPrecision(3)}`,0,0);ctx.restore();
}
function drawMonitor() {
  if(!monitorData)return;
  const spectrum=monitorData.spectrum,max=Number($("#k-max").value),rows=monitorData.scalars,quantity=$("#scalar-quantity").value;
  const points=spectrum?spectrum.k.map((k,i)=>[k,spectrum.nk[i],spectrum.complete[i]]).filter(p=>!max || p[0]<=max):[];
  plot($("#spectrum-canvas"),points,{log:$("#spectrum-log").checked,xlabel:"k",ylabel:"n(k)"});
  $("#spectrum-caption").textContent=spectrum?`t = ${fmt(spectrum.time)} · step ${spectrum.step} · grey: incomplete shells` : "No spectrum saved yet.";
  plot($("#scalar-canvas"),rows.map(row=>[row.time,row[quantity]]),{xlabel:"Stage time",ylabel:quantity});
  drawImage(monitorData.image);
  $("#image-caption").textContent=monitorData.image?`${monitorData.image.kind} · t = ${fmt(monitorData.image.time)} · display sampled to ≤256 pixels per axis` : "No image saved yet.";
  const latest=rows.at(-1);
  $("#scalar-cards").innerHTML=latest?[['Stage time','time'],['Norm','norm'],['Kinetic energy','E_kin'],['Angular momentum','Lz']].map(([label,key])=>`<div class="scalar-card"><small>${label}</small><strong>${escapeHTML(fmt(latest[key]))}</strong></div>`).join(""):"";
}

pollState();
setInterval(async()=>{
  await pollState();
  if(selectedView==="monitor" && $("#auto-refresh").checked && !document.hidden && !workflowState?.operation.busy && Date.now()-lastRefresh>=Math.max(5,Number($("#refresh-seconds").value)||15)*1000) refreshMonitor();
},1800);

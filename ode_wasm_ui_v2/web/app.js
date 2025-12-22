import init, { integrate_linear_2x2, integrate_vanderpol } from './pkg/ode_wasm.js';
const el = id => document.getElementById(id);
const log = (msg) => { const L = el('log'); L.textContent = msg + "\n" + L.textContent; };
const modelSel = el('model');
const phase = el('phase'); const ctx = phase.getContext('2d');
const td = el('td'); const tdctx = td.getContext('2d');
let seeds = [{x:1,y:0}];
const getView = () => ({ xmin: parseFloat(el('xmin').value), xmax: parseFloat(el('xmax').value),
  ymin: parseFloat(el('ymin').value), ymax: parseFloat(el('ymax').value), });
function worldToScreen(x, y) { const v=getView(); const X=(x-v.xmin)/(v.xmax-v.xmin)*phase.width;
  const Y=phase.height - (y-v.ymin)/(v.ymax-v.ymin)*phase.height; return [X,Y]; }
function screenToWorld(X, Y) { const v=getView(); const x=v.xmin+(X/phase.width)*(v.xmax-v.xmin);
  const y=v.ymin+((phase.height-Y)/phase.height)*(v.ymax-v.ymin); return [x,y]; }
function drawAxes(){ const v=getView(); ctx.strokeStyle='#2a313b'; ctx.lineWidth=1; ctx.beginPath();
  let [X0,Y0]=worldToScreen(v.xmin,0), [X1,Y1]=worldToScreen(v.xmax,0); ctx.moveTo(X0,Y0); ctx.lineTo(X1,Y1);
  [X0,Y0]=worldToScreen(0,v.ymin); [X1,Y1]=worldToScreen(0,v.ymax); ctx.moveTo(X0,Y0); ctx.lineTo(X1,Y1); ctx.stroke(); }
function drawVectorFieldLinear(a11,a12,a21,a22,density=20){ const v=getView(); ctx.strokeStyle='#7aa2f7'; ctx.lineWidth=1;
  for(let i=0;i<density;i++){ for(let j=0;j<density;j++){ const x=v.xmin+(i+0.5)*(v.xmax-v.xmin)/density;
    const y=v.ymin+(j+0.5)*(v.ymax-v.ymin)/density; const u=a11*x+a12*y; const w=a21*x+a22*y;
    let mag=Math.hypot(u,w)||1e-6; const scale=0.07*Math.max(phase.width,phase.height)/density;
    const dx=(u/mag)*scale, dy=(w/mag)*scale; const [X,Y]=worldToScreen(x,y);
    ctx.beginPath(); ctx.moveTo(X-dx/2,Y+dy/2); ctx.lineTo(X+dx/2,Y-dy/2); ctx.stroke(); } } }
function drawVectorFieldVdP(mu,density=20){ const v=getView(); ctx.strokeStyle='#7aa2f7'; ctx.lineWidth=1;
  for(let i=0;i<density;i++){ for(let j=0;j<density;j++){ const x=v.xmin+(i+0.5)*(v.xmax-v.xmin)/density;
    const s=v.ymin+(j+0.5)*(v.ymax-v.ymin)/density; const dx=s; const ds=mu*(1-x*x)*s - x;
    let mag=Math.hypot(dx,ds)||1e-6; const sc=0.07*Math.max(phase.width,phase.height)/density;
    const ddx=(dx/mag)*sc, dds=(ds/mag)*sc; const [X,Y]=worldToScreen(x,s);
    ctx.beginPath(); ctx.moveTo(X-ddx/2,Y+dds/2); ctx.lineTo(X+ddx/2,Y-dds/2); ctx.stroke(); } } }
function drawLineImplicit(a,b,c){ const v=getView(); let pts=[];
  for(let x of [v.xmin,v.xmax]){ if(Math.abs(b)>1e-12){ const y=-(a*x+c)/b; if(y>=v.ymin&&y<=v.ymax) pts.push([x,y]); } }
  for(let y of [v.ymin,v.ymax]){ if(Math.abs(a)>1e-12){ const x=-(b*y+c)/a; if(x>=v.xmin&&x<=v.xmax) pts.push([x,y]); } }
  if(pts.length<2) return; const [X0,Y0]=worldToScreen(pts[0][0],pts[0][1]); const [X1,Y1]=worldToScreen(pts[1][0],pts[1][1]);
  ctx.beginPath(); ctx.moveTo(X0,Y0); ctx.lineTo(X1,Y1); ctx.stroke(); }
function drawNullclinesLinear(a11,a12,a21,a22){ ctx.strokeStyle='#f7768e'; ctx.lineWidth=1.5; drawLineImplicit(a11,a12,0);
  ctx.strokeStyle='#9ece6a'; drawLineImplicit(a21,a22,0); }
function drawNullclinesVdP(mu){ ctx.strokeStyle='#f7768e'; ctx.lineWidth=1.5; const v=getView();
  let [X0,Y0]=worldToScreen(v.xmin,0), [X1,Y1]=worldToScreen(v.xmax,0); ctx.beginPath(); ctx.moveTo(X0,Y0); ctx.lineTo(X1,Y1); ctx.stroke();
  ctx.strokeStyle='#9ece6a'; ctx.beginPath(); let first=true;
  for(let xi=0;xi<=400;xi++){ const x=(xi/400)*(v.xmax-v.xmin)+v.xmin; const denom=mu*(1-x*x);
    if(Math.abs(denom)<1e-6){ first=true; continue;} const s=x/denom; if(!isFinite(s)||s<v.ymin||s>v.ymax){ first=true; continue;}
    const [X,Y]=worldToScreen(x,s); if(first){ ctx.moveTo(X,Y); first=false;} else { ctx.lineTo(X,Y); } } ctx.stroke(); }
function classifyLinear(a11,a12,a21,a22){ const tr=a11+a22; const det=a11*a22-a12*a21; const disc=tr*tr-4*det; let typ='';
  if(det<0) typ='saddle'; else if(Math.abs(det)<1e-12) typ='degenerate / line of eq.';
  else { if(disc<0) typ=tr<0?'spiral (stable)':'spiral (unstable)'; else if(Math.abs(disc)<1e-12) typ='repeated real, star/degenerate';
    else typ=tr<0?'node (stable)':'node (unstable)'; } return {tr,det,disc,typ}; }
function drawTDPlanePoint(tr,det){ tdctx.clearRect(0,0,td.width,td.height); tdctx.strokeStyle='#2a313b'; tdctx.lineWidth=1; tdctx.beginPath();
  tdctx.moveTo(0, td.height-20); tdctx.lineTo(td.width, td.height-20); tdctx.moveTo(30, td.height); tdctx.lineTo(30, 0); tdctx.stroke();
  tdctx.fillStyle='#e6e9ee'; tdctx.font='12px monospace'; tdctx.fillText('det', td.width-24, td.height-6); tdctx.fillText('tr', 6, 12);
  const detMax=10, trMax=10; const X=30+(Math.max(-detMax,Math.min(det,detMax))+detMax)/(2*detMax)*(td.width-40);
  const Y=(td.height-20) - (Math.max(-trMax,Math.min(tr,trMax))+trMax)/(2*trMax)*(td.height-30);
  tdctx.fillStyle='#e0af68'; tdctx.beginPath(); tdctx.arc(X,Y,4,0,2*Math.PI); tdctx.fill(); }
function redraw(){ ctx.clearRect(0,0,phase.width,phase.height); drawAxes(); const density=parseInt(el('dense').value);
  const model=modelSel.value; if(model==='linear'){ const a11=parseFloat(el('a11').value), a12=parseFloat(el('a12').value),
    a21=parseFloat(el('a21').value), a22=parseFloat(el('a22').value);
    drawVectorFieldLinear(a11,a12,a21,a22,density); drawNullclinesLinear(a11,a12,a21,a22);
    const {tr,det,disc,typ}=classifyLinear(a11,a12,a21,a22); drawTDPlanePoint(tr,det);
    el('td-stats').textContent=`tr=${tr.toFixed(3)}  det=${det.toFixed(3)}  Δ=${disc.toFixed(3)}\n${typ}`;
  } else { const mu=parseFloat(el('mu').value); drawVectorFieldVdP(mu,density); drawNullclinesVdP(mu); tdctx.clearRect(0,0,td.width,td.height);
    el('td-stats').textContent='—'; }
  ctx.fillStyle='#e0af68'; for(const s of seeds){ const [X,Y]=worldToScreen(s.x,s.y); ctx.beginPath(); ctx.arc(X,Y,3,0,2*Math.PI); ctx.fill(); } }
function integrateSeeds(){ const model=modelSel.value; ctx.lineWidth=1.6; ctx.strokeStyle='#e0af68';
  if(model==='linear'){ const a11=parseFloat(el('a11').value), a12=parseFloat(el('a12').value),
      a21=parseFloat(el('a21').value), a22=parseFloat(el('a22').value);
      const T=parseFloat(el('T_lin').value), dt=parseFloat(el('dt_lin').value);
      for(const s of seeds){ try{ const res=integrate_linear_2x2(a11,a12,a21,a22,s.x,s.y,0.0,T,dt); drawTrajectory(res.t,res.y,res.dim); }
        catch(e){ log('Linear integrate error: '+e); } } }
  else { const mu=parseFloat(el('mu').value); const T=parseFloat(el('T_vdp').value), dt=parseFloat(el('dt_vdp').value);
      for(const s of seeds){ try{ const res=integrate_vanderpol(mu,s.x,s.y,0.0,T,dt); drawTrajectory(res.t,res.y,res.dim); }
        catch(e){ log('VdP integrate error: '+e); } } } }
function drawTrajectory(ts, flatY, dim){ ctx.strokeStyle='#e0af68'; ctx.beginPath();
  for(let i=0;i<ts.length;i++){ const x=flatY[i*dim+0], y=flatY[i*dim+1]; const [X,Y]=worldToScreen(x,y);
    if(i===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y); } ctx.stroke(); }
phase.addEventListener('click',(ev)=>{ const r=phase.getBoundingClientRect(); const [x,y]=screenToWorld(ev.clientX-r.left, ev.clientY-r.top);
  if(ev.shiftKey){ seeds.push({x,y}); } else { seeds=[{x,y}]; } redraw(); });
el('clearSeeds').onclick=()=>{ seeds=[]; redraw(); };
el('resetView').onclick=()=>{ el('xmin').value=-4; el('xmax').value=4; el('ymin').value=-4; el('ymax').value=4; redraw(); };
el('run').onclick=()=>{ redraw(); integrateSeeds(); };
modelSel.onchange=()=>{ const linear=modelSel.value==='linear'; el('linear-params').classList.toggle('hidden',!linear);
  el('vdp-params').classList.toggle('hidden',linear); redraw(); };
['a11','a12','a21','a22','mu','xmin','xmax','ymin','ymax','dense','T_lin','dt_lin','T_vdp','dt_vdp']
  .forEach(id=>{ const e=el(id); e && (e.oninput=redraw); });
await init().then(()=>log('WASM loaded')).catch(err=>log('WASM init failed: '+err)); redraw();

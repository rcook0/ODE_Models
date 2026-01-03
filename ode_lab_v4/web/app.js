import init, {
  integrate_linear_2x2, integrate_vanderpol, integrate_predator_prey,
  integrate_duffing, integrate_sir, integrate_lorenz63, integrate_rossler,
  integrate_lorenz63_adaptive, integrate_rossler_adaptive,
  integrate_chua, integrate_thomas
} from './pkg/ode_wasm.js';

const el = (id) => document.getElementById(id);
const phase = el('phase'); const ctx = phase.getContext('2d');
const td = el('td'); const tdctx = td.getContext('2d');
const view3d = el('view3d'); const v3 = view3d.getContext('2d');
const modelSel = el('model'); const paramArea = el('paramArea');
const log = (msg) => { el('log').textContent = msg + "\n" + el('log').textContent; };

let seeds = [{x:1,y:0}];
let latestAttractor = null;
const trajCache = new Map();      // key -> Series {t,y,dim}
const attractorCache = new Map(); // key -> [[x,y,z],...]

function stableHash(obj){
  const seen = new WeakSet();
  const norm = (v) => {
    if (v && typeof v === 'object'){
      if (seen.has(v)) return null;
      seen.add(v);
      if (Array.isArray(v)) return v.map(norm);
      const out = {};
      Object.keys(v).sort().forEach(k => out[k]=norm(v[k]));
      return out;
    }
    return v;
  };
  return JSON.stringify(norm(obj));
}

class SVGRecorder {
  constructor(w,h){
    this.w=w; this.h=h;
    this.elems=[];
  }
  line(x1,y1,x2,y2, stroke, strokeWidth){
    this.elems.push(`<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="${stroke}" stroke-width="${strokeWidth}" stroke-linecap="round"/>`);
  }
  polyline(points, stroke, strokeWidth){
    const d = points.map(p=>`${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');
    this.elems.push(`<polyline points="${d}" fill="none" stroke="${stroke}" stroke-width="${strokeWidth}" stroke-linejoin="round" stroke-linecap="round"/>`);
  }
  circle(cx,cy,r, stroke, strokeWidth, fill='none'){
    this.elems.push(`<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${r.toFixed(2)}" stroke="${stroke}" stroke-width="${strokeWidth}" fill="${fill}"/>`);
  }
  toBlob(){
    const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${this.w}" height="${this.h}" viewBox="0 0 ${this.w} ${this.h}">
  <rect width="100%" height="100%" fill="#0f1318"/>
  ${this.elems.join('\n  ')}
</svg>`;
    return new Blob([svg], {type:'image/svg+xml'});
  }
}
let svgRec = null;


const presets = {
  _solver:{adaptive3D:true, rtol:1e-6, atol:1e-9, max_steps:200000, dt0:0.01},
  linear:{a11:-1,a12:-5,a21:2,a22:-3,T:15,dt:0.01},
  vdp:{mu:3,T:40,dt:0.01},
  lv:{alpha:1,beta:0.1,delta:0.075,gamma:1.5,T:40,dt:0.01},
  duffing:{delta:0.2,alpha:-1,beta:1,gamma:0.3,omega:1.2,T:40,dt:0.01},
  sir:{beta:0.5,gamma:0.2,T:160,dt:0.2,s0:0.99,i0:0.01,r0:0},
  lorenz:{sigma:10,rho:28,beta:8/3,z0:1,T:25,dt:0.01},
  rossler:{a:0.2,b:0.2,c:5.7,z0:0.1,T:80,dt:0.01},
  chua:{alpha:15.6,beta:28.0,m0:-1.143,m1:-0.714,z0:0.0,T:200,dt:0.01},
  thomas:{a:0.208186,z0:0.0,T:400,dt:0.02},
};

const getView = () => ({xmin:+el('xmin').value,xmax:+el('xmax').value,ymin:+el('ymin').value,ymax:+el('ymax').value});
const worldToScreen = (x,y)=>{const v=getView();return [(x-v.xmin)/(v.xmax-v.xmin)*phase.width, phase.height-(y-v.ymin)/(v.ymax-v.ymin)*phase.height];};
const screenToWorld = (X,Y)=>{const v=getView();return [v.xmin+(X/phase.width)*(v.xmax-v.xmin), v.ymin+((phase.height-Y)/phase.height)*(v.ymax-v.ymin)];};

function debounce(fn, ms=220){let t=null;return (...a)=>{if(t)clearTimeout(t);t=setTimeout(()=>fn(...a),ms);}}

function redrawBase(){
  if(svgRec){ svgRec.elems=[]; }
  ctx.clearRect(0,0,phase.width,phase.height);
  ctx.strokeStyle='#2a313b'; ctx.lineWidth=1;
  const v=getView();
  let [X0,Y0]=worldToScreen(v.xmin,0), [X1,Y1]=worldToScreen(v.xmax,0);
  ctx.beginPath();
  const pts = svgRec ? [] : null; ctx.moveTo(X0,Y0); ctx.lineTo(X1,Y1); if(svgRec){ svgRec.line(X0,Y0,X1,Y1,'#2a313b',1); }
  [X0,Y0]=worldToScreen(0,v.ymin); [X1,Y1]=worldToScreen(0,v.ymax);
  ctx.moveTo(X0,Y0); ctx.lineTo(X1,Y1); if(svgRec){ svgRec.line(X0,Y0,X1,Y1,'#2a313b',1); }
  ctx.stroke();
  if(svgRec && pts && pts.length>1){ svgRec.polyline(pts,'#e0af68',1.6); }


  tdctx.clearRect(0,0,td.width,td.height);
  el('td-stats').textContent='';
}

function drawQuiver(sampleFn, density){
  const v=getView();
  ctx.strokeStyle='#7aa2f7'; ctx.lineWidth=1;
  for(let i=0;i<density;i++)for(let j=0;j<density;j++){
    const x=v.xmin+(i+0.5)*(v.xmax-v.xmin)/density;
    const y=v.ymin+(j+0.5)*(v.ymax-v.ymin)/density;
    const [u,w]=sampleFn(x,y); const mag=Math.hypot(u,w)||1e-9;
    const s=0.07*Math.max(phase.width,phase.height)/density;
    const dx=(u/mag)*s, dy=(w/mag)*s;
    const [X,Y]=worldToScreen(x,y);
    ctx.beginPath(); ctx.moveTo(X-dx/2,Y+dy/2); ctx.lineTo(X+dx/2,Y-dy/2); ctx.stroke();
    if(svgRec){ svgRec.line(X-dx/2,Y+dy/2,X+dx/2,Y-dy/2,'#7aa2f7',1); }
  }
}

function drawSeeds(){
  ctx.fillStyle='#e0af68';
  for(const s of seeds){const [X,Y]=worldToScreen(s.x,s.y);ctx.beginPath();ctx.arc(X,Y,3,0,2*Math.PI);ctx.fill(); if(svgRec){ svgRec.circle(X,Y,3,'#e0af68',1,'#e0af68'); }}
}

function drawTrajectory(res){
  ctx.strokeStyle='#e0af68'; ctx.lineWidth=1.6;
  ctx.beginPath();
  for(let i=0;i<res.t.length;i++){
    const x=res.y[i*res.dim+0], y=res.y[i*res.dim+1];
    const [X,Y]=worldToScreen(x,y);
    if(i===0)ctx.moveTo(X,Y); else ctx.lineTo(X,Y);
    if(pts) pts.push([X,Y]);
  }
  ctx.stroke();
}

function classifyLinear(a11,a12,a21,a22){
  const tr=a11+a22, det=a11*a22-a12*a21, disc=tr*tr-4*det;
  let typ = det<0?'saddle':(Math.abs(det)<1e-12?'degenerate':(disc<0?(tr<0?'spiral stable':'spiral unstable'):(tr<0?'node stable':'node unstable')));
  return {tr,det,disc,typ};
}

function drawTD(tr,det){
  tdctx.strokeStyle='#2a313b'; tdctx.lineWidth=1;
  tdctx.beginPath(); tdctx.moveTo(0,td.height-20); tdctx.lineTo(td.width,td.height-20); tdctx.moveTo(30,td.height); tdctx.lineTo(30,0); tdctx.stroke();
  const detMax=10,trMax=10;
  const X=30+((Math.max(-detMax,Math.min(det,detMax))+detMax)/(2*detMax))*(td.width-40);
  const Y=(td.height-20)-((Math.max(-trMax,Math.min(tr,trMax))+trMax)/(2*trMax))*(td.height-30);
  tdctx.fillStyle='#e0af68'; tdctx.beginPath(); tdctx.arc(X,Y,4,0,2*Math.PI); tdctx.fill();
}

function mountParams(model){
  const p=presets[model];
  const card=document.createElement('div'); card.className='paramCard';
  const row = (pairs)=>pairs.map(([k,v,step])=>`<label>${k}<input id="${k}" type="number" step="${step}" value="${v}"/></label>`).join('');
  const common = `<div class="grid"><label>T<input id="T" type="number" step="0.5" value="${p.T}"/></label><label>dt<input id="dt" type="number" step="0.001" value="${p.dt}"/></label></div>`;
  if(model==='linear') card.innerHTML=`<div class="mono small">Linear 2×2</div><div class="grid">${row([['a11',p.a11,0.1],['a12',p.a12,0.1],['a21',p.a21,0.1],['a22',p.a22,0.1]])}</div>${common}`;
  if(model==='vdp') card.innerHTML=`<div class="mono small">Van der Pol</div><div class="grid">${row([['mu',p.mu,0.5]])}</div>${common}`;
  if(model==='lv') card.innerHTML=`<div class="mono small">Predator–Prey</div><div class="grid">${row([['alpha',p.alpha,0.01],['beta',p.beta,0.01],['delta',p.delta,0.01],['gamma',p.gamma,0.01]])}</div>${common}`;
  if(model==='duffing') card.innerHTML=`<div class="mono small">Duffing</div><div class="grid">${row([['delta',p.delta,0.01],['alpha',p.alpha,0.1],['beta',p.beta,0.1],['gamma',p.gamma,0.01],['omega',p.omega,0.01]])}</div>${common}`;
  if(model==='sir') card.innerHTML=`<div class="mono small">SIR (phase S–I)</div><div class="grid">${row([['beta',p.beta,0.01],['gamma',p.gamma,0.01],['s0',p.s0,0.01],['i0',p.i0,0.01],['r0',p.r0,0.01]])}</div>${common}`;
  if(model==='lorenz') card.innerHTML=`<div class="mono small">Lorenz63 (3D)</div><div class="grid">${row([['sigma',p.sigma,0.1],['rho',p.rho,0.1],['beta',p.beta,0.01],['z0',p.z0,0.1]])}</div>${common}`;
  if(model==='rossler') card.innerHTML=`<div class="mono small">Rössler (3D)</div><div class="grid">${row([['a',p.a,0.01],['b',p.b,0.01],['c',p.c,0.1],['z0',p.z0,0.1]])}</div>${common}`;
  if(model==='chua') card.innerHTML=`<div class=\"mono small\">Chua (3D)</div><div class=\"grid\">${row([['alpha',p.alpha,0.1],['beta',p.beta,0.1],['m0',p.m0,0.01],['m1',p.m1,0.01],['z0',p.z0,0.1]])}</div>${common}`;
  if(model==='thomas') card.innerHTML=`<div class=\"mono small\">Thomas (3D)</div><div class=\"grid\">${row([['a',p.a,0.001],['z0',p.z0,0.1]])}</div>${common}`;
  paramArea.innerHTML=''; paramArea.appendChild(card);
  card.querySelectorAll('input').forEach(i=>i.oninput=liveUpdate);
}

function currentParams(){
  const m=modelSel.value;
  const get=(id)=>+el(id).value;
  const T=get('T'), dt=get('dt');
  const o={model:m,T,dt};
  ['a11','a12','a21','a22','mu','alpha','beta','delta','gamma','omega','s0','i0','r0','sigma','rho','a','b','c','z0'].forEach(k=>{
    const n=document.getElementById(k); if(n) o[k]=+n.value;
  });
  return o;
}

function draw3D(){
  v3.clearRect(0,0,view3d.width,view3d.height);
  if(!latestAttractor) return;
  const yaw=+el('yaw').value, pitch=+el('pitch').value;
  const rot=(p)=>{
    const [x,y,z]=p;
    const cy=Math.cos(yaw), sy=Math.sin(yaw);
    let x1=cy*x - sy*y, y1=sy*x + cy*y, z1=z;
    const cp=Math.cos(pitch), sp=Math.sin(pitch);
    let y2=cp*y1 - sp*z1;
    return [x1,y2];
  };
  const pts=latestAttractor.map(rot);
  let xmin=1e9,xmax=-1e9,ymin=1e9,ymax=-1e9;
  for(const [x,y] of pts){xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
  const w=view3d.width,h=view3d.height,dx=(xmax-xmin)||1,dy=(ymax-ymin)||1;
  const tx=(x)=> (x-xmin)/dx*(w*0.86)+w*0.07;
  const ty=(y)=> h-((y-ymin)/dy*(h*0.86)+h*0.07);
  v3.strokeStyle='#e0af68'; v3.lineWidth=1; v3.beginPath();
  for(let i=0;i<pts.length;i++){const x=tx(pts[i][0]), y=ty(pts[i][1]); if(i===0)v3.moveTo(x,y); else v3.lineTo(x,y);}
  v3.stroke();
}

function integrateAndDraw(opts={forceIntegrate:true}){
  const p=currentParams();
  const doIntegrate = (opts.forceIntegrate !== false);
  redrawBase();
  const density=+el('dense').value;
  const keyBase = stableHash({model:p.model, params:p, dt:p.dt, T:p.T});
  const seedsKey = stableHash(seeds);

  if(p.model==='linear'){
    drawQuiver((x,y)=>[p.a11*x+p.a12*y,p.a21*x+p.a22*y],density);
    const c=classifyLinear(p.a11,p.a12,p.a21,p.a22); drawTD(c.tr,c.det);
    el('td-stats').textContent=`tr=${c.tr.toFixed(3)} det=${c.det.toFixed(3)} Δ=${c.disc.toFixed(3)}\n${c.typ}`;
  } else if(p.model==='vdp'){
    drawQuiver((x,v)=>[v,p.mu*(1-x*x)*v-x],density);
  } else if(p.model==='lv'){
    drawQuiver((x,y)=>[p.alpha*x-p.beta*x*y,p.delta*x*y-p.gamma*y],density);
  } else if(p.model==='duffing'){
    drawQuiver((x,v)=>[v,-p.delta*v-p.alpha*x-p.beta*x*x*x+p.gamma],density);
  } else if(p.model==='sir'){
    drawQuiver((S,I)=>[-p.beta*S*I,p.beta*S*I-p.gamma*I],density);
  } else if(p.model==='lorenz'){
    drawQuiver((x,y)=>[p.sigma*(y-x), x*(p.rho-p.z0)-y],density);
  } else if(p.model==='rossler'){
    drawQuiver((x,y)=>[-y-p.z0, x+p.a*y],density);
  } else if(p.model==='chua'){
    drawQuiver((x,y)=>{
      const fx = p.m1*x + 0.5*(p.m0-p.m1)*(Math.abs(x+1)-Math.abs(x-1));
      return [p.alpha*(y - x - fx), x - y + p.z0];
    }, density);
  } else if(p.model==='thomas'){
    drawQuiver((x,y)=>[Math.sin(y) - p.a*x, Math.sin(p.z0) - p.a*y], density);
  }

  drawSeeds();

  // integrate
  if(['linear','vdp','lv','duffing','sir'].includes(p.model)){
    latestAttractor=null; draw3D();
    for(const s of seeds){
      const k = keyBase + '::seed=' + stableHash(s);
      if(!doIntegrate && trajCache.has(k)){
        drawTrajectory(trajCache.get(k));
        continue;
      }
      try{
        let res=null;
        if(p.model==='linear') res=integrate_linear_2x2(p.a11,p.a12,p.a21,p.a22,s.x,s.y,0,p.T,p.dt);
        if(p.model==='vdp') res=integrate_vanderpol(p.mu,s.x,s.y,0,p.T,p.dt);
        if(p.model==='lv') res=integrate_predator_prey(p.alpha,p.beta,p.delta,p.gamma,s.x,s.y,0,p.T,p.dt);
        if(p.model==='duffing') res=integrate_duffing(p.delta,p.alpha,p.beta,p.gamma,p.omega,s.x,s.y,0,p.T,p.dt);
        if(p.model==='sir') res=integrate_sir(p.beta,p.gamma,s.x,s.y,Math.max(0,1-s.x-s.y),0,p.T,p.dt);
        if(res){ trajCache.set(k, res); drawTrajectory(res); }
      } catch(e){ log('Integrate error: '+e); }
    }
    return;
  }

  // 3D
  try{
    const k3 = keyBase + '::seeds=' + seedsKey;
    if(!doIntegrate && attractorCache.has(k3)){
      latestAttractor = attractorCache.get(k3);
      draw3D();
      return;
    }
    const x0=seeds.length?seeds[0].x:1, y0=seeds.length?seeds[0].y:1, z0=p.z0??1;
    let res=null;
    const sopts = presets._solver;
    if(p.model==='lorenz'){
      res = sopts.adaptive3D
        ? integrate_lorenz63_adaptive(p.sigma,p.rho,p.beta,x0,y0,z0,0,p.T,sopts.dt0,sopts.rtol,sopts.atol,sopts.max_steps)
        : integrate_lorenz63(p.sigma,p.rho,p.beta,x0,y0,z0,0,p.T,p.dt);
    }
    if(p.model==='rossler'){
      res = sopts.adaptive3D
        ? integrate_rossler_adaptive(p.a,p.b,p.c,x0,y0,z0,0,p.T,sopts.dt0,sopts.rtol,sopts.atol,sopts.max_steps)
        : integrate_rossler(p.a,p.b,p.c,x0,y0,z0,0,p.T,p.dt);
    }
    if(p.model==='chua') res=integrate_chua(p.alpha,p.beta,p.m0,p.m1,x0,y0,z0,0,p.T,p.dt);
    if(p.model==='thomas') res=integrate_thomas(p.a,x0,y0,z0,0,p.T,p.dt);
    const pts=[];
    for(let i=0;i<res.t.length;i++){
      pts.push([res.y[i*res.dim+0],res.y[i*res.dim+1],res.y[i*res.dim+2]]);
    }
    attractorCache.set(k3, pts);
    latestAttractor=pts;
    draw3D();
  }catch(e){ log('3D error: '+e); }
}

const liveUpdate = debounce(()=>{ if(el('live').checked) integrateAndDraw({forceIntegrate:true}); }, 220);

phase.addEventListener('click',(ev)=>{
  const r=phase.getBoundingClientRect(); const [x,y]=screenToWorld(ev.clientX-r.left, ev.clientY-r.top);
  if(ev.shiftKey) seeds.push({x,y}); else seeds=[{x,y}];
  if(el('live').checked) integrateAndDraw({forceIntegrate:true}); else { redrawBase(); drawSeeds(); }
});

el('clearSeeds').onclick=()=>{seeds=[]; integrateAndDraw();};
el('resetView').onclick=()=>{el('xmin').value=-4;el('xmax').value=4;el('ymin').value=-4;el('ymax').value=4; integrateAndDraw();};
el('run').onclick=()=>integrateAndDraw({forceIntegrate:true});
['xmin','xmax','ymin','ymax','dense'].forEach(id=>el(id).oninput=liveUpdate);
['yaw','pitch','scale3'].forEach(id=>el(id).oninput=()=>draw3D());

// Exports
function downloadBlob(blob,name){const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);}
el('exportPng').onclick=()=>{const png=phase.toDataURL('image/png');fetch(png).then(r=>r.blob()).then(b=>downloadBlob(b,`ode_phase_${modelSel.value}.png`));};
el('exportSvg').onclick=()=>{
  svgRec = new SVGRecorder(phase.width, phase.height);
  integrateAndDraw({forceIntegrate:false});
  const blob = svgRec.toBlob();
  svgRec = null;
  downloadBlob(blob, `ode_phase_${modelSel.value}.svg`);
};

// Scenes
function getScene(){return {version:3,model:modelSel.value,view:getView(),dense:+el('dense').value,live:el('live').checked,seeds,params:currentParams(),camera:{yaw:+el('yaw').value,pitch:+el('pitch').value,scale:+el('scale3').value}};}
function applyScene(s){
  modelSel.value=s.model; el('xmin').value=s.view.xmin; el('xmax').value=s.view.xmax; el('ymin').value=s.view.ymin; el('ymax').value=s.view.ymax;
  el('dense').value=s.dense??20; el('live').checked=!!s.live; el('yaw').value=s.camera?.yaw??0.7; el('pitch').value=s.camera?.pitch??0.4; el('scale3').value=s.camera?.scale??80;
  seeds=s.seeds??[{x:1,y:0}];
  mountParams(s.model);
  Object.entries(s.params||{}).forEach(([k,v])=>{const n=document.getElementById(k); if(n && typeof v==='number') n.value=v;});
  integrateAndDraw();
}
el('saveScene').onclick=()=>{const name=prompt("Scene name to save (local):",`scene_${modelSel.value}`); if(!name)return; localStorage.setItem(`ode_scene_${name}`,JSON.stringify(getScene())); log(`Saved scene: ${name}`);};
el('loadScene').onclick=()=>{
  const keys=Object.keys(localStorage).filter(k=>k.startsWith('ode_scene_')).map(k=>k.replace('ode_scene_','')).sort();
  if(!keys.length){alert("No saved scenes.");return;}
  const name=prompt("Scene name:\n"+keys.join('\n')); if(!name)return;
  const raw=localStorage.getItem(`ode_scene_${name}`); if(!raw){alert("Not found.");return;}
  applyScene(JSON.parse(raw)); log(`Loaded scene: ${name}`);
};
el('exportScene').onclick=()=>downloadBlob(new Blob([JSON.stringify(getScene(),null,2)],{type:'application/json'}),`ode_scene_${modelSel.value}.json`);
el('importScene').addEventListener('change',async(ev)=>{const f=ev.target.files?.[0]; if(!f)return; try{applyScene(JSON.parse(await f.text())); log("Imported scene");}catch{alert("Invalid JSON");} ev.target.value="";});

modelSel.onchange=()=>{seeds=modelSel.value==='sir'?[{x:0.99,y:0.01}]:[{x:1,y:0}]; mountParams(modelSel.value); integrateAndDraw();};

await init().then(()=>log("WASM loaded")).catch(e=>log("WASM init failed: "+e));
mountParams(modelSel.value);
integrateAndDraw();

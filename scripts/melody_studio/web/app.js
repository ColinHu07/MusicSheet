const $ = id => document.getElementById(id);
const ticksPerBeat = 48, ticksPerBar = 192;
let project, score, version, selected, busy = false, generation = 0, timer;
let past = [], future = [], glyphs = new Map();
let pendingSave = null;
let libraryScroll = 0;
let audioContext, bank, playing = false, playbackToken = 0, pos = 0, clockStart = 0, index = 0;
const recording = new Audio(), voices = new Set(), buffers = new Map();
let catalogOrigin='';
let engravingOrigin='';
recording.preload = 'auto';

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || 'Request failed (' + res.status + ').');
  }
  return res.json();
}
function showError(e) {
  $('message').textContent = e.message || String(e);
  $('message').hidden = false;
}
function clearError() { $('message').hidden = true; }
const base = () => '/api/projects/' + project.id + '/';
const rate = () => Number($('speed').value);
const step = () => Number($('step').value);
const byId = () => score?.notes.find(n => n.id === selected);
const startSec = tick => score.phase + tick * 60 / (ticksPerBeat * score.bpm);
const time = s => Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
const endSec = () => Math.max(score.duration, startSec(score.notes.at(-1)?.end || 0));
const pitchName = m => (score.fifths > 0 ? ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'] :
  ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B'])[m%12] + (Math.floor(m/12)-1);
function parsePitch(text) {
  const m = /^([a-g])([#♯b♭]?)(-?\d+)$/i.exec(text.trim());
  if (!m) throw Error('Enter a pitch such as C5, F#4 or Bb4.');
  const offset = {'#':1,'♯':1,'b':-1,'♭':-1,'':0}[m[2]];
  const value = 12 * (Number(m[3])+1) + {C:0,D:2,E:4,F:5,G:7,A:9,B:11}[m[1].toUpperCase()] + offset;
  if (!(value >= 0 && value <= 127)) throw Error('This pitch is outside the MIDI range.');
  return value;
}
function controls() {
  const hasNote = !!byId();
  $('undo').disabled = busy || !past.length;
  $('redo').disabled = busy || !future.length;
  $('play').disabled = !score || busy;
  $('exports').disabled = !score || busy;
  $('insert').disabled = !score || busy;
  for (const el of [...$('editor').elements, $('split'), $('join'), $('delete')]) el.disabled = !hasNote || busy;
}
function fields() {
  const n = byId();
  if (n) {
    $('pitch').value = pitchName(n.pitch);
    $('bar').value = 1 + Math.floor(n.start/ticksPerBar);
    $('beat').value = +(1 + n.start%ticksPerBar/ticksPerBeat).toFixed(5);
    $('length').value = +((n.end-n.start)/ticksPerBeat).toFixed(5);
    $('location').textContent = 'Bar ' + $('bar').value + ' · Beat ' + $('beat').value;
  } else $('location').textContent = 'Select a note on the score.';
  controls();
}
function highlight() {
  document.querySelectorAll('.selectionBox').forEach(el=>el.remove());
  document.querySelectorAll('.chosen').forEach(el=>el.classList.remove('chosen'));
  for (const g of glyphs.get(selected) || []) {
    g.classList.add('chosen');
    const head = g.querySelector('.notehead') || g;
    const b = head.getBBox(), rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('class','selectionBox');
    for (const [k,v] of Object.entries({x:b.x-40,y:b.y-35,width:b.width+80,height:b.height+70,rx:35})) rect.setAttribute(k,v);
    head.parentNode.insertBefore(rect,head);
  }
}
function seek(sec) {
  const resume = playing;
  pause();
  pos = Math.max(0,Math.min(endSec(),sec));
  paintClock();
  if (resume) play().catch(showError);
}
function selectNote(id, audition=true) {
  selected = id;
  highlight(); fields();
  const note = byId();
  if (note && !playing) {
    pos = startSec(note.start);
    paintClock();
    if (audition) preview(note).catch(showError);
  }
}
async function render() {
  const gen = generation;
  const pages = await api(engravingOrigin+base()+'pages?v='+version);
  if (gen !== generation) return;
  const top = $('paperArea').scrollTop;
  $('paper').replaceChildren();
  glyphs = new Map();
  pages.forEach((svg,i)=>{
    const section = document.createElement('section');
    section.innerHTML = svg.slice(svg.indexOf('<svg'));
    $('paper').append(section);
    for (const g of section.querySelectorAll('g.note[id]')) {
      const m = /^n(\d+)_/.exec(g.id), n = m && score.notes[Number(m[1])];
      if (!n) continue;
      g.dataset.uid = n.id;
      if (!glyphs.has(n.id)) glyphs.set(n.id,[]);
      glyphs.get(n.id).push(g);
      const label = document.createElementNS('http://www.w3.org/2000/svg','title');
      label.textContent = pitchName(n.pitch)+' · Bar '+(1+Math.floor(n.start/ticksPerBar));
      g.prepend(label);
    }
    const number = document.createElement('div');
    number.className='pageNumber'; number.textContent=(i+1)+' / '+pages.length;
    section.append(number);
  });
  $('paperArea').scrollTop=top;
  highlight();
  for (const [id,format] of [['xml','musicxml'],['mid','mid'],['pdf','pdf']]) $(id).href=(format==='mid'?'':engravingOrigin)+base()+format+'?v='+version;
  $('baseline').href=engravingOrigin+base()+'musicxml?v=0';
}
function jobTiming(p) {
  if(p.status==='failed')return 'Transcription stopped';
  const estimate=p.estimated_seconds;
  if(p.status==='queued'){
    const ahead=p.queue_ahead;
    const queue=ahead ? 'Queued · '+ahead+' ahead' : 'Waiting to start';
    return estimate ? queue+' · about '+Math.max(1,Math.round(estimate/60))+' min once started' : queue;
  }
  const parts=[p.stage || 'Starting transcription'];
  if(p.started_at){
    const elapsed=Math.max(0,Math.floor(Date.now()/1000-p.started_at));
    parts.push(time(elapsed)+' elapsed');
    if(estimate){
      const left=estimate-elapsed;
      parts.push(left<=0 ? 'Taking longer than estimated' : left<60 ? 'Under a minute left' : 'About '+Math.round(left/60)+' min left');
    }
  }
  return parts.join(' · ');
}
function galleryCard(p) {
  const card=document.createElement('article');
  card.className='sheet-card';
  const link=document.createElement('a');
  link.className='sheet-link';
  link.href='#'+p.id;
  const cover=document.createElement('div');
  cover.className='sheet-cover';
  const paper=document.createElement('div');
  paper.className='mini-page';
  if(p.status==='ready'){
    const image=document.createElement('img');
    image.alt=''; image.loading='lazy';
    if(Number.isInteger(p.version))image.src=catalogOrigin+'/api/projects/'+p.id+'/preview.svg?v='+p.version;
    else{
      // An already-running server may only expose the full engraving endpoint.
      image.dataset.pages='/api/projects/'+p.id+'/pages';
      legacyPreviews.observe(image);
    }
    paper.append(image);
  }
  cover.append(paper);
  const summary=document.createElement('div');
  summary.className='card-description';
  const title=document.createElement('strong');
  title.textContent=title.title=p.title;
  const detail=document.createElement('small');
  detail.textContent=p.artist || '';
  summary.append(title,detail);
  if(p.status!=='ready'){
    const progress=document.createElement('small');
    progress.dataset.progress=p.id;
    progress.textContent=jobTiming(p);
    summary.append(progress);
  }
  link.append(cover,summary);
  const edit=document.createElement('button');
  edit.className='edit-details'; edit.textContent='Edit'; edit.dataset.edit=p.id;
  edit.setAttribute('aria-label','Edit details for '+p.title);
  edit.onclick=()=>{
    $('detailsForm').dataset.project=p.id;
    $('sheetTitle').value=p.title; $('sheetAuthor').value=p.artist || '';
    $('detailsError').textContent=''; $('sheetDetails').showModal(); $('sheetTitle').select();
  };
  card.append(link,edit);
  return card;
}
$('detailsCancel').onclick=()=>$('sheetDetails').close();
$('detailsForm').onsubmit=async event=>{
  event.preventDefault();
  const id=$('detailsForm').dataset.project;
  $('detailsSave').disabled=true;
  try{
    await api(catalogOrigin+'/api/projects/'+id+'/labels',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({title:$('sheetTitle').value,artist:$('sheetAuthor').value})});
    $('sheetDetails').close();
    await showLibrary();
    document.querySelector('[data-edit="'+id+'"]')?.focus({preventScroll:true});
  }catch(e){if($('sheetDetails').open)$('detailsError').textContent=e.message;else showError(e);}
  finally{$('detailsSave').disabled=false;}
};
const legacyPreviews=new IntersectionObserver(entries=>{
  for(const {target:image,isIntersecting} of entries){
    if(!isIntersecting)continue;
    legacyPreviews.unobserve(image);
    api(image.dataset.pages).then(pages=>{
      if(!pages.length || !image.isConnected)return;
      const url=URL.createObjectURL(new Blob([pages[0]],{type:'image/svg+xml'}));
      image.onload=image.onerror=()=>URL.revokeObjectURL(url);
      image.src=url;
    }).catch(()=>{ image.remove(); });
  }
});
async function showLibrary() {
  const gen=++generation;
  clearTimeout(timer); pause(); recording.removeAttribute('src'); recording.load();
  score=null; project=null; selected=null; glyphs.clear(); clearError();
  $('paper').replaceChildren(); document.title='Melody Studio';
  const refresh=async()=>{
    const list=await api(catalogOrigin+'/api/projects');
    if(gen!==generation)return;
    const signature=JSON.stringify(list);
    if($('gallery').dataset.signature!==signature || list.some(p=>p.status==='ready' && p.version===undefined)){
      legacyPreviews.disconnect();
      const active=document.activeElement;
      const focused=$('gallery').contains(active) && (active.getAttribute('href') || active.dataset.edit);
      $('gallery').replaceChildren(...list.map(galleryCard));
      $('gallery').dataset.signature=signature;
      if(focused)Array.from($('gallery').querySelectorAll('a,button')).find(a=>a.getAttribute('href')===focused || a.dataset.edit===focused)?.focus({preventScroll:true});
    }
    for(const p of list){
      const progress=$('gallery').querySelector('[data-progress="'+p.id+'"]');
      if(progress)progress.textContent=jobTiming(p);
    }
    $('sheetCount').textContent=String(list.length);
    $('emptyLibrary').hidden=list.length!==0;
    if(list.some(p=>['queued','working'].includes(p.status)))timer=setTimeout(()=>refresh().catch(showError),3000);
  };
  await refresh();
}
function showScoreDetails() {
  const values = [score.bpm+' BPM', '4/4', score.notes.length+' notes',
    score.shift ? 'Violin +'+score.shift+' semitones' : 'Concert pitch'];
  if(project.render_seconds)values.push('Rendered in '+time(project.render_seconds));
  $('meta').replaceChildren(...values.map(text => {
    const chip = document.createElement('span');
    chip.textContent = text;
    return chip;
  }));
}
function loadingScene(song,stage,timing='') {
  $('workspace').classList.add('is-loading');
  $('loadingScene').hidden=false;
  $('credits').hidden=true;
  $('loadingSong').textContent=song;
  if($('loadingStage').textContent!==stage)$('loadingStage').textContent=stage;
  $('loadingTime').textContent=timing;
}
function endLoading() {
  $('loadingScene').hidden=true;
  $('workspace').classList.remove('is-loading');
  $('credits').hidden=$('workspace').hidden;
}
$('pauseAnimation').onclick=()=>{
  const paused=$('loadingScene').classList.toggle('motion-paused');
  const label=paused?'Resume animation':'Pause animation';
  $('pauseAnimation').setAttribute('aria-label',label);
  $('pauseAnimation').title=label;
  $('pauseAnimation').firstElementChild.textContent=paused?'▶':'Ⅱ';
};
async function openProject(id) {
  const gen=++generation;
  clearTimeout(timer); pause(); recording.removeAttribute('src'); recording.load(); clearError();
  busy=true; project=null; score=null; pos=0; past=[]; future=[]; selected=null; glyphs.clear();
  $('workspace').querySelector('.layout').hidden=true;
  $('paper').replaceChildren(); $('title').textContent='Opening score…'; $('artist').hidden=true; $('saved').textContent=''; $('meta').textContent=''; $('clock').textContent='0:00'; $('total').textContent='0:00'; $('scrub').value=0;
  controls();
  if(id==='loading-preview'){
    project=null; busy=false; document.title='Loading animation · Melody Studio';
    loadingScene('','Animation preview');
    return;
  }
  let stage='Opening the score';
  const slowLoad=setTimeout(()=>{if(gen===generation)loadingScene(project?.title || '',stage);},350);
  const failed=e=>{
    clearTimeout(slowLoad);
    if(gen!==generation)return;
    endLoading(); busy=false; controls(); $('title').textContent=project?.title || 'Unable to open score';
    showError(e);
  };
  const refresh=async()=>{
    const p=await api(catalogOrigin+'/api/projects/'+id+'/state');
    if(gen!==generation)return;
    project=p; $('title').textContent=p.title; document.title=p.title+' · Melody Studio';
    $('artist').textContent=p.artist || '';
    $('artist').hidden=!p.artist;
    if(p.status!=='ready'){
      busy=false; controls(); fields();
      $('meta').textContent=jobTiming(p);
      if(p.status==='failed'){
        clearTimeout(slowLoad); endLoading();
        showError(p.error || 'Transcription failed.');
        const a=document.createElement('a');a.href=base()+'log';a.textContent=' Open processing log';$('message').append(a);
      }else{
        clearTimeout(slowLoad);
        const [step,...timing]=jobTiming(p).split(' · ');
        loadingScene(p.title,step,timing.join(' · '));
        timer=setTimeout(()=>refresh().catch(failed),2200);
      }
      return;
    }
    score=p.score; version=p.version;
    stage='Drawing your sheet music';
    if(!$('loadingScene').hidden)loadingScene(p.title,stage);
    $('workspace').querySelector('.layout').hidden=false;
    recording.src=base()+'audio'; recording.volume=Number($('mix').value)/100;
    $('scrub').max=endSec(); $('total').textContent=time(endSec());
    showScoreDetails();
    await render(); if(gen!==generation)return;
    clearTimeout(slowLoad); endLoading();
    busy=false; selected=score.notes[0]?.id; fields(); highlight(); paintClock();
    $('saved').textContent=version?'Saved · r'+version:'Model original';
  };
  await refresh().catch(failed);
}
function save(notes,nextId,action='edit') {
  if(busy || !score)return;
  pendingSave=saveRevision(notes,nextId,action).finally(()=>{pendingSave=null;});
  return pendingSave;
}
async function saveRevision(notes, nextId, action) {
  if(busy || !score)return;
  busy=true; const previous=score.notes, previousId=selected;
  pause();clearError();fields();$('saved').textContent='Saving…';
  let committed=false;
  try{
    const result=await api(base()+'score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({version,notes})});
    committed=true;
    if(action==='undo'){past.pop();future.push([previous,previousId]);}
    else if(action==='redo'){future.pop();past.push([previous,previousId]);}
    else {past.push([previous,previousId]);if(past.length>60)past.shift();future=[];}
    score.notes=[...notes].sort((a,b)=>a.start-b.start);version=result.version;
    selected=nextId || score.notes[0]?.id; $('scrub').max=endSec(); $('total').textContent=time(endSec());
    showScoreDetails();
    await render();
    $('saved').textContent='Saved · r'+version;
  } catch(e) {showError(e);$('saved').textContent=committed?'Saved · reload to redraw':'Edit not saved';}
  finally {busy=false;fields();highlight();}
}
function applyDraft() {
  const n=byId(); if(!n || busy)return;
  const pitch=parsePitch($('pitch').value), bar=Number($('bar').value), beat=Number($('beat').value);
  const start=Math.round((bar-1)*ticksPerBar+(beat-1)*ticksPerBeat), length=Math.round(Number($('length').value)*ticksPerBeat);
  if(!Number.isInteger(bar)||bar<1||beat<1||beat>=5||length<=0||start<0)throw Error('Check the bar, beat and positive note length.');
  const end=start+length;
  if(start===n.start && end===n.end && pitch===n.pitch)return;
  const delta=$('ripple').checked?end-n.end:0;
  const changed=score.notes.map(x=>x.id===n.id?{...x,start,end,pitch}:delta && x.start>=n.end?{...x,start:x.start+delta,end:x.end+delta}:x);
  return save(changed,n.id);
}
$('editor').onsubmit=e=>{e.preventDefault();try{applyDraft()}catch(err){showError(err)}};
for(const b of document.querySelectorAll('[data-offset]'))b.onclick=()=>{
  try {const p=parsePitch($('pitch').value)+Number(b.dataset.offset);if(p<0||p>127)throw Error('Pitch is outside the MIDI range.');
    $('pitch').value=pitchName(p);preview({...byId(),pitch:p}).catch(showError);applyDraft();}catch(e){showError(e);}
};
function moveDraft(dx) {
  const n=byId(),t=n.start+dx;if(t<0)throw Error('The note is already at the start of the score.');
  $('bar').value=1+Math.floor(t/ticksPerBar);$('beat').value=+(1+t%ticksPerBar/ticksPerBeat).toFixed(5);return applyDraft();
}
$('earlier').onclick=()=>{try{moveDraft(-step())}catch(e){showError(e)}};
$('later').onclick=()=>{try{moveDraft(step())}catch(e){showError(e)}};
$('undo').onclick=()=>{if(past.length)save(...past.at(-1),'undo')};
$('redo').onclick=()=>{if(future.length)save(...future.at(-1),'redo')};
$('delete').onclick=()=>{const i=score.notes.findIndex(n=>n.id===selected);save(score.notes.filter(n=>n.id!==selected),score.notes[i+1]?.id||score.notes[i-1]?.id)};
$('insert').onclick=()=>{
  const t=Math.max(0,Math.round((pos-score.phase)*score.bpm*ticksPerBeat/(60*step()))*step());
  const near=byId();const start=near && Math.abs(pos-startSec(near.start))<.001?near.start:t;
  const note={id:crypto.randomUUID(),pitch:near?.pitch??72,start,end:start+step(),velocity:near?.velocity??80};
  const notes=score.notes.map(n=>$('ripple').checked && n.start>=start?{...n,start:n.start+step(),end:n.end+step()}:n);
  save([...notes,note],note.id);
};
$('split').onclick=()=>{
  const n=byId(),len=n.end-n.start;
  const unit=(n.start%3||n.end%3)?8:3,mid=n.start+Math.round(len/2/unit)*unit;
  if(mid<=n.start||mid>=n.end)return showError(Error('This note is too short to split on its grid.'));
  const second={...n,id:crypto.randomUUID(),start:mid};
  save([...score.notes.map(x=>x.id===n.id?{...x,end:mid}:x),second],second.id);
};
$('join').onclick=()=>{
  const n=byId(),next=score.notes[score.notes.indexOf(n)+1];
  if(!next||next.pitch!==n.pitch)return showError(Error('Join needs two consecutive notes of the same pitch.'));
  save(score.notes.filter(x=>x.id!==next.id).map(x=>x.id===n.id?{...x,end:next.end}:x),n.id);
};
let drag;
$('paper').addEventListener('pointerdown',e=>{
  const g=e.target.closest('g.note[data-uid]');if(!g||busy||e.button!==0)return;
  const box=g.closest('.measure')?.getBoundingClientRect();
  drag={uid:g.dataset.uid,x:e.clientX,y:e.clientY,width:Math.max(45,box?.width||160)};
});
window.addEventListener('pointercancel',()=>drag=null);
window.addEventListener('pointerup',e=>{
  if(!drag)return;
  const d=drag;drag=null;const dx=e.clientX-d.x,dy=e.clientY-d.y;
  const moved=Math.abs(dx)>6||Math.abs(dy)>6;
  selectNote(d.uid,!moved);
  if(!moved)return;
  try{
    const n=byId();const pitch=Math.max(0,Math.min(127,n.pitch-Math.round(dy/7)));
    $('pitch').value=pitchName(pitch);moveDraft(Math.round(dx*ticksPerBar/d.width/step())*step());
  }catch(err){showError(err)}
});

async function prepareSound(){
  if(!audioContext)audioContext=new AudioContext();
  await audioContext.resume();
  bank ??= fetch('/samples/manifest.json').then(r=>r.ok?r.json():{samples:[]}).then(m=>m.samples);
  return await bank;
}
async function sample(m){
  const files=await prepareSound();
  if(!files.length)return [null,m];
  const f=files.reduce((a,b)=>Math.abs(m-a.midi)<=Math.abs(m-b.midi)?a:b);
  if(!buffers.has(f.name))buffers.set(f.name,fetch('/samples/'+f.name).then(r=>r.arrayBuffer()).then(b=>audioContext.decodeAudioData(b)).catch(e=>{buffers.delete(f.name);throw e}));
  return [await buffers.get(f.name),f.midi];
}
function sound(n,buf,anchor,start,duration){
  const ctx=audioContext,node=buf?ctx.createBufferSource():ctx.createOscillator(),amp=ctx.createGain();
  if(buf){node.buffer=buf;node.playbackRate.value=2**((n.pitch-anchor)/12)}
  else{node.type="triangle";node.frequency.value=440*2**((n.pitch-69)/12)}
  amp.gain.setValueAtTime(.62*n.velocity/90,start);amp.gain.setTargetAtTime(.0001,start+duration,.027);
  node.connect(amp).connect(ctx.destination);node.start(start);node.stop(start+duration+.14);
  voices.add(node);node.onended=()=>{voices.delete(node);node.disconnect();amp.disconnect()};
}
async function preview(n) {const [b,m]=await sample(n.pitch);if(!playing)sound(n,b,m,audioContext.currentTime,.30);}
function current(){return playing?Math.min(endSec(),Math.max(0,(audioContext.currentTime-clockStart)*rate())):pos}
function paintClock(){
  if(!score)return;
  const t=current();$('clock').textContent=time(t);$('scrub').value=t;
  $('scrub').style.setProperty('--fill', (100*t/endSec())+'%');
}
function pause(){
  if(playing)pos=current();
  playing=false;playbackToken++;recording.pause();$('play').textContent='▶';$('play').ariaLabel='Play';
  for(const n of voices){try{n.stop()}catch{}}
  document.querySelectorAll('.sounding').forEach(g=>g.classList.remove('sounding'));
}
async function play(){
  if(!score||playing)return;
  const token=++playbackToken;
  if(pos>=endSec()-.03)pos=0;
  const pitches=[...new Set(score.notes.map(n=>n.pitch))];
  const loaded=new Map(await Promise.all(pitches.map(async p=>[p,await sample(p)])));
  if(token!==playbackToken)return;
  await prepareSound();if(token!==playbackToken)return;
  recording.playbackRate=rate();recording.currentTime=Math.min(pos,score.duration);
  if(pos<score.duration)await recording.play();
  if(token!==playbackToken){recording.pause();return;}
  clockStart=audioContext.currentTime-pos/rate();playing=true;
  $('play').textContent='Ⅱ';$('play').ariaLabel='Pause';
  index=score.notes.findIndex(n=>startSec(n.end)>pos);if(index<0)index=score.notes.length;
  let lastId,lastSync=audioContext.currentTime;
  function pump(){
    if(!playing||token!==playbackToken)return;
    const t=current();
    while(index<score.notes.length && startSec(score.notes[index].start)<=t+.13*rate()){
      const n=score.notes[index++],start=Math.max(t,startSec(n.start)),duration=(startSec(n.end)-start)/rate();
      if(duration>0)sound(n,...loaded.get(n.pitch),audioContext.currentTime+(start-t)/rate(),duration);
    }
    if(audioContext.currentTime-lastSync>1){
      if(!recording.paused && Math.abs(recording.currentTime-t)>.09)recording.currentTime=t;
      lastSync=audioContext.currentTime;
    }
    const n=score.notes.slice(Math.max(0,index-12),index).find(n=>startSec(n.start)<=t && t<startSec(n.end));
    if(lastId!==n?.id){for(const g of glyphs.get(lastId)||[])g.classList.remove('sounding');for(const g of glyphs.get(n?.id)||[])g.classList.add('sounding');lastId=n?.id;}
    paintClock();if(t>=endSec()-.004){pause();return;}setTimeout(pump,30);
  }pump();
}
$('play').onclick=()=>playing?pause():play().catch(showError);
$('scrub').onchange=()=>seek(Number($('scrub').value));
let previousRate=1;
$('speed').onchange=()=>{const target=rate(),resume=playing;if(resume){$('speed').value=String(previousRate);pause();$('speed').value=String(target);}previousRate=target;if(resume)play().catch(showError)};
function updateMix() {
  recording.volume=Number($('mix').value)/100;
  $('mix').style.setProperty('--fill', $('mix').value+'%');
}
$('mix').oninput=updateMix;
updateMix();
$('scrub').addEventListener('input',()=>{
  $('scrub').style.setProperty('--fill',(100*Number($('scrub').value)/Number($('scrub').max))+'%');
});
function showExports(open) {
  $('exportMenu').hidden=!open;
  $('exports').setAttribute('aria-expanded', String(open));
}
$('exports').onclick=()=>showExports($('exportMenu').hidden);
$('exportMenu').onclick=()=>showExports(false);
document.addEventListener('click',e=>{
  if (!$('exportMenu').hidden && !e.target.closest('.document')) showExports(false);
});
document.addEventListener('keydown',e=>{
  if(e.key==='Escape' && !$('exportMenu').hidden){showExports(false);$('exports').focus();}
});
function showImport() {
  $('uploadError').textContent='';
  $('upload').showModal();
}
$('uploadOpen').onclick=showImport;
const dropArea=$('dropArea');
dropArea.ondragover=e=>{if(e.dataTransfer.types.includes('Files')){e.preventDefault();dropArea.classList.add('drop-active');}};
dropArea.ondragleave=e=>{if(!dropArea.contains(e.relatedTarget))dropArea.classList.remove('drop-active');};
dropArea.ondrop=e=>{
  e.preventDefault();dropArea.classList.remove('drop-active');
  const files=e.dataTransfer.files;
  if(files.length!==1 || !/\.mp3$/i.test(files[0].name)){showError(Error('Choose one MP3 file.'));return;}
  clearError();$('file').files=files;showImport();
};
$('cancel').onclick=()=>$('upload').close();
$('uploadForm').onsubmit=async e=>{
  e.preventDefault();const f=$('file').files[0];if(!f)return;
  $('send').disabled=true;$('uploadError').textContent='Uploading…';
  try{const p=await api('/api/projects?'+new URLSearchParams({name:f.name,language:$('language').value}),{method:'POST',body:f});
    $('upload').close();location.hash=p.id;
  }catch(err){$('uploadError').textContent=err.message;}finally{$('send').disabled=false}
};
document.addEventListener('keydown',e=>{
  if(e.target.closest('input,textarea,select,button,dialog')||busy||!score)return;
  if(e.code==='Space'){e.preventDefault();$('play').click();}
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();$(e.shiftKey?'redo':'undo').click();}
  if(['ArrowLeft','ArrowRight'].includes(e.key)){
    const i=score.notes.findIndex(n=>n.id===selected)+(e.key==='ArrowLeft'?-1:1);
    if(score.notes[i]){e.preventDefault();selectNote(score.notes[i].id);}
  }
});
async function route(navigated=false){
  if(pendingSave)await pendingSave;
  const id=location.hash.slice(1);
  const home=!id || id==='library';
  if(!home && !$('home').hidden)libraryScroll=window.scrollY;
  $('home').hidden=!home; $('localLabel').hidden=!home;
  $('workspace').hidden=home; $('backToLibrary').hidden=home; $('credits').hidden=home;
  endLoading();
  showExports(false);
  if(home){
    await showLibrary();
    if(navigated && !$('home').hidden){
      window.scrollTo(0,libraryScroll);
      $('homeTitle').focus({preventScroll:true});
    }
  }else{
    window.scrollTo(0,0);
    if(navigated)$('title').focus({preventScroll:true});
    await openProject(id);
  }
}
window.addEventListener('hashchange',()=>route(true).catch(showError));
await route().catch(showError);

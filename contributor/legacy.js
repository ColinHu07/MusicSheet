const $ = id => document.getElementById(id);
const audio = $('audio');
const relations = ['New attack','Same pitch, new attack','Same syllable, new pitch','Continue same note','Non-vocal / instrumental','Uncertain'];
let state, key, url, ticket=0;
const status = text => $('status').textContent=text;
const phrase = () => state.phrases[state.current];
function save(){
  try{localStorage.setItem(key,JSON.stringify(state));status('Saved on this device.')}
  catch{status('Browser storage unavailable. Export your annotations to keep them.')}
}
function choose(){
  const length=Math.min(20,audio.duration);
  const anchors=state.score.filter(n=>n.time+state.offset>=0 && n.time+state.offset<=audio.duration-length);
  const start=anchors.length ? anchors[Math.floor(Math.random()*anchors.length)].time+state.offset : Math.random()*(audio.duration-length);
  const end=Math.min(audio.duration,start+length);
  const notes=state.score.map(n=>({...n,time:n.time+state.offset})).filter(n=>n.time>=start && n.time<end)
    .map(n=>({...n,syllable:'',relation:'New attack'}));
  state.current=state.phrases.push({start,end,notes,reviewed:false})-1;
  save();show();
}
function parseScore(text){
  const doc=new DOMParser().parseFromString(text,'application/xml');
  const parts=doc.querySelectorAll('score-partwise > part');
  if(parts.length!==1 || doc.querySelector('parsererror, backup, forward, chord, repeat, ending'))throw Error('Use an unfolded, single-part, single-voice MusicXML melody.');
  let divisions=1,bpm=120,time=0,last;
  const notes=[];
  for(const measure of parts[0].children){
    for(const item of measure.children){
      const tempo=item.matches('sound[tempo]')? item:item.querySelector('sound[tempo]');
      if(tempo)bpm=Number(tempo.getAttribute('tempo'));
      if(item.matches('attributes') && item.querySelector('divisions'))divisions=Number(item.querySelector('divisions').textContent);
      if(!item.matches('note') || item.querySelector('grace'))continue;
      const length=Number(item.querySelector('duration')?.textContent)/divisions*60/bpm;
      if(!(length>0 && Number.isFinite(length)))throw Error('Unsupported note duration or tempo.');
      const p=item.querySelector('pitch');
      if(p){
        const step=p.querySelector('step').textContent, alter=Number(p.querySelector('alter')?.textContent||0);
        const pitch=step+({'-2':'bb','-1':'b','0':'','1':'#','2':'##'}[alter]??'?')+p.querySelector('octave').textContent;
        if(item.querySelector('tie[type="stop"]') && last?.pitch===pitch)last.duration+=length;
        else {last={time,pitch,duration:length};notes.push(last)}
      }else last=null;
      time+=length;
    }
  }
  if(!notes.length)throw Error('No pitched notes in that file.');
  return notes;
}
$('file').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  const current=++ticket; audio.pause();$('panel').hidden=true;
  try{
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');
    if(ticket!==current)return;
    key='musicsheet:phrase:v1:'+hash;
    state=JSON.parse(localStorage.getItem(key)||'null')||{version:1,song:file.name,audio_sha256:hash,score:[],offset:0,phrases:[],current:0};
    if(url)URL.revokeObjectURL(url);
    url=URL.createObjectURL(file);
    audio.onloadedmetadata=()=>{
      if(current!==ticket)return;
      if(!Number.isFinite(audio.duration)||!audio.duration)return status('Unsupported recording duration.');
      $('panel').hidden=false;$('title').textContent=state.song;$('offset').value=state.offset;$('xml').value='';
      if(state.phrases.length)show();else choose();
    };
    audio.onerror=()=>status('This browser could not decode the recording.');
    audio.src=url;
  }catch(err){status(err.message)}
};
$('xml').onchange=async e=>{
  const selected=state;
  try{const score=parseScore(await e.target.files[0].text());if(selected!==state)return;state.score=score;choose()}
  catch(err){status(err.message)}
};
$('offset').onchange=()=>{state.offset=Number($('offset').value)||0;choose()};
$('another').onclick=choose;
$('phrases').onchange=()=>{state.current=Number($('phrases').value);save();show()};
function draw(){
  const notes=phrase().notes.filter(n=>/^([A-G])([#b]{0,2})([0-8])$/.test(n.pitch));
  $('staff').replaceChildren();
  if(!notes.length)return;
  const tokens=notes.map(n=>{
    const [,step,acc,oct]=n.pitch.match(/^([A-G])([#b]{0,2})([0-8])$/);
    const o=Number(oct);
    const pitch=acc.replaceAll('#','^').replaceAll('b','_')+(o<5?step:step.toLowerCase())+(o<4?',':"'").repeat(o<4?4-o:Math.max(0,o-5));
    const label=n.syllable.replace(/[^\p{L}\p{M}\p{N}' -]/gu,'');
    return (label?'"^'+label+'"':'')+pitch;
  });
  ABCJS.renderAbc('staff','X:1\nT:Pitch map\nM:none\nL:1/4\nK:C\n'+tokens.join(' '),{responsive:'resize',staffwidth:720,clickListener:(_,__,___,____,analysis)=>{const i=analysis.clickedName==='note'?analysis.selectableElement?.counters?.note:undefined;if(notes[i])audio.currentTime=notes[i].time}});
}
function show(){
  audio.pause();audio.currentTime=phrase().start;
  $('range').textContent='Phrase '+(state.current+1)+' Â· '+phrase().start.toFixed(2)+'â€“'+phrase().end.toFixed(2)+' s';
  $('phrases').replaceChildren(...state.phrases.map((p,i)=>new Option('Phrase '+(i+1)+(p.reviewed?' ✓':''),String(i))));
  $('phrases').value=String(state.current);
  $('done').checked=phrase().reviewed;$('rows').replaceChildren();
  for(const n of phrase().notes){
    const row=$('rows').insertRow();const jump=document.createElement('button');jump.textContent=n.time.toFixed(2);jump.onclick=()=>audio.currentTime=n.time;row.insertCell().append(jump);
    for(const k of ['pitch','syllable']){
      const input=document.createElement('input');input.value=n[k];input.ariaLabel=k;input.placeholder=k==='pitch'?'C#4':'Syllable';row.insertCell().append(input);
      input.oninput=()=>{n[k]=input.value;save();draw()};
    }
    const sel=document.createElement('select');for(const r of relations)sel.add(new Option(r,r));sel.value=n.relation;sel.onchange=()=>{n.relation=sel.value;save()};row.insertCell().append(sel);
    const remove=document.createElement('button');remove.textContent='Remove';remove.onclick=()=>{phrase().notes.splice(phrase().notes.indexOf(n),1);save();show()};row.insertCell().append(remove);
  }
  draw();
}
$('add').onclick=()=>{const time=audio.currentTime;if(time<phrase().start || time>=phrase().end)return status('Seek inside the selected phrase first.');phrase().notes.push({time,pitch:'',syllable:'',relation:'New attack'});phrase().notes.sort((a,b)=>a.time-b.time);save();show();audio.currentTime=time};
$('done').onchange=()=>{phrase().reviewed=$('done').checked;save()};
$('speed').onchange=()=>audio.playbackRate=Number($('speed').value);
$('replay').onclick=()=>{audio.currentTime=phrase().start;audio.play().catch(e=>status(e.message))};
audio.ontimeupdate=()=>{if(!audio.paused && $('loop').checked && audio.currentTime>=phrase().end)audio.currentTime=phrase().start};
audio.onended=()=>{if($('loop').checked)$('replay').click()};
$('export').onclick=()=>{
  const {score,...review}=state;
  const link=document.createElement('a');link.download=state.song+'.annotations.json';link.href=URL.createObjectURL(new Blob([JSON.stringify(review,null,2)],{type:'application/json'}));link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};

const dbReady=new Promise((resolve,reject)=>{
  const req=indexedDB.open('musicsheet-contributor-v2',1);
  req.onupgradeneeded=()=>{for(const store of ['songs','settings'])req.result.createObjectStore(store)};
  req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);
});
async function dbGet(store,key){
  const db=await dbReady;
  return new Promise((resolve,reject)=>{
    const objects=db.transaction(store).objectStore(store);
    const req=key===undefined?objects.getAll():objects.get(key);
    req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);
  });
}
async function writeReview(expected,data){
  const db=await dbReady;
  return new Promise((resolve,reject)=>{
    const tx=db.transaction('settings','readwrite'),store=tx.objectStore('settings'),req=store.get('review');
    let result;
    req.onsuccess=()=>{
      if((req.result?.revision||0)!==expected){result={status:409};return}
      store.put({revision:expected+1,data},'review');result={ok:true,revision:expected+1};
    };
    tx.oncomplete=()=>resolve(result);tx.onabort=()=>reject(tx.error);
  });
}
function playable(entry){return {...entry.clip,audio_url:URL.createObjectURL(entry.file)}}
async function storedClips(){return (await dbGet('songs')).sort((a,b)=>a.created-b.created).map(playable)}
function storeSong(db,id,entry){
  return new Promise((resolve,reject)=>{
    const tx=db.transaction('songs','readwrite');tx.objectStore('songs').add(entry,id);
    tx.oncomplete=resolve;tx.onabort=()=>reject(tx.error);
  });
}
function chooseWindow(buffer,id){
  let seed=parseInt(id.slice(0,8),16);
  const length=Math.min(20,buffer.duration),samples=buffer.getChannelData(0);
  // Choose randomly among audible windows so leading/trailing silence is less likely.
  let start=0;
  for(let i=0;i<16;i++){
    seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    start=seed/4294967296*(buffer.duration-length);
    let audible=0;
    for(let j=0;j<200;j++)if(Math.abs(samples[Math.floor((start+j*length/200)*buffer.sampleRate)])>.008)audible++;
    if(audible>50)break;
  }
  return {start,end:start+length};
}
async function analyze(buffer,start,end){
  const render=new OfflineAudioContext(1,Math.ceil((end-start)*8000),8000);
  const source=render.createBufferSource();source.buffer=buffer;source.connect(render.destination);source.start(0,start,end-start);
  const samples=(await render.startRendering()).getChannelData(0);
  return new Promise((resolve,reject)=>{
    const worker=new Worker('pitch-worker.js');
    worker.onmessage=e=>{worker.terminate();resolve(e.data)};
    worker.onerror=e=>{worker.terminate();reject(Error(e.message))};
    worker.postMessage(samples);
  });
}
function notation(notes){
  const images=[];
  for(let offset=0;offset<notes.length;offset+=12){
    const group=notes.slice(offset,offset+12);
    const holder=document.createElement('div');holder.style.cssText='position:absolute;left:-10000px;width:900px';document.body.append(holder);
    try{
      const code=group.map(n=>{
        const octave=Math.floor(n.midi/12)-1;
        const name=['C','^C','D','^D','E','F','^F','G','^G','A','^A','B'][n.midi%12];
        return (octave>=5?name.toLowerCase():name)+(octave>=5?"'".repeat(octave-5):','.repeat(Math.max(0,4-octave)));
      });
      ABCJS.renderAbc(holder,'X:1\nM:none\nL:1/4\nK:C\n'+code.join(' '),{staffwidth:800,add_classes:true,paddingtop:65});
      const svg=holder.querySelector('svg'),box=svg.getBoundingClientRect();
      const heads=Array.from(svg.querySelectorAll('.abcjs-note')).map((el,i)=>{
        const rect=el.querySelector('.abcjs-notehead')?.getBoundingClientRect()||el.getBoundingClientRect();
        const x=rect.left-box.left+rect.width/2,y=rect.top-box.top+rect.height/2;
        return {id:group[i].id,number:group[i].index,x,y,label_y:18+(i%2)*23};
      });
      images.push({width:box.width,height:box.height,url:'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(new XMLSerializer().serializeToString(svg)),notes:heads});
    }finally{holder.remove()}
  }
  return images;
}
async function importBatch(files){
  const context=new AudioContext(),db=await dbReady,errors=[];
  try{
    for(const [i,file] of files.entries()){
      try{
        announce('Preparing '+(i+1)+' of '+files.length+' · '+file.name);
        const bytes=await file.arrayBuffer();
        const id=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
        if(clipMap.has(id))continue;
        const existing=await dbGet('songs',id);
        if(existing){clips.push(playable(existing));clipMap.set(id,clips.at(-1));continue}
        const buffer=await context.decodeAudioData(bytes);
        const {start,end}=chooseWindow(buffer,id),events=await analyze(buffer,start,end);
        const notes=events.map((e,i)=>({...e,id:id+':'+(i+1),index:i+1,start_s:start+e.start_s,end_s:Math.min(end,start+e.end_s),pitch:['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'][e.midi%12]+(Math.floor(e.midi/12)-1)}));
        const clip={id,song_id:id,title:file.name.replace(/\.[^.]+$/,''),start_s:start,end_s:end,notes,images:notation(notes),source_hashes:{audio_sha256:id},score_role:'Estimated pitch map',alignment_note:'Note spacing is schematic. Check pitch, attacks and instrumental sounds by ear.'};
        const entry={file,clip,created:Date.now()};await storeSong(db,id,entry);clips.push(playable(entry));clipMap.set(id,clips.at(-1));
      }catch(e){errors.push(file.name+' — '+e.message)}
      render();
    }
    $('overallCount').textContent=clips.length+' songs';announce(errors.join(' · '),!!errors.length);
  }finally{await context.close()}
}
$('songs').disabled=true;
init().then(()=>{$('songs').disabled=false;if(!clips.length)$('progressText').textContent='Import songs to begin.'});
$('songs').onchange=async e=>{
  const input=e.target;input.disabled=true;
  try{await importBatch(Array.from(input.files))}catch(err){announce(err.message,true)}
  finally{input.value='';input.disabled=false}
};

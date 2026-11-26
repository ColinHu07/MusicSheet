let mapLibrary;
async function originalLibrary(){
  if(mapLibrary)return mapLibrary;
  mapLibrary=await dbGet('settings','originalMaps');
  if(!mapLibrary && location.hostname==='127.0.0.1' && location.port==='18892'){
    const response=await fetch('/original-maps.json');
    if(response.ok)mapLibrary=await response.json();
  }
  return mapLibrary;
}
document.getElementById('mapFile').onchange=async event=>{
  const input=event.target;
  try{
    const library=JSON.parse(await input.files[0].text());
    if(library.format!=='original-score-maps-1'||!library.pairs)throw Error('Choose an original score map library.');
    const db=await dbReady;
    await new Promise((yes,no)=>{
      const tx=db.transaction('settings','readwrite');
      tx.objectStore('settings').put(library,'originalMaps');tx.oncomplete=yes;tx.onabort=()=>no(tx.error);
    });
    mapLibrary=library;announce('Original maps loaded. Import a matching score or pick another passage.');
  }catch(error){announce(error.message,true)}finally{input.value=''}
};
async function originalPassage(source,id){
  const map=(await originalLibrary())?.pairs[source.song_id];
  if(!map)return null;
  const pages=source.fullScorePages||source.images;
  const pdf=source.source_hashes.score_sha256===map.pdf;
  const pageIndex=map.pages.map((hash,i)=>pdf?i:(source.pageHashes||[]).indexOf(hash));
  const options=[];
  for(let i=0;i<map.systems.length;i++){
    const selection=[],ids=new Set();
    for(const system of map.systems.slice(i)){
      if(!pages[pageIndex[system.page]])break;
      selection.push(system);
      for(const [key] of system.pins)ids.add(key);
      if(ids.size>=30){options.push(selection);break}
    }
  }
  if(!options.length)return null;
  const fresh=options.filter(systems=>!systems.some(s=>clips.some(c=>c.song_id===source.song_id&&c.originalKeys?.some(key=>s.pins.some(([k])=>k===key)))));
  const pool=fresh.length?fresh:options;
  const chosen=pool[Math.floor(Math.random()*pool.length)];
  const keys=[...new Set(chosen.flatMap(s=>s.pins.map(p=>p[0])))].sort((a,b)=>a-b);
  const notes=keys.map((key,i)=>{
    const [start_s,end_s,pitch,midi,measure]=map.notes[key];
    return {id:id+':ref:'+key,index:i+1,start_s,end_s,pitch:pitch||['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'][midi%12]+(Math.floor(midi/12)-1),midi,measure};
  });
  const images=[];
  for(const s of chosen){
    const img=new Image();img.src=pages[pageIndex[s.page]].url;await img.decode();
    const w=img.naturalWidth,h=img.naturalHeight,y=Math.floor(s.top*h),height=Math.ceil((s.bottom-s.top)*h);
    const canvas=document.createElement('canvas');canvas.width=w;canvas.height=height+72;
    const ctx=canvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,w,canvas.height);
    ctx.drawImage(img,0,y,w,height,0,72,w,height);
    const pins=s.pins.map(([key,x,py],i)=>({id:notes[keys.indexOf(key)].id,number:keys.indexOf(key)+1,x:x*w,y:py*h-y+72,label_y:12+i%3*22}));
    images.push({width:w,height:canvas.height,url:canvas.toDataURL(),notes:pins});
  }
  return {...source,id,originalKeys:keys,fullScorePages:pages,notes,images,start_s:notes[0].start_s,end_s:notes.at(-1).end_s,alignment_note:'Original score map · tied heads select one annotation row.'};
}

async function resolvePrintedPassage(source,buffer,id){
  const original=await originalPassage(source,id);
  if(original)return original;
  if(await originalLibrary()){
    throw Error('No matching original note map for this recording and score. Select the corresponding original PDF or original page images.');
  }
  const result=await printedPassage(source,buffer,id);
  result.alignment_note='Raster detector · estimated printed heads and audio timing. Load original maps to use the 17-song extractor results.';
  return result;
}

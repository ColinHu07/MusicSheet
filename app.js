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
async function storedClips(){
  const entries=(await dbGet('songs')).sort((a,b)=>a.created-b.created);
  const db=await dbReady;
  for(const entry of entries){
    const clip=entry.clip;
    if(clip.pdfReference || clip.notes.length || clip.pitchVersion>=2)continue;
    try{
      const decoder=new OfflineAudioContext(1,1,44100);
      const sound=await decoder.decodeAudioData(await entry.file.arrayBuffer());
      const events=await analyze(sound,clip.start_s,clip.end_s);
      const notes=makeNotes(events,clip.id,clip.start_s,clip.end_s);
      Object.assign(clip,{notes,images:notation(notes),pitchVersion:2});
      await new Promise((resolve,reject)=>{
        const tx=db.transaction('songs','readwrite');
        tx.objectStore('songs').put(entry,clip.id);
        tx.oncomplete=resolve;tx.onabort=()=>reject(tx.error);
      });
    }catch(error){console.warn('Could not refresh empty pitch map',error)}
  }
  return entries.filter(entry=>entry.clip.notes.length || entry.clip.id!==entry.clip.song_id || !entries.some(other=>other.clip.song_id===entry.clip.id && other.clip.source_hashes?.score_sha256 && other.clip.id!==entry.clip.id)).map(playable);
}
function storeSong(db,id,entry){
  return new Promise((resolve,reject)=>{
    const tx=db.transaction('songs','readwrite');tx.objectStore('songs').add(entry,id);
    tx.oncomplete=resolve;tx.onabort=()=>reject(tx.error);
  });
}
function chooseWindow(buffer,id){
  let seed=parseInt(id.slice(0,8),16);
  const length=Math.min(8,buffer.duration),samples=buffer.getChannelData(0);
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
    const worker=new Worker('pitch-worker.js?v=2');
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
function makeNotes(events,id,start,end){
  return events.map((e,i)=>({...e,id:id+':'+(i+1),index:i+1,start_s:start+e.start_s,end_s:Math.min(end,start+e.end_s),pitch:['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'][e.midi%12]+(Math.floor(e.midi/12)-1)}));
}
async function importBatch(files){
  const context=new AudioContext(),db=await dbReady,errors=[];
  try{
    for(const [i,file] of files.entries()){
      try{
        announce('Preparing '+(i+1)+' of '+files.length+' Â· '+file.name);
        const bytes=await file.arrayBuffer();
        const id=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
        if(clipMap.has(id))continue;
        const existing=await dbGet('songs',id);
        if(existing && clips.some(c=>c.song_id===id))continue;
        if(existing){clips.push(playable(existing));clipMap.set(id,clips.at(-1));continue}
        const buffer=await context.decodeAudioData(bytes);
        const {start,end}=chooseWindow(buffer,id),events=[];
        const notes=makeNotes(events,id,start,end);
        const clip={pitchVersion:2,id,song_id:id,title:file.name.replace(/\.[^.]+$/,''),start_s:start,end_s:end,notes,images:notation(notes),source_hashes:{audio_sha256:id},score_role:'Waiting for reference score',alignment_note:'Import the sheet that goes with this recording.'};
        const entry={file,clip,created:Date.now()};await storeSong(db,id,entry);clips.push(playable(entry));clipMap.set(id,clips.at(-1));
      }catch(e){errors.push(file.name+' â€” '+e.message)}
      render();
    }
    $('overallCount').textContent=clips.length+' songs';announce(errors.join(' Â· '),!!errors.length);
  }finally{await context.close()}
}
$('songs').disabled=true;
init().then(()=>{$('songs').disabled=false;if(!clips.length)$('progressText').textContent='Import songs to begin.'});
$('songs').onchange=async e=>{
  const input=e.target;input.disabled=true;
  try{await importBatch(Array.from(input.files))}catch(err){announce(err.message,true)}
  finally{input.value='';input.disabled=false}
};

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
        else {last={time,pitch,duration:length,measure:measure.getAttribute('number'),midi:12*(Number(p.querySelector('octave').textContent)+1)+({C:0,D:2,E:4,F:5,G:7,A:9,B:11}[step])+alter};notes.push(last)}
      }else last=null;
      time+=length;
    }
  }
  if(!notes.length)throw Error('No pitched notes in that file.');
  return notes;
}
async function saveSheet(entry){
  const db=await dbReady;
  await new Promise((resolve,reject)=>{
    const tx=db.transaction('songs','readwrite');
    tx.objectStore('songs').put(entry,entry.clip.id);
    tx.oncomplete=resolve;tx.onabort=()=>reject(tx.error);
  });
}
async function pdfImages(file){
  const pdfjs=await import('./vendor/pdf.min.js');
  pdfjs.GlobalWorkerOptions.workerSrc=new URL('./vendor/pdf.worker.min.js',location.href).href;
  const pdf=await pdfjs.getDocument({data:await file.arrayBuffer(),isEvalSupported:false}).promise;
  const images=[];
  try{
    for(let i=1;i<=pdf.numPages;i++){
      const page=await pdf.getPage(i),base=page.getViewport({scale:1});
      const viewport=page.getViewport({scale:1200/base.width});
      const canvas=document.createElement('canvas');
      canvas.width=Math.ceil(viewport.width);canvas.height=Math.ceil(viewport.height);
      await page.render({canvasContext:canvas.getContext('2d'),viewport}).promise;
      images.push({width:canvas.width,height:canvas.height,url:canvas.toDataURL('image/webp',.9),notes:[]});
      page.cleanup();
    }
  }finally{await pdf.destroy()}
  return images;
}
async function screenshotPages(files){
  const pages=[];
  for(const file of files){
    const url=await new Promise((resolve,reject)=>{
      const reader=new FileReader();
      reader.onload=()=>resolve(reader.result);
      reader.onerror=()=>reject(Error('Could not read '+file.name));
      reader.readAsDataURL(file);
    });
    const image=new Image();
    image.src=url;
    await image.decode();
    pages.push({width:image.naturalWidth,height:image.naturalHeight,url,notes:[]});
  }
  return pages;
}
async function attachScore(input,files,source){
  input.disabled=true;
  try{
    const offset=Number(input.closest('.row').querySelector('[data-offset]').value);
    if(!Number.isFinite(offset))throw Error('Enter the score start time in seconds.');
    const pages=Array.from(files).sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}));
    const file=pages[0], screenshots=pages.every(page=>/\.(png|jpe?g|webp)$/i.test(page.name));
    if(pages.length>1 && !screenshots)throw Error('Choose one score document or a group of screenshot pages.');
    let bytes=await file.arrayBuffer();
    if(pages.length>1){
      const digests=new Uint8Array(32*pages.length);
      for(const [index,page] of pages.entries()){
        const digest=await crypto.subtle.digest('SHA-256',await page.arrayBuffer());
        digests.set(new Uint8Array(digest),32*index);
      }
      bytes=digests;
    }
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
    const id=source.id+'-'+hash+'-'+offset;
    if(clipMap.has(id)){scrollToClip(clipMap.get(id));return}
    const entry=await dbGet('songs',source.id);
    const pdf=/\.pdf$/i.test(file.name);
    const clip={...source,id,notes:[],pdfReference:pdf||screenshots,pitchVersion:3,score_role:file.name,source_hashes:{...entry.clip.source_hashes,score_sha256:hash}};
    delete clip.writtenNotes;
    if(screenshots){
      clip.images=await screenshotPages(pages);
      clip.alignment_note='Score screenshots in filename order · click the printed notes you hear in the loop.';
    }else if(pdf){
      clip.images=await pdfImages(file);
      clip.alignment_note='Original PDF · select the printed notes you hear in the loop.';
    }else{
      const notes=parseScore(new TextDecoder().decode(bytes));
      clip.writtenNotes=notes;clip.scoreOffset=offset;
      const starts=notes.filter(n=>n.time+offset>=0);
      const media=new Audio();media.preload='metadata';media.src=source.audio_url;
      await new Promise((yes,no)=>{media.onloadedmetadata=yes;media.onerror=()=>no(Error('Could not read song duration'))});
      const duration=media.duration;media.removeAttribute('src');media.load();
      const eligible=starts.filter(n=>n.time+offset<duration);
      if(!eligible.length)throw Error('The score starts outside the recording. Check the offset.');
      const anchor=eligible.reduce((a,b)=>Math.abs(a.time+offset-clip.start_s)<=Math.abs(b.time+offset-clip.start_s)?a:b);
      clip.start_s=anchor.time+offset;clip.end_s=Math.min(duration,clip.start_s+8);
      clip.notes=eligible.filter(n=>n.time>=anchor.time && n.time+offset<clip.end_s).map((n,i)=>({...n,id:id+':'+(i+1),index:i+1,start_s:n.time+offset,end_s:Math.min(duration,n.time+offset+n.duration)}));
      clip.images=notation(clip.notes);
      clip.alignment_note='Written pitches from MusicXML · check playback alignment against the recording. Pitch map spacing is schematic.';
    }
    delete clip.phraseImages;
    if(clip.pdfReference){
      announce('Aligning printed score to recording…');
      await new Promise(resolve=>setTimeout(resolve,20));
      try{
        const decoder=new OfflineAudioContext(1,1,8000);
        Object.assign(clip,await printedPassage(clip,await decoder.decodeAudioData(await entry.file.arrayBuffer()),id));
      }catch(error){clip.alignment_note=error.message+' Mark printed notes manually.'}
    }
    const newEntry={...entry,created:Date.now(),clip};await saveSheet(newEntry);
    const live=playable(newEntry);if(source.id===source.song_id && !source.notes.length)clips=clips.filter(c=>c!==source);clips.push(live);clipMap.set(id,live);render();$('overallCount').textContent=new Set(clips.map(c=>c.song_id)).size+' songs';scrollToClip(live);announce('');
  }catch(e){announce(e.message,true)}finally{input.disabled=false;input.value=''}
}
document.addEventListener('change',event=>{
  const input=event.target;
  if(input.matches('[data-sheet]') && input.files.length)attachScore(input,input.files,clipMap.get(input.dataset.sheet));
});
$('scoreFirst').onchange=()=>{
  const file=$('scoreFirst').files[0];
  $('pairSongLabel').hidden=!file;
  if(file)announce('Choose the matching recording for '+file.name);
};
$('pairSong').onchange=async()=>{
  const song=$('pairSong').files[0],sheet=$('scoreFirst').files;
  if(!song||!sheet.length)return;
  $('pairSong').disabled=$('scoreFirst').disabled=true;
  try{
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await song.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');
    await importBatch([song]);
    const clip=clips.find(c=>c.song_id===hash);
    if(!clip)throw Error('Could not import the matching recording.');
    const input=document.querySelector('[data-sheet="'+clip.id+'"]');
    await attachScore(input,sheet,clip);
  }catch(error){announce(error.message,true)}
  finally{$('pairSong').disabled=$('scoreFirst').disabled=false;$('pairSong').value=$('scoreFirst').value='';$('pairSongLabel').hidden=true}
};
document.addEventListener('click',async event=>{
  if(!event.target.matches('[data-score-page] img'))return;
  const article=event.target.closest('[data-clip]'),clip=clipMap.get(article.dataset.clip);
  if(!clip.pdfReference)return;
  const t=audio.currentTime,b=bounds(clip);
  if(active!==clip || t<b.start || t>b.end){announce('Play this reference phrase and pause at the note you want to label.');return}
  audio.pause();
  const page=Number(event.target.closest('[data-score-page]').dataset.scorePage),rect=event.target.getBoundingClientRect(),image=clip.images[page];
  const index=clip.notes.length+1,id=clip.id+':'+crypto.randomUUID();
  const x=(event.clientX-rect.left)/rect.width*image.width,y=(event.clientY-rect.top)/rect.height*image.height;
  clip.notes.push({id,index,pitch:'',start_s:t,end_s:t});
  image.notes.push({id,number:index,x,y,label_y:Math.max(12,y-30)});
  try{
    const entry=await dbGet('songs',clip.id);
    const {audio_url,...persisted}=clip;entry.clip=persisted;await saveSheet(entry);
    render();selectNote(id);
  }catch(e){announce(e.message,true)}
});


async function estimatePhrase(clip,buffer){
  const events=await analyze(buffer,clip.start_s,clip.end_s);
  clip.notes=makeNotes(events,clip.id,clip.start_s,clip.end_s);
  clip.phraseImages=notation(clip.notes);
}
async function anotherPassage(source){
  const entry=await dbGet('songs',source.id);
  const decoder=new OfflineAudioContext(1,1,8000);
  const buffer=await decoder.decodeAudioData(await entry.file.arrayBuffer());
  const existing=clips.filter(c=>c.song_id===source.song_id);
  const gaps=[];
  let cursor=0;
  for(const {start,end} of existing.map(bounds).sort((a,b)=>a.start-b.start)){
    if(start>cursor)gaps.push([cursor,start]);
    cursor=Math.max(cursor,end);
  }
  if(cursor<buffer.duration)gaps.push([cursor,buffer.duration]);
  const id=crypto.randomUUID(), offset=source.scoreOffset||0;
  if(source.source_hashes.score_sha256 && !source.pdfReference && !source.writtenNotes){
    announce('Reimport this MusicXML score once to use its full melody for additional passages.');return;
  }
  if(source.pdfReference){
    announce('Reading printed noteheads and matching the recording…');
    await new Promise(resolve=>setTimeout(resolve,20));
    const clip=await printedPassage(entry.clip,buffer,id);
    const next={...entry,created:Date.now(),clip};await storeSong(await dbReady,id,next);
    const live=playable(next);clips.push(live);clipMap.set(id,live);render();scrollToClip(live);announce('');return;
  }
  const candidates=[];
  for(const [a,b] of gaps){
    for(let start=a;start+6<=b;start+=8){
      candidates.push({start,end:Math.min(start+40,b)});
    }
  }
  let clip;
  announce('Looking for a passage with at least 30 notes…');
  while(candidates.length){
    const index=Math.floor(Math.random()*candidates.length);
    const {start,end}=candidates.splice(index,1)[0];
    const notes=source.writtenNotes
      ? source.writtenNotes.filter(n=>n.time+offset>=start && n.time+offset<end).map((n,i)=>({...n,id:id+':'+(i+1),index:i+1,start_s:n.time+offset,end_s:Math.min(end,n.time+offset+n.duration)}))
      : makeNotes(await analyze(buffer,start,end),id,start,end);
    if(notes.length<30)continue;
    clip={...entry.clip,id,start_s:start,end_s:end,notes,images:[],pitchVersion:3};
    if(source.writtenNotes)clip.images=notation(notes);
    else{
      clip.phraseImages=notation(notes);
      if(source.pdfReference)clip.images=source.images.map(image=>({...image,notes:[]}));
    }
    break;
  }
  if(!clip){announce('No remaining passage yielded 30 notes.');return}
  announce('');
  const next={...entry,created:Date.now(),clip};await storeSong(await dbReady,id,next);
  const live=playable(next);clips.push(live);clipMap.set(id,live);render();scrollToClip(live);
}
async function deleteSong(source){
  if(!confirm('Delete this phrase of “'+source.title+'” and its annotations?'))return;
  await save();if(dirty||saving||conflicted)throw Error('Finish saving before deleting this song.');
  const db=await dbReady, expected=revision;
  const remaining=copy(state);
  await new Promise((resolve,reject)=>{
    const tx=db.transaction(['songs','settings'],'readwrite'),songs=tx.objectStore('songs'),settings=tx.objectStore('settings');
    const req=settings.get('review');
    req.onsuccess=()=>{
      if((req.result?.revision||0)!==expected){tx.abort();return}
      const scan=songs.openCursor();
      scan.onsuccess=()=>{
        const cursor=scan.result;
        if(!cursor){settings.put({revision:expected+1,data:remaining},'review');return}
        const c=cursor.value.clip;
        if(c.id===source.id){
          cursor.delete();delete remaining.clips[c.id];
          for(const [key,r] of Object.entries(remaining.records))if(r.clip_id===c.id)delete remaining.records[key];
        }
        cursor.continue();
      };
    };
    tx.oncomplete=resolve;tx.onabort=()=>reject(Error('Could not delete. Reload the notebook and retry.'));
  });
  if(active?.id===source.id){audio.pause();audio.removeAttribute('src');audio.load();active=null;$('playerTitle').textContent='Choose a phrase below';$('playerRange').textContent='';$('replay').disabled=$('back').disabled=true}
  for(const [id,c] of clipMap)if(c.id===source.id){clipMap.delete(id);URL.revokeObjectURL(c.audio_url)}
  clips=clips.filter(c=>c.id!==source.id);revision=expected+1;applyState(remaining);backup();
  $('overallCount').textContent=new Set(clips.map(c=>c.song_id)).size+' songs';
  if(!clips.length){$('progressText').textContent='Import songs to begin.';$('progressBar').style.width='0%'}
  announce('Phrase deleted.');
}
document.addEventListener('click',async event=>{
  const button=event.target.closest('[data-more],[data-delete-song]');if(!button)return;
  button.disabled=true;
  try{
    if(button.dataset.more)await anotherPassage(clipMap.get(button.dataset.more));
    else await deleteSong(clipMap.get(button.dataset.deleteSong));
  }catch(error){announce(error.message,true)}finally{button.disabled=false}
});

document.addEventListener('click',async event=>{
  const button=event.target.closest('[data-picker]');
  if(!button)return;
  const input=button.parentElement.querySelector('#scoreFirst,[data-sheet]');
  const images=button.dataset.picker==='images';
  input.accept=images?'.png,.jpg,.jpeg,.webp':'.pdf,.xml,.musicxml';
  input.multiple=images;
  if(!window.showOpenFilePicker){input.click();return}
  try{
    const handles=await window.showOpenFilePicker({
      id:images?'score-screenshots':'score-documents',
      startIn:images?'pictures':'downloads',
      multiple:images,
      types:images?[{description:'Score screenshots',accept:{'image/png':['.png'],'image/jpeg':['.jpg','.jpeg'],'image/webp':['.webp']}}]:
        [{description:'PDF or MusicXML score',accept:{'application/pdf':['.pdf'],'application/xml':['.xml','.musicxml']}}]
    });
    const selected=new DataTransfer();
    for(const handle of handles)selected.items.add(await handle.getFile());
    input.files=selected.files;
    input.dispatchEvent(new Event('change',{bubbles:true}));
  }catch(error){
    if(error.name!=='AbortError')input.click();
  }
});

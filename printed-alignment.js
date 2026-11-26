// Read single treble staves from printed pages. Positions stay in original image pixels.
async function readPrintedPage(page,pageIndex){
  const img=new Image();img.src=page.url;await img.decode();
  const canvas=document.createElement('canvas');canvas.width=page.width;canvas.height=page.height;
  const ctx=canvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0,canvas.width,canvas.height);
  const w=canvas.width,h=canvas.height,pixels=ctx.getImageData(0,0,w,h).data;
  const ink=(x,y)=>pixels[4*(Math.round(y)*w+Math.round(x))]<128;
  const lines=[];
  for(let y=0;y<h;y++){
    let count=0;for(let x=0;x<w;x+=2)if(pixels[4*(y*w+x)]<234)count++;
    if(count>w*.26){if(lines.length && lines.at(-1).end===y-1)lines.at(-1).end=y;else lines.push({start:y,end:y})}
  }
  const centers=lines.map(l=>(l.start+l.end)/2),systems=[];
  for(let i=0;i+4<centers.length;i++){
    const top=centers[i],d=(centers[i+4]-top)/4;
    if(d<3||centers.slice(i+1,i+5).some((v,k)=>Math.abs(v-top-(k+1)*d)>1.6))continue;
    i+=4;
    let left=0,right=w-1;
    while(left<w&&pixels[4*(Math.round(top)*w+left)]>=234)left++;while(right>left&&pixels[4*(Math.round(top)*w+right)]>=234)right--;
    const bars=[left];
    for(let x=left+4*d;x<right-3;x++){
      let hits=0;for(let k=1;k<=8;k++)if(ink(x,top+(k-.5)*d/2))hits++;
      let clear=0;
      for(let k=0;k<4;k++)if(!ink(x-.32*d,top+(k+.5)*d)&&!ink(x+.32*d,top+(k+.5)*d))clear++;
      if(hits===8 && clear===4 && x-bars.at(-1)>5*d)bars.push(x);
    }
    if(right-bars.at(-1)>3*d)bars.push(right);else bars[bars.length-1]=right;
    const candidates=[];
    for(let x=Math.round(left+4*d);x<right-d/2;x++){
      for(let step=-8;step<=18;step++){
        const y=top+(8-step)*d/2;
        let inside=0,outside=0;
        for(const u of [-.32,0,.32])for(const v of [-.22,.22]){
          if(ink(x+u*d,y+(v-u*.25)*d))inside++;
          if(!ink(x+u*d,y+(Math.sign(v)*.69-u*.25)*d))outside++;
        }
        if(inside>=5&&outside>=4){
          let stem=false;
          for(const side of [-1,1])for(let dx=.36;dx<=.74;dx+=.08){
            let hits=0;for(let k=0;k<5;k++)if(pixels[4*(Math.round(y-side*(.72+.29*k)*d)*w+Math.round(x+side*dx*d))]<205)hits++;
            if(hits>=4)stem=true;
          }
          if(stem)candidates.push({x,y,step,quality:outside+Number(ink(x-.49*d,y+.18*d))+Number(ink(x+.49*d,y-.18*d))});
        }
      }
    }
    const heads=[];
    for(const n of candidates.sort((a,b)=>b.quality-a.quality))if(!heads.some(o=>Math.abs(o.x-n.x)<.88*d))heads.push(n);
    heads.sort((a,b)=>a.x-b.x);
    if(heads.length)systems.push({pageIndex,top,d,bars,heads,canvas});
  }
  return systems;
}

// Chromatic energy from the local recording in 80 ms steps.
function pitchFrames(buffer){
  const samples=buffer.getChannelData(0),rate=buffer.sampleRate,hop=Math.round(rate*.08),size=Math.round(rate*.12);
  const frames=Array.from({length:Math.ceil(samples.length/hop)},()=>new Float32Array(12));
  const taper=Float32Array.from({length:size},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/size));
  for(let midi=48;midi<=83;midi++){
    const k=2*Math.cos(2*Math.PI*440*2**((midi-69)/12)/rate);
    for(let f=0;f<frames.length;f++){
      let a=0,b=0;
      for(let i=0,j=f*hop;i<size&&j+i<samples.length;i++){const c=taper[i]*samples[j+i]+k*a-b;b=a;a=c}
      frames[f][midi%12]+=Math.sqrt(Math.max(0,a*a+b*b-k*a*b));
    }
  }
  for(const frame of frames){const max=Math.max(...frame,.001);for(let i=0;i<12;i++)frame[i]/=max}
  return frames;
}
async function printedPassage(source,buffer,id){
  const pages=source.fullScorePages||source.images;
  const systems=(await Promise.all(pages.map(readPrintedPage))).flat();
  let bar=0;
  for(const system of systems){
    const {bars,heads}=system;
    for(let j=1;j<bars.length;j++,bar++){
      const group=heads.filter(n=>n.x>=bars[j-1]&&n.x<bars[j]);
      const first=group[0]?.x??bars[j-1];
      for(const n of group){
        n.beat=4*(bar+(n.x-first)/Math.max(1,bars[j]-first));
        const scale=[0,2,4,5,7,9,11],degree=n.step+2;
        n.midi=60+12*Math.floor(degree/7)+scale[(degree%7+7)%7];
      }
    }
  }
  const all=systems.flatMap(s=>s.heads).filter(n=>Number.isFinite(n.beat));
  if(all.length<30)throw Error('Could not identify 30 printed noteheads on these pages.');
  const frames=pitchFrames(buffer);
  // Search tempo and first-bar offset against the written melody, allowing a transposed arrangement.
  const probe=all.filter((_,i)=>i%Math.max(1,Math.floor(all.length/180))===0);
  let best={score:-Infinity};
  for(let bpm=48;bpm<=190;bpm+=.5){
    const beat=60/bpm;
    for(let t=0;t<=Math.min(45,buffer.duration*.3);t+=.32){
      if(t+probe.at(-1).beat*beat>buffer.duration+2)continue;
      const scores=new Float32Array(12);
      for(const n of probe){
        const frame=frames[Math.floor((t+(n.beat+.15)*beat)/.08)];if(!frame)continue;
        for(let shift=0;shift<12;shift++)scores[shift]+=frame[(n.midi+shift)%12];
      }
      const score=Math.max(...scores)/probe.length;
      if(score>best.score)best={score,beat,t,shift:scores.indexOf(Math.max(...scores))};
    }
  }
  if(!best.beat)throw Error('Could not align these pages to this recording.');
  // Follow the optical note sequence monotonically through the recording.
  // A soft duration cost lets a rest or uneven engraving shift subsequent notes.
  const lattice=[];
  for(const [index,n] of all.entries()){
    const center=Math.round((best.t+n.beat*best.beat)/.08);
    const lo=Math.max(0,center-110),hi=Math.min(frames.length-1,center+110);
    if(hi<lo)throw Error('The printed melody extends beyond this recording.');
    const costs=new Float64Array(hi-lo+1).fill(-Infinity),parents=new Int32Array(costs.length);
    const prev=lattice.at(-1),duration=index?Math.max(1,(n.beat-all[index-1].beat)*best.beat/.08):1;
    const pitch=(n.midi+best.shift)%12;
    for(let f=lo;f<=hi;f++){
      const energy=(frames[f][pitch]+(frames[f+1]?.[pitch]??0))/2;
      let value=prev?-Infinity:0;
      if(prev){
        for(let earlier=Math.max(prev.lo,f-Math.ceil(duration*2+12));earlier<Math.min(f,prev.lo+prev.costs.length);earlier++){
          const trial=prev.costs[earlier-prev.lo]-.24*Math.abs(Math.log((f-earlier)/duration));
          if(trial>value){value=trial;parents[f-lo]=earlier}
        }
      }
      costs[f-lo]=value+energy-.002*Math.abs(center-f);
    }
    lattice.push({lo,costs,parents});
  }
  const tail=lattice.at(-1),peak=Math.max(...tail.costs);
  if(!Number.isFinite(peak))throw Error('Could not trace the printed melody through this recording.');
  let frame=tail.lo+tail.costs.indexOf(peak);
  for(let i=all.length-1;i>=0;i--){
    all[i].time=frame*.08;frame=lattice[i].parents[frame-lattice[i].lo];
  }
  const choices=[];
  for(let i=0;i<systems.length;i++){
    let count=0;
    for(let j=i;j<systems.length;j++){
      count+=systems[j].heads.length;
      if(count>=30){
        const slice=systems.slice(i,j+1);
        const events=slice.flatMap(s=>s.heads).filter(n=>Number.isFinite(n.time));
        if(events.length>=30 && events.at(-1).time<buffer.duration && events.every((n,k)=>!k||n.time>events[k-1].time))choices.push(slice);
        break;
      }
    }
  }
  if(!choices.length)throw Error("No ordered 30-note section could be aligned.");
  const fresh=choices.filter(section=>!clips.some(c=>c.song_id===source.song_id && c.fullScorePages && c.start_s<section.at(-1).heads.at(-1).time && c.end_s>section[0].heads[0].time));
  const pool=fresh.length?fresh:choices;
  const chosen=pool[Math.floor(Math.random()*pool.length)],notes=[],images=[];
  for(const system of chosen){
    const {top,d,heads,canvas}=system;
    const y=Math.max(0,Math.floor(Math.min(top,...heads.map(n=>n.y))-4*d));
    const bottom=Math.min(canvas.height,Math.ceil(Math.max(top+4*d,...heads.map(n=>n.y))+3*d));
    const crop=document.createElement('canvas');crop.width=canvas.width;crop.height=bottom-y;
    crop.getContext('2d').drawImage(canvas,0,-y);
    const pins=[];
    for(const n of heads.filter(n=>Number.isFinite(n.beat))){
      const index=notes.length+1,noteId=id+':'+index,time=n.time;
      notes.push({id:noteId,index,pitch:'',start_s:time,end_s:time});
      pins.push({id:noteId,number:index,x:n.x,y:n.y-y,label_y:10+index%3*14});
    }
    images.push({url:crop.toDataURL(),width:crop.width,height:crop.height,notes:pins});
  }
  const last=notes.at(-1);
  return {...source,id,alignmentTempo:60/best.beat,alignmentOffset:best.t,fullScorePages:pages,phraseImages:undefined,notes,images,start_s:notes[0].start_s,end_s:Math.min(buffer.duration,last.start_s+best.beat),alignment_note:'Printed score · estimated audio alignment. Check the loop by ear.'};
}



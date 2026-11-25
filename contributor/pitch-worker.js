/* Monophonic pitch candidates from normalized autocorrelation at 8 kHz. */
onmessage=({data:samples})=>{
  const hop=160,window=320,frames=[];
  for(let pos=0;pos+window+100<=samples.length;pos+=hop){
    let power=0;for(let j=0;j<window;j++)power+=samples[pos+j]**2;
    if(power/window<.000035){frames.push(null);continue}
    const correlations=[];
    for(let lag=0;lag<=100;lag++){
      let cross=0,other=0;
      for(let j=0;j<window;j++){const y=samples[pos+j+lag];cross+=samples[pos+j]*y;other+=y*y}
      correlations.push(cross/Math.sqrt(power*other||1));
    }
    const cutoff=Math.max(.78,.94*Math.max(...correlations.slice(8)));
    const lag=correlations.findIndex((x,i)=>i>=8 && x>=cutoff && x>=correlations[i-1] && x>correlations[i+1]);
    frames.push(lag<0?null:Math.round(69+12*Math.log2(8000/lag/440)));
  }
  const notes=[];
  for(let i=0;i<frames.length;){
    let j=i+1;while(j<frames.length&&frames[j]===frames[i])j++;
    if(frames[i]!==null && j-i>=4)notes.push({midi:frames[i],start_s:i*.02,end_s:j*.02});
    i=j;
  }
  postMessage(notes);
};

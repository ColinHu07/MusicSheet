"""Tempo-relative monophonic notation, with attack identity kept separate from release.

Input notes and their acoustic start/end times are never modified or merged.
The written rhythm is an auditable interpretation. It uses attack intervals and
phrase continuity; brief vocal releases usually become articulation, not rests.
Triplets require a coherent three-attack pattern. Ambiguous dense material is
retained on a finer grid and explicitly marked for review.
"""
from __future__ import annotations
import math
from pathlib import Path
import xml.etree.ElementTree as ET

DIV = 48
BAR = 4 * DIV
VERSION = 'learned-context-grid-r1'

def _learned_cost(duration, probability):
    from .learned_rhythm import TICKS
    return min(-math.log(max(float(p),1e-8)) + 8*abs(float(t)-duration)/DIV for t,p in zip(TICKS,probability))


def _validate(events):
    result = [dict(event) for event in events]
    for i, note in enumerate(result):
        if not all(math.isfinite(float(note[key])) for key in ('start_s', 'end_s', 'midi')):
            raise ValueError('Finite note times and pitches required')
        if note['end_s'] <= note['start_s'] or int(note['midi']) != note['midi'] or not 0 <= note['midi'] <= 127:
            raise ValueError('Invalid note duration or MIDI pitch')
        if i and note['start_s'] <= result[i-1]['start_s']:
            raise ValueError('Attacks must be strictly ordered; notes are never silently merged')
        if i and note['start_s'] < result[i-1]['end_s']-1e-6:
            raise ValueError('Overlapping monophonic notes require explicit resolution')
    return result


def _triplets(beats):
    """Three attacks at equal thirds, anchored to an ordinary pulse subdivision."""
    groups = []
    used = set()
    for i in range(len(beats)-2):
        if any(j in used for j in (i,i+1,i+2)):
            continue
        a,b,c = beats[i:i+3]
        for step, anchor in ((1/3,1.), (1/6,.5), (2/3,2.)):
            error=max(abs((b-a)-step), abs((c-b)-step))
            anchor_error=abs(a-round(a/anchor)*anchor)
            # Ordinary 16th-16th-eighth patterns fail this test. The tolerance
            # scales with musical duration, never a millisecond cutoff.
            terminal_clear=i+3>=len(beats) or beats[i+3]>=round(a/anchor)*anchor+3*step-.055
            if error<=min(.055,step*.16) and anchor_error<=.08 and terminal_clear:
                members=[i,i+1,i+2]
                groups.append(dict(id=len(groups),event_indices=members,step_beats=step,
                    ratio=[3,2],anchor_beat=round(a/anchor)*anchor,
                    ioi_max_error_beats=error,anchor_error_beats=anchor_error))
                used.update(members)
                break
    return groups


def _phrase_ids(events, period):
    ids=[0]
    for previous,note in zip(events,events[1:]):
        gap=(note['start_s']-previous['end_s'])/period
        interval=(note['start_s']-previous['start_s'])/period
        ids.append(ids[-1]+int(gap>=.65 and gap/max(interval,1e-9)>=.28))
    return ids


def refine_phase(events, bpm, phase):
    """Estimate vocal-grid latency from all attacks; never consult score labels."""
    import cmath
    if len(events)<12:return phase,dict(applied=False,reason='too few attacks')
    period=60/bpm
    z=sum(cmath.exp(4j*math.pi*(n['start_s']-phase)/period) for n in events)/len(events)
    offset=cmath.phase(z)/(4*math.pi)
    applied=abs(z)>=.25 and abs(offset)<=.24
    updated=phase+offset*period if applied else phase
    if updated>events[0]['start_s']:updated-=4*period
    if updated<0:updated=phase
    return updated,dict(applied=applied,coherence=abs(z),offset_beats=(updated-phase)/period,method='Circular vocal attack phase relative to eighth-note pulse; no score input')


def _block_candidate(indices, beats, start, step, family, probabilities=None, strength=0., learned_costs=None):
    """Ordered assignment to a single two-beat grid, keeping every attack."""
    if family!='binary':
        # Require a recurring subdivision across neighboring beats. A dotted
        # binary figure must not become a triplet solely to reduce glyph count.
        local=[b for b in beats if start/DIV-4<=b<=start/DIV+6]
        run=best=0
        target=step/DIV
        tolerance=.085 if step==16 else .10 if step==32 else .04
        for a,b in zip(local,local[1:]):
            run=run+1 if abs((b-a)-target)<=tolerance else 0
            best=max(best,run)
        if best<4:
            learned_support=probabilities is not None and len(indices)>=3 and sum(float(probabilities[i,0,{16:5,32:8,8:2}[step]]) for i in indices[:-1])/len(indices[:-1])>.65
            if not learned_support:return None
    slots=list(range(start,start+96,step))
    if len(indices)>len(slots):return None
    rows=[];backs=[]
    for pos,i in enumerate(indices):
        row=[];back=[]
        for j,t in enumerate(slots):
            delta=t/DIV-beats[i]
            if abs(delta)>.34:row.append(float('inf'));back.append(-1);continue
            cost=(delta/.14)**2
            # Prefer simple written subdivisions, not arbitrary fine ticks.
            if family=='binary':cost+=.4*(t%24!=0)+2.0*(t%12!=0)+3.0*(t%6!=0)
            if not pos:row.append(cost);back.append(-1);continue
            options=[]
            for k,previous in enumerate(slots[:j]):
                duration=t-previous
                nearby=beats[max(0,i-3):i+2]
                dense=sum(.09<=b-a<=.19 for a,b in zip(nearby,nearby[1:]))>=3
                if dense and abs(duration/DIV-(beats[i]-beats[indices[pos-1]]))>.075:continue
                complexity=(.5 if duration%12 else .04 if duration%24 else 0) if family=='binary' else 0
                learned=strength*learned_costs[indices[pos-1],0,duration] if probabilities is not None else 0
                options.append((rows[-1][k]+complexity+learned,k))
            if options:
                v,k=min(options);row.append(v+cost);back.append(k)
            else:row.append(float('inf'));back.append(-1)
        rows.append(row);backs.append(back)
    j=min(range(len(slots)),key=lambda j:rows[-1][j])
    cost=rows[-1][j]
    if not math.isfinite(cost):return None
    selected=[]
    for pos in range(len(indices)-1,-1,-1):selected.append(slots[j]);j=backs[pos][j]
    selected.reverse()
    # A tuple is evidence of a recurring pulse, not a way to spell one stray note.
    if family!='binary':
        if len(indices)<3:return None
        cost+=.2
        if step==8:cost+=2.
    return dict(cost=cost,ticks=selected,step=step,family=family,indices=indices,start=start)


def quantize_rhythm(events,bpm,phase=0.,model=None,strength=0.):
    if not math.isfinite(float(bpm)) or not 30<=bpm<=240:raise ValueError('Invalid tempo')
    events=_validate(events)
    if not events:return [],dict(version=VERSION,events=0,triplet_groups=[])
    probabilities=model.probabilities(events,bpm) if model is not None and strength>0 else None
    learned_costs=None
    if probabilities is not None:
        import numpy as np
        from .learned_rhythm import TICKS
        learned_costs=(-np.log(np.maximum(probabilities[:,:,:,None],1e-8))+8*np.abs(TICKS[:,None]-np.arange(193))/DIV).min(axis=2)
    period=60/bpm
    beats=[(n['start_s']-phase)/period for n in events]
    if min(beats)<-.35:raise ValueError('Notation origin lies after input attack')
    blocks={}
    for i,b in enumerate(beats):blocks.setdefault(max(0,math.floor((b+.12)/2)),[]).append(i)
    candidates=[];keys=sorted(blocks)
    for block in keys:
        options=[]
        for step,family in [(24,'binary'),(12,'binary'),(6,'binary'),(3,'binary'),(16,'triplet'),(32,'quarter_triplet'),(8,'sixteenth_triplet')]:
            c=_block_candidate(blocks[block],beats,block*96,step,family,probabilities,strength,learned_costs)
            if c is not None:options.append(c)
        if not options:raise ValueError('Unresolved dense block; attacks preserved for review')
        candidates.append(options)
    costs=[];backs=[]
    for j,options in enumerate(candidates):
        row=[];back=[]
        for c in options:
            if not j:row.append(c['cost']);back.append(-1);continue
            choices=[]
            for k,prev in enumerate(candidates[j-1]):
                same=(c['family']==prev['family'])
                penalty=0 if same or keys[j]>keys[j-1]+1 else .45
                choices.append((costs[-1][k]+penalty,k))
            v,k=min(choices);row.append(v+c['cost']);back.append(k)
        costs.append(row);backs.append(back)
    k=min(range(len(costs[-1])),key=lambda k:costs[-1][k]);selected=[]
    for j in range(len(candidates)-1,-1,-1):selected.append(candidates[j][k]);k=backs[j][k]
    selected.reverse()
    assigned={};groups=[]
    for c in selected:
        for i,t in zip(c['indices'],c['ticks']):assigned[i]=(t,c)
        if c['family']!='binary':
            width=96 if c['step']==32 else 24 if c['step']==8 else 48
            for a in range(c['start'],c['start']+96,width):
                members=[i for i,t in zip(c['indices'],c['ticks']) if a<=t<a+width]
                if members:groups.append(dict(id=len(groups),event_indices=members,start_tick=a,end_tick=a+width,step_beats=c['step']/DIV))
    group_map={i:g for g in groups for i in g['event_indices']}
    results=[]
    for i,event in enumerate(events):
        start,c=assigned[i];release=(event['end_s']-phase)/period*DIV
        step=c['step'] if c['family']!='binary' else 12
        end=max(start+step,round(release/step)*step)
        decision='phrase_final_release';gap=0.
        if i+1<len(events):
            next_start=assigned[i+1][0]
            gap=(events[i+1]['start_s']-event['end_s'])/period
            ratio=gap/max((events[i+1]['start_s']-event['start_s'])/period,1e-9)
            meaningful=event.get('rest_after') is True or gap>=.45 or (gap>=.22 and ratio>=.5)
            if meaningful and end<next_start:decision='notated_rest'
            else:end=next_start;decision='articulation_release'
            end=min(end,next_start)
        # Grid transitions end on their shared pulse boundary; rests stay expressible.
        if c['family']!='binary':end=min(end,c['start']+96)
        if i+1<len(events) and c['family']=='binary' and assigned[i+1][1]['family']!='binary':end=min(end,assigned[i+1][1]['start'])
        if probabilities is not None and not event.get('rest_after'):
            from .learned_rhythm import TICKS
            limit=assigned[i+1][0] if i+1<len(events) else end+48
            if c['family']!='binary':limit=min(limit,c['start']+96)
            if i+1<len(events) and c['family']=='binary' and assigned[i+1][1]['family']!='binary':limit=min(limit,assigned[i+1][1]['start'])
            unit=3 if c['family']=='binary' else 8
            legal=[start+int(t) for t in TICKS if int(t)%unit==0 and start+int(t)<=limit and abs(start+int(t)-release)<=48]
            # Duration probabilities cannot invent a written rest after the
            # source-gap policy classified this as articulation.
            if decision=='articulation_release':legal=[e for e in legal if e>=end]
            if legal:
                old_end=end
                end=min(legal+[end],key=lambda e:strength*_learned_cost(e-start,probabilities[i,1])+.4*abs(e-old_end)/DIV)
                if end!=old_end:decision='learned_written_duration'
        if end<=start:raise ValueError('Attack lost at grid transition')
        n=dict(event,tick_start=start,tick_end=end,source_event_index=event.get('source_event_index',i),acoustic_start_s=event['start_s'],acoustic_end_s=event['end_s'],rhythm_grid=c['family'],rhythm_step_ticks=c['step'],onset_deviation_beats=start/DIV-beats[i],release_interpretation=decision,extra_review=bool(event.get('extra_review') or c['step']<12 or abs(start/DIV-beats[i])>.20))
        if i in group_map:
            g=group_map[i];n.update(tuplet_group=g['id'],tuplet_ratio=[3,2],tuplet_start_tick=g['start_tick'],tuplet_end_tick=g['end_tick'],tuplet_block_start=c['start'],tuplet_block_width=96 if c['step']==32 else 24 if c['step']==8 else 48)
        results.append(n)
    # Quarter-triplet runs may begin on an odd beat, crossing the two-beat
    # classifier's window boundary. Fit complete runs before engraving them.
    i=0
    while i+5<len(events):
        anchor=round(beats[i]);limit=i+1
        while limit<len(events) and limit<i+18 and .48<=beats[limit]-beats[limit-1]<=.86:
            limit+=1
        count=(limit-i)//3*3
        accepted=0
        while count>=6:
            target=[anchor+2*j/3 for j in range(count)]
            err=[target[j]-beats[i+j] for j in range(count)]
            end=(anchor+2*count/3)*DIV
            clean=all(n['rhythm_grid']=='binary' and not n.get('rest_after') for n in results[i:i+count]) and all((events[k+1]['start_s']-events[k]['end_s'])/period<.4 for k in range(i,i+count-1))
            bounds=(i==0 or anchor*DIV>results[i-1]['tick_start']) and (i+count==len(results) or end<=results[i+count]['tick_start'])
            context=count>=9 or any(g['end_tick']<=anchor*DIV and g['end_tick']>=anchor*DIV-8*DIV for g in groups)
            if clean and bounds and context and max(map(abs,err))<=.28 and math.sqrt(sum(e*e for e in err)/count)<=.16:
                accepted=count;break
            count-=3
        if not accepted:i+=1;continue
        if i and results[i-1]['tick_end']>anchor*DIV:results[i-1]['tick_end']=anchor*DIV
        for j in range(accepted):
            n=results[i+j];start=round((anchor+2*j/3)*DIV);group_start=round((anchor+2*(j//3))*DIV)
            n.update(tick_start=start,tick_end=start+32,rhythm_grid='quarter_triplet',rhythm_step_ticks=32,tuplet_ratio=[3,2],tuplet_group=len(groups)+j//3,tuplet_start_tick=group_start,tuplet_end_tick=group_start+96,tuplet_block_start=group_start,tuplet_block_width=96,onset_deviation_beats=start/DIV-beats[i+j],extra_review=bool(n['extra_review'] or abs(start/DIV-beats[i+j])>.20))
        for j in range(0,accepted,3):
            a=round((anchor+2*j/3)*DIV);groups.append(dict(id=len(groups),event_indices=list(range(i+j,i+j+3)),start_tick=a,end_tick=a+96,step_beats=2/3,method='complete quarter-triplet run across grid windows'))
        i+=accepted
    audit=dict(version=VERSION,input_attacks=len(events),output_attacks=len(results),attacks_merged=0,attacks_inserted=0,pitches_changed=False,triplet_groups=groups,blocks=selected,review_events=sum(n['extra_review'] for n in results),policy='Contextual straight/triplet grid selection; all attacks and source pitches preserved; no reference score read')
    if probabilities is not None:
        audit['learned_rhythm']=dict(weights_sha256=model.sha256,strength=strength,version=model.meta['feature_version'],inputs='audio-derived note context only')
    return results,audit


def _add(parent,tag,text=None,**attrs):
    node=ET.SubElement(parent,tag,{key:str(value) for key,value in attrs.items()})
    if text is not None: node.text=str(text)
    return node


def _pieces(duration, triplet=False):
    """Fewest ordinary/dotted values; barlines are handled by the caller."""
    regular={192:'whole',96:'half',48:'quarter',24:'eighth',12:'16th',6:'32nd',3:'64th'}
    values=[]
    for length,kind in regular.items():
        base=round(length*2/3) if triplet else length
        if triplet and length<12: continue
        values.append((base,kind,0))
        if not triplet and base*3%2==0:
            values.append((base*3//2,kind,1))
    memo={0:(0.,[])}
    for count in range(1,duration+1):
        candidates=[]
        for length,kind,dots in values:
            if length<=count and count-length in memo:
                cost,parts=memo[count-length]
                candidates.append((cost+1+.03*dots,[(length,kind,dots)]+parts))
        if candidates:
            memo[count]=min(candidates,key=lambda item:(item[0],-item[1][0][0]))
    if duration not in memo:
        raise ValueError(f'Duration {duration} cannot be written without inventing an unsupported tuplet')
    return memo[duration][1]


def _notation_triplet_contexts(notes):
    contexts={}
    for n in notes:
        if not n.get('tuplet_ratio'):continue
        start=n['tuplet_block_start'];width=n['tuplet_block_width']
        for a in range(start,start+96,width):
            contexts[a]=dict(start=a,end=a+width,group=str(a),kind='context_grid')
    return sorted(contexts.values(),key=lambda c:c['start'])


def musicxml(notes,bpm,phase,title,end_tick=None,key_fifths=0):
    """Write one violin part; identical-pitch attacks are never tied together."""
    root=ET.Element('score-partwise',version='4.0')
    _add(_add(root,'work'),'work-title',title+' - violin melody')
    _add(_add(root,'identification'),'creator','Local audio-derived melody; rhythm interpretation v3',type='arranger')
    score_part=_add(_add(root,'part-list'),'score-part',id='P1')
    _add(score_part,'part-name','Violin');_add(score_part,'part-abbreviation','Vln.')
    _add(_add(score_part,'score-instrument',id='P1-I'),'instrument-name','Violin')
    instrument=_add(score_part,'midi-instrument',id='P1-I');_add(instrument,'midi-channel',1);_add(instrument,'midi-program',41)
    part=_add(root,'part',id='P1')
    end_tick=max(BAR,math.ceil(max((n['tick_end'] for n in notes),default=BAR)/BAR)*BAR) if end_tick is None else end_tick
    velocity_state=None
    triplet_contexts=_notation_triplet_contexts(notes)
    for bar in range(math.ceil(end_tick/BAR)):
        left,right=bar*BAR,(bar+1)*BAR
        measure=_add(part,'measure',number=bar+1)
        accidental_state={}
        if bar==0:
            attributes=_add(measure,'attributes');_add(attributes,'divisions',DIV)
            _add(_add(attributes,'key'),'fifths',key_fifths)
            time=_add(attributes,'time');_add(time,'beats',4);_add(time,'beat-type',4)
            clef=_add(attributes,'clef');_add(clef,'sign','G');_add(clef,'line',2)
            direction=_add(measure,'direction',placement='above')
            _add(_add(direction,'direction-type'),'words',f'Quarter note = {bpm:g}; provisional 4/4')
            _add(direction,'sound',tempo=f'{bpm:.9f}')
        active=[(i,n) for i,n in enumerate(notes) if n['tick_end']>left and n['tick_start']<right]
        cursor=left
        fragments=[]
        def append_fragments(a,b,index,note):
            cuts=sorted({a,b}|{edge for context in triplet_contexts for edge in (context['start'],context['end']) if a<edge<b})
            for begin,end in zip(cuts,cuts[1:]):
                context=next((c for c in triplet_contexts if c['start']<=begin<c['end']),None)
                at=begin
                for piece in _pieces(end-begin,context is not None):
                    fragments.append((at,piece,index,note,context));at+=piece[0]
        for index,note in active+[(None,None)]:
            start=max(left,note['tick_start']) if note else right
            if cursor<start:
                append_fragments(cursor,start,None,None)
                cursor=start
            if note:
                stop=min(right,note['tick_end'])
                append_fragments(start,stop,index,note)
                cursor=stop
        beam_items=[]
        for cursor,(length,kind,dots),index,note,context in fragments:
            if note and cursor==note['tick_start']:
                velocity=int(note.get('velocity',82))
                if velocity!=velocity_state:
                    direction=_add(measure,'direction',placement='below')
                    dynamics=_add(_add(direction,'direction-type'),'dynamics')
                    _add(dynamics,'p' if velocity<61 else 'mp' if velocity<76 else 'mf' if velocity<91 else 'f')
                    _add(direction,'sound',dynamics=f'{velocity/90*100:.6f}')
                    velocity_state=velocity
            attrs={'id':f'n{index}_{cursor}'} if note else {}
            if note:
                attrs['dynamics']=f'{int(note.get("velocity",82))/90*100:.6f}'
                if note.get('extra_review'):attrs['color']='#A45316'
            element=_add(measure,'note',**attrs)
            if note:
                midi=int(note['midi']);names=[('C',0),('C',1),('D',0),('E',-1),('E',0),('F',0),('F',1),('G',0),('A',-1),('A',0),('B',-1),('B',0)]
                if key_fifths>0:names=[('C',0),('C',1),('D',0),('D',1),('E',0),('F',0),('F',1),('G',0),('G',1),('A',0),('A',1),('B',0)]
                step,alter,octave=spell_pitch(midi,key_fifths)
                pitch=_add(element,'pitch');_add(pitch,'step',step)
                if alter:_add(pitch,'alter',alter)
                _add(pitch,'octave',octave)
            else:_add(element,'rest')
            _add(element,'duration',length)
            ties=[]
            if note:
                if cursor>note['tick_start']:ties.append('stop')
                if cursor+length<note['tick_end']:ties.append('start')
                for tie in ties:_add(element,'tie',type=tie)
            _add(element,'voice',1);_add(element,'type',kind)
            for unused in range(dots):_add(element,'dot')
            if note:
                if cursor==note['tick_start'] and accidental_state.get((step,octave),(1 if key_fifths>0 else -1) if step in ('FCGDAEB' if key_fifths>0 else 'BEADGCF')[:abs(key_fifths)] else 0)!=alter:
                    _add(element,'accidental',{-1:'flat',0:'natural',1:'sharp'}[alter])
                accidental_state[(step,octave)]=alter
            if context:
                modification=_add(element,'time-modification');_add(modification,'actual-notes',3);_add(modification,'normal-notes',2)
            beam_items.append((element,cursor,length,kind,note,context))
            if ties or context:
                notation=_add(element,'notations')
                for tie in ties:_add(notation,'tied',type=tie)
                # Each coherent run is bracketed explicitly. The exact duration
                # remains in MusicXML; playback does not guess from the bracket.
                if context:
                    if cursor==context['start']:
                        _add(notation,'tuplet',type='start',number='1',bracket='yes')
                    if cursor+length==context['end']:
                        _add(notation,'tuplet',type='stop',number='1')
        beam_groups=[];group=[];group_key=None
        levels={'eighth':1,'16th':2,'32nd':3,'64th':4}
        for item in beam_items:
            element,tick,length,kind,note,context=item
            key=('triplet',context['group']) if context else ('beat',tick//DIV)
            if note is not None and kind in levels:
                if group and (key!=group_key or tick!=group[-1][1]+group[-1][2]):
                    beam_groups.append(group);group=[]
                group.append(item);group_key=key
            elif group:
                beam_groups.append(group);group=[];group_key=None
        if group:beam_groups.append(group)
        for group in beam_groups:
            if len(group)<2:continue
            for i,(element,tick,length,kind,note,context) in enumerate(group):
                for level in range(1,levels[kind]+1):
                    before=i>0 and levels[group[i-1][3]]>=level
                    after=i+1<len(group) and levels[group[i+1][3]]>=level
                    value='continue' if before and after else 'end' if before else 'begin' if after else 'forward hook' if i<len(group)-1 else 'backward hook'
                    beam=ET.Element('beam',number=str(level));beam.text=value
                    notation=element.find('notations')
                    element.insert(list(element).index(notation) if notation is not None else len(element),beam)
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(root,encoding='unicode')


def read_musicxml(path,origin_s=0.):
    """Strict round-trip reader for this module's monophonic, constant-tempo XML."""
    root=ET.parse(path).getroot()
    if any(root.find('.//'+tag) is not None for tag in ('repeat','grace','backup','chord')):
        raise ValueError('Unsupported navigation or polyphony')
    tempos={float(n.get('tempo')) for n in root.findall('.//sound') if n.get('tempo')}
    if len(tempos)!=1:raise ValueError('Exactly one tempo required')
    bpm=tempos.pop();period=60/bpm
    cursor=0;events=[];held=None;divisions=None
    for measure in root.findall('./part/measure'):
        measure_start=cursor
        for child in measure:
            if child.tag=='attributes':
                divisions=int(child.findtext('divisions',str(divisions)))
            if child.tag!='note':continue
            duration=int(child.findtext('duration'))
            if not divisions or duration<=0:raise ValueError('Invalid duration/divisions')
            if child.find('rest') is None:
                pitch=child.find('pitch');step=pitch.findtext('step');alter=int(pitch.findtext('alter','0'));octave=int(pitch.findtext('octave'))
                midi=12*(octave+1)+dict(C=0,D=2,E=4,F=5,G=7,A=9,B=11)[step]+alter
                a=origin_s+cursor/divisions*period;b=origin_s+(cursor+duration)/divisions*period
                ties={node.get('type') for node in child.findall('tie')}
                if 'stop' in ties:
                    if held is None or held['midi']!=midi or abs(held['end_s']-a)>1e-7:raise ValueError('Broken tie')
                    held['end_s']=b
                else:
                    if held is not None:raise ValueError('Unclosed tie before distinct attack')
                    held=dict(midi=midi,start_s=a,end_s=b,velocity=round(float(child.get('dynamics','91.111111'))*.9))
                    events.append(held)
                if 'start' not in ties:held=None
            elif held is not None:raise ValueError('Tie crosses a rest')
            cursor+=duration
        if cursor-measure_start!=4*divisions:raise ValueError('Measure does not sum to four beats')
    if held is not None:raise ValueError('Unclosed final tie')
    return dict(bpm=bpm,parts={'P1':events},duration_s=cursor/divisions*period)


def spell_pitch(midi,key_fifths):
    natural=dict(C=0,D=2,E=4,F=5,G=7,A=9,B=11)
    changed=('FCGDAEB' if key_fifths>0 else 'BEADGCF')[:abs(key_fifths)]
    signature={letter:(1 if key_fifths>0 else -1) if letter in changed else 0 for letter in natural}
    preferred={(pc+signature[letter])%12:(letter,signature[letter]) for letter,pc in natural.items()}
    flats=[('C',0),('D',-1),('D',0),('E',-1),('E',0),('F',0),('G',-1),('G',0),('A',-1),('A',0),('B',-1),('B',0)]
    sharps=[('C',0),('C',1),('D',0),('D',1),('E',0),('F',0),('F',1),('G',0),('G',1),('A',0),('A',1),('B',0)]
    letter,alter=preferred.get(midi%12,(sharps if key_fifths>0 else flats)[midi%12])
    return letter,alter,(midi-natural[letter]-alter)//12-1

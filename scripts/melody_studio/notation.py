"""Keep edit identity, fixed score ticks and the source clock separate."""
from pathlib import Path
from io import BytesIO
import json
import math
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from musicsheet.rhythm import (
    DIV, BAR, musicxml, read_musicxml, refine_phase, quantize_rhythm, _notation_triplet_contexts)
from musicsheet.midi import midi_bytes
from musicsheet.audio import sha256
from scipy.signal import correlate
from musicsheet.learned_rhythm import RhythmModel


def estimate_grid(features, bpm=None):
    """Integer tempo and pulse phase from recording onset flux only."""
    times = np.asarray(features['times_s'], float)
    if len(times) < 20:
        raise ValueError('Audio feature sequence is too short for notation')
    hop = float(np.median(np.diff(times)))
    flux = np.asarray(features['onset_flux'], float)
    flux = np.maximum(flux - np.percentile(flux, 35), 0)
    centered = flux - flux.mean()
    autocorrelation = correlate(centered, centered, mode='full', method='fft')[len(flux)-1:]
    autocorrelation /= np.maximum(np.arange(len(flux), 0, -1), 1)
    autocorrelation /= max(float(autocorrelation[0]), 1e-9)
    candidates = []
    for candidate in range(55, 201):
        lag = 60 / candidate / hop
        periodicity = float(np.interp(lag, np.arange(len(flux)), autocorrelation))
        periodicity += .4 * float(np.interp(2 * lag, np.arange(len(flux)), autocorrelation))
        coherence = float(abs(np.sum(flux*np.exp(2j*np.pi*times*candidate/60))/max(float(flux.sum()),1e-9)))
        # A near half-tempo integer can win short-lag autocorrelation while
        # drifting over the recording. Long-span phase coherence penalizes it.
        candidates.append(dict(bpm=candidate, score=periodicity+coherence,
                               autocorrelation=periodicity, phase_coherence=coherence))
    candidates.sort(key=lambda x: -x['score'])
    if bpm is None:
        bpm = candidates[0]['bpm']
        method = 'Onset-flux autocorrelation plus long-span circular phase coherence, integer 55..200 BPM; octave ambiguity remains'
    else:
        if not isinstance(bpm, int) or not 30 <= bpm <= 240:
            raise ValueError('BPM override must be an integer in 30..240')
        method = 'Explicit user BPM override; phase estimated from recording onset flux'
    period = 60 / bpm
    phase_vector = np.sum(flux * np.exp(2j * np.pi * times / period)) / max(float(flux.sum()), 1e-9)
    phase = float((np.angle(phase_vector) / (2 * np.pi) % 1) * period)
    return dict(bpm=bpm, beat_phase_s=phase, phase_concentration=float(abs(phase_vector)),
                method=method, candidate_tempos=candidates[:8], score_used=False,
                meter='4/4 provisional; downbeat not independently detected', key='No key signature asserted; explicit accidentals')

def phrase_dynamics(notes, features_dir, bpm):
    """Bounded source-RMS expression; one written velocity per short phrase block."""
    path=Path(features_dir)/'vocal_features.npz'
    if not path.exists() or not notes:
        return [dict(n,velocity=82) for n in notes],dict(method='constant_velocity_missing_vocal_features',velocity=82)
    with np.load(path) as features:
        times=np.asarray(features['times_s']);rms=np.asarray(features['rms'])
    period=60/bpm
    groups=[];current=[];start=None;previous=None
    for i,note in enumerate(notes):
        if current and (note['start_s']-previous['end_s']>=.65*period or note['start_s']-start>=8*period):
            groups.append(current);current=[]
        if not current:start=note['start_s']
        current.append(i);previous=note
    if current:groups.append(current)
    levels=[]
    for group in groups:
        mask=np.zeros(len(times),bool)
        for i in group:mask|=(times>=notes[i]['start_s'])&(times<notes[i]['end_s'])
        samples=rms[mask]
        levels.append(float(np.median(samples[samples>1e-5])) if (samples>1e-5).any() else 1e-5)
    db=20*np.log10(np.maximum(levels,1e-5))
    center=float(np.median(db));span=max(float(np.percentile(db,85)-np.percentile(db,15)),6.)
    desired=np.clip(82+(db-center)/span*16,70,96)
    # Smooth neighboring phrase levels and limit abrupt MIDI jumps. This changes
    # velocity only; no random pitch, onset, duration or timing humanization.
    smooth=[]
    for i,value in enumerate(desired):
        local=float(np.median(desired[max(0,i-1):min(len(desired),i+2)]))
        smooth.append(int(round(np.clip(local,smooth[-1]-6,smooth[-1]+6))) if smooth else int(round(local)))
    output=[dict(n) for n in notes];records=[]
    for group,level,velocity in zip(groups,levels,smooth):
        for i in group:output[i]['velocity']=velocity
        records.append(dict(first_event=group[0],last_event=group[-1],vocal_rms=level,velocity=velocity))
    return output,dict(method='Local vocal RMS, median-smoothed phrase blocks up to eight beats; velocities 70..96, adjacent change at most six',
        source_features_sha256=sha256(path),written_to_musicxml=True,timing_changed=False,groups=records)


def arrange(raw, features, duration):
    with np.load(features/'audio_features.npz') as z:
        grid = estimate_grid(z)
        chroma = z['chroma']
        chroma = chroma.mean(axis=1 if chroma.shape[0] == 12 else 0)
    profiles = [[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88],
                [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]]
    _, tonic, minor = max((float(np.corrcoef(chroma,np.roll(profile,k))[0,1]), k, mode)
                          for mode,profile in enumerate(profiles) for k in range(12))
    fifths = [0,-5,2,-3,4,-1,6,1,-4,3,-2,5][(tonic+3*minor)%12]
    bpm = grid['bpm']
    phase = refine_phase(raw,bpm,min(raw[0]['start_s'],grid['beat_phase_s']))[0] if raw else 0
    shift = max(0,12*math.ceil((55-min((n['midi'] for n in raw),default=55))/12))
    voiced,_ = phrase_dynamics([dict(n,midi=n['midi']+shift) for n in raw],features,bpm)
    model_dir = ROOT/'checkpoints/rhythm_network_r01_20261031_run3'
    strength = json.loads((model_dir/'FROZEN_BEFORE_EVALUATION.json').read_text())['decoder_strength']
    quantized,_ = quantize_rhythm(voiced,bpm,phase,RhythmModel(model_dir),strength)
    notes = [dict(id=f'n{i}',start=n['tick_start'],end=n['tick_end'],pitch=n['midi'],
                  velocity=n.get('velocity',80)) for i,n in enumerate(quantized)]
    return dict(bpm=bpm,phase=phase,shift=shift,fifths=fifths,duration=duration,
                contexts=_notation_triplet_contexts(quantized),notes=notes,
                review_ids=[f'n{i}' for i,n in enumerate(quantized) if n.get('extra_review')])


def import_example(xml, meta):
    notes = read_musicxml(xml)['parts']['P1']
    scale = DIV * meta['bpm']/60
    contexts, cursor, beginning = [], 0, None
    for el in ET.parse(xml).findall('./part/measure/note'):
        ticks = int(el.findtext('duration'))
        if el.find('./notations/tuplet[@type="start"]') is not None:
            beginning = cursor
        cursor += ticks
        if el.find('./notations/tuplet[@type="stop"]') is not None and beginning is not None:
            contexts.append(dict(start=beginning,end=cursor,group=str(beginning)))
            beginning = None
    review_ids = sorted({el.get('id','').split('_')[0] for el in ET.parse(xml).findall('.//note[@color]')})
    return dict(bpm=meta['bpm'],phase=meta['phase'],shift=meta['violin_shift_semitones'],review_ids=review_ids,
                fifths=meta['written_key_fifths'],duration=meta['duration_s'],contexts=contexts,
                notes=[dict(id=f'n{i}',start=round(n['start_s']*scale),end=round(n['end_s']*scale),
                            pitch=n['midi'],velocity=n.get('velocity',80)) for i,n in enumerate(notes)])


def validate(notes):
    if not isinstance(notes,list) or len(notes)>20000:
        raise ValueError('Expected a list of up to 20,000 notes.')
    clean, ids = [], set()
    for n in notes:
        if not isinstance(n,dict):
            raise ValueError('Invalid note.')
        uid = n.get('id')
        if not isinstance(uid,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',uid) or uid in ids:
            raise ValueError('Each note needs a unique ID.')
        ids.add(uid)
        if any(type(n.get(k)) is not int for k in ('start','end','pitch','velocity')):
            raise ValueError('Pitch and timing must use whole MIDI notes and score ticks.')
        if not (0<=n['start']<n['end']<=BAR*2048 and 0<=n['pitch']<=127 and 1<=n['velocity']<=127):
            raise ValueError('Check the pitch, position and length of the selected note.')
        clean.append({k:n[k] for k in ('id','start','end','pitch','velocity')})
    clean.sort(key=lambda n:n['start'])
    if any(a['end']>b['start'] for a,b in zip(clean,clean[1:])):
        raise ValueError('These notes overlap. Shorten the previous note or enable “Shift following notes”.')
    return clean


def engrave(score, title):
    notes = validate(score['notes'])
    edges = {n[k] for n in notes for k in ('start','end')}
    # Preserve original tuplet groups unless an edit introduces a straight subdivision inside.
    groups = [g for g in score['contexts'] if any(g['start']<t<g['end'] and t%3 for t in edges)
              and all((t-g['start'])%8==0 for t in edges if g['start']<t<g['end'])]
    for t in sorted(edges):
        if t%3 and not any(g['start']<=t<=g['end'] for g in groups):
            a=(t//24)*24
            if any((v-a)%8 for v in edges if a<v<a+24):
                raise ValueError('This edit mixes straight and triplet subdivisions within half a beat. Choose another position or length.')
            groups.append(dict(start=a,end=a+24,group=str(a)))
    flagged=set(score.get('review_ids',[]))
    events = [dict(tick_start=n['start'],tick_end=n['end'],midi=n['pitch'],velocity=n['velocity'],
                   extra_review=n['id'] in flagged) for n in notes]
    return musicxml(events,score['bpm'],score['phase'],title,key_fifths=score['fifths'],triplet_contexts=groups)


def midi(score):
    unit=60/(score['bpm']*DIV)
    return midi_bytes([dict(start_s=n['start']*unit,end_s=n['end']*unit,midi=n['pitch'],
                            velocity=n['velocity']) for n in score['notes']],score['bpm'])


def with_tempo_header(xml):
    """Reserve a page heading for Studio's opening tempo instead of sharing the bar-number row."""
    root=ET.fromstring(xml)
    measure=root.find('part/measure')
    part_list=root.find('part-list')
    if measure is None or part_list is None:
        return xml
    for index,direction in enumerate(measure):
        if direction.tag=='note': break
        if direction.tag!='direction': continue
        words=direction.find('direction-type/words')
        sound=direction.find('sound[@tempo]')
        if words is None or sound is None: continue
        caption=(words.text or '').strip()
        match=re.fullmatch(r'Quarter note = ([\d.]+); provisional 4/4',caption)
        if match:
            caption=f'{match[1]} BPM  ·  provisional 4/4'
        elif not re.fullmatch(r'[\d.]+ BPM',caption):
            continue
        credit=ET.Element('credit',page='1')
        ET.SubElement(credit,'credit-words',justify='left',valign='top').text=caption
        root.insert(list(root).index(part_list),credit)
        measure.remove(direction)
        measure.insert(index,sound)
        return ET.tostring(root,encoding='unicode')
    return xml


def render_pages(xml):
    import verovio
    # Verovio stores its default resource directory per thread on Windows.
    verovio.setDefaultResourcePath(str(Path(verovio.__file__).parent/'data'))
    toolkit=verovio.toolkit()
    toolkit.setOptions(dict(pageWidth=2000,pageHeight=2200,scale=45,header='encoded',footer='none',
                            svgViewBox=True,adjustPageHeight=True,mnumInterval=1,spacingSystem=8))
    if not toolkit.loadData(xml):
        raise ValueError('Could not engrave the edited notation.')
    return [toolkit.renderToSVG(i) for i in range(1,toolkit.getPageCount()+1)]


def pdf_drawing(svg):
    from copy import deepcopy
    from svglib.svglib import svg2rlg
    SVG_NS = "{http://www.w3.org/2000/svg}"
    # svglib does not resolve Verovio's nested SVG viewport/use elements reliably.
    doc = ET.fromstring(svg)
    defs = {n.get('id'): n for n in doc.iter() if n.get('id')}
    for parent in list(doc.iter()):
        for child in list(parent):
            if child.tag != SVG_NS + 'use':
                continue
            href = child.get('{http://www.w3.org/1999/xlink}href', child.get('href', ''))
            group = ET.Element(SVG_NS + 'g', {k: v for k, v in child.attrib.items() if not k.endswith('href')})
            group.append(deepcopy(defs[href[1:]]))
            i = list(parent).index(child)
            parent.remove(child)
            parent.insert(i, group)
    outer = list(map(float, doc.attrib['viewBox'].split()))
    for child in doc.findall(SVG_NS + 'svg'):
        box = list(map(float, child.attrib['viewBox'].split()))
        child.tag = SVG_NS + 'g'
        child.set('transform', f'scale({outer[2]/box[2]} {outer[3]/box[3]})')
        child.attrib.pop('viewBox')
    return svg2rlg(BytesIO(ET.tostring(doc)))


def pdf(pages,title):
    import os
    from xml.sax.saxutils import escape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.graphics import renderPDF
    from reportlab.lib.pagesizes import A4
    output=BytesIO()
    c=Canvas(output,pagesize=A4)
    c.setTitle(title)
    width,height=A4
    font='Helvetica-Bold'
    local_font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/malgunbd.ttf'
    if local_font.is_file():
        font='StudioSheetTitle'
        if font not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(font,str(local_font)))
    heading=Paragraph(escape(title),ParagraphStyle('sheetTitle',fontName=font,fontSize=15,leading=19,wordWrap='CJK'))
    _,heading_height=heading.wrap(width-50,height)
    heading_bottom=height-20-heading_height
    music_top=heading_bottom-26
    for i,page in enumerate(pages,1):
        drawing=pdf_drawing(page)
        scale=min((width-50)/drawing.width,(music_top-28)/drawing.height)
        heading.drawOn(c,25,heading_bottom)
        c.setFont('Helvetica',9)
        c.drawRightString(width-25,heading_bottom-16,f'{i} / {len(pages)}')
        c.translate(25,music_top-scale*drawing.height)
        c.scale(scale,scale)
        renderPDF.draw(drawing,c,0,0)
        c.showPage()
    c.save()
    return output.getvalue()

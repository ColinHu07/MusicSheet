"""Separate, auditable notation export. Raw note events are never overwritten."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
from .audio import save_json, sha256
from .rhythm import quantize_rhythm, musicxml, read_musicxml, BAR
from .midi import midi_bytes


def export_score(predictions, output, *, bpm, phase=0., key_sharps=0, title='Melody draft',
                 rhythm_model=None, rhythm_strength=1., pdf=False):
    if not math.isfinite(bpm) or not 10 <= bpm <= 400 or not math.isfinite(phase) or phase < 0:
        raise ValueError('Provide a finite tempo of 10..400 BPM and nonnegative score origin')
    if not -7 <= key_sharps <= 7:
        raise ValueError('Key signature must have -7..7 sharps (negative means flats)')
    source = Path(predictions)
    document = json.loads(source.read_text(encoding='utf-8'))
    events = document['notes']
    if not events:
        raise ValueError('No note events to engrave')
    if events[0]['start_s'] < phase:
        raise ValueError('Score origin cannot discard leading note events')
    model = None
    if rhythm_model:
        from .learned_rhythm import RhythmModel
        model = RhythmModel(rhythm_model)
    notes, audit = quantize_rhythm(events, bpm, phase, model, rhythm_strength)
    if [n['midi'] for n in notes] != [n['midi'] for n in events]:
        raise ValueError('Notation changed the pitches or note count')
    end = max(BAR, math.ceil(max(n['tick_end'] for n in notes) / BAR) * BAR)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    xml = output / 'melody.musicxml'
    xml.write_text(musicxml(notes, bpm, phase, title, end, key_sharps), encoding='utf-8')
    parsed = read_musicxml(xml, phase)['parts']['P1']
    if [n['midi'] for n in parsed] != [n['midi'] for n in events]:
        raise ValueError('MusicXML round trip lost a pitch or attack')
    (output / 'raw.mid').write_bytes(midi_bytes(events, bpm))
    (output / 'notated.mid').write_bytes(midi_bytes(parsed, bpm))
    save_json(output / 'notation_audit.json', dict(source_sha256=sha256(source), bpm=bpm, phase_s=phase,
              key_sharps=key_sharps, meter='4/4', assumptions='Caller-specified, not independently verified',
              source_pitches_and_attack_count_preserved=True, raw_timing_changed=False,
              pitches_below_violin_g3=sum(n['midi'] < 55 for n in events), notation=audit))
    if pdf:
        render_pdf(xml, output)


def render_pdf(xml, output):
    import verovio
    from svglib.svglib import svg2rlg
    from reportlab.pdfgen import canvas
    from reportlab.graphics import renderPDF
    from reportlab.lib.pagesizes import A4
    toolkit = verovio.toolkit()
    toolkit.setOptions(dict(pageWidth=2100, pageHeight=2900, scale=40, footer='none', header='none',
                           breaks='auto', spacingStaff=12, svgViewBox=True))
    if not toolkit.loadFile(str(xml)) or toolkit.getPageCount() < 1:
        raise RuntimeError('Verovio could not engrave MusicXML')
    pdf = canvas.Canvas(str(output / 'melody.pdf'), pagesize=A4)
    width, height = A4
    for page in range(1, toolkit.getPageCount() + 1):
        svg = toolkit.renderToSVG(page)
        (output / f'page-{page}.svg').write_text(svg, encoding='utf-8')
        doc = ET.fromstring(svg)
        ns = '{http://www.w3.org/2000/svg}'
        definitions = {node.get('id'): node for node in doc.iter() if node.get('id')}
        for parent in list(doc.iter()):
            for child in list(parent):
                if child.tag == ns + 'use':
                    href = child.get('{http://www.w3.org/1999/xlink}href', child.get('href', ''))
                    group = ET.Element(ns + 'g', {key: value for key, value in child.attrib.items() if not key.endswith('href')})
                    group.append(deepcopy(definitions[href[1:]]))
                    index = list(parent).index(child)
                    parent.remove(child)
                    parent.insert(index, group)
        outer = list(map(float, doc.attrib['viewBox'].split()))
        for child in doc.findall(ns + 'svg'):
            box = list(map(float, child.attrib['viewBox'].split()))
            child.tag = ns + 'g'
            child.set('transform', f'scale({outer[2]/box[2]} {outer[3]/box[3]})')
            child.attrib.pop('viewBox')
        drawing = svg2rlg(BytesIO(ET.tostring(doc)))
        if drawing is None or drawing.width <= 0 or drawing.height <= 0:
            raise RuntimeError('Unable to render nonempty PDF page')
        scale = min((width - 60) / drawing.width, (height - 75) / drawing.height)
        pdf.saveState()
        pdf.translate(30, height - 25 - drawing.height * scale)
        pdf.scale(scale, scale)
        renderPDF.draw(drawing, pdf, 0, 0)
        pdf.restoreState()
        pdf.setFont('Helvetica', 8)
        pdf.drawString(30, 22, 'Provisional rhythm; raw source pitches and attacks retained.')
        pdf.drawRightString(width - 30, 22, f'{page}/{toolkit.getPageCount()}')
        pdf.showPage()
    pdf.save()

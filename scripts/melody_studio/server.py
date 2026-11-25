"""Run with .venv/Scripts/python.exe scripts/melody_studio/server.py"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from statistics import median
import shutil
import subprocess
import sys
import tempfile
import threading
from time import time
from urllib.parse import urlsplit, parse_qs
from uuid import uuid4
import xml.etree.ElementTree as ET

from pipeline import BACKEND, check_backend, ROOT, read, write, run_job
from notation import engrave, import_example, midi, pdf, render_pages, validate, read_musicxml, with_tempo_header

STATIC=Path(__file__).parent/'web'
BANK=BACKEND/'data/instruments/salamander_subset'
lock=threading.RLock()
worker=ThreadPoolExecutor(max_workers=1)
MAX_UPLOAD=160*1024*1024


def latest(job):
    return max((int(p.name[1:]) for p in job.glob('r[0-9]*') if p.is_dir() and p.name[1:].isdigit()),default=-1)


def commit(job, version, score, title, xml=None):
    score=dict(score,notes=validate(score['notes']))
    xml=with_tempo_header(xml or engrave(score,title))
    with tempfile.TemporaryDirectory(dir=job) as temp:
        stage=Path(temp)/'revision'
        stage.mkdir()
        path=stage/'score.musicxml'
        path.write_text(xml,encoding='utf-8')
        notes=read_musicxml(path)['parts']['P1']
        scale=score['bpm']*48/60
        if [(n['midi'],round(n['start_s']*scale),round(n['end_s']*scale)) for n in notes]!=[(n['pitch'],n['start'],n['end']) for n in score['notes']]:
            raise ValueError('The score could not preserve these note positions.')
        write(stage/'score.json',score)
        (stage/'score.mid').write_bytes(midi(score))
        stage.rename(job/f'r{version}')


def update(job,**changes):
    with lock:
        info=read(job/'info.json')
        write(job/'info.json',dict(info,**changes))


@lru_cache(maxsize=256)
def first_render(job):
    """Read the immutable first revision, including imports made before timing was recorded."""
    first=job/'r0'/'score.json'
    info=read(job/'info.json')
    log=job/'pipeline.log'
    if not log.is_file() or not first.is_file():
        return None
    started=info.get('started_at') or getattr(log.stat(),'st_birthtime',None) or (job/'original.mp3').stat().st_mtime
    elapsed=info.get('finished_at',first.stat().st_mtime)-started
    duration=info.get('duration_s') or read(first).get('duration')
    if elapsed>0 and duration and duration>0:
        return elapsed,duration


def library(store):
    projects=[]
    rates=[]
    for f in store.glob('*/info.json'):
        p=read(f)
        job=f.parent
        rev=latest(job)
        saved=job/f'r{rev}'/'score.json' if rev>=0 else f
        p.update(version=rev,updated=saved.stat().st_mtime)
        labels=job/'labels.json'
        if labels.is_file(): p.update(read(labels))
        if p['status'] in ('queued','working') and 'queued_at' not in p:
            # Active imports from an older worker have file timestamps instead.
            audio=job/'original.mp3'
            p['queued_at']=audio.stat().st_mtime
            if p['status']=='working':
                log=job/'pipeline.log'
                p['started_at']=getattr(log.stat(),'st_birthtime',p['queued_at']) if log.exists() else p['queued_at']
            p['duration_s']=clip_duration(audio)
        if p['status']=='ready':
            timing=first_render(job)
            if timing:
                elapsed,duration=timing
                p['render_seconds']=round(elapsed)
                rates.append(elapsed/duration)
        projects.append(p)
    queued=sorted((p for p in projects if p['status']=='queued'),key=lambda p:p.get('queued_at',p['updated']))
    for index,p in enumerate(queued):
        p['queue_ahead']=index+sum(x['status']=='working' for x in projects)
    for p in projects:
        if p['status'] in ('queued','working') and rates and p.get('duration_s'):
            p['estimated_seconds']=round(median(rates)*p['duration_s'])
    return sorted(projects,key=lambda p:(-p['updated'],p['title'].casefold()))


@lru_cache(maxsize=64)
def clip_duration(audio):
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(audio)],
                         capture_output=True,timeout=30)
    try: return float(probe.stdout.strip())
    except ValueError: return None


def process(job):
    try:
        update(job,status='working',started_at=time())
        info=read(job/'info.json')
        score=run_job(job,lambda stage:update(job,status='working',stage=stage))
        commit(job,0,score,info['title'])
        update(job,status='ready',stage='',finished_at=time())
    except Exception as error:
        with (job/'pipeline.log').open('a',encoding='utf8') as log:
            log.write('\n'+str(error))
        update(job,status='failed',stage=read(job/'info.json')['stage'],error='Transcription stopped. See the local job log for details.')


def initialize(store):
    store.mkdir(parents=True,exist_ok=True)
    for p in store.glob('*/info.json'):
        if read(p)['status'] in ('queued','working'):
            if latest(p.parent)>=0:
                update(p.parent,status='ready',stage='')
            else:
                update(p.parent,status='failed',error='The server was stopped during transcription. Upload again to restart.')
@lru_cache(maxsize=6)
def score_xml(revision):
    return with_tempo_header((revision/'score.musicxml').read_text(encoding='utf8'))


@lru_cache(maxsize=6)
def pages(revision):
    return render_pages(score_xml(revision))


@lru_cache(maxsize=96)
def thumbnail(revision):
    """Engrave a few measures for a library card without rendering the full song."""
    root=ET.fromstring(score_xml(revision))
    part=root.find('part')
    measures=part.findall('measure')
    first=next((i for i,m in enumerate(measures) if m.find('note/pitch') is not None),0)
    keep=measures[first:first+8]
    if first and keep and keep[0].find('attributes') is None:
        attrs=measures[0].find('attributes')
        if attrs is not None: keep[0].insert(0,attrs)
    for m in measures:
        if m not in keep: part.remove(m)
    return render_pages(ET.tostring(root,encoding='unicode'))[0].encode('utf-8')


class Handler(BaseHTTPRequestHandler):
    def catalog_origin(self):
        port=getattr(self.server,'library_for',None)
        origin=self.headers.get('Origin')
        if port and origin in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
            return origin

    def end_headers(self):
        origin=self.catalog_origin()
        if origin:
            self.send_header('Access-Control-Allow-Origin',origin)
            self.send_header('Vary','Origin')
        super().end_headers()

    def do_OPTIONS(self):
        if not self.catalog_origin(): return self.obj({'error':'Local studio only.'},403)
        self.reply(b'','text/plain',204,{'Access-Control-Allow-Methods':'GET, POST',
                                       'Access-Control-Allow-Headers':'Content-Type'})

    def reply(self, data, mime, code=200, extra=None):
        self.send_response(code)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        for k,v in (extra or {}).items():
            self.send_header(k,v)
        self.end_headers()
        self.wfile.write(data)

    def obj(self,data,code=200):
        self.reply(json.dumps(data,ensure_ascii=False).encode(),'application/json; charset=utf-8',code)

    def file(self,path):
        size=path.stat().st_size
        start,end=0,size-1
        extra={'Accept-Ranges':'bytes'}
        if 'Range' in self.headers:
            m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
            if not m or int(m[1])>=size:
                return self.reply(b'','text/plain',416,{'Content-Range':f'bytes */{size}'})
            start=int(m[1])
            if m[2]: end=min(end,int(m[2]))
            if start>end: return self.reply(b'','text/plain',416)
            extra['Content-Range']=f'bytes {start}-{end}/{size}'
        self.send_response(206 if 'Range' in self.headers else 200)
        self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1))
        for k,v in extra.items():self.send_header(k,v)
        self.end_headers()
        with path.open('rb') as f:
            f.seek(start)
            remaining=end-start+1
            while remaining:
                buf=f.read(min(remaining,262144))
                if not buf:break
                self.wfile.write(buf)
                remaining-=len(buf)

    def route(self):
        host=self.headers.get('Host','')
        if host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):
            return self.obj({'error':'Local connections only.'},403)
        if self.command=='POST' and self.headers.get('Origin') not in (None,'http://'+host) and not self.catalog_origin():
            return self.obj({'error':'Use the local editor to save changes.'},403)
        u=urlsplit(self.path)
        path=u.path
        catalog_port=getattr(self.server,'library_for',None)
        if catalog_port:
            allowed=(self.command=='GET' and (path in ('/api/health','/api/projects') or path.endswith(('/state','/preview.svg','/pages','/pdf','/musicxml')))) or (self.command=='POST' and path.endswith('/labels'))
            if not allowed: raise FileNotFoundError()
        if self.command=='GET' and path=='/api/health':
            return self.obj({'service':'melody-studio','sheet_details':True,'tempo_header':True,'library_for':str(catalog_port or '')})
        if self.command=='GET' and path in ('/','/app.js','/studio.css'):
            return self.file(STATIC/('index.html' if path=='/' else path[1:]))
        if self.command=='GET' and re.fullmatch(r'/samples/[A-Za-z0-9_.-]+',path):
            return self.file(BANK/path.split('/')[-1])
        if path=='/api/projects':
            if self.command=='POST':return self.upload(parse_qs(u.query))
            return self.obj(library(self.server.store))
        m=re.fullmatch(r'/api/projects/(taeyang|[0-9a-f]{32})/(state|labels|audio|score|pages|preview\.svg|musicxml|mid|pdf|log)',path)
        if not m:raise FileNotFoundError()
        job=self.server.store/m[1]
        info=read(job/'info.json')
        action=m[2]
        if action=='state':
            info=next(p for p in library(self.server.store) if p['id']==m[1])
        if action=='labels' and self.command=='POST':
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096:raise ValueError('Invalid sheet details.')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict) or not all(isinstance(data.get(k),str) for k in ('title','artist')):
                raise ValueError('Enter a title and author.')
            labels={k:data[k].strip() for k in ('title','artist')}
            if not labels['title'] or any(len(s)>120 for s in labels.values()):
                raise ValueError('Enter a title and keep each field within 120 characters.')
            with lock: write(job/'labels.json',labels)
            return self.obj(labels)
        if action=='labels': raise FileNotFoundError()
        if self.command=='POST':
            if action!='score' or info['status']!='ready':raise ValueError('This project is not editable yet.')
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4000000:raise ValueError('Invalid score payload size.')
            edit=json.loads(self.rfile.read(length))
            with lock:
                v=latest(job)
                if edit['version']!=v:return self.obj({'error':'This score was edited in another tab. Reload its latest revision.'},409)
                score=read(job/f'r{v}/score.json')
                commit(job,v+1,dict(score,notes=edit['notes']),info['title'])
            return self.obj({'version':v+1})
        if action=='audio':return self.file(job/'original.mp3')
        if action=='log':return self.file(job/'pipeline.log')
        v=latest(job)
        rev=int(parse_qs(u.query).get('v',[v])[0])
        if rev>v or rev<0:
            if action=='state':return self.obj(info)
            raise FileNotFoundError()
        folder=job/f'r{rev}'
        if action=='state':
            return self.obj(dict(info,version=rev,score=read(folder/'score.json')))
        if action=='preview.svg':
            with lock: svg=thumbnail(folder)
            return self.reply(svg,'image/svg+xml')
        if action=='pages':
            with lock: rendered=pages(folder)
            return self.obj(rendered)
        if action in ('musicxml','mid','pdf'):
            suffix=action
            with lock:
                if suffix=='musicxml':
                    content=score_xml(folder).encode('utf8')
                elif suffix=='pdf':
                    path=folder/'score-header-v1.pdf'
                    if not path.exists():
                        content=pdf(pages(folder),info['title'])
                        stage=path.with_suffix('.pdf.tmp')
                        stage.write_bytes(content)
                        stage.replace(path)
                    content=path.read_bytes()
                else:
                    content=(folder/'score.mid').read_bytes()
            return self.reply(content,{'pdf':'application/pdf','musicxml':'application/vnd.recordare.musicxml+xml','mid':'audio/midi'}[suffix],extra={'Content-Disposition':f'attachment; filename="melody-r{rev}.{suffix}"'})
        raise FileNotFoundError()

    def upload(self,q):
        check_backend()
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=MAX_UPLOAD:raise ValueError('Choose an MP3 up to 160 MB.')
        name=q.get('name',['Untitled'])[0]
        language=q.get('language',['ko'])[0]
        if language not in ('ko','en'):raise ValueError('Choose Korean or English.')
        title=Path(name).stem.strip()[:120] or 'Untitled'
        job=self.server.store/uuid4().hex
        job.mkdir()
        try:
            remaining=length
            with (job/'original.mp3').open('wb') as audio:
                while remaining:
                    chunk=self.rfile.read(min(262144,remaining))
                    if not chunk:raise ValueError('Upload was interrupted.')
                    audio.write(chunk)
                    remaining-=len(chunk)
            result=subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
                  'stream=codec_name:format=duration','-of','json',str(job/'original.mp3')],capture_output=True,timeout=45)
            probe=json.loads(result.stdout) if not result.returncode else {}
            if not any(s.get('codec_name')=='mp3' for s in probe.get('streams',[])):
                raise ValueError('This file does not contain a readable MP3 audio track.')
            info=dict(id=job.name,title=title,language=language,status='queued',stage='Waiting for the local GPU')
            info['queued_at']=time()
            duration=probe.get('format',{}).get('duration')
            if duration and duration!='N/A': info['duration_s']=float(duration)
            write(job/'info.json',info)
        except Exception:
            shutil.rmtree(job)
            raise
        worker.submit(process,job)
        return self.obj(info,202)

    def handle_request(self):
        try:self.route()
        except (ConnectionResetError,BrokenPipeError,ConnectionAbortedError):pass
        except FileNotFoundError:self.obj({'error':'Item not found.'},404)
        except (ValueError,KeyError,TypeError) as exc:self.obj({'error':str(exc)},400)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.obj({'error':f'Unable to complete this request: {exc}'},500)

    do_GET=do_POST=handle_request


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8784)
    parser.add_argument('--store',type=Path,default=ROOT/'outputs/melody_studio')
    parser.add_argument('--library-for',type=int,help='Serve only library details alongside an existing local Studio worker')
    parser.add_argument('--open',action='store_true',help='Open the local editor in the default browser')
    args=parser.parse_args()
    url=f'http://127.0.0.1:{args.port}/'
    try:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    except OSError:
        if args.open:
            import urllib.request, webbrowser
            with urllib.request.urlopen(url+'api/health',timeout=3) as response:
                if json.load(response).get('service')=='melody-studio':
                    webbrowser.open(url)
                    sys.exit(0)
        raise
    server.store=args.store.resolve()
    server.library_for=args.library_for
    if not args.library_for: initialize(server.store)
    print(f'Melody Studio is ready at {url}',flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    server.serve_forever()

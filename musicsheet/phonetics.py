"""Automatic phone/syllable hypotheses, not verified singing labels."""
import re
import numpy as np
VOWELS = set("aeiouyɑɒæɛɜɞəɚɐɪɨʉɯʊʌɔœøɵɶ")

def ctc_align(log_probs,tokens,blank=0):
    """Viterbi with explicit blanks and legal skip/repeat transitions."""
    if not tokens:return []
    states=np.zeros(len(tokens)*2+1,np.int64);states[1::2]=tokens
    if log_probs.shape[0]<len(tokens)+sum(a==b for a,b in zip(tokens,tokens[1:])):return None
    scores=np.full(len(states),-np.inf);scores[0]=log_probs[0,blank];scores[1]=log_probs[0,tokens[0]]
    back=np.zeros((len(log_probs),len(states)),np.int8)
    for t in range(1,len(log_probs)):
        one=np.r_[-np.inf,scores[:-1]];two=np.r_[-np.inf,-np.inf,scores[:-2]]
        allowed=(np.arange(len(states))>=2)&(states!=blank)&(states!=np.roll(states,2));two[~allowed]=-np.inf
        candidates=np.stack([scores,one,two]);choice=candidates.argmax(0)
        scores=candidates[choice,np.arange(len(states))]+log_probs[t,states];back[t]=choice
    state=len(states)-1 if scores[-1]>=scores[-2] else len(states)-2
    if not np.isfinite(scores[state]):return None
    path=np.empty(len(log_probs),int)
    for t in range(len(log_probs)-1,-1,-1):path[t]=state;state-=back[t,state]
    spans=[]
    for i,token in enumerate(tokens):
        frames=np.flatnonzero(path==2*i+1)
        if not len(frames):return None
        spans.append(dict(first=int(frames[0]),last=int(frames[-1])+1,
                          peak=int(frames[np.argmax(log_probs[frames,token])]),
                          confidence=float(np.exp(log_probs[frames,token].mean()))))
    return spans


def make_g2p():
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    import espeakng_loader
    EspeakWrapper.set_library(espeakng_loader.get_library_path());EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    engines={lang:EspeakBackend(lang,with_stress=False,preserve_punctuation=False,language_switch='remove-flags') for lang in ('en-us','ko')}
    sep=Separator(phone=' ',word=' | ',syllable='')
    def phones(word):
        parts=re.findall(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?",word)
        result=[]
        for part in parts:
            lang='ko' if re.search('[가-힣]',part) else 'en-us'
            text=engines[lang].phonemize([part],separator=sep,strip=True)[0]
            result.extend(text.replace('|',' ').split())
        return result
    return phones


def align_lyrics(asr,ppg,times,vocab):
    g2p=make_g2p();words=[];phone_rows=[];syllables=[];rejected=[]
    for seg in asr['segments']:
        for word in seg.get('words',[]):
            text=word['word'].strip();phones=g2p(text)
            unknown=[p for p in phones if p not in vocab]
            row=dict(text=text,asr_start_s=word['start'],asr_end_s=word['end'],asr_confidence=word['probability'],phones=phones,syllables=[])
            words.append(row)
            if not phones or unknown or word['end']-word['start']<.02:
                row['status']='unresolved_pronunciation_or_asr_time';row['unsupported_phones']=unknown;continue
            lo=max(0,word['start']-.18);hi=word['end']+.18
            indexes=np.flatnonzero((times>=lo)&(times<hi))
            if len(indexes)<len(phones):row['status']='insufficient_audio_frames';continue
            spans=ctc_align(np.log(np.maximum(ppg[indexes],1e-8)),[vocab[p] for p in phones])
            if spans is None:row['status']='alignment_failed';continue
            nuclei=[i for i,p in enumerate(phones) if any(c in VOWELS for c in p)]
            hangul=re.findall('[가-힣]',text)
            for k,(ph,span) in enumerate(zip(phones,spans)):
                entry=dict(phone=ph,word_index=len(words)-1,evidence_start_s=float(times[indexes[span['first']]]),
                     evidence_end_s=float(times[indexes[min(span['last'],len(indexes)-1)]]+.02),
                     peak_s=float(times[indexes[span['peak']]]),confidence=span['confidence'])
                phone_rows.append(entry)
            for j,k in enumerate(nuclei):
                span=spans[k];at=float(times[indexes[span['peak']]])
                first=0 if j==0 else nuclei[j-1]+1
                start=float(times[indexes[spans[first]['first']]])
                confidence=float(min(word['probability'],span['confidence']))
                supported=word['probability']>=.65 and span['confidence']>=.15 and word['start']-.10<=at<=word['end']+.10
                syll=dict(word_index=len(words)-1,word=text,index_in_word=j+1,
                    label=hangul[j] if len(hangul)==len(nuclei) else f'{text} [{j+1}/{len(nuclei)}]',
                    phones=phones[first:k+1],vowel=phones[k],attack_estimate_s=start,nucleus_s=at,
                    end_s=None,confidence=confidence,supported_input=supported,
                    human_verified=False,timing_uncertainty_s=.12,
                    status='automatic_phone_alignment_candidate',note_count=None)
                syllables.append(syll);row['syllables'].append(len(syllables)-1)
            row['status']='automatic_unverified_alignment';row['syllable_count']=len(nuclei)
            row['estimated_syllables_per_second']=len(nuclei)/max(.02,word['end']-word['start'])
    syllables.sort(key=lambda s:s['nucleus_s'])
    # Rebuild indexes after chronological sorting. Overlapping word alignments
    # are flagged, not silently forced into one lyric sequence.
    for row in words:row['syllables']=[]
    for i,s in enumerate(syllables):
        words[s['word_index']]['syllables'].append(i)
        if i and s['nucleus_s']-syllables[i-1]['nucleus_s']<.06:
            s['supported_input']=False;syllables[i-1]['supported_input']=False
            s['status']='overlapping_or_too_close_alignment_review'
    return dict(words=words,phonemes=phone_rows,syllables=syllables,
        limits=['Automatic ASR and speech-trained phone alignment on singing; not human truth',
                'CTC phone spikes are not sung vowel duration measurements',
                'Text-derived syllable count does not impose a note count',
                'Pronunciation can differ in singing; unsupported phones remain unknown',
                'Overlapping vocals and rap have not been assigned a melody role'],
        lyrics_extracted=True,lyrics_used_for_alignment=True,score_read=False,tempo_used=False)

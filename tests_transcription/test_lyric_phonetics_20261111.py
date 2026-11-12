import numpy as np
from musicsheet.phonetics import ctc_align

def test_repeated_phone_requires_blank_and_preserves_two_events():
    p=np.full((5,3),.01);p[[0,2,4],0]=.98;p[[1,3],1]=.98
    spans=ctc_align(np.log(p),[1,1]);assert [s['peak'] for s in spans]==[1,3]
    assert ctc_align(np.log(p[:2]),[1,1]) is None

def test_no_equal_division_of_word_time():
    p=np.full((20,4),.001);p[:,0]=.99;p[2,1]=.999;p[2,0]=.001;p[16,2]=.999;p[16,0]=.001
    spans=ctc_align(np.log(p),[1,2]);assert [s['peak'] for s in spans]==[2,16]

def test_low_evidence_is_not_high_confidence():
    spans=ctc_align(np.log(np.full((30,8),1/8)),[1,2,3])
    assert all(s['confidence']<.15 for s in spans)

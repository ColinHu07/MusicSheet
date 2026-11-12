import numpy as np
import torch
from musicsheet.events import NoteEventNetwork, targets


def note(a,b,midi=60):
    return dict(start_s=a,end_s=b,midi=midi,supervise=True,alignment_confidence=.9)


def item(notes,negatives=()):
    return dict(id='fixture',pp=np.ones((61,200),np.float32)/61,
                alignment=dict(notes=notes,hard_negative_spans=list(negatives)))


def test_unknown_gaps_are_not_rests_and_note_interior_continues():
    result=targets(item([note(1,2)]))
    assert result['event_weights'][:,10].sum()==0
    assert result['events'][4,75]==1
    assert result['pitch'][75]==25
    assert result['event_weights'][4,75]>0


def test_repeat_and_pitch_transition_are_distinct_without_lyric_inference():
    result=targets(item([note(.5,1),note(1,1.5),note(1.5,2,62)]))
    assert result['events'][2,50]==1
    assert result['events'][3,75]==1
    assert result['events'][3,50]==0


def test_verified_quiet_does_not_overwrite_positive_note():
    result=targets(item([note(1,2)],[dict(start_s=.5,end_s=1.5,verified=True,confidence=.8,provenance='acoustic_quiet')]))
    assert result['pitch'][60]==25
    assert result['pitch'][30]==0 and result['pitch_weights'][30]>0


def test_zero_initialized_residual_preserves_baseline_evidence():
    torch.set_num_threads(2)
    model=NoteEventNetwork().eval();p=torch.randn(1,61,100);b=torch.randn(1,2,100)
    with torch.no_grad(): pp,bb,aux,c=model(torch.randn(1,229,100),p,b)
    assert torch.equal(pp,p) and torch.equal(bb,b) and torch.count_nonzero(c)==0
    assert aux.shape==(1,3,100)

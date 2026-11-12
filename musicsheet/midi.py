"""MIDI serialization preserving source times and pitches."""
import struct

def midi_bytes(notes, bpm, program=40):
    """Absolute source-time MIDI; the initial silence is retained."""
    def vlq(value):
        result = [value & 127]
        while value >> 7:
            value >>= 7
            result.insert(0, (value & 127) | 128)
        return bytes(result)
    tempo = round(60_000_000 / bpm)
    # Use the actual encoded integer microsecond tempo when mapping seconds.
    ticks_per_s = 480_000_000 / tempo
    track = b'\x00\xff\x51\x03' + tempo.to_bytes(3, 'big') + bytes([0, 0xc0, program])
    events = []
    for note in notes:
        a = max(0, round(note['start_s'] * ticks_per_s))
        b = max(a + 1, round(note['end_s'] * ticks_per_s))
        events += [(a, 1, bytes([0x90, int(note['midi']), int(note.get('velocity', 82))])),
                   (b, 0, bytes([0x80, int(note['midi']), 0]))]
    previous = 0
    for tick, _, event in sorted(events):
        track += vlq(tick - previous) + event
        previous = tick
    track += b'\x00\xff\x2f\x00'
    return b'MThd' + struct.pack('>IHHH', 6, 0, 1, 480) + b'MTrk' + struct.pack('>I', len(track)) + track

import midi
import numpy as np
import glob
from tqdm import tqdm

#midi note range ( 0 - 127)
midi_range = 127

def notes_from_one (midifile):
    pattern = midi.read_midifile(midifile)

    drum_notes = np.zeros([1,128])
    main_notes = np.zeros([1,128])

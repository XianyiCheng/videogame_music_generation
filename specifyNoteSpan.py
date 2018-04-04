import midi
import numpy as np
import glob
from tqdm import tqdm

#midi note range ( 0 - 127)
midi_range = 127

def notes_from_one (midifile):
    pattern = midi.read_midifile(midifile)

    drum_notes = np.zeros([128])
    main_notes = np.zeros([128])
    ticks = main_notes = np.zeros([12800])
    print(pattern.resolution)

    for i in [1,2]:
        notes = np.zeros([128])
        track = pattern[i]

        for evt in range(len(track)):
            if isinstance(track[evt],midi.NoteEvent):
                notes[track[evt].pitch] = notes[track[evt].pitch] + 1
                if track[evt].tick < 12800:
                    ticks[track[evt].tick] = ticks[track[evt].tick] + 1


        if i == 1:
            main_notes = notes

        if i == 2:
            drum_notes = notes

    return (main_notes, drum_notes, ticks)

def findNoteSpan (path):
    files = glob.glob('{}/*.mid*'.format(path))
    all_drum_notes = np.zeros([128])
    all_main_notes = np.zeros([128])
    all_ticks = 1000
    for f in tqdm(files):
        [main_notes, drum_notes, ticks] = notes_from_one(f)
        all_drum_notes = all_drum_notes + drum_notes
        all_main_notes = all_main_notes + main_notes
        all_ticks = all_ticks + ticks
    """
    print(np.argwhere(all_main_notes == 0))
    print(all_main_notes)
    print(np.argwhere(all_drum_notes == 0))
    print(all_drum_notes)
    print(all_min_tick)
    """
    #print(np.argsort(all_ticks)[12750:])
    #print(np.sort(all_ticks)[12700:])

    return None


if __name__ == '__main__':
    findNoteSpan('./Game_Music_Midi')

import midi
import numpy as np
import glob
from tqdm import tqdm
import math

lowerBound_main = 26 #The lowest note
upperBound_main = 101 #The highest note
span_main = upperBound_main - lowerBound_main #The note range

lowerBound_drum = 35 #The lowest note
upperBound_drum = 81 #The highest note
span_drum = upperBound_drum - lowerBound_drum #The note range

note_res = 60 #The note resolution, number of ticks at one row of note play matrix

num_timesteps      = 32 #The number of note timesteps that we produce with each RNN

def midiToNotePlayMatrix(midifile, squash=True, span_main=span_main, span_drum = span_drum):
    pattern = midi.read_midifile(midifile)
    main_track = pattern[1]
    drum_track = pattern[2]

    ticks = [0,0]
    for i in [1,2]:
        for j in range(len(pattern[i])):
            ticks[i-1] = ticks[i-1] + pattern[i][j].tick

    num_notes = math.ceil(max(ticks)/note_res)
    NotePlayMatrix = np.zeros([num_notes,span_drum + span_main + 2])
    note_pos = 0

    for i in range(len(main_track)):
        evt = main_track[i]
        if isinstance(evt, midi.NoteEvent):
            note_pos = note_pos + round(evt.tick/note_res)
            if evt.velocity > 0:
                NotePlayMatrix[note_pos:,evt.pitch - lowerBound_main] = 1
            if evt.velocity == 0:
                NotePlayMatrix[note_pos:,evt.pitch - lowerBound_main] = 0
    note_pos = 0
    for i in range(len(drum_track)):
        evt = drum_track[i]
        if isinstance(evt, midi.NoteEvent):
            note_pos = note_pos + round(evt.tick/note_res)
            if evt.velocity > 0:
                NotePlayMatrix[note_pos:,evt.pitch - lowerBound_drum + span_main + 1] = 1
            if evt.velocity == 0:
                NotePlayMatrix[note_pos:,evt.pitch - lowerBound_drum + span_main + 1] = 0
    """
    if np.sum(NotePlayMatrix[-1,:]) != 0:
        NotePlayMatrix = np.vstack((NotePlayMatrix,np.zeros([span_drum + span_main + 2])))
    """
    return NotePlayMatrix


def notePlayMatrixToMidi(NotePlayMatrix, name = "example"):
    pattern = midi.Pattern()
    pattern.resolution = 480
    main_track = midi.Track()
    drum_track = midi.Track()
    pattern.append(main_track)
    pattern.append(drum_track)

    main_ticks_count= note_res
    drum_ticks_count= note_res
    print(NotePlayMatrix.shape[0])
    for pos in range(NotePlayMatrix.shape[0]):
        if pos == 0:
            notes = np.nonzero(NotePlayMatrix[0,:])[0]
            for n in notes:
                if n < span_main + 1:
                    main_track.append(midi.NoteOnEvent(tick=0, velocity=40,
                     pitch=n+lowerBound_main))
                else:
                    drum_track.append(midi.NoteOnEvent(tick=0, velocity=40,
                     pitch=n-span_main-1+lowerBound_drum))

        else:
            noteChanges = NotePlayMatrix[pos,:] != NotePlayMatrix[pos-1,:]

            #main track
            if sum(noteChanges[:span_main+1]) == 0 :
                main_ticks_count = main_ticks_count + note_res
            else:
                notes = np.where(noteChanges[:span_main+1])[0]
                notes_status = NotePlayMatrix[pos,:span_main+1][noteChanges[:span_main+1]]

                for n in range(len(notes)):
                    if notes_status[n] == 1:
                        main_track.append(midi.NoteOnEvent(tick=main_ticks_count, velocity=40,
                        pitch=notes[n]+lowerBound_main))
                    if notes_status[n] == 0:
                        main_track.append(midi.NoteOffEvent(tick=main_ticks_count,
                        pitch=notes[n]+lowerBound_main))
                    main_ticks_count = 0

                main_ticks_count= note_res

            # drum track
            if sum(noteChanges[span_main+1:]) == 0:
                drum_ticks_count = drum_ticks_count + note_res
            else:
                notes = np.where(noteChanges[span_main+1:])[0]
                notes_status = NotePlayMatrix[pos,span_main+1:][noteChanges[span_main+1:]]

                for n in range(len(notes)):
                    if notes_status[n] == 1:
                        drum_track.append(midi.NoteOnEvent(tick=drum_ticks_count, velocity=40,
                        pitch=notes[n] + lowerBound_drum))
                    if notes_status[n] == 0:
                        drum_track.append(midi.NoteOffEvent(tick=drum_ticks_count,
                        pitch=notes[n] + lowerBound_drum))

                    drum_ticks_count = 0

                drum_ticks_count= note_res


    eot = midi.EndOfTrackEvent(tick=0)
    main_track.append(eot)
    drum_track.append(eot)

    print(np.sum(NotePlayMatrix,axis = 1))

    midi.write_midifile("{}.mid".format(name), pattern)






# the following functions needs to be modified

def write_song(path, song):
    #Reshape the song into a format that midi_manipulation can understand, and then write the song to disk
    song = np.reshape(song, (song.shape[0]*num_timesteps, span_drum + span_main + 2))
    notePlayMatrixToMidi(song, name=path)

def get_song(path):
    #Load the song and reshape it to place multiple timesteps next to each other
    song = midiToNotePlayMatrix(path)
    song = song[:int(np.floor(song.shape[0]/num_timesteps)*num_timesteps)]
    song = np.reshape(song, [int(song.shape[0]/num_timesteps), song.shape[1]*num_timesteps])
    return song

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = get_song(f)
            if np.array(song).shape[0] > 50/num_timesteps:
                songs.append(song)
        except Exception as e:
            print (f, e)
    return songs

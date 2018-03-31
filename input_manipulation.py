import midi
import numpy as np
import glob
from tqdm import tqdm

lowerBound_main = 24 #The lowest note
upperBound_main = 102 #The highest note
span_main = upperBound_main - lowerBound_main #The note range

lowerBound_drum = 24 #The lowest note
upperBound_drum = 102 #The highest note
span_drum = upperBound_drum - lowerBound_drum #The note range

num_timesteps      = 5 #The number of note timesteps that we produce with each RNN

def midiToNotePlayMatrix(midifile, squash=True, span_main=span_main, span_drum = span_drum):
    pass

def notePlayMatrixToNoteStateMatrix(NotePlayMatrix):
    pass

def noteStateMatrixToMidi():
    pass

# the following functions needs to be modified

def write_song(path, song):
    #Reshape the song into a format that midi_manipulation can understand, and then write the song to disk
    song = np.reshape(song, (song.shape[0]*num_timesteps, 2*span))
    noteStateMatrixToMidi(song, name=path)

def get_song(path):
    #Load the song and reshape it to place multiple timesteps next to each other
    song = np.array(midiToNoteStateMatrix(path))
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

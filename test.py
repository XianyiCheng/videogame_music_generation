import input_manipulation as im

midifile = './Game_Music_Midi/1943sab_m.mid'

NotePlayMatrix = im.midiToNotePlayMatrix(midifile)

im.notePlayMatrixToMidi(NotePlayMatrix)

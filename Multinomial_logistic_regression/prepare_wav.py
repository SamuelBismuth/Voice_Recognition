from pydub import AudioSegment


def divide_audio(path, begin_track, end_track):
    begin_track = begin_track * 1000
    end_track = end_track * 1000
    song = AudioSegment.from_wav(path)
    song[begin_track:end_track].export('newSong.wav', format="wav")
    return 'newSong.wav'


def song_time(path):
    song = AudioSegment.from_wav(path)
    return song.duration_seconds

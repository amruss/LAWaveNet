import os
import glob
from pydub import AudioSegment

#TODO: put path to your fma_small unzipped dir below
video_dir = '/Users/lauragustafson/6867/finalProject/fma_small/015/'

#TODO: put path to where you want the resulting wav files to be saved below
wav_video_dir = '/Users/lauragustafson/6867/finalProject/fma_small_wav/015/'


extension = '*.mp3'
os.chdir(video_dir)
for video in glob.glob(extension):
    wav_filename = os.path.splitext(os.path.basename(video))[0] + '.wav'
    wav_filename = wav_video_dir + wav_filename
    print wav_filename
    AudioSegment.from_file(video).export(wav_filename, format='wav')

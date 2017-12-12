#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:21:52 2017

@author: lauragustafson
"""
import pandas as pd
import ast

import os
import glob
from pydub import AudioSegment


x = pd.read_csv('../fma_metadata/tracks.csv', index_col=0, header=[0, 1])
x['index1'] = x.index

df = x[[('track', 'genre_top'),('set', 'subset'), ('index1', '')]]

'''
df2 = df[(
    df[('track', 'genre_top')] == 'Electronic' &
    (df[('set', 'subset')] == 'small' | df[('set', 'subset')] == 'medium')
    )]
'''

dfsmall = df[(df[('track', 'genre_top')] == 'Electronic') &
    (df[('set', 'subset')] == 'small')]

dfmedium = df[(df[('track', 'genre_top')] == 'Electronic') &
    (df[('set', 'subset')] == 'medium')]

tracks_small = dfsmall[('index1', '')].tolist()

filenames1 = []
filenames2 =[]
for track in tracks_small:
    tid_str = '{:06d}'.format(track)
    filenames1.append(tid_str[:3])
    filenames2.append(tid_str)

video_dir = '/Users/lauragustafson/6867/finalProject/fma_small/'

#TODO: put path to where you want the resulting wav files to be saved below
wav_video_dir = '/Users/lauragustafson/6867/finalProject/fma_small_techno_wav/'

extension = '*.mp3'
#os.chdir(video_dir)
i = 0
for i in range (len(filenames1)):
    try:
        wav_filename = filenames1[i] + filenames2[i] + '.wav'
        wav_filename = wav_video_dir + wav_filename
        AudioSegment.from_file(video_dir+filenames1[i]+ "/" + filenames2[i]+ '.mp3').export(wav_filename, format='wav')
        print(wav_filename)
        i += 1
    except Exception as e:
        print(e)
        continue
print("Finished", i, "wav files")

''''
COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
           ('track', 'genres'), ('track', 'genres_all'),
           ('track', 'genre_top')]
for column in COLUMNS:
    x[column] = x[column].map(ast.literal_eval)

COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
           ('album', 'date_created'), ('album', 'date_released'),
           ('artist', 'date_created'), ('artist', 'active_year_begin'),
           ('artist', 'active_year_end')]
for column in COLUMNS:
    x[column] = pd.to_datetime(x[column])

SUBSETS = ('small', 'medium', 'large')
x['set', 'subset'] = x['set', 'subset'].astype(
        'category', categories=SUBSETS, ordered=True)

COLUMNS = [('track', 'license'), ('artist', 'bio'),
           ('album', 'type'), ('album', 'information')]
for column in COLUMNS:
    x[column] = x[column].astype('category's)
'''

#!/usr/bin/env python
# coding: utf-8


## Specify save file path
dir_video_list = './video_list.xlsx'
dir_video_save_root = '../videos/videos_original/'
dir_video_save_logs_root = './logs'
dir_video_new_save_root = '../videos/videos_320x180' 
res_size_new = (320, 180)
fps = 25

## Import libraries
import os
import glob
import re
import pandas as pd
from collections import defaultdict

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pytube
from pytube import YouTube
from pytube.cli import on_progress

import ffmpeg

import torchvision 
from moviepy.editor import *


## Read video_list
df_video_list = pd.read_excel(dir_video_list)

## Read existing logs and add
def read_and_save_logs(dir_df_video_save_logs, df_video_save_logs_new):
    ## Read existing logs
    try:
            df_video_save_logs = pd.read_csv(dir_df_video_save_logs)
    except:
            df_video_save_logs = pd.DataFrame()

    ## Append to existing logs (if exists)
    df_video_save_logs = pd.concat([df_video_save_logs, 
                                    df_video_save_logs_new], axis=0).drop_duplicates().reset_index(drop=True)
    
    ## Save logs
    df_video_save_logs.to_csv(dir_df_video_save_logs, index=False)
        
## Download video
def download_video(video_uid, video_url, dir_video_save_root, dir_video_save_logs_root, save_video=False):
    dir_df_video_save_logs = dir_video_save_logs_root + '/videos_downloaded.xlsx'
    dir_df_video_save_logs_error = dir_video_save_logs_root + '/videos_download_errors.xlsx'
    
    ## Create new row of dataframe
    df_video_save_logs_new = pd.DataFrame({'video_uid': video_uid, 'video_url': video_url}, index=[0])
    
    ## Create directory
    if os.path.exists(dir_video_save_logs_root)==False:
        os.makedirs(dir_video_save_logs_root)
    
    ## Check if video was previously downloaded
    if ((os.path.exists(dir_video_save_root+ '/' + video_uid + '_video.mp4')==False) or
        (os.path.exists(dir_video_save_root+ '/' + video_uid + '.mp4')==False)):
        
        ## Download video
        yt = pytube.YouTube(video_url, on_progress_callback=on_progress)

        try:
            ## Download video and audio separately
            if (save_video == True):
                print(f'\t Downloading {video_url}')
                download_video = (yt.streams.filter(adaptive=True, file_extension='mp4')
                           .order_by('resolution')
                           .desc()
                           .first())
                download_video.download(output_path=dir_video_save_root, 
                                     filename=video_uid+'_video'+'.mp4')
                download_audio = (yt.streams.get_audio_only())

                download_audio.download(output_path=dir_video_save_root, 
                                     filename=video_uid+'_audio'+'.mp4')
                
                ## Save download log
                read_and_save_logs(dir_df_video_save_logs=dir_df_video_save_logs, 
                                   df_video_save_logs_new=df_video_save_logs_new)

        except:
            print(f'Error downloading video {video_uid}: {video_url}')
            
            ## Save download error log
            if os.path.exists(dir_video_save_logs_root):
                read_and_save_logs(dir_df_video_save_logs=dir_df_video_save_logs_error, 
                                   df_video_save_logs_new=df_video_save_logs_new)
            pass
        
    else:
        print(f'\t Already downloaded {video_uid}.mp4')

## Iterate through video_list and download videos
for row, data in df_video_list.iterrows():
    print(f'{row+1} out of {df_video_list.shape[0]}', end='')
    video_uid = data['video_uid']
    video_url = data['video_url']
    
    ## Download video
    download_video(video_uid, video_url, dir_video_save_root, dir_video_save_logs_root, save_video=True)
    
print('Download videos completed')


df_video_list['dir_filename'] = dir_video_save_root + df_video_list['video_uid']
df_video_list['dir_video_exists'] = [True if os.path.exists(x + '_video.mp4') else False for x in df_video_list['dir_filename']]
df_video_list['dir_audio_exists'] = [True if os.path.exists(x + '_audio.mp4') else False for x in df_video_list['dir_filename']]

df_video_list


## Merge audio and video
for row, data in df_video_list.iterrows():
    print(f'{row+1} out of {df_video_list.shape[0]}', end='')
    filename = data['dir_filename'] + '.mp4'
    
    dir_df_video_process_logs_error = dir_video_save_logs_root + '/videos_process_errors.xlsx'
    
    ## Create new row of dataframe
    df_video_process_logs_new = pd.DataFrame({'filename': filename}, index=[0])

    if os.path.exists(filename)==False:
        print(f'\t Processing {filename}')

        try:
            ## Merge video and audio file
            video = ffmpeg.input(df_video_list['dir_filename'][row] + '_video.mp4')
            audio = ffmpeg.input(df_video_list['dir_filename'][row] + '_audio.mp4')

            try:
                out = ffmpeg.output(video, audio, filename, vcodec='copy', acodec='aac', strict='experimental', loglevel='quiet')
                out.run(overwrite_output=True)
                
            except:
                print(f'\t Error processing {filename}')
                read_and_save_logs(dir_df_video_save_logs=dir_df_video_process_logs_error, 
                               df_video_save_logs_new=df_video_process_logs_new)

                pass
            
        except:
            print(f'\t Error processing {filename}')
            read_and_save_logs(dir_df_video_save_logs=dir_df_video_process_logs_error, 
                               df_video_save_logs_new=df_video_process_logs_new)
            continue
            
    else:
        print(f'\t Already processed {filename}')
            
print('Merging videos completed')

## Resample video down to same resolution and fps
def resample_video(video_uid, dir_filename, dir_video_new_save_root, res_size_new=res_size_new, fps=fps):
    dir_video_old = dir_filename
    dir_video_new = dir_video_new_save_root + '/' + video_uid + '.mp4'
    
    os.makedirs(dir_video_new_save_root, exist_ok=True)
    
    ## Search for processed file
    if not re.match(r'.*_video|.*_audio', dir_video_old):
        print(f'\t Processing {dir_video_old}')

        if os.path.exists(dir_video_new)==False:
            ## Resample video and audio
            video_audio = VideoFileClip(dir_video_old)
            video_fps = video_audio.fps
            video_size = video_audio.size #video_audio.w, video_audio.h
            
            video_audio.set_fps(fps)
            video_audio = video_audio.resize(newsize=res_size_new)
            video_audio.write_videofile(dir_video_new, fps=fps, verbose=False, logger=None)

            print(f'\t\t File saved: {dir_video_new}')

        else:
            print(f'\t\t Already exists: {dir_video_new}')


for row, data in df_video_list.iterrows():
    print(f'{row+1} out of {df_video_list.shape[0]}', end='')  
    
    video_uid = data['video_uid']
    dir_filename = data['dir_filename'] + '.mp4'
    
    dir_df_video_resample_logs_error = dir_video_save_logs_root + '/videos_resample_errors.xlsx'
    
    ## Create new row of dataframe
    df_video_resample_logs_new = pd.DataFrame({'dir_filename': dir_filename}, index=[0])

    try:
        resample_video(video_uid, dir_filename, dir_video_new_save_root, res_size_new=res_size_new, fps=fps)
    except:
        read_and_save_logs(dir_df_video_save_logs=dir_df_video_resample_logs_error, 
                               df_video_save_logs_new=df_video_resample_logs_new)
        pass

print('Resampling videos completed')


## Extract frames using ffmpeg and parallel
print('Extracting videos')

cmd = '''
num_threads=20
src_path=''' + dir_video_new_save_root + '''
dst_path=''' + dir_video_new_save_root.replace('videos', 'frames')'''

mkdir $dst_path

parallel -j $num_threads "mkdir ${dst_path}/{};ffmpeg -i ${src_path}/{}.mp4 -start_number 1 ${dst_path}/{}/'{}_%06d.png' -loglevel error" ::: `ls ${src_path} |cut -d '.' -f1`
'''
subprocess.check_output(cmd, shell=True) # frames are 1-indexed

print('Extracting videos completed')
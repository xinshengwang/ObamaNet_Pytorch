import os
import json
import numpy as np
from pydub import AudioSegment
from moviepy.editor import *
from pydub.silence import split_on_silence_with_timestamp

def get_timestamp(chunks):
    timestamp=[]
    for i, data in enumerate(chunks):
        _,ss,ed = data
        timestamp.append([ss,ed])
    return timestamp

def open_text(sent_path):
    with open(sent_path,'r',encoding='utf-8') as f:
        sents = f.read().splitlines()
    return sents

def segment_texts(index_start,index_end,sents,name,text_save_root):
    for i in range(len(index_start)):
        save_path = text_save_root + name + '_' + str(i) + '.txt'
        new_sents = []
        start = index_start[i]
        end = index_end[i]
        for j in range(start,end):
            sent = sents[j].split('[')[0]
            new_sents.append(sent)

        new_sent = ' '.join(new_sents)
        new_sent = new_sent.replace('  ',' ')
        with open(save_path,'w') as f:
            f.writelines(new_sent)

def save_timestamp(timestamp,save_root,name):
    save_path = save_root + name + '_' + str(i) + '.json'
    with open(save_path,'w') as f:
        json.dump(timestamp,f)


def segment_video(timestamp,video_path,save_root,name):
    video_whole = VideoFileClip(video_path)
    i = 0
    for st,ed in timestamp:
        save_file = save_root + name + '_' + str(i) + '.mp4'
        st = float(st) / 1000
        ed = float(ed) / 1000     
        video = video_whole.subclip(st,ed)
        video.write_videofile(save_file)
        i += 1


audio_root = '/home/work_nfs3/xswang/data/avatar/Obama/subset_for_develop/raw/audios/'
save_audio_root = '/home/work_nfs3/xswang/data/avatar/Obama/subset_for_develop/clip/audios/'
text_root = '/home/work_nfs3/xswang/data/avatar/Obama/subset_for_develop/raw/texts/'
save_text_root = '/home/work_nfs3/xswang/data/avatar/Obama/subset_for_develop/clip/texts/'
save_time_root = '/home/work_nfs3/xswang/data/avatar/Obama/subset_for_develop/clip/timestamp/'
video_root = '/home/work_nfs3/xswang/data/avatar/Obama/subset_for_develop/raw/videos/'
save_video_root = '/home/work_nfs3/xswang/data/avatar/Obama/subset_for_develop/clip/videos/'

names = os.listdir(audio_root)
audio_names = []
for name in names:
    audio_names.append(name)
for i, name in enumerate(audio_names):
    audiopath = os.path.join(audio_root, name)
    text_path = os.path.join(text_root,name.replace('.wav','.txt'))
    video_path = os.path.join(video_root,name.replace('.wav','.mp4'))
    sents = open_text(text_path)
    print(audiopath)
    sound = AudioSegment.from_file(audiopath, format='wav')
    chunks = split_on_silence_with_timestamp(sound, min_silence_len=1000, silence_thresh=-45,keep_silence=500)
#     filepath = os.path.split(audiopath)[0]
    for j in range(len(chunks)):
        new = chunks[j][0]
        save_name = save_audio_root + '{}_{}.{}'.format(name.split('.')[0], j, 'wav')
        new.export(save_name, format='wav')
    timestamp = get_timestamp(chunks)
    text_path = text_root + name.replace('.wav','.txt')
    sents = open_text(text_path)
    starts=[]
    ends =[]
    for sent in sents:
        st,ed=sent.split()[-1].replace('[','').replace(']','').split(',')
        start_time = float(st) * 1000
        end_time = float(ed) *1000  
        starts.append(start_time)
        ends.append(end_time)
    starts = np.array(starts)
    ends = np.array(ends)
    index_start = []
    index_end = []
    for ss,_ in timestamp:
        index_s = np.where(starts>ss)[0][0]
        if index_s > 0:
            gap = starts[index_s]-ss
            gap2 = ss - starts[index_s-1]
            if gap > gap2:
                index_s -= 1
        index_start.append(index_s)
    index_end = index_start[1:]
    index_end.append(len(starts))
    segment_texts(index_start,index_end,sents,name.split('.')[0],save_text_root)
    save_timestamp(timestamp,save_time_root,name.split('.')[0])  
    segment_video(timestamp,video_path,save_video_root,name.split('.')[0])
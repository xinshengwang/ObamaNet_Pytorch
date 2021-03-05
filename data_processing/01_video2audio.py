import os
from tqdm import tqdm
from moviepy.editor import *

root = '/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip/videos'
save_root = '/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip/audios'
names = os.listdir(root)
for name in tqdm(names):
    video_path = os.path.join(root,name)
    audio_path = os.path.join(save_root,name.replace('.mp4','.wav'))
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)    
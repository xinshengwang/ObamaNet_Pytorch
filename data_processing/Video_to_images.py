import subprocess
import os
from tqdm import tqdm

video_root = '/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip/videos'
image_root = '/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip/images'
prevdir = '/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip/images'
video_names = os.listdir(video_root)
for video_name in tqdm(video_names):
    video_path = os.path.join(video_root,video_name)
    name = video_name.split('.')[0]
    save_dir = os.path.join(image_root,name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cmd = 'ffmpeg -i %s -r 20 -vf scale=-1:720 %s/%s/'%(video_path,image_root,name) + '%05d.bmp'
    subprocess.call(cmd ,shell=True)
    image_file = '%s/%s/'%(image_root,name)
    os.chdir(image_file)
    cmd = 'mogrify -format jpg *.bmp'
    subprocess.call(cmd ,shell=True)
    cmd ='rm -rf *.bmp'
    subprocess.call(cmd ,shell=True)
    os.chdir(prevdir)

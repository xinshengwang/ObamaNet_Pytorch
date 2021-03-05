# ObamaNet_Pytorch

This is a PyTroch implementation of the paper "ObamaNet: Photo-realistic lip-sync from text". 

## Requirements

* Python 3.x
* PyTorch
* OpenCV
* Dlib
* Python Speech Features 

For video data processing, the following tools are used:

* ffmpeg (sudo apt-get install ffmpeg)
* YouTube-dl

## Dataprocessing

* Download videos from YouTube

```
youtube-dl --batch-file data/obama_addresses.txt -o 'data/videos/%(autonumber)s.%(ext)s' -f "best[height=720]" --autonumber-start 1
```

* Download Subtitle from YouTube
```
youtube-dl --sub-lang en --skip-download --write-sub --output 'data/captions/%(autonumber)s.%(ext)s' --batch-file data/obama_addresses.txt --ignore-config
```

* Video segmentation

As I'd like to train a TTS system, I segmented the video based on the silence, and then the resulted video clips were aligned to the subtitles. However, if you just want to train an audio driven talking head generation system, you can segment the video to fiexed-length clips as you want. The silence-based segmentation process can refer to the [segment code](data_processing/segment_video_text_based_silence.py).

* Video to Images

```
cd data_processing
python Video_to_images.py
```

* Crop image

Here, I used the Dlib to detect the face region, based on which the image were croped into 256*256. 
```
cd data_processing
python crop_image.py
```
* Landmarks (key points) / audio feature extration and processing

```
cd data_processing
python extract_audio_feature.py
python extract_keypoint.py
```

* Mask the mouth region and draw lips for training pix2pix
```
cd data_processing
python mask_mouth_region_and_draw_lips.py

```

## Training

* lip sync 
```
python audio2keypoint.py --train
```
* pix2pix
```
python audio2keypoint.py --train
```

## Reference 

* [pix2pix](https://github.com/taey16/pix2pixBEGAN.pytorch)
* [Keras implementation of lip sync from audio](https://github.com/amtsai96/Learning-Lip-Sync-from-Audio)


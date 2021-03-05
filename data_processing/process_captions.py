import os
from webvtt import WebVTT

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

caption_path = 'captions/00001.en.vtt'
captions = WebVTT().read(caption_path)

k = 0
for idx, caption in enumerate(captions):
    ss = get_sec(caption.start)
    end = get_sec(caption.end)
    caption = caption.text.replace('\n',' ')  #+ '[' + str(ss) + ',' + str(end)  + ']'
    caption = caption.replace('The President: ','')
    if caption[-1] != '.' or k !=0 :
        if k == 0 :
            start = ss
            text = caption
        else:
            end = end
            text = text + ' ' + caption
        k += 1
        if caption[-1] == '.':
            k = 0
    else:
        start = ss
        text = caption
    if text[-1] == '.':
        text = text + '[' + str(start) + ',' + str(end)  + ']'
        print(text)
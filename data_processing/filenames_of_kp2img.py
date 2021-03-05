import json
import os

def get_split_files(split_name):
    root = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/split'
    train_file = os.path.join(root,split_name)

    with open(train_file) as f:
        t_f = json.load(f)

    _files = []
    for vname in t_f:
        vpath = os.path.join('/home/work_nfs3/xswang/data/avatar/Obama/clip/select/for_ObamaNet/images_lip/',vname)
        inames = os.listdir(vpath)
        for iname in inames:
            _file = os.path.join(vname,iname.replace('.png',''))
            _files.append(_file)
    save_path = os.path.join(root,'for_kp2img',split_name)
    with open(save_path,'w') as f:
        json.dump(_files,f)

def downsample(split_file):
    file_name = split_file.split('/')[-1]
    with open(split_file,'rb') as f:
        filenames = json.load(f)
    select_names = []
    for name in sorted(filenames):
        num = int(name.split('/')[-1])
        if (num - 1) % 5 == 0:
            select_names.append(name)
    save_path = os.path.join('/home/work_nfs3/xswang/data/avatar/Obama/clip/select/split/for_kp2img/5fps',file_name)
    with open(save_path,'w') as f:
        json.dump(select_names,f)



get_split_files('train.json')
get_split_files('test.json')
get_split_files('val.json')

downsample('/home/work_nfs3/xswang/data/avatar/Obama/clip/select/split/for_kp2img/train.json')
downsample('/home/work_nfs3/xswang/data/avatar/Obama/clip/select/split/for_kp2img/test.json')
downsample('/home/work_nfs3/xswang/data/avatar/Obama/clip/select/split/for_kp2img/val.json')


print('pause')
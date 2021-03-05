import os
from glob import glob
import json
import random

# 5% val and test
def data_split(full_list):
    n_total = len(full_list)
    tr_num = int(n_total * 0.9)
    tr_ts_num = int(n_total * 0.95)
    random.shuffle(full_list)
    train_list = full_list[:tr_num]
    test_list = full_list[tr_num:tr_ts_num]
    val_list = full_list[tr_ts_num:]
    return sorted(train_list),sorted(val_list),sorted(test_list)

root = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/for_ObamaNet/images_lip/'
# names = sorted(glob(root + '/*.wav'))
name_list = sorted(os.listdir(root))
# name_list = [name[len(root):-len('.wav')] for name in names]

train_list,val_list,test_list = data_split(name_list)

save_root = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/split/'
with open(save_root + 'train.json','w') as f:
    json.dump(train_list,f)
with open(save_root + 'val.json','w') as f:
    json.dump(val_list,f)
with open(save_root + 'test.json','w') as f:
    json.dump(test_list,f)
    
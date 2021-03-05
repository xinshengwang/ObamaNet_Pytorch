import numpy as np
import yaml
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# audio2keypoint parameter
__C.in_channel = 26
__C.hidden_size = 64
__C.lstm_layer = 2
__C.drop = 0.25
__C.out_channel = 8
__C.time_delay = 20

# Unet parameter
__C.Unet = edict()
__C.Unet.inputChannelSize = 3
__C.Unet.outputChannelSize = 3
__C.Unet.ngf = 64
__C.Unet.ndf = 64
__C.Unet.poolSize = 50


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        raise TypeError('{} is not a valid edict type'.format(a))

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise TypeError(('Type mismatch ({} vs. {}) for config key: {}'.format(type(b[k]), type(v), k)))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                raise KeyError('Error under config key: {}'.format(k))
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)

from utils import *

inputFolder = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/audios/'
outputFolder = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/for_ObamaNet/audio_kp/'
resumeFrom = 0
frame_rate = 5
kp_type = 'mel'

if not(os.path.exists(outputFolder)):
    # Create directory
    subprocess.call('mkdir -p ' + outputFolder, shell=True)

filelist = sorted(glob(inputFolder+'*.wav'))

d = {}

for idx, file in enumerate(tqdm(filelist[resumeFrom:])):
    key = file[len(inputFolder):-len('.wav')]

    if(kp_type == 'world'):
        x, fs = sf.read(file)
        # 2-1 Without F0 refinement
        f0, t = pw.dio(x, fs, f0_floor=50.0, f0_ceil=600.0,
                        channels_in_octave=2,
                        frame_period=frame_rate,
                        speed=1.0)
        sp = pw.cheaptrick(x, f0, t, fs)
        ap = pw.d4c(x, f0, t, fs)
        features = np.hstack((f0.reshape((-1, 1)), np.hstack((sp, ap))))

    elif (kp_type == 'mel'):
        (rate, sig) = wav.read(file)
        features = logfbank(sig,rate)

    d[key] = features

saveFilename = outputFolder + 'audio_kp' + '_' + kp_type + '.pickle'
    
with open(saveFilename, "wb") as output_file:
    pkl.dump(d, output_file)
    """
    oldSaveFilename = outputFolder + 'audio_kp' + str(idx+resumeFrom-2) + '_' + kp_type + '.pickle'

    if not (os.path.exists(saveFilename)):
        with open(saveFilename, "wb") as output_file:
            pkl.dump(d, output_file)
            # print('Saved output for', (idx+resumeFrom+1), 'file.')
    else:
        # Resume
        with open(saveFilename, "rb") as output_file:
            d = pkl.load(output_file)
            print('Loaded output for ', (idx+resumeFrom+1), ' file.')

    # Keep removing stale versions of the files
    if (os.path.exists(oldSaveFilename)):
        cmd = 'rm -rf ' + oldSaveFilename
        subprocess.call(cmd, shell=True)
    """

print('Saved Everything audio kp')
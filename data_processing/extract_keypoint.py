from utils import *
np.seterr(divide='ignore', invalid='ignore')

EPSILON = 1e-8
video_fps = 20

def extract_keypoint(inputFolder,outputFolder,resumeFrom = 0):
    if not(os.path.exists(outputFolder)):
        # Create directory
        subprocess.call('mkdir -p ' + outputFolder, shell=True)

    directories = sorted(glob(inputFolder+'*/'))

    d = {}

    for idx, directory in tqdm(enumerate(directories[resumeFrom:])):
        key = directory[len(inputFolder):-1]
        imglist = sorted(glob(directory+'*.jpg'))
        big_list = []
        for file in tqdm(imglist):
            keypoints = get_facial_landmarks(file)
            if not (keypoints.shape[0] == 1): # if there are some kp then
                l = getKeypointFeatures(keypoints)
                unit_kp, N, tilt, mean = l[0], l[1], l[2], l[3]
                kp_mouth = unit_kp[48:68]
                store_list = [kp_mouth, N, tilt, mean, unit_kp, keypoints]
                prev_store_list = store_list
                big_list.append(store_list)
            else:
                big_list.append(prev_store_list)
        d[key] = big_list
        print('processing the %d-video'%(idx))
    saveFilename = outputFolder + 'kp' + '.pickle'
    with open(saveFilename, "wb") as output_file:
        pkl.dump(d, output_file)



def extract_pca(inputFolder,outputFolder):
    new_list = []

    filename = inputFolder + 'kp' + '.pickle'

    if not(os.path.exists(outputFolder)):
        # Create directory
        subprocess.call('mkdir -p ' + outputFolder, shell=True)

    if (os.path.exists(filename)):
        with open(filename, 'rb') as file:
            big_list = pkl.load(file)
        print('Keypoints file loaded')
    else:
        print('Input keypoints not found')
        sys.exit(0)

    print('Unwrapping all items from the big list')

    for key in tqdm(sorted(big_list.keys())):
        for frame_kp in big_list[key]:
            kp_mouth = frame_kp[0]
            x = kp_mouth[:, 0].reshape((1, -1))
            y = kp_mouth[:, 1].reshape((1, -1))
            X = np.hstack((x, y)).reshape((-1)).tolist()
            new_list.append(X)

    X = np.array(new_list)

    pca = PCA(n_components=8)
    pca.fit(X)
    with open(outputFolder + 'pca' + '.pickle', 'wb') as file:
        pkl.dump(pca, file)

    with open(outputFolder + 'explanation' + '.pickle', 'wb') as file:
        pkl.dump(pca.explained_variance_ratio_, file)

    print('Explanation for each dimension:', pca.explained_variance_ratio_)
    print('Total variance explained:', 100*sum(pca.explained_variance_ratio_))
    print('')
    print('Upsampling...')

    # Upsample the lip keypoints
    upsampled_kp = {}
    for key in tqdm(sorted(big_list.keys())):
        # print('Key:', key)
        try: 
            nFrames = len(big_list[key])
            factor = int(np.ceil(100/video_fps))
            # Create the matrix
            new_unit_kp = np.zeros((int(factor*nFrames), big_list[key][0][0].shape[0], big_list[key][0][0].shape[1]))
            new_kp = np.zeros((int(factor*nFrames), big_list[key][0][-1].shape[0], big_list[key][0][-1].shape[1]))

            # print('Shape of new_unit_kp:', new_unit_kp.shape, 'new_kp:', new_kp.shape)

            for idx, frame in enumerate(big_list[key]):
                # Create two lists, one with original keypoints, other with unit keypoints
                new_kp[(idx*(factor)), :, :] = frame[-1]
                new_unit_kp[(idx*(factor)), :, :] = frame[0]

                if (idx > 0):
                    start = (idx-1)*factor + 1
                    end = idx*factor
                    for j in range(start, end):
                        new_kp[j, :, :] = new_kp[start-1, :, :] + ((new_kp[end, :, :] - new_kp[start-1, :, :])*(np.float(j+1-start)/np.float(factor)))
                        # print('')
                        l = getKeypointFeatures(new_kp[j, :, :])
                        # print('')
                        new_unit_kp[j, :, :] = l[0][48:68, :]
            
            upsampled_kp[key] = new_unit_kp
        except:
            print(key)

    # Use PCA to de-correlate the points
    d = {}
    keys = sorted(upsampled_kp.keys())
    for key in tqdm(keys):
        x = upsampled_kp[key][:, :, 0]
        y = upsampled_kp[key][:, :, 1]
        X = np.hstack((x, y))
        X_trans = pca.transform(X)
        d[key] = X_trans

    with open(outputFolder + 'pkp' + '.pickle', 'wb') as file:
        pkl.dump(d, file)
    print('Saved Everything')


inputFolder = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/images_crop/'
outputFolder = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/for_ObamaNet/image_kp_raw/'
extract_keypoint(inputFolder,outputFolder)

inputFolder = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/for_ObamaNet/image_kp_raw/'
outputFolder = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/for_ObamaNet/pca/'
extract_pca(inputFolder,outputFolder)
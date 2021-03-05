
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle as pkl
from time import time
import cv2

def subsample(y, fps_from = 100.0, fps_to = 29.97):
	factor = int(np.ceil(fps_from/fps_to))
	# Subsample the points
	new_y = np.zeros((int(y.shape[0]/factor), 20, 2)) #(timesteps, 20) = (500, 20x2)
	for idx in range(new_y.shape[0]):
		if not (idx*factor > y.shape[0]-1):
			# Get into (x, y) format
			new_y[idx, :, 0] = y[idx*factor, 0:20]
			new_y[idx, :, 1] = y[idx*factor, 20:]
		else:
			break
	# print('Subsampled y:', new_y.shape)
	new_y = [np.array(each) for each in new_y.tolist()]
	# print(len(new_y))
	return new_y

def drawLips(keypoints, new_img, c = (255, 255, 255), th = 1, show = False):

	keypoints = np.float32(keypoints)

	for i in range(48, 59):
		cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[59]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[60]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[54]), tuple(keypoints[64]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[67]), tuple(keypoints[60]), color=c, thickness=th)
	for i in range(60, 67):
		cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)

	if (show == True):
		cv2.imshow('lol', new_img)
		cv2.waitKey(10000)

def getOriginalKeypoints(kp_features_mouth, N, tilt, mean):
	# Denormalize the points
	kp_dn = N * kp_features_mouth
	# Add the tilt
	x, y = kp_dn[:, 0], kp_dn[:, 1]
	c, s = np.cos(tilt), np.sin(tilt)
	x_dash, y_dash = x*c + y*s, -x*s + y*c
	kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1,1))))
	# Shift to the mean
	kp = kp_tilt + mean
	return kp

#########################################################################################

# Load the files
with open('data/pca/pkp1467.pickle', 'rb') as pkl_file:
	video_kp = pkl.load(pkl_file)
with open('data/pca/pca1467.pickle', 'rb') as pkl_file:
	pca = pkl.load(pkl_file)
# Get the original keypoints file
with open('data/a2key_data/kp_test.pickle', 'rb') as pkl_file:
	kp = pkl.load(pkl_file)

# Get the data
X, y = [], [] # Create the empty lists
# audio = audio_kp[key_audio]
video = video_kp['00001-000']


# Get audio features
(rate, sig) = wav.read(key_audio)
audio = logfbank(sig,rate)


# if (len(audio) > len(video)):
# 	audio = audio[0:len(video)]
# else:
# 	video = video[0:len(audio)]
start = (time_delay-look_back) if (time_delay-look_back > 0) else 0
for i in range(start, len(audio)-look_back):
	a = np.array(audio[i:i+look_back])
	# v = np.array(video[i+look_back-time_delay]).reshape((1, -1))
	X.append(a)
	# y.append(v)

for i in range(start, len(video)-look_back):
	v = np.array(video[i+look_back-time_delay]).reshape((1, -1))
	y.append(v)

X = np.array(X)
y = np.array(y)
shapeX = X.shape
shapey = y.shape
print('Shapes:', X.shape)
X = X.reshape(-1, X.shape[2])
y = y.reshape(-1, y.shape[2])
print('Shapes:', X.shape)

scalerX = MinMaxScaler(feature_range=(0, 1))
scalery = MinMaxScaler(feature_range=(0, 1))

X = scalerX.fit_transform(X)
y = scalery.fit_transform(y)


X = X.reshape(shapeX)
# y = y.reshape(shapey[0], shapey[2])

# print('Shapes:', X.shape, y.shape)
# print('X mean:', np.mean(X), 'X var:', np.var(X))
# print('y mean:', np.mean(y), 'y var:', np.var(y))

y_pred = model.predict(X)

# Scale it up
y_pred = scalery.inverse_transform(y_pred)
# y = scalery.inverse_transform(y)

y_pred = pca.inverse_transform(y_pred)
# y = pca.inverse_transform(y)

print('Upsampled number:', len(y_pred))

y_pred = subsample(y_pred, 100, 34)

# y = subsample(y, 100, 100)

# error = np.mean(np.square(np.array(y_pred) - np.array(y)))

# print('Error:', error)

print('Subsampled number:', len(y_pred))

# Visualization
# Cut the other stream according to whichever is smaller
if (len(kp) < len(y_pred)):
	n = len(kp)
	y_pred = y_pred[:n]
else:
	n = len(y_pred)
	kp = kp[:n]


for idx, (x, k) in enumerate(zip(y_pred, kp)):

	unit_mouth_kp, N, tilt, mean, unit_kp, keypoints = k[0], k[1], k[2], k[3], k[4], k[5]
	kps = getOriginalKeypoints(x, N, tilt, mean)
	keypoints[48:68] = kps

	imgfile = 'data/a2key_data/images/' + str(idx+1).rjust(5, '0') + '.png'
	im = cv2.imread(imgfile)
	drawLips(keypoints, im, c = (255, 255, 255), th = 1, show = False)

	# make it pix2pix style
	im_out = np.zeros_like(im)
	im1 = np.hstack((im, im_out))
	# print('Shape: ', im1.shape)
	cv2.imwrite(outputFolder + str(idx) + '.png', im1)

print('Done writing', n, 'images')

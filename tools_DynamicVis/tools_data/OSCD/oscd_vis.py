import glob
import os
import mmcv
import numpy as np

in_path = '/Users/kyanchen/Downloads/OSCD/train'
labels = glob.glob(os.path.join(in_path, 'label', '*.png'))
h, w = [], []
for label in labels:
	imgA = os.path.join(in_path, 'A', os.path.basename(label))
	imgB = os.path.join(in_path, 'B', os.path.basename(label))
	label = mmcv.imread(label)
	imgA = mmcv.imread(imgA)
	imgB = mmcv.imread(imgB)
	h.append(label.shape[0])
	w.append(label.shape[1])
	# mmcv.imshow(np.concatenate([label, imgA, imgB], axis=1), 'label | imgA | imgB', 0)
print(np.mean(h), np.mean(w))
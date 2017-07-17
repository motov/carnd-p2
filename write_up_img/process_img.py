import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

def gray_scale(X):
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    X = X.reshape(X.shape + (1,))
    return X


validation_file = 'valid.p'
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
X_valid, y_valid = valid['features'], valid['labels']

img1 = X_valid[400:401]
fig = plt.figure()
plt.imshow(img1[0])
fig.savefig('valid_img1.png')
img1_bk = gray_scale(img1)
img1_bk.shape
fig2 = plt.figure()
plt.imshow(img1_bk[0,:,:,0], cmap='gray')
fig2.savefig('valid_img1_gray.png')

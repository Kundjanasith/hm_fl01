import utils 
import tenseal as ts 
import matplotlib.pyplot as plt 
import numpy as np
import time


def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx

def context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, 4096, coeff_mod_bit_sizes=[20, 20, 20])
    context.global_scale = pow(2,10)
    context.generate_galois_keys()
    return context 

def model_encryption(context,model):
    encrypted_model = utils.model_init()
    model_weights = []
    for l in model.layers:
        l_idx = getLayerIndexByName(model, l.name)
        w_arr = []
        for w_idx in range(len(encrypted_model.get_layer(index=l_idx).get_weights())):
            w = encrypted_model.get_layer(index=l_idx).get_weights()[w_idx]
            start = time.time()
            w = ts.plain_tensor(w)
            w = ts.ckks_tensor(context, w)
            end = time.time()
            print(l.name,w_idx,w.shape,end-start)
            w_arr.append(w)
        model_weights.append(w_arr)
    return model_weights
            

(X_train, Y_train), (X_test, Y_test) = utils.load_dataset()
model = utils.model_init()
# model.load_weights('res_0.78550.h5')
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# res = model.evaluate(X_test, Y_test)
# print(res)
context = context()
encrypted_model = model_encryption(context,model)


# print(res)
# tensor_msg = ts.plain_tensor(model)
# encrypted_tensor_msg = ts.ckks_tensor(context, tensor_msg)
# decrypted_tensor_msg = encrypted_tensor_msg.decrypt()
# list_decrypted_tensor_msg = np.array(decrypted_tensor_msg.tolist()).reshape(32,96)
# img2 = list_decrypted_tensor_msg
# plt.imshow(img2)
# plt.show() 

"""
import sys
from scipy.linalg import norm
from scipy import sum, average
def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr
def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng
def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)
n_m, n_0 = compare_images(img1, img2)
print("Manhattan norm:", n_m, "/ per pixel:", n_m/img1.size)
print("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)
"""
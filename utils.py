from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import clear_session
import random
import numpy as np

def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
            
def load_dataset():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    return (X_train, Y_train), (X_test, Y_test)

def sampling_data(num_samples):
    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(len(x_train))
    num_of_each_dataset = num_samples
    print(num_of_each_dataset)
    split_data_index = []
    while len(split_data_index) < num_of_each_dataset:
        item = random.choice(range(x_train.shape[0]))
        if item not in split_data_index:
            split_data_index.append(item)
    new_x_train = np.asarray([x_train[k] for k in split_data_index])
    new_y_train = np.asarray([y_train[k] for k in split_data_index])
    return new_x_train, new_y_train

def model_init():
    model = MobileNet(include_top=False,input_tensor=Input(shape=(32,32,3)))
    x = model.output
    x = Flatten()(x)
    x = Dense(512,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(10,activation='softmax')(x)
    model = Model(model.input,x)
    return model

def aggregated(model_path):
    global_model = model_init()
    global_model.load_weights(model_path[0])
    model_dict = {}
    count = 0
    for l in global_model.layers:
        l_idx = getLayerIndexByName(global_model, l.name)
        for w_idx in range(len(global_model.get_layer(index=l_idx).get_weights())):
            w = global_model.get_layer(index=l_idx).get_weights()[w_idx]
            model_dict[count] = []
            model_dict[count].append(w)
            count = count + 1
    clear_session()
    for p in model_path[1:]:
        count = 0
        client_model = model_init()
        print(p)
        client_model.load_weights(p)
        for l in client_model.layers:
            l_idx = getLayerIndexByName(client_model, l.name)
            for w_idx in range(len(client_model.get_layer(index=l_idx).get_weights())):
                w = client_model.get_layer(index=l_idx).get_weights()[w_idx]
                model_dict[count].append(w)
                count = count + 1
    clear_session()
    aggregated_model = model_init()
    count = 0
    for l in aggregated_model.layers:
        l_idx = getLayerIndexByName(aggregated_model, l.name)
        w_arr = []
        for w_idx in range(len(aggregated_model.get_layer(index=l_idx).get_weights())):
            w = aggregated_model.get_layer(index=l_idx).get_weights()[w_idx]
            w_avg = np.nanmean(np.array(model_dict[count]),axis=0)
            count = count + 1
            w_arr.append(w_avg)
        aggregated_model.get_layer(index=l_idx).set_weights(w_arr)
    return aggregated_model
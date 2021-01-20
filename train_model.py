import sys 
import pandas as pd
import logging 
from ast import literal_eval
from tqdm import tqdm
import numpy as np 
import pickle 
logging.getLogger().setLevel(logging.INFO)

from IPython import embed
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import activations
import keras 
from sklearn.model_selection import train_test_split


from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())



def read_pickle(name):
    logging.info(f'|-> Reading data from {name} \n')
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data 
     
def read_csv(path):
    logging.info(f"|-> Reading path {path}")
    data = pd.read_csv(path) 
    logging.info(f"|-> Successfully read data of shape {data.shape} \n")
    return data

def normalize(X):
    logging.info('|-> Normalizing data ')

    X = [ x / 255  for x in tqdm(X) ]

    return X 

def build_model(data):

    print(data.shape)
    
    X = data['img_data'].values
     
    Y = data[['s_0', 
            's_1', 
            's_2', 
            's_3',
            's_4',
            's_5',
            's_6',
            's_7',
            's_8',
            's_9',
            's_10',
            's_11',
            's_12',
            's_13',
            's_14',
            's_15',
            's_16',
            's_17',
            's_18',
            's_19',
            's_20',
            's_21',
            's_22',
            's_23']].values


    X = [x[0] for x in X]
    # X = normalize(X)
    X = np.array(X)
    Y = np.array(Y)

    print(X[0].shape)
    print(Y.shape)
    print(X.shape)
    
    # reshape 
    # Split test train 
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state = 100)

    print('X_train shape : ', X_train.shape)
    print('Y_train shape : ', Y_train.shape)
    print('X_test shape  : ', X_test.shape )
    print('Y_test shape  : ', Y_test.shape )
    model = keras.Sequential([
       
            layers.Conv2D(64, kernel_size= (3,3), input_shape= (330,490,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Activation(activations.relu), 
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(64, kernel_size= (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Activation(activations.relu), 
            layers.MaxPooling2D(pool_size=(2,2)),

            
            layers.Conv2D(64, kernel_size= (3,3), input_shape= (330,490,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Activation(activations.relu), 
            layers.MaxPooling2D(pool_size=(2,2)),


            layers.GlobalAveragePooling2D(),

            layers.Flatten(),
            layers.Dense(24, activation='softmax')

            ])
    
     
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train,Y_train,epochs=100, batch_size =1)

def main():
    # print('Num GPUs : ', len(tf.config.experimental.list_physical_devices('GPU')))
    # Load data 
    data = read_pickle('train_img_spec.pkl') 
    print(data.head())
    logging.info(f'|-> Read data of shpae {data.shape}')
    
    build_model(data)
    
if __name__=='__main__':
    main()

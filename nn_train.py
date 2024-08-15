import os
import gensim
from pyvi import ViTokenizer
from nn_utils import *
import warnings
from sklearn.model_selection import train_test_split
import json
import pandas as pd

warnings.filterwarnings("ignore")

def read_data(file):
    data = pd.read_csv(file, encoding='utf-8')
    data['text'].dropna(inplace=True)
    return data['text'], data['label']

def get_data(X_data, y_data):
    data = []
    for X in X_data:
        X = gensim.utils.simple_preprocess(X)
        X = ' '.join(X)
        X = ViTokenizer.tokenize(X) #Tokenizer
        X = X.lower() #Lowercase
        data.append(X)
    return data, y_data


def nn_train(labId:str=None, data_dir:str=None, model_type:str=None, max_sample_length:int=100, learning_rate:float=2e-3, activation:str='relu', epochs:int=2, batch_size:int=2, val_size:float=0.2, layers_num:int=1, rnn_units:int=100, rnn_bidirectional:bool=False, num_filters:int=2, kernel_size=2, pooling_size:int=2, padding:bool=True):
    '''
    params:
    model_type: str - tên loại model sử dụng
    data_dir : str - đường dẫn thư mục chứa dữ liệu
    model_path: str - đường dẫn thư mục lưu model
    epochs : int - số epoch 
    val_size : float - kích thước chia tập train, val
    learning_rate: float - tỉ lệ học
    max_sample_length: kích thước tối đa của 1 sample
    activation: hàm kích hoạt sử dụng
    layers_num: Số lượng layer được chồng lên nhau
    rnn_units: Sử dụng cho LSTM và GRU
    rnn_bidirectional: có sử dụng mã hóa 2 chiều với LSTM và GRU
    num_filters: Số lượng filters được sử dụng với CNN
    kernel_size: kích thước kernel_size
    pooling_size: kích thước của pooling_size
    padding: có sử dụng padding để giữ kích thước của đặc trưng CNN
    '''
    #Đọc dữ liệu
    X_data, y_data = read_data(data_dir)
    X_train, y_train = get_data(X_data, y_data)

    checkpoint_dir = f'./modelDir/{labId}/log_train/{model_type["modelName"]}/'

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42)
    
    X_train, y_train = preprocess_data(
        X_train, y_train, max_sample_length)
    
    X_val, y_val = preprocess_data(
        X_val, y_val, max_sample_length)


    
    classifier = None
    if model_type['backbone'] == "lstm":
        classifier = create_lstm_model(max_sample_length=max_sample_length, num_layers=layers_num, bidirectional=rnn_bidirectional, units=rnn_units, activation=activation, learning_rate=learning_rate)
    elif model_type['backbone'] == "cnn":
        classifier = create_cnn_model(max_sample_length=max_sample_length, num_layer=layers_num, num_filters=num_filters, kernel_size=kernel_size, pooling_size=pooling_size, padding=padding, activation=activation, learning_rate=learning_rate)
    elif model_type['backbone'] == "gru":
        classifier = create_gru_model(max_sample_length=max_sample_length, num_layers=layers_num, bidirectional=rnn_bidirectional, units=rnn_units, activation=activation, learning_rate=learning_rate)

    print(classifier.summary())
    history = classifier.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, 
        batch_size=batch_size
    )

    model_json = classifier.to_json()
    with open(checkpoint_dir + f"model_{model_type}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights(checkpoint_dir + f"model_{model_type}.h5")

    print("Train history", history)
  
    return {
        "history": json.dumps(history.history),
    }
  

if __name__ == "__main__":
    model_type = {}
    model_type['backbone'] = "cnn"
    model_type['modelName'] = "_cnn"
    nn_train(
        data_dir="datasets/_train.csv",
        learning_rate=0.01,
        epochs=10,
        batch_size=2,
        val_size=0.2,
        model_type=model_type,
        labId='6ca93b2-3142-428f-9ffb-bd52a8fae21d',
        layers_num = 2,
        rnn_units=100,
        activation='relu',
        rnn_bidirectional=True
    )

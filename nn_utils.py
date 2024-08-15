
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM, GRU, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.utils import to_categorical
import fasttext
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np

print("Download FastText Begin")
model_path = hf_hub_download(repo_id="facebook/fasttext-vi-vectors", filename="model.bin", local_dir="models")
word2vec = fasttext.load_model(model_path)
print("Download FastText End")

#layer_num: số lớp cần thêm
#units : list chiều của không gian đầu ra tương ứng với các layer
#activation: relu, linear, sigmioid,tanh,softmax

def create_lstm_model(max_sample_length=100,num_layers=1, bidirectional=False, units=150, activation='relu', learning_rate=0.001):
    model = Sequential()

    for i in range(num_layers):
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=(i < num_layers - 1), activation=activation), 
                                        input_shape=(max_sample_length, 300)))
            else:
                model.add(LSTM(units, return_sequences=(i < num_layers - 1), activation=activation,
                               input_shape=(max_sample_length, 300)))
        else:
            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=(i < num_layers - 1), activation=activation)))
            else:
                model.add(LSTM(units, return_sequences=(i < num_layers - 1), activation=activation))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer = Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(max_sample_length=100, num_layers=1, bidirectional=False, units=150, activation='relu', learning_rate=0.001):
    model = Sequential()

    for i in range(num_layers):
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(GRU(units, return_sequences=(i < num_layers - 1), activation=activation), 
                                        input_shape=(max_sample_length, 300)))
            else:
                model.add(GRU(units, return_sequences=(i < num_layers - 1), activation=activation,
                               input_shape=(max_sample_length, 300)))
        else:
            if bidirectional:
                model.add(Bidirectional(GRU(units, return_sequences=(i < num_layers - 1), activation=activation)))
            else:
                model.add(GRU(units, return_sequences=(i < num_layers - 1), activation=activation))

    model.add(Dense(2, activation='softmax'))

    model.compile( optimizer = Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model(max_sample_length=100, num_layer=1, num_filters=3, kernel_size=2, pooling_size=2, padding=True, learning_rate=0.001, activation='relu'):
    model = Sequential()
    if padding:
        padding='same'
    else:
        padding='valid'

    # Convolutional layers
    for i in range(num_layer):
        filters = num_filters[i] if isinstance(num_filters, list) else num_filters
        if i == 0:
            model.add(Conv2D(filters, kernel_size, input_shape=(max_sample_length, 300, 1), padding=padding))
        else:
            model.add(Conv2D(filters, kernel_size, padding=padding))
        model.add(MaxPooling2D(pool_size=pooling_size))

    model.add(Flatten())

    # Output layer
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def remove_stopwords(data, stopwords):
    for i in range(len(data)):
        text = data[i]
        try:
            split_words = [
                x.strip('0123456789%@$.,=+-!;/()*"&^:#|\n\t\'').lower() for x in text.split()]
        except TypeError:
            split_words = []
        data[i] = ' '.join(
            [word for word in split_words if word not in stopwords])

    return data


# Tiền xử lý dữ liệu
def preprocess_data(X_train, y_train, max_len):
    
    with open('./vietnamese_stopwords/vietnamese-stopwords-dash.txt', 'r', encoding="utf8") as f:
        stopwords = set([w.strip() for w in f.readlines()])

    # Loại bỏ stopwords
    X_train = remove_stopwords(X_train, stopwords)
    
    embeddings = []
    for sample in X_train:
        sample_embedding = []
        words = sample.split()
        for word in words:
            if word in word2vec:  # Chỉ thêm từ nếu có trong từ điển word2vec
                sample_embedding.append(word2vec[word])
            else:
                sample_embedding.append(np.zeros(300))  # Thêm vector 0 nếu từ không tồn tại

        embeddings.append(sample_embedding)
    # Padding hoặc truncation để đảm bảo tất cả các chuỗi có cùng độ dài max_len
    padded_embeddings = tf.keras.preprocessing.sequence.pad_sequences(embeddings, maxlen=max_len, padding='post', truncating='post', dtype='float32')

    # Chuyển đổi thành tensor của Keras
    X_train_tensor = tf.convert_to_tensor(padded_embeddings, dtype=tf.float32)
    y_train = to_categorical(y_train, num_classes=2)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int32)  # Chuyển đổi nhãn y_train thành tensor

    return X_train_tensor, y_train_tensor

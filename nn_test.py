import gensim
from pyvi import ViTokenizer
import pandas as pd
from nn_utils import *
import warnings
from sklearn import metrics
from keras.models import model_from_json
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


def nn_test(data_dir, model_type, labId, max_sample_length):
    X_data, y_data = read_data(data_dir)
    X_test, y_test = get_data(X_data, y_data)
    X_test, _ = preprocess_data(X_test, y_test, max_sample_length)


    checkpoint_dir = f'./modelDir/{labId}/log_train/{model_type["modelName"]}/'
    json_file = open(checkpoint_dir + f"model_{model_type}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights(checkpoint_dir + f"model_{model_type}.h5")
    test_predictions = classifier.predict(X_test)
    test_predictions = test_predictions.argmax(axis=-1)
    test_acc = metrics.accuracy_score(test_predictions, y_test)
    return {
        "test_acc": test_acc,
        'texts': X_data.to_list(),
        'predicts': list(test_predictions),
        'labels': y_test.to_list()
    }


if __name__ == "__main__":
    model_type = {}
    model_type['backbone'] = "cnn"
    model_type['modelName'] = "_cnn"
    print(nn_test(data_dir='datasets/test.csv', model_type=model_type, labId='6ca93b2-3142-428f-9ffb-bd52a8fae21d', max_sample_length=100))
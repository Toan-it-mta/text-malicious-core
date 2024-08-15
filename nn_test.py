import gensim
from pyvi import ViTokenizer
import pandas as pd
from nn_utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from sklearn import metrics
from keras.models import model_from_json
warnings.filterwarnings("ignore")


def read_data(file):
    data = pd.read_csv(file, encoding='latin-1')
    data['text'].dropna(inplace=True)
    return data['text'], data['label']

def get_data(X_data, y_data):
    data = []
    for X in X_data:
        X = X.lower()
        X = gensim.utils.simple_preprocess(X)
        X = ' '.join(X)
        X = ViTokenizer.tokenize(X)
        data.append(X)
    return data, y_data


async def nn_test(data_dir, model_type, ckpt_number, labId):
    X_data, y_data = read_data(data_dir)
    X_test, y_test = get_data(X_data, y_data)
    with open('./vietnamese_stopwords/vietnamese-stopwords-dash.txt', 'r',encoding="utf8") as f:
        stopwords = set([w.strip() for w in f.readlines()])

    X_test = remove_stopwords(X_data, stopwords)

    checkpoint_dir=f'./modelDir/{labId}/log_train/'
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
    tfidf_vect = pickle.load(open(checkpoint_dir + "test_vectorizer.pickle", "rb"))
    #tfidf_vect.fit(X_test)

    
    #pickle.dump(tfidf_vect, open(checkpoint_dir + "test_vectorizer.pickle", "wb"))
    tfidf_X_test =  tfidf_vect.transform(X_test)

    svd = TruncatedSVD(n_components=300, random_state=42)
    #svd.fit(tfidf_X_test)
    #pickle.dump(svd, open(checkpoint_dir + "test_selector.pickle", "wb"))
    svd = pickle.load(open(checkpoint_dir + "test_selector.pickle", "rb"))
    X_test_tfidf_svd = svd.transform(tfidf_X_test)
    
    encoder = preprocessing.LabelEncoder()
    y_test = encoder.fit_transform(y_test)
    numpy.save(checkpoint_dir + 'test_classes.npy', encoder.classes_)
 
    checkpoint_dir = f'./modelDir/{labId}/log_train/{model_type}/'
    json_file = open(checkpoint_dir + f"model_{model_type}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights(checkpoint_dir + f"model_{model_type}.h5")
    test_predictions = classifier.predict(X_test_tfidf_svd)
    test_predictions = test_predictions.argmax(axis=-1)
    test_acc = metrics.accuracy_score(test_predictions, y_test)
    return {
        "test_acc": test_acc,
    }


if __name__ == "__main__":
    nn_test(model_path="model\\",data_path="data\\", is_neuralnet=True)
from pyvi import ViTokenizer  # thư viện NLP tiếng Việt
import numpy as np
import gensim  # thư viện NLP
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from keras.models import model_from_json
import warnings
from nn_utils import *
warnings.filterwarnings("ignore")


async def nn_infer(text, model_type, ckpt_number, labId):
    lines = text.splitlines()
    lines = ' '.join(lines)
    lines = gensim.utils.simple_preprocess(lines)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    with open('./vietnamese_stopwords/vietnamese-stopwords-dash.txt', 'r', encoding="utf8") as f:
        stopwords = set([w.strip() for w in f.readlines()])

    try:
        split_words = [
            x.strip('0123456789%@$.,=+-!;/()*"&^:#|\n\t\'').lower() for x in lines.split()]
    except TypeError:
        split_words = []

    lines = ' '.join([word for word in split_words if word not in stopwords])
    x = [lines]

    checkpoint_dir=f'./modelDir/{labId}/log_train/'
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load(checkpoint_dir + 'train_classes.npy')

    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
    tfidf_vect = pickle.load(open(checkpoint_dir + "train_vectorizer.pickle", "rb"))

    tfidf_x = tfidf_vect.transform(x)
    svd = TruncatedSVD(n_components=300, random_state=42)

    svd = pickle.load(open(checkpoint_dir + "train_selector.pickle", "rb"))
    tfidf_x_svd = svd.transform(tfidf_x)

    checkpoint_dir = f'./modelDir/{labId}/log_train/{model_type}/'
    
    json_file = open(checkpoint_dir + f"model_{model_type}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(checkpoint_dir + f"model_{model_type}.h5")

    label = int(encoder.inverse_transform(
        [np.argmax(loaded_model.predict(np.array(tfidf_x_svd))[0])])[0])
    print("Label", label)

    return {
        "text": text,
        "model_checkpoint_number": ckpt_number,
        "label": label
    }

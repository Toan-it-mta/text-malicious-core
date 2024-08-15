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


def nn_infer(text, model_type, ckpt_number, labId, max_sample_length):
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
    X = [lines]
    X_infer, _ = preprocess_data(X, [1], max_sample_length)

   
    checkpoint_dir = f'./modelDir/{labId}/log_train/{model_type["modelName"]}/'
    
    json_file = open(checkpoint_dir + f"model_{model_type}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(checkpoint_dir + f"model_{model_type}.h5")
    id2label = {0: "normal", 1: "malicious"}
    idx = np.argmax(loaded_model.predict(np.array(X_infer))[0])
    label = id2label[idx]
    return {
        "text": text,
        "model_checkpoint_number": ckpt_number,
        "label": label
    }
if __name__ == "__main__":
    model_type = {}
    model_type['backbone'] = "cnn"
    model_type['modelName'] = "_cnn"
    print(nn_infer("đụ má mày", model_type, 1, labId='6ca93b2-3142-428f-9ffb-bd52a8fae21d', max_sample_length=100 ))
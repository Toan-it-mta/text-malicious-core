from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tr_feature import TraditionalFeatures
from nn_feature import NeuronFeatures
from sklearn.model_selection import train_test_split
from tr_utils import compute_metrics, make_dir_if_not_exists
import os
import pickle
import torch
from transformers import AutoModel, AutoTokenizer
import json
import pandas as pd

def create_svm_model(**kargs):
    """## Khởi tạo mô hình SVM
    """
    classifier = SVC(**kargs)
    return classifier

def create_navie_bayes_model(**kargs):
    """## Khởi tạo mô hình Navie Bayes
    """
    classifier = GaussianNB(**kargs)
    return classifier

def create_knn_model(**kargs):
    """## Khởi tạo mô hình K-NN
    """
    classifier = KNeighborsClassifier(**kargs)
    return classifier

def tr_train(
        path_file_train_csv:str='datasets/train.csv',
        model_name:str='k-nn', 
        feature_name:str='tf-idf', 
        val_size:float = 0.1, 
        labId:str='abc',
        **kargs):
    """## Huấn luyện các mô hình truyền thống với các đặc trưng cụ thể

    ### Args:
        - `model_name (str, optional)`: Mô hình sử dụng. Các giá trị có thể nhận là: 'k-nn', 'svm', 'navie-bayes'. Defaults to 'k-nn'.
        - `feature_name (str, optional)`: Loại đặc trưng cần trích xuất. Các giá trị có thể nhận là: 'tf-idf', 'count-vectorizing', 'vinai/phobert-base', 'vinai/bartpho-word' . Defaults to 'tf-idf'.
        - `path_file_train_csv (str, optional)`: Đường dần tới bộ dữ liệu train. Defaults to 'datasets/train.csv'.
        - `val_size (float, optional)`: Tỷ lệ chia bộ Validation . Defaults to 0.1.

    ### Returns:
        - `Valid_score`: Thang đo độ chính xác của tập Valid
    """
    if feature_name=='tf-idf' or feature_name=='count-vectorizing':
        traditionalfeature = TraditionalFeatures()
        
        feature_path = f'./modelDir/{labId}/log_train/tr_feature'
        vocab_path = f'{feature_path}/vocab'
        # if os.path.exists(vocab_path):
        #     traditionalfeature.load_vocab_from_file(vocab_path)
        # else:
        traditionalfeature.create_vocab_from_corpus(path_file_train_csv, path_vocab_file=vocab_path)

        vectorizers_path = f'{feature_path}/vectorizers' 
        vector_path = f'{feature_path}/vector' 

        result, _, _ = traditionalfeature.get_features(
            path_file_csv=path_file_train_csv, 
            feature_name=feature_name, 
            path_vector_save=vector_path, 
            path_vectorizer_save=vectorizers_path)
        
        str_result = json.dumps(result)
        df_train = pd.read_json(path_or_buf=str_result, orient='records')
        X, y = df_train[feature_name].to_list(), df_train['label'].to_list()
    else:
        neuronfeature = NeuronFeatures()
        feature_path = f'./modelDir/{labId}/log_train/nn_feature'
        vector_path = f'{feature_path}/vector'

        result,_,_ = neuronfeature.get_features(
            path_file_csv=path_file_train_csv, 
            feature_name=feature_name,
            path_vector_save=vector_path)
        
        str_result = json.dumps(result)
        df_train = pd.read_json(path_or_buf=str_result, orient='records')
        X, y = df_train["arr_sentence_vector"].to_list(), df_train['label'].to_list()
        
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=val_size, random_state=42)
    if model_name == 'k-nn':
        classifier = create_knn_model(**kargs)
    elif model_name == 'navie-bayes':
        classifier = create_navie_bayes_model(**kargs)
    else:
        classifier = create_svm_model(**kargs) 
    
    classifier.fit(X_train, y_train)
    
    ckpt_path = f'./modelDir/{labId}/log_train/{model_name}'
    make_dir_if_not_exists(ckpt_path)
    model_name_ = f"model_{feature_name.replace('/','_')}.pkl"
    path_model_save = os.path.join(ckpt_path, model_name_)
    
    with open(path_model_save, 'wb') as f:
        pickle.dump(classifier, f)
        
    val_predicts = classifier.predict(X_val).tolist()
    score = compute_metrics(val_predicts, y_val)
    
    return {
        'val_acc': score['accuracy'],
        'val_f1': score['f1_score']
    }
    

def tr_test(
        path_file_test_csv:str='datasets/test.csv',
        model_name:str='k-nn', 
        feature_name:str='tf-idf',
        labId:str='abc',
    ):
    
    """## Test các mô hình truyền thống với các đặc trưng cụ thể

    ### Args:
        - `path_model (str, optional)`: Đường dẫn tới mô hình. Defaults to None.
        - `feature_name (str, optional)`: Loại đặc trưng cần trích xuất. Các giá trị có thể nhận là: 'tf-idf', 'count-vectorizing', 'vinai/phobert-base', 'vinai/bartpho-word' . Defaults to 'tf-idf'.
        - `path_file_test_csv (str, optional)`: Đường dần tới bộ dữ liệu test. Defaults to 'datasets/test.csv'.

    ### Yields:
        - - `Test_score`: Thang đo độ chính xác của tập Test
    """
    if feature_name=='tf-idf' or feature_name=='count-vectorizing':
        traditionalfeature = TraditionalFeatures()
        feature_path = f'./modelDir/{labId}/log_train/tr_feature'
        #vocab_path = f'{feature_path}/vocab'
        #traditionalfeature.load_vocab_from_file(vocab_path)

        vectorizers_path = f'{feature_path}/vectorizers' 
        vector_path = f'{feature_path}/vector' 

        result,_,_ = traditionalfeature.get_features(
            path_file_csv=path_file_test_csv, 
            feature_name=feature_name, 
            path_vector_save=vector_path, 
            path_vectorizer_save=vectorizers_path)
        
        str_result = json.dumps(result)
        df_train = pd.read_json(path_or_buf=str_result, orient='records')
        X, y = df_train[feature_name].to_list(), df_train['label'].to_list()
        
    else:
        neuronfeature = NeuronFeatures()
        feature_path = f'./modelDir/{labId}/log_train/nn_feature'
        vector_path = f'{feature_path}/vector'

        result,_,_ = neuronfeature.get_features(
            path_file_csv=path_file_test_csv, 
            feature_name=feature_name,
            path_vector_save=vector_path)
        
        str_result = json.dumps(result)
        df_train = pd.read_json(path_or_buf=str_result, orient='records')
        X, y = df_train["arr_sentence_vector"].to_list(), df_train['label'].to_list()
        
    ckpt_path = f'./modelDir/{labId}/log_train/{model_name}'
    model_name_ = f"model_{feature_name.replace('/','_')}.pkl"
    path_model_save = os.path.join(ckpt_path, model_name_)    
    with open(path_model_save, 'rb') as f:
        classifier = pickle.load(f)
    
    texts = df_train['text'].to_list()
    val_predicts = classifier.predict(X).tolist()
    score = compute_metrics(val_predicts, y)
    
    return {
        'test_acc': score['accuracy'],
        # 'test_f1': score['f1_score'],
        'texts': texts,
        'predicts': val_predicts,
        'labels': y
    }

def tr_infer(
        text:str='',
        model_name:str='k-nn', 
        feature_name:str='tf-idf',
        labId:str='abc'):
    """## Infer mô hình với đoạn text cụ thể

    ### Args:
        - `path_model (str, optional)`: Đường dẫn tới mô hình. Defaults to None.
        - `feature_name (str, optional)`: Loại đặc trưng cần trích xuất. Các giá trị có thể nhận là: 'tf-idf', 'count-vectorizing', 'vinai/phobert-base', 'vinai/bartpho-word' . Defaults to 'tf-idf'.
        - `text (str, optional)`: Đoạn văn bản cần test.

    ### Yields:
        - - `label`:  Nhãn của đoạn text {0: "normal", 1: "malicious"}
    """
    id2label = {0: "normal", 1: "malicious"}
    
    if feature_name=='tf-idf' or feature_name=='count-vectorizing':
        feature_path = f'./modelDir/{labId}/log_train/tr_feature'
        vectorizers_path = f'{feature_path}/vectorizers/{feature_name}-vectorizer.pkl' 
 
        with open(vectorizers_path, 'rb') as f:
            vectorizer_ = pickle.load(f)
        embedding = vectorizer_.transform([text]).toarray()
          
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(feature_name, cache_dir='models')
        model = AutoModel.from_pretrained(feature_name, cache_dir='models').to(device)
        neuronfeature = NeuronFeatures()
        embedding = neuronfeature.get_embedding_from_text(text=text, tokenizer=tokenizer, model=model, number_words_per_sample_logs= 1)[1].reshape(1, -1)
    
    ckpt_path = f'./modelDir/{labId}/log_train/{model_name}'
    model_name_ = f"model_{feature_name.replace('/','_')}.pkl"
    path_model_save = os.path.join(ckpt_path, model_name_)  
    with open(path_model_save, 'rb') as f:
        classifier = pickle.load(f)
    
    id = classifier.predict(embedding).tolist()[0]
    return {
        'text': text,
        'label': id2label[id]
    } 

if __name__ == "__main__":
    # print(tr_train(model_name='navie-bayes', feature_name='tf-idf'))
    # print(tr_test(labId='abc',feature_name='tf-idf', model_name='navie-bayes'))
    # print(tr_infer('models/log-train/model_vinai_phobert-base_navie-bayes.pkl', feature_name='vinai/phobert-base', 
    #                   path_vectorizer_save='datasets/vectorizers/tf-idf-vectorizer.pkl', text='"""đm thô nhưng thật vl"""'))
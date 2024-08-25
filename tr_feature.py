from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle
import os
import json
from tr_utils import make_dir_if_not_exists, visual_embedding, array2string

class TraditionalFeatures:
    """
    Lớp thực hiện tạo bộ từ điển và trích xuất các đặc trưng truyền thống theo từ điển
    """
    def __init__(self, path_file_stopwords:str='./stopwords.txt'):
        """
            Khởi tạo các tham số và load bộ stopwords
        ### Args:
            - `path_file_stopwords (str, optional)`: Đường dẫn tới bộ stopwords. Defaults to 'stopwords.txt'.
        """
        self.vocab = None
        with open(path_file_stopwords, encoding='utf-8') as f:
            self.stop_wors = [word.strip() for word in f.readlines()]

    def create_vocab_from_corpus(
            self, 
            path_file_corpus:str = 'datasets/train.csv', 
            min_frequency:float = 1, 
            max_frequency:float = 1, 
            lower_case:bool=True, 
            max_features=None, 
            path_vocab_file:str = './modelDir/labId/log_train/tr_feature/vocab'
        ):
        """## Sinh bộ từ điển từ một corpus cụ thể

        ### Args:
            - `path_file_corpus (str, optional)`: Đường dẫn tới Corpus. Defaults to 'datasets/train.csv'.
            - `min_frequency (float, optional)`: Tần suất xuất hiện tối thiểu của từ để được cho vào từ điển. Defaults to 1.
            - `max_frequency (float, optional)`: Tần suất xuất hiện tối đa của từ để được cho vào từ điển. Defaults to 1.0.
            - `lower_case (bool, optional)`: Các từ có được xử lý lower_case. Defaults to True.
            - `max_features (_type_, optional)`: Kích thước tối đa của bộ từ điển. Defaults to None.
            - `path_vocab_file (str, optional)`: Đường dẫn bộ từ điển sẽ được lưu lại. Defaults to './vocabs/vocab.txt'.

        ### Returns:
            - `vocab`: Bộ từ điển được sinh ra
        """
        corpus = pd.read_csv(path_file_corpus)
        vectorizer = CountVectorizer(
            min_df=min_frequency, 
            max_df=max_frequency, 
            lowercase=lower_case, 
            stop_words=self.stop_wors, 
            max_features=max_features
        )
        vectorizer.fit_transform(corpus['text'].to_list())
        self.vocab = vectorizer.get_feature_names_out().tolist()
        self.write_vocab_to_file(f'{path_vocab_file}/vocab.txt')

        return self.vocab
    
    def write_vocab_to_file(self, path_vocab_file='/vocab/vocab.txt'):
        """## Lưu trữ bộ từ điển dưới dạng file 

        ### Args:
            - `path_vocab_file (str, optional)`: Đường dẫn file lưu trữ. Defaults to 'vocabs/vocab.txt'.
        """
        try:
            with open(path_vocab_file, 'w', encoding='utf-8') as f:
                for word in self.vocab:
                    f.write(word+'\n')
        except Exception as e:
            print("Error: ", e)
    
    def load_vocab_from_file(self, path_vocab_file='./modelDir/labId/log_train/tr_feature/vocab'):
        """## Load bộ từ điển sẵn có

        ### Args:
            - `path_vocab_file (str, optional)`: Đường dẫn tới bộ từ điển. Defaults to 'vocab/vocab.txt'.

        ### Returns:
            - `vocab`: Bộ từ điển được lưu trữ dưới dạng List
        """
        try:
            with open(f'{path_vocab_file}/vocab.txt', 'r', encoding='utf-8') as f:
                self.vocab = [word.strip() for word in f.readlines()]
            vocab = self.vocab
            return vocab
        except Exception as e:
            print("Error: ", e)

    def add_word_to_vocab(self, word: str):
        """## Thêm từ mới vào bộ từ điển

        ### Args:
            - `word (str)`: Từ cần thêm
        """
        if word not in self.vocab:
            self.vocab.append(word)

    def remove_word_from_vocab(self, word: str):
        """## Xóa một từ khỏi bộ từ điển

        ### Args:
            - `word (str)`: Từ cần xóa
        """
        if word in self.vocab:
            self.vocab.remove(word)
    
    def get_features(
            self, 
            path_file_csv:str = 'datasets/train.csv', 
            feature_name:str = 'count-vectorizing',
            path_vector_save:str = './modelDir/labId/log_train/tr_feature/vector', 
            path_vectorizer_save:str = './modelDir/labId/log_train/tr_feature/vectorizers',
            number_samples_logs:int=10
        ):
        """## Trích xuất đặc trưng truyền thống. Cần thực hiện load Vocab trước nếu không sẽ tự động sinh Vocab theo kho dữ liệu

        ### Args:
            - `path_file_csv (str, optional)`: Đường dẫn tới kho dữ liệu. Defaults to 'datasets/train.csv'.
            - `feature_name (str, optional)`: Tên của đặc trưng cần trích xuất có thể là 'count-vectorizing' và 'tf-idf'. Defaults to 'count-vectorizing'.
            - `path_vector_save (str, optional)`: Đường dẫn file lưu trữ véc-tơ biểu diễn của văn bản. 
            Không chỉ định sẽ được tạo tự động với với tên được tạo từ (path_file_csv, feature_name) Defaults to None.
            - `path_vectorizer_save (str, optional)`: Đường dẫn lưu trữ công cụ trích xuất đặc trưng tương ứng. Defaults to None.

        ### Returns:
            - `df`: Một dataframe có 2 cột (text, feature_name) tương ứng với văn bản và đặc trưng tương ứng
        """
        try:
            df = pd.read_csv(path_file_csv)
            corpus = df['text'].to_list()

            vectorizer_path = f'{path_vectorizer_save}/{feature_name}-vectorizer.pkl'
            vector_path = f'{path_vector_save}/{feature_name}-vector.pkl'
            
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    vectorizer_ = pickle.load(f)
                    
            elif feature_name == 'count-vectorizing':
                vectorizer_ = CountVectorizer(vocabulary=self.vocab)
                vectorizer_.fit(corpus)
                with open(vectorizer_path, 'wb') as f:
                    pickle.dump(vectorizer_, f)
                              
            else:
                vectorizer_ = TfidfVectorizer(vocabulary=self.vocab)
                vectorizer_.fit(corpus)
                with open(vectorizer_path, 'wb') as f:
                    pickle.dump(vectorizer_, f)
                    
            def vectorizer_text(text):
                X = vectorizer_.transform([text]).toarray()
                return X[0]
            
            df[feature_name] = df['text'].apply(vectorizer_text)
            df.to_pickle(vector_path) 

            embedings = df[feature_name].to_numpy()
            words = [None] * len(embedings)
            embedings = np.stack(embedings)
            path_file_sentences_visual = visual_embedding(embedings, words,f'{path_vector_save}/{feature_name}-img.png')

            _df = df
            _df['sentence_vector'] = _df[feature_name].apply(array2string)
            _df_to_show = _df[:number_samples_logs]
            return json.loads(_df.to_json(orient="records")), path_file_sentences_visual, json.loads(_df_to_show.to_json(orient="records"))
        except Exception as e:
            print("Error: ", e)

async def tr_get_dict(data_dir, min_frequency, max_frequency, lower_case, labId):
    traditionalFeature = TraditionalFeatures()
    feature_path = f'./modelDir/{labId}/log_train/tr_feature'

    vocab_path = f'{feature_path}/vocab'
    make_dir_if_not_exists(vocab_path)

    traditionalFeature.create_vocab_from_corpus(
        path_file_corpus=data_dir,
        min_frequency=min_frequency, 
        max_frequency = max_frequency, 
        lower_case=lower_case, 
        path_vocab_file=vocab_path
        )
    vocab = traditionalFeature.load_vocab_from_file(path_vocab_file=vocab_path)

    return {
        "vocab": vocab
    }

async def tr_add_dict(word, labId):
    traditionalFeature = TraditionalFeatures()
    vocab_path = f'./modelDir/{labId}/log_train/tr_feature/vocab'
    traditionalFeature.load_vocab_from_file(path_vocab_file=vocab_path)
    traditionalFeature.add_word_to_vocab(word)
    traditionalFeature.write_vocab_to_file(f'{vocab_path}/vocab.txt')
    vocab = traditionalFeature.load_vocab_from_file(path_vocab_file=vocab_path)

    return {
        "vocab": vocab
    }

async def tr_remove_dict(word, labId):
    traditionalFeature = TraditionalFeatures()
    vocab_path = f'./modelDir/{labId}/log_train/tr_feature/vocab'
    traditionalFeature.load_vocab_from_file(path_vocab_file=vocab_path)
    traditionalFeature.remove_word_from_vocab(word)
    traditionalFeature.write_vocab_to_file(f'{vocab_path}/vocab.txt')
    vocab = traditionalFeature.load_vocab_from_file(path_vocab_file=vocab_path)

    return {
        "vocab": vocab
    }

async def tr_feature(data_dir, feature_name, labId): 
    traditionalFeature = TraditionalFeatures()
    feature_path = f'./modelDir/{labId}/log_train/tr_feature'

    vocab_path = f'{feature_path}/vocab'
    vectorizers_path = f'{feature_path}/vectorizers' 
    vector_path = f'{feature_path}/vector'
    make_dir_if_not_exists(vocab_path)
    make_dir_if_not_exists(vectorizers_path)
    make_dir_if_not_exists(vector_path)

    
    traditionalFeature.create_vocab_from_corpus(path_file_corpus=data_dir, path_vocab_file=vocab_path)
    traditionalFeature.load_vocab_from_file(path_vocab_file=vocab_path)

    _, img_path, word_vector = traditionalFeature.get_features(
        path_file_csv=data_dir,
        feature_name=feature_name, 
        path_vector_save=vector_path, 
        path_vectorizer_save=vectorizers_path
    )

    return {
        "word_vector": word_vector,
        "img_path":img_path
    }

# if __name__ == "__main__":
#     tr_feature("datasets/train.csv",'tf-idf','abc')


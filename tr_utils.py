import re
import os
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _array2string(array, pre_num=3, suf_num=3):
    pre_ = array[:pre_num].tolist()
    suf_ = array[-suf_num:].tolist()
    output = []
    for i in pre_:
        output.append(str(i))
    output.append("...")
    for i in suf_:
        output.append(str(i))
    return output

def array2string(array, words=False, pre_num=3, suf_num=3):
    _array2string(array)
    if words:
        words_embedding = []
        for i in array:
            output = _array2string(i)
            words_embedding.append(output)
        return words_embedding
    
    result = _array2string(array, pre_num, suf_num)
    return result

def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_metrics_acc_score(predictions, labels):
    score = accuracy_score(labels, predictions)
    return {'accuracy': float(score)}

def compute_metrics_f_score(predictions, labels):
    score = f1_score(labels,predictions, average='macro')
    return {"f1_score": float(score)}

def compute_metrics(predictions, labels):
    """## Tính toán các thang đo

    ### Args:
        - `predictions (_type_)`: Đầu ra mô hình dự đoán
        - `labels (_type_)`: Nhãn chính xác

    ### Returns:
        - 'score': Hai điểm accuracy và f-1_score
    """
    metric_results = {}
    for metric_function in [compute_metrics_acc_score, compute_metrics_f_score]:
        metric_results.update(metric_function(predictions, labels))
    return metric_results

def preprocessing_text(text):
    text = re.sub("\r", "\n", text)
    text = re.sub("\n{2,}", "\n", text)
    text = re.sub("…", ".", text)
    text = re.sub("/.{2,}", ".", text)
    text = text.strip()
    text = text.lower()
    return text

def plot_matrix(embeddings, texts, filename):
    for i, text in enumerate(texts):
        x, y = embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(text, (x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")

    plt.savefig(filename)
    plt.close()

def visual_embedding(embedings, texts, path):
    # Giảm chiều dữ liệu về 2 chiều bằng PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(embedings)

    plot_matrix(reduced_data, texts, path)
    return path
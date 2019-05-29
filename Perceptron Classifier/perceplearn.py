import os
import re
import operator
import sys
import math
import numpy as np
import random
import json


def parse_data(input_path):
    input_dict = dict()
    input_dict_list = dict()
    positive_negative_directory_list = next(os.walk(input_path))[1]
    for positive_negative in positive_negative_directory_list:
        if positive_negative.startswith("positive"):
            classify = "positive"
        elif positive_negative.startswith("negative"):
            classify = "negative"
        x = os.path.join(input_path, positive_negative)
        truthful_dec_directory_list = next(os.walk(x))[1]
        for truthful_dec in truthful_dec_directory_list:
            if truthful_dec.startswith("deceptive"):
                label_final = classify + "_deceptive"
            elif truthful_dec.startswith("truthful"):
                label_final = classify + "_truthful"
            y = os.path.join(x, truthful_dec)
            folders = next(os.walk(y))[1]
            str_list = list()
            input_dict_list[label_final] = list()

            for fold in folders:
                z = os.path.join(y, fold)
                files = next(os.walk(z))[2]
                for fileRead in files:
                    file_open = open(os.path.join(z, fileRead), "r")
                    document = file_open.read().lower().rstrip('\n')
                    document = preprocessing(document)
                    str_list.append(document)
                    input_dict_list[label_final].append(document)
            text = ' '.join(str_list)
            input_dict[label_final]=(text)
    return input_dict, input_dict_list


def compute_dict(classify1, classify2, bias1, bias2, weight1, weight2, features1, features2):
    dictionary = dict()
    dictionary[classify1 + '_bias'] = float(bias1[0])
    dictionary[classify2 + '_bias'] = float(bias2[0])
    top_features_val_1 = weight1.tolist()
    top_features_dict_1 = dict(zip(features1, top_features_val_1))
    dictionary[classify1] = top_features_dict_1
    top_features_val_2 = weight2.tolist()
    top_features_dict_2 = dict(zip(features2, top_features_val_2))
    dictionary[classify2] = top_features_dict_2
    return dictionary

def preprocessing(document):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    document= re.sub(r'[^\w\s]', '', document)
    document= re.sub('\\d', '', document)
    document = " ".join([x for x in document.split() if x not in stop_words])
    return document



def word_count(str):
    count_word_dict = dict()
    words = str.split()
    for word in words:
        if word in count_word_dict:
            count_word_dict[word] += 1
        else:
            count_word_dict[word] = 1
    return count_word_dict


def compute_class_label(input_dict):
    class_dict = dict()
    class_dict['positive'] = input_dict['positive_deceptive'] + input_dict['positive_truthful']
    class_dict['negative'] = input_dict['negative_deceptive'] + input_dict['negative_truthful']
    class_dict['truthful'] = input_dict['positive_truthful'] + input_dict['negative_truthful']
    class_dict['deceptive'] = input_dict['positive_deceptive'] + input_dict['negative_deceptive']
    return class_dict


def find_vocabulary(class_dict):
    str2 = class_dict['truthful'] + class_dict['deceptive']
    str1 = class_dict['positive'] + class_dict['negative']
    word_count_2 = word_count(str2)
    word_count_1 = word_count(str1)
    return word_count_1, word_count_2


def build_tf_idf(count_word, no_of_words, idf_dict, word):
    if count_word is None:
        count_word = 0
    tf = count_word / float(no_of_words)
    tf_idf = tf * idf_dict[word]
    return tf_idf

def build_idf(word_dict):
    idf_dict = {}
    N = len(word_dict)
    for k, v in word_dict.items():
        idf_dict[k] = math.log10(N/float(v))
    return idf_dict


def find_vocabulary_corpus(input_dict):
    neg_dec = input_dict['negative_deceptive']
    pos_dec = input_dict['positive_deceptive']
    neg_tru = input_dict['negative_truthful']
    pos_tru = input_dict['positive_truthful']
    str = pos_dec + neg_dec + pos_tru + neg_tru
    word_count_str = word_count(str)
    return word_count_str

def feature_selection_using_tfidf(word_dict, class_label, label, no_of_features):
    word_list = list(word_dict.keys())
    if label=='pos_neg':
        class_1 = word_count(class_label['positive'])
        class_2 = word_count(class_label['negative'])
    elif label=='tru_dec':
        class_1 = word_count(class_label['truthful'])
        class_2 = word_count(class_label['deceptive'])
    idf_dict = build_idf(word_dict)
    result_list = [[0], [0]]
    tf_idf_arr = np.array(result_list)
    for word in word_list:
        tf_idf1 = build_tf_idf(class_1.get(word), len(class_1), idf_dict, word)
        tf_idf2 = build_tf_idf(class_2.get(word), len(class_2), idf_dict, word)
        array_list = [[tf_idf1], [tf_idf2]]
        array = np.array(array_list)
        tf_idf_arr = np.append(tf_idf_arr, array, axis=1)
    tf_idf_arr = tf_idf_arr[:, 1:]

    avg_tf_idf = tf_idf_arr.mean(axis=0)
    avg_tf_idf_list = avg_tf_idf.tolist()
    avg_tf_idf_dict = dict(zip(word_list, avg_tf_idf_list))
    sort_list = sorted(avg_tf_idf_dict.items(), key=operator.itemgetter(1), reverse=True)
    sort_list = sort_list[0:no_of_features]
    top_features = [i[0] for i in sort_list]
    return top_features


def build_input_feature_vector(label1, label2, no_of_features, top_features, labels):
    X = np.zeros(shape=(1, no_of_features))
    label = label1
    for k in range(0, 2):
        for j in range(len(labels[label])):
            feature_vector = build_feature_vector(top_features, labels[label][j])
            feature_vector = feature_vector.reshape((1, no_of_features))
            X = np.concatenate((X, feature_vector), axis=0)
        label = label2
    X = X[1:, :]
    return X


def build_feature_vector(features, doc):
    doc_words = word_count(doc)
    feature_vector = np.zeros(len(features))
    i=0
    for word in features:
        if doc_words.get(word) is not None:
            feature_vector[i] = doc_words.get(word)
        i+=1
    return feature_vector


def build_average_perceptron_weight(feature_count, iteration_count, X, y):
    X = np.where(X > 0, 1, 0)
    #initialize weights
    w = np.zeros(shape=(feature_count,))
    cw = np.zeros(shape=(feature_count,))
    b = 0
    cb = 0
    cnt = 1
    for i in range(iteration_count):
        num = random.randint(0, len(X)-1)
        activation = np.dot(w, X[num]) + b
        if y[num] * activation <= 0:
            w = np.sum((w, y[num]*X[num]), axis=0)
            cw = np.sum((cw, cnt*y[num]*X[num]), axis=0)
            b = b + y[num]
            cb = cb + y[num]*cnt
        cnt+=1
    in_cnt = 1/cnt
    return np.subtract(w, in_cnt*cw), b-(in_cnt*cb)

def build_vanilla_perceptron_weight(feature_count, iteration_count, X, y):
    X=np.where(X>0,1,0)
    #initialize weights
    w = np.zeros(shape=(feature_count,))
    b = 0
    for i in range(iteration_count):
        n = random.randint(0, len(X)-1)
        activation = np.dot(w, X[n]) + b
        if y[n] * activation <= 0:
            w = np.sum((w, y[n]*X[n]), axis=0)
            b = b + y[n]
    return w, b

if __name__ == "__main__":
    model_file = "vanillamodel.txt"
    avg_model_file = "averagemodel.txt"

    input_path = str(sys.argv[1])
    label_2 = 'tru_dec'
    iteration_count = 850
    feature_count = 800
    label_1 = 'pos_neg'

    input_dict, input_dict_list = parse_data(input_path)
    word_class = compute_class_label(input_dict)

    word_dict_pos, word_dict_tru = find_vocabulary(word_class)
    top_features_pos_neg = feature_selection_using_tfidf(word_dict_pos, word_class, label_1, feature_count)
    top_features_tru_dec = top_features_pos_neg

    label_list = compute_class_label(input_dict_list)
    X_1 = build_input_feature_vector('positive', 'negative', feature_count, top_features_pos_neg, label_list)
    X_2 = build_input_feature_vector('truthful', 'deceptive', feature_count, top_features_tru_dec, label_list)

    length = len(X_1)
    y = list()
    y[:length] = [1] * length
    y = np.array(y)
    index = int(length/2)
    y[index:length] = -1
    y = y.reshape((length, 1))

    #run average perceptron
    weight_average_pos_neg, bias_average_pos_neg = build_average_perceptron_weight(feature_count, iteration_count, X_1, y)
    weight_average_tru_dec, bias_average_tru_dec = build_average_perceptron_weight(feature_count, iteration_count, X_2, y)
    master_dict_average = compute_dict(label_1, label_2, bias_average_pos_neg, bias_average_tru_dec, weight_average_pos_neg, weight_average_tru_dec, top_features_tru_dec, top_features_tru_dec)
    json_average = json.dumps(master_dict_average, indent=2)
    f2 = open(avg_model_file, "w")
    f2.write(json_average)
    f2.close()

    #run vanilla perceptron
    weight_vanilla_pos_neg, bias_vanilla_pos_neg = build_vanilla_perceptron_weight(feature_count, iteration_count, X_1, y)
    weight_vanilla_tru_dec, bias_vanilla_tru_dec = build_vanilla_perceptron_weight(feature_count, iteration_count, X_2, y)
    master_dict_vanilla = compute_dict(label_1, label_2, bias_vanilla_pos_neg, bias_vanilla_tru_dec, weight_vanilla_pos_neg, weight_vanilla_tru_dec, top_features_pos_neg, top_features_tru_dec)
    json_vanilla = json.dumps(master_dict_vanilla, indent=2)
    f1 = open(model_file, "w")
    f1.write(json_vanilla)
    f1.close()

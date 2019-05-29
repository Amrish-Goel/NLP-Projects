import sys
import os
import numpy as np
import re
import json

def parse_data(input_path):
    input_dict = dict()
    positive_negative_directory_list = next(os.walk(input_path))[1]
    for positive_negative in positive_negative_directory_list:
        x = os.path.join(input_path, positive_negative)
        truthful_deceptive_directory_list = next(os.walk(x))[1]
        for truthful_deceptive in truthful_deceptive_directory_list:
            y = os.path.join(x, truthful_deceptive)
            folders = next(os.walk(y))[1]
            for fold in folders:
                z = os.path.join(y, fold)
                files = next(os.walk(z))[2]
                for fileRead in files:
                    file_open = open(os.path.join(z, fileRead), "r")
                    document = file_open.read().lower().rstrip('\n')
                    document = preprocessing(document)
                    file_path = z+'/'+fileRead
                    input_dict[file_path] = document
    return input_dict


def predict_class(text, model):
    with open(model) as f:
        model_data = json.load(f)
    dict1 = model_data['tru_dec']
    bias1 = model_data['tru_dec_bias']
    features1 = list(dict1.keys())
    weights1 = list(dict1.values())
    weights1 = np.array(weights1)
    feature_vector1 = build_feature_vector(features1, text)
    feature_vector1 = np.where(feature_vector1 > 0, 1, 0)
    activation = np.dot(weights1, feature_vector1) + bias1
    if activation>0:
        result = 'truthful'
    else:
        result = 'deceptive'

    dict2 = model_data['pos_neg']
    bias2 = model_data['pos_neg_bias']
    features2 = list(dict2.keys())
    weights2 = list(dict2.values())
    weights2 = np.array(weights2)
    feature_vector2 = build_feature_vector(features2, text)
    feature_vector2 = np.where(feature_vector2 > 0, 1, 0)
    activation = np.dot(weights2, feature_vector2)+bias2
    if activation>0:
        result += ' positive'
    else:
        result += ' negative'
    f.close()
    return result

def preprocessing(document):
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    document= re.sub('\\d', '', document)
    document= re.sub(r'[^\w\s]', '', document)
    document = " ".join([x for x in document.split() if x not in stop])
    return document


def build_feature_vector(features, document):
    words = word_count(document)
    feature_vector = np.zeros(len(features))
    i=0
    for word in features:
        if words.get(word) is not None:
            feature_vector[i] = words.get(word)
        i+=1
    return feature_vector


def word_count(string):
    counts = dict()
    words = string.split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


# def evaluate(result, label_pos, label_neg):
#     true_positive = 0
#     false_positive = 0
#     true_negative = 0
#     false_negative = 0
#     for key, val in result.items():
#         if label_pos in key and label_pos in val:
#             true_positive+=1
#         if label_neg in key and label_neg in val:
#             false_positive+=1
#         if label_pos in key and label_neg in val:
#             true_negative+=1
#         if label_neg in key and label_pos in val:
#             false_negative+=1
#     recall = true_positive / float(true_positive + false_negative)
#     precision = true_positive / float(true_positive + false_positive)
#     fscore = 2 * precision * recall / (precision + recall)
#     accuracy = (true_positive+true_negative) / float(true_negative+true_positive+false_positive+false_negative)
#     return fscore
#
# def print_f1_scores():
#     f1_truthful = evaluate(result, 'truthful', 'deceptive')
#     print('truthful-f1: ' + str(f1_truthful))
#     f1_deceptive = evaluate(result, 'deceptive', 'truthful')
#     print('deceptive-f1: ' + str(f1_deceptive))
#     f1_positive = evaluate(result, 'positive', 'negative')
#     print('positive-f1: ' + str(f1_positive))
#     f1_negative = evaluate(result, 'negative', 'positive')
#     print('negative-f1: ' + str(f1_negative))

if __name__ == "__main__":
    model_file = str(sys.argv[1])
    output_file = "percepoutput.txt"
    input_path = str(sys.argv[2])

    testing_dict = parse_data(input_path)
    ans = dict()
    for key, v in testing_dict.items():
        ans[key] = predict_class(v, model_file)
    with open(output_file, 'w') as f:
        for key, value in ans.items():
            f.write('%s %s\n' % (value, key))
    f.close()
    # print_f1_scores()

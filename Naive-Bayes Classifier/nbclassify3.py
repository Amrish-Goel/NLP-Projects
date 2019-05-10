# use this file to classify using naive-bayes classifier
# Expected: generate nboutput.txt
import sys
import json
import math
import os
import re

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your',
              'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it',
              "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
              'this',
              'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
              'has', 'had',
              'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
              'until',
              'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
              'before',
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very',
              's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
              've',
              'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
              "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't",
              'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
              "wouldn't"]


with open('nbmodel.txt') as f:
    model_data = json.load(f)

def remove_stop_words(text):
    # text = ' '.join([word for word in text.split() if word not in stop_words])  # removing stop words
    # return text
    words = re.sub("[^\w]", " ",  text).split()
    cleaned_text = ' '.join([w.lower() for w in words if w not in stop_words])
    return cleaned_text


def remove_punctuations(text):
    # translator = str.maketrans("", "", string.punctuation)
    # text.translate(translator)  # removing punctuations to check not working
    text.replace('[^\w\s]', '')
    return text


def pre_processing(text):
    text = remove_punctuations(text)
    text = remove_stop_words(text)
    return text

def calculate_probability(sentence, model_data, category):
    words = sentence.split()
    x=0.25
    prob = math.log(x)
    for word in words:
        ## ignore unseen words
        val = model_data[category].get(word)
        if val is not None:
            #val = model_data[word].get(category)
            prob = prob + math.log(val)
    return prob

def predict_class(sentence):
    predict_dict = dict()
    predict_dict['deceptive positive'] = calculate_probability(sentence, model_data, 'pos_dec')
    predict_dict['deceptive negative'] = calculate_probability(sentence, model_data, 'neg_dec')
    predict_dict['truthful positive'] = calculate_probability(sentence, model_data, 'pos_tru')
    predict_dict['truthful negative'] = calculate_probability(sentence, model_data, 'neg_tru')
    predicted_class = max(predict_dict, key=predict_dict.get)
    return predicted_class


def read_testing_data(testing_data_path):
    testing_data = list()
    path_list = list()
    polarity_directory_list = next(os.walk(testing_data_path))[1]
    for polarity_directory in polarity_directory_list:
        x = os.path.join(testing_data_path, polarity_directory)
        deceptive_truthful_directory_list = next(os.walk(x))[1]
        for deceptive_truthful_directory in deceptive_truthful_directory_list:
            y = os.path.join(x, deceptive_truthful_directory)
            folders = next(os.walk(y))[1]
            for fold in folders:
                z = os.path.join(y, fold)
                file_list = next(os.walk(z))[2]
                for file_no in file_list:
                    file_open = open(os.path.join(z, file_no), "r")
                    text = file_open.read().lower().rstrip('\n')
                    data = pre_processing(text)
                    testing_data.append(data)
                    path_list.append(z+'/'+file_no)
    return testing_data, path_list

if __name__ == "__main__":
    model_file = "nbmodel.txt"
    output_file = "nboutput.txt"
    result_dict = dict()
    # input_path = str(sys.argv[1])
    input_path = '/Users/amrish/Documents/NLP/HW1/op_spam_testing_data'
    testing_data, path_list = read_testing_data(input_path)

    for text, path in zip(testing_data, path_list):
        label = predict_class(text)
        result_dict[path] = label

    with open("nboutput.txt", 'w') as outfile:
        for key, value in result_dict.items():
            outfile.write('%s %s\n' % (value, key))


    outfile.close()
    f.close()

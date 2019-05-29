import json
import numpy as np

def read_input(input_file):
    document = []
    with open(input_file, 'r', encoding='utf-8') as fp:
        for each_line in fp:
            each_line = each_line.strip()
            document.append(each_line)
    return document


def read_model_file(model_file):
    with open(model_file) as fp:
        model_data = json.load(fp)
    return model_data


def calculate_transition_matrix(model_data):
    transition_list = list()
    transition_states = model_data['transition_states']
    for key, val in transition_states.items():
        transition_list.append(list(val.values()))
    transition_matrix = np.array(transition_list)
    return transition_matrix

def calculate_emission_matrix(model_data):
    emission_list = list()
    emission_states = model_data['emission_states']
    for key, val in emission_states.items():
        emission_list.append(list(val.values()))
    emission_matrix = np.array(emission_list)
    return emission_matrix


def create_word_dict(model_data):
    word_list = list(model_data['emission_states'].keys())
    word_index = list(range(0, len(word_list)))
    word_dict = dict(zip(word_list,word_index))
    return word_dict


def viterbi_decoding(line, states, transition_matrix, emission_matrix, initial_states, word_dict):
    time_points = line.split(' ')
    states_len = len(states)
    time_points_len = len(time_points)
    probability = np.zeros((states_len, time_points_len))
    back_pointer = np.zeros((states_len, time_points_len))
    if word_dict.get(time_points[0]) is not None:
        emission_index = word_dict.get(time_points[0])
        probability[:, 0] = initial_states * emission_matrix[emission_index, :]
    else:
        probability[:, 0] = initial_states
    for t in range(1, time_points_len):
        for i in range(0, states_len):
            emission_val = emission_matrix[word_dict.get(time_points[t])][i] if word_dict.get(time_points[t]) is not None else 1
            mul_val = probability[:, t-1] * transition_matrix[:, i]
            probability[i, t] = np.max(emission_val * mul_val)
            back_pointer[i, t] = np.argmax([mul_val])
    most_probable_state = np.argmax([probability[:, time_points_len-1]])

    tags=[]
    for j in range(time_points_len-1, -1, -1):
        tags.append(states[most_probable_state])
        most_probable_state = int(back_pointer[most_probable_state, j])
    tags.reverse()
    ans_list = []
    for i in range(0, len(tags)):
        ans_list.append(time_points[i]+'/'+tags[i])
    ans = ' '.join(ans_list)
    return ans



if __name__ == '__main__':
    input_file = '/Users/amrish/PycharmProjects/NLP-HW1/hmm-training-data/it_isdt_dev_raw.txt'
    model_file = "hmmmodel_amrish.txt"
    output_file = "hmmoutput_amrish.txt"
    document = read_input(input_file)
    model_data = read_model_file(model_file)
    transition_matrix = calculate_transition_matrix(model_data)
    emission_matrix = calculate_emission_matrix(model_data)
    # creating dict with words corresponding their indexes
    word_dict = create_word_dict(model_data)
    states = list(model_data['start_states'].keys())
    states_val = np.array(list(model_data['start_states'].values()))
    fout = open(output_file, "w")
    for line in document:
        result = viterbi_decoding(line, states, transition_matrix, emission_matrix, states_val, word_dict)
        fout.write('%s\n' % result)

    fout.close()

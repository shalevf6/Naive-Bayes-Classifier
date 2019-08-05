import pandas as pd
import numpy as np
from sklearn import preprocessing


class NB_classifier:
    def build_model(self, data_structure, train_data, bins):#MILON, CSV FILE, NUM OF BINS
        #clean data(preprocessing)
        train_data = train_data.fillna(train_data.mean())
        le = preprocessing.LabelEncoder()
        for header in train_data:
            if data_structure[header][0] == "NUMERIC":
                train_data[header] = pd.cut(train_data[header], bins)
                data_structure[header].append(le.fit(train_data[header]))
                train_data[header] = le.transform(train_data[header])
            else:
                train_data[header] = train_data[header].fillna(train_data[header].mode()[0])
        return train_data

    #classify the input from test csv
    def clssify_input(self, model, to_class, data_structure, bins):#model-cleanTrainData, testdata from csv, milon(Gender-male,female), bins
        class_yes_probs = []
        class_no_probs = []
        outputString =""
        m = 2
        p_val_by_att = {}
        data_T = to_class.fillna(to_class.mean())
        headers_list = []
        for key in data_structure.keys():
            if key != 'class':
                headers_list.append(key)
        le = preprocessing.LabelEncoder()
        for header in data_T:
            if data_structure[header][0][0] == "NUMERIC":
                p_val_by_att[header] = 1 / bins
                data_T[header] = pd.cut(data_T[header], bins)
                le.fit(data_T[header])
                data_T[header] = le.transform(data_T[header])
            else:
                p_val_by_att[header] = 1 / len(data_structure[header][0])
                data_T[header] = data_T[header].fillna(data_T[header].mode()[0])
        class_counts = model.groupby(by=['class'])['class'].count()
        total_rows = class_counts['Y'] + class_counts['N']
        class_yes_prob = class_counts['Y'] / total_rows
        class_no_prob = class_counts['N'] / total_rows
        Y_rows_count = model[(model['class'] == 'Y')].shape[0]
        N_rows_count = model[(model['class'] == 'N')].shape[0]
        for ind, row in data_T.iterrows():
            for header in headers_list:
                p = p_val_by_att[header]
                nc_y = model.loc[(model['class'] == 'Y') & (model[header] == row[header])].shape[0]
                prob_M_E_Class_yes = (nc_y + m*p) / (Y_rows_count + m)
                class_yes_probs.append(prob_M_E_Class_yes)
                nc_n = model.loc[(model['class'] == 'N') & (model[header] == row[header])].shape[0]
                prob_M_E_Class_no = (nc_n + m * p) / (N_rows_count + m)
                class_no_probs.append(prob_M_E_Class_no)
            list_class_no_mult = 1
            list_class_yes_mult = 1
            for pXk in class_yes_probs:
                list_y_mult = list_class_yes_mult * pXk
            for pXk in class_no_probs:
                list_n_mult = list_class_no_mult * pXk
            Cnb_y = class_yes_prob * list_class_yes_mult
            Cnb_n = class_no_prob * list_class_no_mult
            if Cnb_y>Cnb_n:
                class_result = 'yes'
            else:
                class_result = 'no'
            class_yes_probs = []
            class_no_probs = []
            outputString += str(ind+1) + ' ' + class_result + '\n'
        with open("output.txt",'w') as output_file:
            output_file.write(outputString)
import pandas as pd
import numpy as np
from sklearn import preprocessing

class NB_classifier:
    def build_model(self, data_headers_and_types, train_data, bins):#MILON, CSV FILE, NUM OF BINS
        # clean data(preprocessing)
        train_data = train_data.fillna(train_data.mean())
        le = preprocessing.LabelEncoder()
        for feature in train_data:
            if data_headers_and_types[feature][0] == "NUMERIC":
                train_data[feature] = pd.cut(train_data[feature], bins)
                data_headers_and_types[feature].append(le.fit(train_data[feature]))
                train_data[feature] = le.transform(train_data[feature])
            else:
                train_data[feature] = train_data[feature].fillna(train_data[feature].mode()[0])
        return train_data

    # classify the input from test csv
    def clssify_input(self, model, to_class, data_headers_and_types, bins):#model-cleanTrainData, testdata from csv, milon(Gender-male,female), bins
        y_probList = []
        n_probList = []
        results_to_file =""
        m = 2
        p_val_by_att = {}
        test_data = to_class.fillna(to_class.mean())
        headers_list = []
        for key in data_headers_and_types.keys():
            if key != 'class':
                headers_list.append(key)
        le = preprocessing.LabelEncoder()
        for header in test_data:
            if data_headers_and_types[header] == "NUMERIC":
                p_val_by_att[header] = 1/bins
                test_data[header] = pd.cut(test_data[header], bins)
                le.fit(test_data[header])
                test_data[header] = le.transform(test_data[header])
            else:
                p_val_by_att[header] = 1 / len(data_headers_and_types[header])
                test_data[header] = test_data[header].fillna(test_data[header].mode())

        class_counts = model.groupby(by=['class'])['class'].count()
        total_rows = class_counts['Y'] + class_counts['N']
        p_N = class_counts['N'] / total_rows
        N_rows_count = model[(model['class'] == 'N')].shape[0]
        p_Y = class_counts['Y'] / total_rows
        Y_rows_count = model[(model['class'] == 'Y')].shape[0]
        for index, row in test_data.iterrows():
            for header in headers_list:
                p = p_val_by_att[header]
                nc_y = model.loc[(model['class'] == 'Y') & (model[header] == row[header])].shape[0]
                p_mEstimate_y = (nc_y + m*p) / (Y_rows_count + m)
                y_probList.append(p_mEstimate_y)
                nc_n = model.loc[(model['class'] == 'N') & (model[header] == row[header])].shape[0]
                p_mEstimate_n = (nc_n + m * p) / (N_rows_count + m)
                n_probList.append(p_mEstimate_n)

            #class no
            list_n_mult = 1
            for pXk in n_probList:
                list_n_mult = list_n_mult * pXk
            Cnb_n = p_N * list_n_mult
            n_probList = []

            #class yes
            list_y_mult = 1
            for pXk in y_probList:
                list_y_mult = list_y_mult * pXk
            Cnb_y = p_Y * list_y_mult
            y_probList = []

            if Cnb_n>Cnb_y:
                class_result = 'no'
            else:
                class_result = 'yes'

            results_to_file += str(index+1) + ' ' + class_result + '\n'

        with open("output.txt",'w') as output_file:
            output_file.write(results_to_file)


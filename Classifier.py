import pandas as pd
import numpy as np

class Classifier:

    def __init__(self, path, data_structure, bins=2):
        self.file_path = path
        self.propabilities = dict()
        self.num_of_bins = int(bins)
        self.structure = data_structure

    # the Main function that build the model by call semi functions.
    # called from the GUI class
    def build_model(self, train_raw_data):
        endURL = '\\train.csv'
        train_data = self.pre_process_data(train_raw_data)
        self.__makeDiscretization(train_data)
        self.__naiveBayes(train_data)

    #the function is discretization and calc the binMinMax
    def __makeDiscretization(self, dataFrame):
        struct = self.structure
        bins = self.num_of_bins
        for (key, value) in struct.items():
            if value['num']:
                maximum = dataFrame[key].max()
                minimum = dataFrame[key].min()
                #the pace size
                pace = (maximum - minimum) / bins
                #calc by minmax formula
                binMinMax = [float('-inf')] + \
                               [(minimum + index * pace) for index in range(1, bins)] if 0 < pace else [minimum]
                binMinMax += [float('inf')]
                dataFrame[key] = pd.cut(dataFrame[key], bins=binMinMax, include_lowest=True, labels=range(len(binMinMax) - 1), duplicates='drop')
                struct[key]['val'] = binMinMax

    def classify_input(self, test_raw_data):
        test_data = self.pre_process_data(test_raw_data)
        self.__makeDiscretization(test_data)
        endURL = '\\output.txt'
        with open(self.file_path + endURL, 'w') as outputFile:
            indexRow = 1
            for index, record in test_data.iterrows():
                row = str(indexRow) + " " + self.predict(record)+"\n"
                outputFile.write(row)
                indexRow += 1

    def __naiveBayes(self, dataFrame):
        struct = self.structure
        bins = self.num_of_bins
        totalClass = {}

        for classValues in struct['class']['val']:
            totalClass[classValues] = len(dataFrame[dataFrame['class'] == classValues])

        for (key, val) in struct.items():
            if key == 'class':
                continue
            df = dataFrame.groupby([key, 'class']).size().reset_index(name='counts')
            df.insert(3, "Prob", float(0.0), True)
            #propabilities[key] = df

            attributeValues = {}
            if struct[key]['num'] == False:
                attributeValues = struct[key]['val']
            else:
                attributeValues = range(bins)

            for attVal in attributeValues:
                for classValue in struct['class']['val']:
                    groupByData = dataFrame[(dataFrame[key] == attVal) & (dataFrame["class"] == classValue)]
                    n = totalClass[classValue]
                    nc = len(groupByData)
                    p = float(1) / len(attributeValues)
                    mEstimateValue = float(nc+2*p)/(n+2)
                    the_Key = (key, attVal, classValue)
                    self.propabilities[the_Key] = mEstimateValue

    def predict(self, record):
        struct = self.structure

        probs = dict()
        for classValue in struct['class']['val']:
            total_prob = 1
            for (key, val) in struct.items():
                if key == 'class':
                    continue
                key_value = (key, record[key], classValue)
                total_prob = total_prob * self.propabilities[key_value]
            probs[classValue] = total_prob
        return max(probs, key=probs.get)

    # pre processing the data
    def pre_process_data(self, data):
        for feature in self.structure.keys():
            if len(self.structure[feature]['attributes']) == 1 and self.structure[feature]['attributes'][0] == "NUMERIC":
                # the numeric case
                data[feature] = data[feature].fillna(data[feature].mean())
            else:
                # the cateogorical case
                data[feature] = data[feature].fillna(data[feature].mode()[0])
        return data
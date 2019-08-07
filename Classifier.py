import pandas as pd
import numpy as np

class NB_classifier:


    propabilities = None
    file_path = None
    num_of_bins = None
    structure = None

    def __init__(self, path, bins=2):
        self.file_path = path
        self.propabilities = dict()
        self.num_of_bins = int(bins)
        self.structure = {}

    # the Main function that build the model by call semi functions.
    # called from the GUI class
    def build_model(self):
        endURL = '\\train.csv'
        self.__buildStructure()
        train_data = self.__insertData(endURL, True)
        self.__makeDiscretization(train_data)
        self.__naiveBayes(train_data)

    # the function build the structure dictionary
    def __buildStructure(self):
        endURL = '\\structure.txt'
        struct = self.structure
        the_path = self.file_path
        decre1 = -1
        decre2 = -2
        with open(the_path + endURL) as struc_file:
            for row in struc_file.readlines():
                row = row.replace('{', '')
                row = row.replace('}', '')
                splits = row.split()
                while 3 < len(splits):
                    splits[len(splits) + decre2] += ' ' + splits.pop(len(splits) + decre1)

                struct[splits[-decre1]] = {
                    'num': splits[-decre2] == 'NUMERIC',
                    'val': splits[-decre2].split(',')
                }

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

    def clssify_input(self):
        endURL = '\\test.csv'
        test_data = self.__insertData(endURL, False)
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

    #preproccessing the data
    def __insertData(self, train_test_path, isTrain):
        struct = self.structure
        the_path = self.file_path
        if (isTrain):
            train_data = pd.read_csv(filepath_or_buffer=the_path + train_test_path)
        else:
            test_data = pd.read_csv(filepath_or_buffer=the_path + train_test_path)
        for value in struct.items():
            feature = value[0]
            isNumeric = value[1]['num']
            if isNumeric:
                # handdle numeric by avr
                if (isTrain):
                    train_data[feature] = train_data[feature].fillna(train_data[feature].mean())
                else:
                    test_data[feature] = test_data[feature].fillna(test_data[feature].mean())
            else:
                if (isTrain):
                    # handdle not numeric by max of appearance
                    train_data[feature] = train_data[feature].fillna(train_data[feature].mode()[0])
                else:
                    test_data[feature] = test_data[feature].fillna(test_data[feature].mode()[0])
        if (isTrain):
            return train_data
        else:
            return test_data
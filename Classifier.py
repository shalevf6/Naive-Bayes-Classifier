import sys
import pandas as pd

class NB_classifier:


    probabilities = None
    file_path = None
    num_of_bins = None
    structure = None

    def __init__(self, path, bins=2):
        self.file_path = path
        self.probabilities = dict()
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
    def __makeDiscretization(self, dataStr):
        struct = self.structure
        bins = self.num_of_bins
        labels = []
        for label in range(self.num_bins):
            labels.append(str(label))
        for (key, value) in struct.items():
            if len(struct[key]['attributes'][0])==1 and struct[key]['attributes'][0] == "NUMERIC":
                maximum = self.findMAX(dataStr, key)
                minimum = self.findMIN(dataStr, key)
                #the pace size
                pace = self.calcPace(maximum, minimum, bins)
                #calc by minmax formula
                binMinMax = [float(sys.minint)] + \
                               [(minimum + index * pace) for index in range(1, bins)] if 0 < pace else [minimum]
                binMinMax += [float(sys.maxint)]
                dataStr[key] = pd.cut(dataStr[key], bins=binMinMax, include_lowest=True, labels=labels)
                struct[key]['attributes'][0] = binMinMax

    def findMAX(self, dataStr, attribute):
        return dataStr[attribute].max()

    def findMIN(self, dataStr, attribute):
        return dataStr[attribute].min()

    def calcPace(self, maximum, minimum, bins):
        return (maximum - minimum) / bins

    def clssify_input(self):
        test_data = self.preProcces4TestSet()
        endURL = '\\output.txt'
        with open(self.file_path + endURL, 'w') as outputFile:
            for index, record in test_data.iterrows():
                rowIndexAsStr = str(index+1)
                row = rowIndexAsStr + " " + self.calc_M_est_for_record(record)+"\n"
                outputFile.write(row)

    def preProcces4TestSet(self):
        endURL = '\\test.csv'
        test_data = self.__insertData(endURL, False)
        self.__makeDiscretization(test_data)
        return test_data

    def __naiveBayes(self, train_data):
        struct = self.structure
        bins = self.num_of_bins
        probabilities = self.probabilities

        for (key, val) in struct.items():
            if key != 'class':
                df = train_data.groupby([key, 'class']).size().reset_index(name='counts')
                df.insert(3, "Prob", float(0.0), True)

                if len(struct[key]['attributes'][0]) == 1 and struct[key]['attributes'][0] == "NUMERIC":
                    values = range(bins)
                else: values = struct[key]['attributes'][0]
                for attVal in values:
                    for classValue in struct['class']['attributes'][0]:
                        mergeAttributsWithClass = train_data[(train_data[key] == attVal) & (train_data["class"] == classValue)]
                        p = float(1) / len(values)
                        nc = len(mergeAttributsWithClass)
                        n = len(train_data[train_data["class"] == classValue])
                        mEstimateValue = float(nc+2*p)/(n+2)
                        the_Key = (key, attVal, classValue)
                        probabilities[the_Key] = mEstimateValue

    def calc_M_est_for_record(self, row_in_test):
        struct = self.structure
        probabilities = self.probabilities
        probs = dict()
        for Yes_No_Class in struct['class']['attributes'][0]:
            m_est_prob = 1
            for (key, val) in struct.items():
                att_val = row_in_test[key]
                if key == 'class':
                    continue
                key_value = (key, att_val, Yes_No_Class)
                m_est_prob = m_est_prob * probabilities[key_value]
            probs[Yes_No_Class] = m_est_prob
        maximum2return = max(probs, key=probs.get)
        return maximum2return

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
            if len(struct[feature]['attributes'][0]) == 1 and struct[feature]['attributes'][0] == "NUMERIC":
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
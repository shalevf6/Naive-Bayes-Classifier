import sys
import pandas as pd

class Classifier:

    def __init__(self, path, data_structure, bins=2):
        self.file_path = path
        self.probabilities = dict()
        self.num_of_bins = int(bins)
        self.structure = data_structure

    # the Main function that build the model by call semi functions.
    # called from the GUI class
    def build_model(self, train_raw_data):
        endURL = '\\train.csv'
        train_data = self.pre_process_data(train_raw_data)
        self.discretization(train_data)
        self.build_NB_model(train_data)

    #the function is discretization and calc the binMinMax
    def discretization(self, dataStr):
        struct = self.structure
        bins = self.num_of_bins
        for (key, value) in struct.items():
            if (len(struct[key]['attributes']) == 1 and struct[key]['attributes'][0] == "NUMERIC") or struct[key]['attributes'][0] == float('-inf'):
                maximum = self.findMAX(dataStr, key)
                minimum = self.findMIN(dataStr, key)
                # the pace size
                pace = self.calcPace(maximum, minimum, bins)
                # calc by minmax formula
                binMinMax = [float('-inf')] + \
                               [(minimum + index * pace) for index in range(1, bins)] if 0 < pace else [minimum]
                binMinMax += [float('inf')]
                labels = []
                for label in range(bins):
                    labels.append(str(label))
                dataStr[key] = pd.cut(dataStr[key], bins=binMinMax, include_lowest=True, labels=labels)
                struct[key]['attributes'] = binMinMax

    def findMAX(self, dataStr, attribute):
        return dataStr[attribute].max()

    def findMIN(self, dataStr, attribute):
        return dataStr[attribute].min()

    def calcPace(self, maximum, minimum, bins):
        return (maximum - minimum) / bins

    def classify_input(self, test_raw_data):
        test_data = self.preProcces4TestSet(test_raw_data)
        endURL = '\\output.txt'
        with open(self.file_path + endURL, 'w') as outputFile:
            for index, record in test_data.iterrows():
                rowIndexAsStr = str(index+1)
                row = rowIndexAsStr + " " + self.calc_M_est_for_record(record)+"\n"
                outputFile.write(row)

    def preProcces4TestSet(self, test_raw_data):
        test_data = self.pre_process_data(test_raw_data)
        self.discretization(test_data)
        return test_data

    def build_NB_model(self, train_data):
        struct = self.structure
        bins = self.num_of_bins
        probabilities = self.probabilities

        for (key, val) in struct.items():
            if key != 'class':
                df = train_data.groupby([key, 'class']).size().reset_index(name='counts')
                df.insert(3, "Prob", float(0.0), True)

                if (len(struct[key]['attributes']) == 1 and struct[key]['attributes'][0] == "NUMERIC") or \
                                struct[key]['attributes'][0] == float('-inf'):
                    values = range(bins)
                else: values = struct[key]['attributes']
                for attVal in values:
                    for classValue in struct['class']['attributes']:
                        mergeAttributsWithClass = train_data[(train_data[key] == str(attVal)) & (train_data["class"] == classValue)]
                        p = float(1) / len(values)
                        nc = len(mergeAttributsWithClass)
                        n = len(train_data[train_data["class"] == classValue])
                        mEstimateValue = float(nc + 2*p) / (n + 2)
                        the_Key = (key, str(attVal), classValue)
                        probabilities[the_Key] = mEstimateValue

    def calc_M_est_for_record(self, row_in_test):
        struct = self.structure
        probabilities = self.probabilities
        probs = dict()
        for Yes_No_Class in struct['class']['attributes']:
            m_est_prob = 1
            for (key, val) in struct.items():
                att_val = str(row_in_test[key])
                if key == 'class':
                    continue
                key_value = (key, att_val, Yes_No_Class)
                m_est_prob = m_est_prob * probabilities[key_value]
            probs[Yes_No_Class] = m_est_prob
        maximum2return = max(probs, key=probs.get)
        return maximum2return

    # pre processing the data
    def pre_process_data(self, data):
        struct = self.structure
        for feature in struct.keys():
            if (len(struct[feature]['attributes']) == 1 and struct[feature]['attributes'][0] == "NUMERIC") or \
                            struct[feature]['attributes'][0] == float('-inf'):                # the numeric case
                data[feature] = data[feature].fillna(data[feature].mean())

            else:
                # the cateogorical case
                data[feature] = data[feature].fillna(data[feature].mode()[0])
        return data
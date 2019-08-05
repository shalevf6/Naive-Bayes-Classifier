from tkinter import IntVar, Label, Entry, Button, Tk, StringVar, Frame

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import os
import sys
import pandas as pd
from nb_imp import NB_classifier


class GUI(object):
    def __init__(self):
        self.model = ""
        self.data_structure = {}
        self.classifier = NB_classifier()

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Naive Bayes Classifier"))
        self.dir_path_lineEdit.setPlaceholderText(_translate("Form", "Enter Path"))
        self.dir_path_browse.setText(_translate("Form", "Browse"))
        self.num_bins_lineEdit.setPlaceholderText(_translate("Form", "# Bins"))
        self.build_button.setText(_translate("Form", "Build"))
        self.label_2.setText(_translate("Form", "Naive Bayes Classifier"))
        self.label_3.setText(_translate("Form", "Directory Path:"))
        self.label_4.setText(_translate("Form", "Discretization Bins:"))
        self.classify_button.setText(_translate("Form", "Classify"))

    # def browse_dir(self):
    #     self.dir_path_lineEdit.clear()
    #     self.dir_path_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))

    def get_path(self):
        self.dir_path_lineEdit.clear()
        self.dir_path_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))

    def build(self):
        structure_file = ""
        train_file = ""
        # data_headers_and_types = {}
        self.dir_path = self.dir_path_lineEdit.text()
        if not os.path.exists(self.dir_path):
            self.alert_message("Folder does not exist in path.",
                               "please enter directory path again.", "user input Error")
        else:
            try:
                structure_file = open(self.dir_path + "/Structure.txt", "r")
                train_file = open(self.dir_path + "/train.csv", "r")
                structure_file_content = structure_file.readlines()
                structure_file.close()
                for line in structure_file_content:
                    line = line.split(" ")
                    types = line[2]
                    i = 3
                    if '{' in types:
                        while '}' not in types:
                            types = types + ' ' + line[i]
                            i += 1
                    types = types.replace("\n", "")
                    types = types.replace("{", "")
                    types = types.replace("}", "")
                    types = types.split(",")
                    self.data_headers_and_types[line[1]] = []
                    self.data_headers_and_types[line[1]].append(types)
                train_data = pd.read_csv(filepath_or_buffer=train_file, usecols=self.data_headers_and_types.keys(),
                                         squeeze=True)
                if self.num_bins_lineEdit.text()!= '' and self.num_bins_lineEdit.text()!='0':
                    self.bins = int(self.num_bins_lineEdit.text())
                    self.NB_model = self.nb_c.build_model(self.data_headers_and_types, train_data, self.bins)
                    self.alert_message("Building classifier using train-set is done!",
                                       "You can now use it to classify new records.", "Cats likes Fish.")
                    self.classify_button.setEnabled(True)
                else:
                    self.alert_message("Invalid bins num.",
                                       "Please insert proper amount and try again.", "user input Error")
                train_file.close()
            except IOError:
                print('An error occured trying to read the file.')
                if not structure_file:
                    self.alert_message("Structure file does not exist in path.",
                                       "Please supply the file and try again.", "user input Error")
                if not train_file:
                    self.alert_message("Training file does not exist in path.",
                                       "Please supply the file and try again.", "user input Error")

    def classify(self):
        test_file = ""
        try:
            test_file = open(self.dir_path + "/test.csv", "r")
            test_data = pd.read_csv(filepath_or_buffer=test_file, usecols=self.data_headers_and_types.keys(),
                                    squeeze=True)

            nb_result = self.nb_c.clssify_input(self.NB_model, test_data, self.data_headers_and_types, self.bins)
            self.alert_message("Classification complete.",
                               "You can see the results in 'output.txt' @ the working folder. ",
                               "We drink Cookies and eat coffee.")
            if os.path.isfile("output.txt"):
                os.system("output.txt")

        except IOError:
            print('An error occured trying to read the file.')
            self.alert_message("Test file does not exist in path.",
                               "Please supply the file and try again.", "user input Error")

    def alert_message(self, message, aditional_info, details):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setInformativeText(aditional_info)
        msg.setWindowTitle("System Notification")
        msg.setDetailedText(details)
        # msg.exec()

    def check_function(self):
        return False

    def show_gui(self):

        # create the main window
        root = Tk()
        root.title('Flower Classification Interface')
        root.geometry('500x300')

        # create the frame
        frame = Frame(root)
        frame.pack(side='top', fill='both', expand=True)

        # initialize the 2 global input variables
        self.bins = IntVar()
        self.folder_path = StringVar()

        # initialize labels and buttons
        browse_label = Label(frame, text="Directory Path:")
        browse_label.grid(column=0, row=1, padx=4, pady=4)
        browse_text_box = Entry(frame, width=40, textvariable=self.folder_path)
        browse_text_box.grid(column=1, row=1, padx=4, pady=4)
        browse_button = Button(frame, text="Browse", command=lambda: self.get_path())
        browse_button.grid(column=2, row=1, padx=4, pady=4)
        bins_label = Label(frame, text="Discretization Bins:")
        bins_label.grid(column=0, row=2, padx=4, pady=4)
        self.bins.trace("w", self.check_function)
        bins_text_box = Entry(frame, width=40, textvariable=self.bins)
        bins_text_box.grid(column=1, row=2, padx=4, pady=4)
        build_button = Button(frame, text="Build", command=lambda: self.build())
        build_button.grid(column=1, row=3, padx=5, pady=8)
        classify_button = Button(frame, text="Classify", command=lambda: self.classify())
        classify_button.grid(column=1, row=4, padx=5, pady=8)

        root.mainloop()

from tkinter import IntVar, Label, Entry, Button, Tk, StringVar, Frame, filedialog, messagebox

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

    # checks whether the files exist in the given folder and are not empty
    def check_files_exist(self):
        return os.path.isfile(self.folder_path.get() + "/Structure.txt") and os.path.isfile(self.folder_path.get() + "/train.csv") and os.path.isfile(self.folder_path.get() + "/test.csv") and os.path.getsize(self.folder_path.get() + "/Structure.txt") > 0 and os.path.getsize(self.folder_path.get() + "/train.csv") > 0 and os.path.getsize(self.folder_path.get() + "/test.csv") > 0

    # gets the path for the folder
    def get_path(self, root):
        try:
            path = filedialog.askdirectory(parent=root, title='Naive Bayes Classifier')
            if path is None:
                messagebox.showerror("Error", "Couldn't get the path!")
            else:
                if self.check_files_exist():
                    self.folder_path.set(path)
                else:
                    messagebox.showerror("Error", "Missing / Empty Files!")
        except:
            messagebox.showerror("Error", "Couldn't get the path!")

    def build(self):
        structure_file = ""
        training_file = ""
        if self.folder_path == "":
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
                        types = types.replace("\n", "")
                        types = types.replace("{", "")
                        types = types.replace("}", "")
                        types = types.split(",")
                    else:
                        types = ['NUMERIC']
                    self.data_structure[line[1]] = types
                train_data = pd.read_csv(filepath_or_buffer=training_file, usecols=self.data_structure.keys(),
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



    def show_gui(self):

        # create the main window
        root = Tk()
        root.title('Naive Bayes Classifier')
        root.geometry('500x300')

        # create the frame
        frame = Frame(root)
        frame.pack(side='top', fill='both', expand=True)

        # initialize the 2 global input variables
        self.bins = IntVar(root)
        self.folder_path = StringVar(root)

        def check_function(*args):
            dir_ok = False
            bin_ok = False
            # check if the directory and its contents are ok
            if self.folder_path.get() is not None and os.path.isdir(self.folder_path.get()):
                if self.check_files_exist():
                    dir_ok = True

            def represents_int(s):
                try:
                    int(s)
                    return True
                except ValueError:
                    return False

            # check if the entered bins are ok
            if represents_int(bins_text_box.get()):
                if int(bins_text_box.get()) > 0:
                    bin_ok = True

            # enable the Build button if both are ok
            if dir_ok and bin_ok:
                build_button.config(state='normal')
            else:
                build_button.config(state='disabled')
            x = browse_text_box.get()
            y = bins_text_box.get()

        # initialize labels and buttons
        browse_label = Label(frame, text="Directory Path:")
        browse_label.grid(column=0, row=1, padx=4, pady=4)
        browse_text_box = Entry(frame, width=40, textvariable=self.folder_path)
        browse_text_box.grid(column=1, row=1, padx=4, pady=4)
        browse_button = Button(frame, text="Browse", command=lambda: self.get_path(root))
        browse_button.grid(column=2, row=1, padx=4, pady=4)
        self.folder_path.trace("w", check_function)
        bins_label = Label(frame, text="Discretization Bins:")
        bins_label.grid(column=0, row=2, padx=4, pady=4)
        self.bins.trace("w", check_function)
        bins_text_box = Entry(frame, width=40, textvariable=self.bins)
        bins_text_box.grid(column=1, row=2, padx=4, pady=4)
        build_button = Button(frame, text="Build", command=lambda: self.build())
        build_button.grid(column=1, row=3, padx=5, pady=8)
        build_button.config(state = 'disabled')
        classify_button = Button(frame, text="Classify", command=lambda: self.classify())
        classify_button.grid(column=1, row=4, padx=5, pady=8)

        root.mainloop()
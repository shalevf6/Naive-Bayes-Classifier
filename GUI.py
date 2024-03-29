from tkinter import IntVar, Label, Entry, Button, Tk, StringVar, Frame, filedialog, messagebox
import os
import pandas as pd
import Classifier


class GUI(object):
    def __init__(self):
        self.classifier = ""

    # checks whether the files exist in the given folder and are not empty
    def check_files_exist(self, path):
        return os.path.isfile(path + "/Structure.txt") and os.path.isfile(path + "/train.csv") and os.path.isfile(path + "/test.csv") and os.path.getsize(path + "/Structure.txt") > 0 and os.path.getsize(path + "/train.csv") > 0 and os.path.getsize(path + "/test.csv") > 0

    # gets the path for the folder
    def get_path(self, root):
        try:
            path = filedialog.askdirectory(parent=root, title='Naive Bayes Classifier')
            if path is None:
                messagebox.showerror("Error", "Couldn't get the path!")
            else:
                if self.check_files_exist(path):
                    self.folder_path.set(path)
                else:
                    messagebox.showerror("Error", "Missing / Empty Files!")
        except:
            messagebox.showerror("Error", "Couldn't get the path!")

    # builds the model from the given structure and training files
    def build(self):
        try:
            data_structure = {}
            structure_file = open(self.folder_path.get() + "/Structure.txt", "r")
            structure_content = structure_file.readlines()
            structure_file.close()

            # get the structure of the model from the structure file
            for line in structure_content:
                initial_split = line.split(" ")

                classifiers = initial_split[2]

                if len(initial_split) > 3:
                    i = 3
                    while i < len(initial_split):
                        classifiers = classifiers + " " + initial_split[i]
                        i = i + 1

                if '{' in classifiers:
                    classifiers = classifiers.replace("\n", "")
                    classifiers = classifiers.replace("{", "")
                    classifiers = classifiers.replace("}", "")
                    classifiers = classifiers.split(",")
                else:
                    classifiers = ['NUMERIC']

                # add a new entry to the structure
                data_structure[line.split(" ")[1]] = {'attributes': classifiers}

            # get the raw data from the train file
            train_data = pd.read_csv(filepath_or_buffer=self.folder_path.get() + "/train.csv")

            # create the classifier from the training file
            self.classifier = Classifier.Classifier(self.folder_path.get(), data_structure, int(self.bins.get()))
            self.classifier.build_model(train_data)

            # enable the "Classify" button to be pressed
            self.classify_button.config(state='normal')

            messagebox.showinfo("Information", "Building classifier using train-set is done!")
        except IOError:
            messagebox.showerror("Error", "There was a problem reading the files!")


    # classify the given test file by using the NB model generated earlier
    def classify(self):
        # get the raw data from the train file
        test_data = pd.read_csv(filepath_or_buffer=self.folder_path.get() + "/test.csv")

        # get the classifications for the test file
        self.classifier.classify_input(test_data)

        answered_ok = messagebox.showinfo("Information","Classification completed!")
        if answered_ok == 'OK':
            self.root.destroy()

    # builds and shows the GUI
    def show_gui(self):

        # create the main window
        self.root = Tk()
        self.root.title('Naive Bayes Classifier')
        self.root.geometry('500x300')

        # create the frame
        frame = Frame(self.root)
        frame.pack(side='top', fill='both', expand=True)

        # initialize the 2 global input variables
        self.bins = IntVar(self.root,value=2)
        self.folder_path = StringVar(self.root)

        # validates the input for enabling the "Build" button to be pressed
        def validate_input(*args):
            dir_ok = False
            bin_ok = False
            # check if the directory and its contents are ok
            if self.folder_path.get() is not None and os.path.isdir(self.folder_path.get()):
                if self.check_files_exist(self.folder_path.get()):
                    dir_ok = True

            # checks if the a given string is a positive integer
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
        browse_button = Button(frame, text="Browse", command=lambda: self.get_path(self.root))
        browse_button.grid(column=2, row=1, padx=4, pady=4)
        self.folder_path.trace("w", validate_input)
        bins_label = Label(frame, text="Discretization Bins:")
        bins_label.grid(column=0, row=2, padx=4, pady=4)
        self.bins.trace("w", validate_input)
        bins_text_box = Entry(frame, width=40, textvariable=self.bins)
        bins_text_box.grid(column=1, row=2, padx=4, pady=4)
        build_button = Button(frame, text="Build", command=lambda: self.build())
        build_button.grid(column=1, row=3, padx=5, pady=8)
        build_button.config(state = 'disabled')
        self.classify_button = Button(frame, text="Classify", command=lambda: self.classify())
        self.classify_button.grid(column=1, row=4, padx=5, pady=8)
        self.classify_button.config(state='disabled')

        self.root.mainloop()
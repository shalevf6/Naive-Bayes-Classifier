

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import os
import pandas as pd
from nb_imp import NB_classifier


class NB_UI(object):
    def __init__(self):
        self.dir_path = ""
        self.data_headers_and_types = {}
        self.NB_model = ""
        self.nb_c = NB_classifier()
        self.bins = 0

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1272, 971)
        self.dir_path_lineEdit = QtWidgets.QLineEdit(Form)
        self.dir_path_lineEdit.setGeometry(QtCore.QRect(360, 260, 651, 41))
        self.dir_path_lineEdit.setStyleSheet("QLineEdit {\n"
                                             " color: #BEBEBE; \n"
                                             "border: 2px solid #cccccc;\n"
                                             "border-radius: 10px;\n"
                                             " }")
        self.dir_path_lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.dir_path_lineEdit.setClearButtonEnabled(True)
        self.dir_path_lineEdit.setObjectName("pictures_path_lineEdit")
        self.dir_path_browse = QtWidgets.QPushButton(Form)
        self.dir_path_browse.setGeometry(QtCore.QRect(1030, 250, 171, 54))
        self.dir_path_browse.setStyleSheet("QPushButton {\n"
                                           "    color: #BEBEBE;\n"
                                           "    border: 2px solid #555;\n"
                                           "    border-radius: 20px;\n"
                                           "    border-style: outset;\n"
                                           "    background: qradialgradient(\n"
                                           "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
                                           "        radius: 1.35, stop: 0 #F5F5F5, stop: 1 #F5F5F5\n"
                                           "        );\n"
                                           "    padding: 5px;\n"
                                           "    }\n"
                                           "\n"
                                           "QPushButton:hover {\n"
                                           "    color: #404040;\n"
                                           "    background: qradialgradient(\n"
                                           "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
                                           "        radius: 1.35, stop: 0 #F5F5F5, stop: 1 #F5F5F5\n"
                                           "        );\n"
                                           "    }\n"
                                           "\n"
                                           "QPushButton:pressed {\n"
                                           "    border-style: inset;\n"
                                           "    background: qradialgradient(\n"
                                           "        cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n"
                                           "        radius: 1.35, stop: 0 #fff, stop: 1 #ddd\n"
                                           "        );\n"
                                           "    }")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/FolderIcon/iconfinder_icon-101-folder-search_314678.ico"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.dir_path_browse.setIcon(icon)
        self.dir_path_browse.setObjectName("dir_path_browse")
        self.num_bins_lineEdit = QtWidgets.QLineEdit(Form)
        self.num_bins_lineEdit.setGeometry(QtCore.QRect(460, 380, 251, 41))
        self.num_bins_lineEdit.setStyleSheet("QLineEdit {\n"
                                             " color: #BEBEBE; \n"
                                             "border: 2px solid #cccccc;\n"
                                             "border-radius: 10px;\n"
                                             " }")
        self.num_bins_lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.num_bins_lineEdit.setClearButtonEnabled(True)
        self.num_bins_lineEdit.setObjectName("num_bins_lineEdit")
        self.build_button = QtWidgets.QPushButton(Form)
        self.build_button.setGeometry(QtCore.QRect(330, 580, 611, 71))
        self.build_button.setStyleSheet("QPushButton {\n"
                                        "    font: 75 11pt \"MS Shell Dlg 2\";\n"
                                        "    color: #F5F5F5;\n"
                                        "    border: 2px solid #555;\n"
                                        "    border-radius: 20px;\n"
                                        "    border-style: outset;\n"
                                        "    background: qradialgradient(\n"
                                        "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
                                        "        radius: 1.35, stop: 0     #75b54f, stop: 1 #75b54f\n"
                                        "        );\n"
                                        "    padding: 5px;\n"
                                        "    }\n"
                                        "\n"
                                        "QPushButton:hover {\n"
                                        "    background: qradialgradient(\n"
                                        "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
                                        "        radius: 1.35, stop: 0 #75b54f, stop: 1 #6B8E23\n"
                                        "        );\n"
                                        "    }\n"
                                        "\n"
                                        "QPushButton:pressed {\n"
                                        "    border-style: inset;\n"
                                        "    background: qradialgradient(\n"
                                        "        cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n"
                                        "        radius: 1.35, stop: 0 #fff, stop: 1 #ddd\n"
                                        "        );\n"
                                        "    }")
        self.build_button.setObjectName("build_button")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(120, 0, 1111, 161))
        self.label_2.setStyleSheet("font: 75 36pt \"Linux Libertine\";")
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(-60, -120, 2301, 1441))
        self.label.setStyleSheet("background-image: url();")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(10, 240, 411, 71))
        self.label_3.setStyleSheet("font: 75 16pt \"Linux Libertine\";")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(10, 370, 491, 61))
        self.label_4.setStyleSheet("font: 75 16pt \"Linux Libertine\";")
        self.label_4.setObjectName("label_4")
        self.classify_button = QtWidgets.QPushButton(Form)
        self.classify_button.setGeometry(QtCore.QRect(330, 710, 621, 71))
        self.classify_button.setStyleSheet("QPushButton {\n"
                                           "    color: #F5F5F5;\n"
                                           "font: 75 11pt \"MS Shell Dlg 2\";\n"
                                           "    border: 2px solid #555;\n"
                                           "    border-radius: 20px;\n"
                                           "    border-style: outset;\n"
                                           "    background: qradialgradient(\n"
                                           "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
                                           "        radius: 1.35, stop: 0     #75b54f, stop: 1 #75b54f\n"
                                           "        );\n"
                                           "    padding: 5px;\n"
                                           "    }\n"
                                           "\n"
                                           "QPushButton:disabled {\n"
                                           "background-color:#d3d3d3;\n"
                                           "}\n"
                                           "QPushButton:hover {\n"
                                           "    background: qradialgradient(\n"
                                           "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
                                           "        radius: 1.35, stop: 0 #75b54f, stop: 1 #6B8E23\n"
                                           "        );\n"
                                           "    }\n"
                                           "\n"
                                           "QPushButton:pressed {\n"
                                           "    border-style: inset;\n"
                                           "    background: qradialgradient(\n"
                                           "        cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n"
                                           "        radius: 1.35, stop: 0 #fff, stop: 1 #ddd\n"
                                           "        );\n"
                                           "    }")
        self.classify_button.setObjectName("classify_button")
        self.label.raise_()
        self.dir_path_lineEdit.raise_()
        self.dir_path_browse.raise_()
        self.num_bins_lineEdit.raise_()
        self.build_button.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.classify_button.raise_()

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.dir_path_browse.clicked.connect(self.browse_dir)
        self.build_button.clicked.connect(self.build_model)
        self.classify_button.clicked.connect(self.classify_input)
        self.classify_button.setEnabled(False)

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

    def browse_dir(self):
        self.dir_path_lineEdit.clear()
        self.dir_path_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))

    def build_model(self):
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

    def classify_input(self):
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
        msg.exec()

    def show_gui(self):
        import sys
        app = QtWidgets.QApplication(sys.argv)
        main_window = QtWidgets.QMainWindow()
        ui = NB_UI()
        ui.setupUi(main_window)
        main_window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    flowers_gui = NB_UI()
    flowers_gui.show_gui()

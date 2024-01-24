import sys
from PyQt6 import QtWidgets,QtGui
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

from PyQt6.QtWidgets import QWidget,QHBoxLayout, QApplication,QSlider, QMainWindow,QLabel, QPushButton,QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon,QPalette, QColor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

from PyQt6 import QtGui


diabetes = pd.read_csv('diabetes_clean_03042021.csv')

'''
class saveFigure():
    pass

class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)


# correlation ---- highest  x age y blood pressure
# Class for matplotlib canvas
class Canvas(FigureCanvas):
    def __init__(self, parent):
        fig, ax = plt.subplots(1, figsize=(6,10))
        super().__init__(fig)
        self.setParent(parent)


        pd.crosstab(diabetes["Age"], diabetes["Outcome"]).plot(figsize=(10,10), xlabel="Age", ylabel = "Passenger Frequency" )


        #pd.crosstab(diabetes.Age[diabetes["BloodPressure"] == 1], diabetes.BloodPressure[diabetes["Blood Pressure"] == 1]).plot(figsize=(5, 5), color="orange", ylabel="Passengers survived", ax=ax[1]);

        # ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
        
        
        
        Here is some code that worked in my example app: (Professor's zoom code)

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        # for description of argument see
        # https://stackoverflow.com/questions/3584805/what-does-the-argument-mean-in-fig-add-subplot111
        # "111" means "1x1 grid, first subplot"
        self.axes = self.figure.add_subplot(111)
        super(MplCanvas, self).__init__(self.figure)
		
# ...

	def plot_rent(self, year=None, rent=None):
	   # ...
	   self.sc.axes.bar(self.df['bjahr'], self.df['mieteqm'])
	   # ...

'''


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=2, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super(MplCanvas, self).__init__(self.figure)

class secondWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # use global data

        global diabetes
        
        widget_histogram = QWidget()

        # Vertical Layout
        histogram_layout = QVBoxLayout()
        # Create a new Canvas
        self.histogram_screen = MplCanvas(self, width=5, height=2, dpi=100)
        # https://stackoverflow.com/questions/32371571/gridspec-of-multiple-subplots-the-figure-containing-the-passed-axes-is-being-cl
        diabetes.hist(ax=self.histogram_screen.axes)
        # save_fig(self.sc2.figure, "histogram", tight_layout=True, fig_extension="png", resolution=300)  # extra code
        histogram_layout.addWidget(self.histogram_screen)
        self.setCentralWidget(widget_histogram)
        widget_histogram.setLayout(histogram_layout)
        self.setMinimumSize(1500, 1000)



class MainWindow(QMainWindow):


    def design(self):
        global minimum
        minimum = 0
        global maximum
        maximum = 100
        global result
        #self.secondWindow = secondWindow()
        

        layout3 = QHBoxLayout()
        MainLayout = QHBoxLayout()
        leftVerticalLayout = QVBoxLayout()
        RightVerticalLayout = QVBoxLayout()
         
        

        label = QLabel(" Diabetics Prediction System ")
        leftVerticalLayout.addWidget(label)


        self.pregnancies = QSlider(Qt.Orientation.Horizontal)
        self.pregnancies.setMinimum(minimum)
        self.pregnancies.setMaximum(maximum)
        self.pregnancies.valueChanged.connect(self.updateSliderValue1)
        self.pregnancies.setGeometry(5, 3, 20, 20)

        self.selectedValue_preg = QLabel(self)
        self.selectedValue_preg.setText(str(minimum))
        label1 = QLabel("Pregnancies", alignment=Qt.AlignmentFlag.AlignCenter)

        leftVerticalLayout.addWidget(label1)
        leftVerticalLayout.addWidget(self.pregnancies)
        leftVerticalLayout.addWidget(self.selectedValue_preg)



        self.Glucose = QSlider(Qt.Orientation.Horizontal)
        self.Glucose.setMinimum(minimum)
        self.Glucose.setMaximum(maximum)
        self.Glucose.valueChanged.connect(self.updateSliderValue2)
        self.Glucose.setGeometry(5, 3, 20, 20)

        self.selectedValue_glucose = QLabel(self)
        self.selectedValue_glucose.setText(str(minimum))
        label2 = QLabel("Glucose", alignment=Qt.AlignmentFlag.AlignCenter)

        leftVerticalLayout.addWidget(label2)
        leftVerticalLayout.addWidget(self.Glucose)
        leftVerticalLayout.addWidget(self.selectedValue_glucose)

        self.BloodPressure = QSlider(Qt.Orientation.Horizontal)
        self.BloodPressure.setMinimum(minimum)
        self.BloodPressure.setMaximum(maximum)
        self.BloodPressure.valueChanged.connect(self.updateSliderValue3)
        self.BloodPressure.setGeometry(5, 3, 20, 20)

        self.selectedValue_blood = QLabel(self)
        self.selectedValue_blood.setText(str(minimum))
        label3 = QLabel("Blood Pressure", alignment=Qt.AlignmentFlag.AlignCenter)

        leftVerticalLayout.addWidget(label3)
        leftVerticalLayout.addWidget(self.BloodPressure)
        leftVerticalLayout.addWidget(self.selectedValue_blood)



        self.SkinThickness = QSlider(Qt.Orientation.Horizontal)
        self.SkinThickness.setMinimum(minimum)
        self.SkinThickness.setMaximum(maximum)
        self.SkinThickness.valueChanged.connect(self.updateSliderValue4)
        self.SkinThickness.setGeometry(5, 3, 20, 20)

        self.selectedValue_skin = QLabel(self)
        self.selectedValue_skin.setText(str(minimum))
        label4 = QLabel("Skin Thickness", alignment=Qt.AlignmentFlag.AlignCenter)

        leftVerticalLayout.addWidget(label4)
        leftVerticalLayout.addWidget(self.SkinThickness)
        leftVerticalLayout.addWidget(self.selectedValue_skin)


        self.Insulin = QSlider(Qt.Orientation.Horizontal)
        self.Insulin.setMinimum(minimum)
        self.Insulin.setMaximum(maximum)
        self.Insulin.valueChanged.connect(self.updateSliderValue5)
        self.Insulin.setGeometry(5, 3, 20, 20)

        self.selectedValue_insulin = QLabel(self)
        self.selectedValue_insulin.setText(str(minimum))
        label5 = QLabel("Insulin", alignment=Qt.AlignmentFlag.AlignCenter)

        leftVerticalLayout.addWidget(label5)
        leftVerticalLayout.addWidget(self.Insulin)
        leftVerticalLayout.addWidget(self.selectedValue_insulin)




        self.BMI = QSlider(Qt.Orientation.Horizontal)
        self.BMI.setMinimum(minimum)
        self.BMI.setMaximum(maximum)
        self.BMI.valueChanged.connect(self.updateSliderValue6)
        self.BMI.setGeometry(5, 3, 20, 20)

        self.selectedValue_BMI = QLabel(self)
        self.selectedValue_BMI.setText(str(minimum))
        label6 = QLabel("BMI", alignment=Qt.AlignmentFlag.AlignCenter)

        leftVerticalLayout.addWidget(label6)
        leftVerticalLayout.addWidget(self.BMI)
        leftVerticalLayout.addWidget(self.selectedValue_BMI)


        self.DiabetesPredegreeFunction = QSlider(Qt.Orientation.Horizontal)
        self.DiabetesPredegreeFunction.setMinimum(minimum)
        self.DiabetesPredegreeFunction.setMaximum(maximum)
        self.DiabetesPredegreeFunction.valueChanged.connect(self.updateSliderValue7)
        self.DiabetesPredegreeFunction.setGeometry(5, 3, 20, 20)

        self.selectedValue_predegree = QLabel(self)
        self.selectedValue_predegree.setText(str(minimum))
        label7 = QLabel("Diabetes Predegree Func", alignment=Qt.AlignmentFlag.AlignCenter)

        leftVerticalLayout.addWidget(label7)
        leftVerticalLayout.addWidget(self.DiabetesPredegreeFunction)
        leftVerticalLayout.addWidget(self.selectedValue_predegree)


        self.age = QSlider(Qt.Orientation.Horizontal)
        self.age.setMinimum(minimum)
        self.age.setMaximum(maximum)
        self.age.valueChanged.connect(self.updateSliderValue8)
        self.age.setGeometry(5, 3, 20, 20)

        self.selectedValue_age = QLabel(self)
        self.selectedValue_age.setText(str(minimum))
        label8 = QLabel("Age", alignment=Qt.AlignmentFlag.AlignCenter)

        leftVerticalLayout.addWidget(label8)
        leftVerticalLayout.addWidget(self.age)
        leftVerticalLayout.addWidget(self.selectedValue_age)


        label9 = QLabel("you are diabatic.", alignment=Qt.AlignmentFlag.AlignCenter)
        self.predict = QPushButton("Predict")
        leftVerticalLayout.addWidget(self.predict)
        self.predict.clicked.connect(self.predictiveSystem)
        self.predictedValue = QLabel(" Prediction: None", self)
        leftVerticalLayout.addWidget(self.predictedValue)


        #### Graph
        self.setWindowTitle('Women Diabetics Prediction')
      
        # x = [1,2,8,3,6]
        # y= [9,3,1,6,3]
        x = diabetes["BMI"]
        y = diabetes["Outcome"]

        #fig = Figure()
        fig = Figure()
        ax =fig.add_subplot(111)
        ax.set_title('The Prevalence of Diabetes Across Different Age Groups in Women')
        x_label = ax.set_ylabel('Outcome')
        y_label= ax.set_xlabel('BMI')
        ax.plot(x,y)
        graph =FigureCanvas(fig)


        ###----------
        RightVerticalLayout.addWidget(graph)
        MainLayout.addLayout(leftVerticalLayout) 
        MainLayout.addLayout(RightVerticalLayout)
        
       
        mainWidget = QWidget()
        mainWidget.setLayout(MainLayout)
        self.setCentralWidget(mainWidget)




        

  
        self.window2 = None  # No external window yet.
        button5 = QPushButton("Histogram")
        layout3.addWidget(button5)
        button5.clicked.connect(self.show_new_window)

        # menu

        # action 1 from menu
        histogram_action = QAction(QIcon("chart-medium.png"), "Histogram", self)
        histogram_action.setStatusTip("View Histogram.")
        histogram_action.triggered.connect(self.show_new_window)

        # action 2 from menu
        save_action = QAction(QIcon("disk-black.png"), "RealTime Graph", self)
        save_action.setStatusTip("Real Time Graph")
        save_action.triggered.connect(self.show_new_window)

        menu = self.menuBar()
        file_menu = menu.addMenu("&Menu")
        file_menu.addAction(histogram_action)
        file_menu.addAction(save_action)


    

    def updateSliderValue1(self):
        val = self.pregnancies.value()
        self.selectedValue_preg.setText(str(val))
    def updateSliderValue2(self):
        val = self.Glucose.value()
        self.selectedValue_glucose.setText(str(val))
    def updateSliderValue3(self):
        val = self.BloodPressure.value()
        self.selectedValue_blood.setText(str(val))
    def updateSliderValue4(self):
        val = self.SkinThickness.value()
        self.selectedValue_skin.setText(str(val))
    def updateSliderValue5(self):
        val = self.Insulin.value()
        self.selectedValue_insulin.setText(str(val))
    def updateSliderValue6(self):
        val = self.BMI.value()
        self.selectedValue_BMI.setText(str(val))
    def updateSliderValue7(self):
        val = self.DiabetesPredegreeFunction.value()
        self.selectedValue_predegree.setText(str(val))
    def updateSliderValue8(self):
        val = self.age.value()
        self.selectedValue_age.setText(str(val))

    def show_new_window(self):
        if self.window2 is None:
            self.window2 = secondWindow()
        self.window2.show()

    def generateData(self):
        global diabetes
        self.diabetes = pd.read_csv('diabetes_clean_03042021.csv')
        print("self.diabetes.head(8):")
        print(self.diabetes.head(10))
        print("self.diabetes.tail(5): ")
        print(self.diabetes.tail(5))
        print("self.diabetes.info():")
        print(self.diabetes.info())

        # number of rows and columns
        print("Rows: ", self.diabetes.shape[0])
        print("Columns: ", self.diabetes.shape[1])
        print("self.diabetes.describe(): ")
        print(self.diabetes.describe())
        print("Total Diabetic(1) and Non-diabetic(0) patient:", self.diabetes['Outcome'].value_counts())
        print("Mean grouped by diabetic(1) and Non-diabetic(0) patient: ")
        print(self.diabetes.groupby('Outcome').mean())

        # ----------- Machine learning ------
        print("-----Training data-----")
        x = self.diabetes.drop(columns='Outcome', axis=1)  # data
        y = self.diabetes['Outcome']  # model

        # Data Standardization
        self.scaler = StandardScaler()
        self.scaler.fit(x)
        standardized_data = self.scaler.transform(x)
        # print(standardized_data)

        x = standardized_data  # data
        y = self.diabetes['Outcome']  # model

        # Train Test Split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
        # test_size = 0.20 means 20% test data  and 80% train data

        # print(x.shape, x_train.shape, x_test.shape)

        # Training The Model with SVM Algorithm
        self. svm_classifier = svm.SVC(kernel='linear')
        # training the support vector machine classifier
        self.svm_classifier.fit(x_train, y_train)

        # Accuracy Check
        # Train Data
        x_train_prediction = self.svm_classifier.predict(x_train)
        training_data_accuracy = accuracy_score(x_train_prediction, y_train)
        print("Accuracy Score of the training data: ", training_data_accuracy)

        # Test data
        x_test_prediction = self.svm_classifier.predict(x_test)
        test_data_accuracy = accuracy_score(x_test_prediction, y_test)
        print("Accuracy Score of the testing data: ", test_data_accuracy)


        # correlation ---- highest  x=age y=blood pressure
        corr_matrix = self.diabetes.corr()
        print("self.diabetes.corr(): ", corr_matrix)

    # Making a predictive System
    def predictiveSystem(self):
        global result
        from random import randint
        count = randint(0, 767)
        Pregnancies = self.pregnancies.value()
        Glucose = self.Glucose.value()
        BloodPressure =self.BloodPressure.value()
        SkinThickness = self.SkinThickness.value()
        Insulin = self.Insulin.value()
        BMI = self.BMI.value()
        DiabetesPedigreeFunction = self.DiabetesPredegreeFunction.value()
        Age = self.age.value()

        # inputData = [value,1, 110, 92, 0, 0, 37.6,0.191, 30]
        sampleData = [count, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                      Age]

        # change the input data to numpy array
        sampleData_numpy = np.asarray(sampleData)

        # reshaping array as we are predicting for one instance
        reshaping = sampleData_numpy.reshape(1, -1)

        # standardize the input data
        standardize_sampleData = self.scaler.transform(reshaping)
        # print(std_data)

        prediction = self.svm_classifier.predict(standardize_sampleData)
        print(prediction)
        if (prediction[0] == 0):
            
            self.predictedValue.setText("Prediction: The person is not diabetic. ")
        else:
            
            self.predictedValue.setText("Prediction: The person is  diabetic. ")




if __name__ == '__main__':
        app = QApplication(sys.argv)
        
        with open("style.css", "r") as file:
            app.setStyleSheet(file.read())
            
        window = MainWindow()
        #window.setFixedSize(1500,1000)
        window.setMinimumSize(1200, 800)
     
        window.design()
        window.generateData()

        window.show()

        sys.exit(app.exec())

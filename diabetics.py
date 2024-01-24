import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# global diabetes

diabetes = pd.read_csv('diabetes_clean_03042021.csv')
print(diabetes.head(10))
print(diabetes.tail(5))

#number of rows and columns
print("Rows: ", diabetes.shape[0])
print("Columns: ", diabetes.shape[1])

print(diabetes.describe())
# print(diabetesData['Outcome'].value_counts())
print(diabetes.groupby('Outcome').mean())
print(diabetes.info())

# ----------- Machine learning ------

x = diabetes.drop(columns = 'Outcome', axis = 1)  # data
y = diabetes['Outcome'] # model

# Data Standardization
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
# print(standardized_data)

x = standardized_data # data
y =diabetes['Outcome'] # model

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42 )
# test_size = 0.20 means 20% test data  and 80% train data

print(x.shape, x_train.shape, x_test.shape)



# Training The Model with SVM Algorithm
svm_classifier = svm.SVC(kernel ='linear')
#training the support vector machine classifier
svm_classifier.fit(x_train, y_train)

# Accuracy Check
# Train Data
x_train_prediction = svm_classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy Score of the training data: ", training_data_accuracy)

# Test data
x_test_prediction = svm_classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy Score of the testing data: ", test_data_accuracy )





# Making a predictive System
def predictiveSystem():
    from random import randint
    count = randint(0, 767)
    # print(value)
    Pregnancies = input("Pregnancies: ")
    Glucose = input("Glucose: ")
    BloodPressure = input("BloodPressure: ")
    SkinThickness = input("SkinThickness: ")
    Insulin = input("Insulin: ")
    BMI = input("BMI: ")
    DiabetesPedigreeFunction = input("DiabetesPedigreeFunction: ")
    Age = input("Age: ")

    # inputData = [value,1, 110, 92, 0, 0,37.6,0.191, 30]
    sampleData = [count, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                  Age]

    # change the input data to numpy array
    sampleData_numpy = np.asarray(sampleData)

    # reshaping array as we are predicting for one instance
    reshaping = sampleData_numpy.reshape(1, -1)

    # standardize the input data
    standardize_sampleData = scaler.transform(reshaping)
    # print(std_data)

    prediction = svm_classifier.predict(standardize_sampleData)
    print(prediction)
    if (prediction[0] == 0):
        result = print("The person is not diabetic. ")
    else:
        result = print("The person is diabetic. ")


def histogram_graph():
    plt.hist(diabetes.Glucose)
    plt.show()
    plt.hist(diabetes.BloodPressure)
    plt.show()
    plt.hist(diabetes.SkinThickness)
    plt.show()
    plt.hist(diabetes.Insulin)
    plt.show()
    plt.hist(diabetes.BMI)
    plt.show()
    plt.hist(diabetes.DiabetesPedigreeFunction)
    plt.show()
    plt.hist(diabetes.Age)
    plt.show()



# histogram_graph()
predictiveSystem()

''' 

The main difference between 
training data and testing data is that training data is 
the subset of original data that is used to train 
the machine learning model, whereas testing data is used to check the 
accuracy of the model. 
The training dataset is generally larger 
in size compared to the testing dataset

'''


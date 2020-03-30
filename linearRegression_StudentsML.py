# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:31:08 2020

@author: Thegood
"""


    # -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:54:29 2020

@author: Thegood
"""
import webbrowser
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn.metrics import mean_absolute_error

#read in  all our data 
data = pd.read_csv("student-mat.csv", sep=";")

#print(data.head())
#attributes we actually want to see, in this case we picked one which are a of data type intergers 
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


# this is also known as a label.
predict = "G3"


# create training data
# set up an to arrays ,array with predict
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Remove the fields from the data set that we don't want to include in our model
del data[predict]


print(data.head())


html = data[0:100].to_html()

# Save the html to a temporary file
with open("data.html", "w") as f:
    f.write(html)
    
'''    # Open the web page in our web browser
full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))'''


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(1000):
    #Taking all attributes and split into 4 arrays.
    #Split our data into testing and training data, 10% testing , 90% training.
    #NB note if we trained the model of all the data the computer would just memorize that patterns because PC are good at see patterns , thus we cant train the model off all the data.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
   # print("Percentage Accuracy: " + str(acc))
    
    
    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)
    


# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: ', linear.coef_)
print('Intercept: ', linear.intercept_)
print("The best accurate percentage:\n",best)


# Find the error rate on the training set
mse = mean_absolute_error(y_train, linear.predict(x_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, linear.predict(x_test))
print("Test Set Mean Absolute Error: %.4f" % mse)


predictions = linear.predict(x_test)

icount=0
for x in range(len(predictions)):
    
    #x_test= x_test[["G1", "G2", "studytime", "failures", "absences"]]
    #print(icount,":", x_test[x])
    # Creating pandas dataframe from numpy array 
    
    #dataset = pd.DataFrame({'G1': x_test[:, x],'G2': x_test[:, x],'studytime': x_test[:, x],'failures': x_test[:, x],'absences': x_test[:, x], 'prediected G2':y_test[:, y]})
    #print(dataset)
    
    print(icount,":", x_test[x],y_test[x],"".format(predictions[x]))
    icount = icount + 1
    
    
'''html = newdata[0:100].to_html()

#Save the html to a temporary file
with open("newdata.html", "w") as f:
    f.write(html)

newdatadata = newdata[["G1", "G2", "studytime", "failures", "absences"]]

# Drawing and plotting model
plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()

# Drawing and plotting model
plot = "studytime"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()

# Drawing and plotting model
plot = "absences"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()'''

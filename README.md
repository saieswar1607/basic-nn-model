# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

<img width="544" alt="DL Output" src="https://github.com/Pavan-Gv/basic-nn-model/assets/94827772/a7e7085f-698d-45dd-b342-b7f3775b9a2e">

## DESIGN STEPS

<b>STEP 1:</b> Loading the dataset.

<b>STEP 2:</b> Split the dataset into training and testing.

<b>STEP 3:</b> Create MinMaxScalar objects ,fit the model and transform the data.

<b>STEP 4:</b> Build the Neural Network Model and compile the model.

<b>STEP 5:</b> Train the model with the training data.

<b>STEP 6:</b> Plot the performance plot.

<b>STEP 7:</b> Evaluate the model with the testing data.

## PROGRAM:
```
Developed By: Sai Eswar Kandukuri
RegNo: 212221240020
```
### Importing Required Packages :
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
```
### Authentication and Creating DataFrame From DataSheet :
```
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dl').sheet1
data = worksheet.get_all_values()
dataset = pd.DataFrame(data[1:], columns=data[0])
dataset = dataset.astype({'Input':'float'})
dataset = dataset.astype({'Output':'float'})
dataset.head()
```
### Assigning X and Y values :
```
X = dataset[['Input']].values
Y = dataset[['Output']].values
```
### Normalizing the data :
```
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state = 20)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train_scale = Scaler.transform(x_train)
```
### Creating and Training the model :
```
my_brain = Sequential([
    Dense(units = 4, activation = 'relu' , input_shape=[1]),
    Dense(units = 6),
    Dense(units = 1)

])
my_brain.compile(optimizer='rmsprop',loss='mse')
my_brain.fit(x=x_train_scale,y=y_train,epochs=20000)
```
### Plot the loss :
```
loss_df = pd.DataFrame(my_brain.history.history)
loss_df.plot()
```
### Evaluate the Model :
```
x_test1 = Scaler.transform(x_test)
my_brain.evaluate(x_test1,y_test)
```
### Prediction for a value :
```
X_n1 = [[30]]
input_scaled = Scaler.transform(X_n1)
my_brain.predict(input_scaled)
```
## Dataset Information:

<img width="137" alt="image" src="https://github.com/Pavan-Gv/basic-nn-model/assets/94827772/6090a65f-2e88-446b-9a0d-88c791ab0e2b">

## OUTPUT:

### Input & Output Data :

<img width="229" alt="image" src="https://github.com/Pavan-Gv/basic-nn-model/assets/94827772/e62a9912-2923-4544-b68c-ea90c0815ef9">

### Training Loss Vs Iteration Plot

<img width="327" alt="image" src="https://github.com/Pavan-Gv/basic-nn-model/assets/94827772/d2cf233c-906c-4cd3-a0bf-0b6569a3898d">


### Test Data Root Mean Squared Error

<img width="326" alt="image" src="https://github.com/Pavan-Gv/basic-nn-model/assets/94827772/1b1d5640-5dc8-4c5a-b378-19888609f810">


### New Sample Data Prediction

<img width="233" alt="image" src="https://github.com/Pavan-Gv/basic-nn-model/assets/94827772/7e72c01d-7ba7-4b2d-b4f4-8409f115b730">

## RESULT:
Therefore We successfully developed a neural network regression model for the given dataset.

# -*- coding: utf-8 -*-
"""
@author: Jeremy Kraisser
"""

from math import exp
import random
import csv

def logistic(x):
    return 1/(1 + exp(-x)) #calculate sigmoid

def dot(x, y):
    s = 0 #dot product over x and y lists
    xSize = len(x)
    ySize = len(y)
    if xSize == ySize:
        for i in range(xSize):
            s += x[i]*y[i]
    else:
        return None
    return s

def predict(model, point):
    dotProduct = dot(model, point['feat']) #dictionary value to be decided when excel file is created
    return logistic(dotProduct) #prediction based off sigmoid

def read_csv(filename):
    lines = []
    #file = open(filename)
    with open(filename) as file:
        read = csv.DictReader(file)
        for line in read:
            lines.append(line)
    return lines

#data = read_csv(filename) with the filename that I have conglomerated


def accuracy(data, predicts):
    accurate = 0 #number of correctly predicted data points
    dataSize = len(data)
    pSize = len(predicts)
    if dataSize == pSize:
        for i in range(dataSize):
            if predicts[i] > 0.5:
                truth = True
            else:
                truth = False
            
            if data[i]['label'] == truth: #if bool of data is same as predicted
                accurate += 1 #dictionary value to be decided based on my csv
    else:
        return None
    
    return float(accurate)/dataSize

def rand_init(num):
    return [random.gauss(0, 1) for i in range(num)]

def train(data, epochs, rate, lam):
    featSize = len(data[0]['feat'])
    model = rand_init(featSize) #initilaize at length of features which will be dependent on my excel file
    
    for e in range(epochs): #run for specified number of epochs to train
        for i in range(featSize):
            s = 0
            for j in range(len(data)):
                xij = data[j]['feat'][i]
                yj = data[j]['label']
                p = predict(model, data[j])
                s += xij*(yj - p)
            
            model[i] += rate*s - rate*lam*model[i] #update model weight
            '''
            learning rate annealing
            if rate > 9e-4:
                rate *= 0.75
            '''
    return model

def feat_engineering(data):
    d = []
    for line in data:
        point = {}
        point['label'] = (line['income'] == '>50K') #this would be the check for revenue/income above a certain limit
        
        featList = []
        featList.append(1.)
        #featList.append(float) include other feature engineering here
        
        d.append(point)
    return d

def tune(data, epochs, rate, lam, seed):  
    random.seed(seed)
    return train(data, epochs, rate, lam) #tune these specific inputs

'''
train_data = extract_features(load_adult_train_data(fn))
valid_data = extract_features(load_adult_valid_data(fn))
model = submission(train_data)
predictions = [predict(model, p) for p in train_data]
print("Training Accuracy:", accuracy(train_data, predictions))
predictions = [predict(model, p) for p in valid_data]
print("Validation Accuracy:", accuracy(valid_data, predictions))
import matplotlib.pyplot as plt
plot training and validation accuracy as epochs increase
                                      as rate increases/decreases
                                      as lam increases/decreases
'''
train_data = read_csv('ConglomerateData.xlsx')
print(train_data)


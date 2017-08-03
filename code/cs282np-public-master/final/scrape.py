#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:43:22 2017

@author: morrisyau
"""
import csv
import numpy as np
init = 1
print('hello world')


with open('factbook.csv', 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        print('hello')



with open('factbook.csv', 'rt') as f:
    reader = csv.reader(f)
    iterate = -1
    for row in reader:
        iterate += 1
        if iterate < 2:
            continue
        #print row[0]
        data = row[0].split(';')
        data = data[1:]
        dimensions = len(data)
        count = 0
        for datum in data:
            if datum == '':
                count += 1
        if count < dimensions/2:
            if init == 1:
                data_set = data
                init = 0
            else:
                data = np.array(data)
                data_set = np.vstack((data_set,data))
#print(np.sum(data_set,axis=0))
print(data_set)
row,col = data_set.shape
remove = []
empty = np.zeros(col)
for i in range(row):
    for j in range(col):
        if data_set[i,j] == '':
            empty[j] += 1

remove_row = []
for j in range(col):
    if empty[j] <= 1:
        for i in range(row):
            if data_set[i,j] == '':
                remove_row.append(i)
remove_row = set(remove_row)
select_row = [i for i in range(row)]
select_row = [s for s in select_row if s not in remove_row]
data_set = data_set[select_row,:]
row,col = data_set.shape

remove = []
for i in range(row):
    for j in range(col):
        if data_set[i,j] == '':
            remove.append(j)
        
print('remove')
remove = set(remove)
print(remove)
select = [i for i in range(col)]
select = [s for s in select if s not in remove]
#print('select')
#print(select)

data_set = data_set[:,select]
row,col = data_set.shape
clean_data = np.zeros((row,col))
for i in range(row):
    for j in range(col):
        #print(float(data_set[i,j]))
        clean_data[i,j] = float(data_set[i,j])
        
#print(clean_data)

#for j in range(col):
#    clean_data[:,j]  = 1.0/np.max(clean_data[:,j]) * clean_data[:,j]
#print(clean_data.shape)
#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sn
import math

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#print(train_data.head())
#print(test_data.head())
# visualizing the data
#sn.lineplot(style="darkgrid",data=train_data)
#plt.show()
ideal_data = pd.read_csv('ideal.csv')

#print(ideal_data.head())

def least_square(train_y,ideal_y):
    array_1 = np.array(train_y)
    array_2 = np.array(ideal_y)
    deviations = array_2 - array_1
    least_sq = np.sum(np.power(deviations,2))
    return least_sq,np.max(np.abs(deviations))

actual_y_data = ideal_data.drop(columns=['x'])
train_y_data = train_data.drop(columns=['x'])
ideal_columns = actual_y_data.columns
train_columns = train_y_data.columns

def compare_list_square():
    results = []
    maximum_deviations = {}
    for i in train_columns:
        least_sq = {}
        max_deviations = {}
        for k in ideal_columns:
            l_square,max_dev = least_square(train_y_data[i],actual_y_data[k])
            #least_sq.append({k:l_square})
            least_sq[k] = l_square
            max_deviations[k] = max_dev
        results.append({i:least_sq})
        maximum_deviations[i] = max_deviations
    return results,maximum_deviations

results,maximum_deviations = compare_list_square()
#print(maximum_deviations)
def get_ideal_function(results,maximum_deviations):
    ideal_functions = {}
    ideal_deviations = {}
    for res in results:
        for key in res:
            my_dict = res[key]
            # get a key with minimum value
            key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
            # add the key to the dictionary
            ideal_functions[key] = key_min
            focus_train = maximum_deviations[key]
            ideal_deviations[key_min] = focus_train[key_min]
    return ideal_functions,ideal_deviations

def print_ideal_functions(ideal_functions,ideal_deviations):
    for key,value in ideal_functions.items():
        print('The ideal function for {} is {}'.format(key,value))
    print(ideal_deviations)

ideal_functions,ideal_deviations = get_ideal_function(results,maximum_deviations)
print_ideal_functions(ideal_functions,ideal_deviations)

# ploting the ideal functions
# sn.lineplot(style="darkgrid",data=ideal_data[['x','y13','y1','y4','y34']])
# plt.show()

# # assigning test data

def assign_test():
    #ideal columns and x value
    ideal_columns = list(ideal_deviations.keys())
    columns = ['x'] + ideal_columns
    ideal_func = ideal_data[columns]

    for index, row in test_data.iterrows():
        #find the row with the x value of the test
        ideal_row = ideal_func.loc[ideal_func['x'] == row['x']]
        #loop through the ideal columns
        assigned = []

        for col in ideal_columns:
            dev = abs(row['y'] - ideal_row[col].values[0])
            max_dev = ideal_deviations[col]*math.sqrt(2)
            if dev <= max_dev:
                assigned.append({col:dev})
        print('The {} x value have been assigned ideal function : {}'.format(row['x'],assigned))
assign_test()

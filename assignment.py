#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sn

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.head()

# visualizing the data
sn.lineplot(style="darkgrid",data=train_data)
plt.show()
ideal_data = pd.read_csv('ideal.csv')

ideal_data.head()

def least_square(train_y,ideal_y):
    array_1 = np.array(train_y)
    array_2 = np.array(ideal_y)
    deviations = array_2 - array_1
    least_sq = np.sum(np.power(deviations,2))
    return least_sq

actual_y_data = ideal_data.drop(columns=['x'])
train_y_data = train_data.drop(columns=['x'])
ideal_columns = actual_y_data.columns
train_columns = train_y_data.columns

def compare_list_square():
    results = []
    for i in train_columns:
        least_sq = {}
        for k in ideal_columns:
            l_square = least_square(train_y_data[i],actual_y_data[k])
            #least_sq.append({k:l_square})
            least_sq[k] = l_square
        results.append({i:least_sq})
    return results

results = compare_list_square()
def get_ideal_function(results):
    ideal_functions = {}
    for res in results:
        for key in res:
            my_dict = res[key]
            # get a key with minimum value
            key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
            # add the key to the dictionary 
            ideal_functions[key] = key_min
    return ideal_functions

def print_ideal_functions(ideal_dict):
    for key,value in ideal_dict.items():
        print('The ideal function for {} is {}'.format(key,value))
print_ideal_functions(get_ideal_function(results))

# ploting the ideal functions
sn.lineplot(style="darkgrid",data=ideal_data[['x','y13','y1','y4','y34']])
plt.show()

# # getting the maximum absolute deviation between train data and ideal functions
# def max_deviation(train_data, ideal_data):
#     deviations = train_data - ideal_data
#     return max(abs(deviations))


# #print(test_data.head())

# def assign_ideal_functions(test_data):
#     for i, row in test_data.iterrows():
#         print(row['y'])

# assign_ideal_functions(test_data)

# # array1 =np.array( [2,3,4])
# # array2 =np.array( [2,6,4])

# # print(max_deviation(array1, array2 ))

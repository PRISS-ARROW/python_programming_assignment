#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sn
import math
import sqlite3
import sqlalchemy as db

class DataStore():
    """ A class to represent a DataStore.

    ...

    Attributes
    ----------
    train_data : DataFrame
        DataFrame that keeps train data
    ideal_data : DataFrame
        DataFrame that keeps ideal functions data
    test_data : DataFrame
        DataFrame that keeps test data

    Methods
    -------
    visualize_train_data():
        Visualizes train data.
    visualize_ideal_functions(columns):
        Visualizes selected ideal functions data.
    insert_train_data():
        Inserts train data to the sqlite database.
    insert_test_data():
        Inserts test data to the sqlite database.
    insert_ideal_data():
        Inserts deal data to the sqlite database.
    get_train_data():
        Gets train data from the sqlite database.
    get_test_data():
        Gets test data from the sqlite database.
    get_ideal_data():
        Gets ideal data from the sqlite database.
    create_assignment_table():
        Create an assignment table in the the sqlite database.
    get_assigned_data():
        Gets assigned ideal data from the sqlite database.
    close_connection():
        Closes sqlite database connection.
    """

    def __init__(self, train_data,ideal_data,test_data):
        """
        Constructs all the necessary attributes for the ideal DataStore object.
        Parameters
        -----------
        train_data: DataFrame
        ideal_data: DataFrame
        test_data: DataFrame
        """
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.test_data = test_data
        self.actual_y_data = self.ideal_data.drop(columns=['x'])
        self.train_y_data = self.train_data.drop(columns=['x'])
        self.engine = db.create_engine('sqlite:////Users/nmalango/Desktop/projects/python_programming_assignment/test.db') #create test.sqlite automatically
        self.connection = self.engine.connect()

    def visualize_train_data(self):
        """
        Visualizes train data using seaborn

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        sn.set_theme(style="darkgrid")
        df = self.train_data.melt('x', var_name='cols',  value_name='y-axis')
        sn.lineplot(x='x', y="y-axis", hue='cols',data=df).set_title('Train Data Scatter Plot')
        plt.show()

    def visualize_ideal_functions(self,columns):
        """
        Visualizes selected ideal functions using seaborn

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        df = self.ideal_data[columns]
        sn.set_theme(style="darkgrid")
        data = df.melt('x', var_name='cols',  value_name='y-axis')
        sn.lineplot(x='x', y="y-axis", hue='cols',data=data).set_title('Chosen Ideal Functions Scatter Plot')
        plt.show()

    def insert_train_data(self):
        """
        Inserts training data into the sqlite database
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        self.train_data.to_sql('train_table',self.connection,if_exists='append',index=False)

    def get_train_data(self):
        """
        Gets training data from the sqlite database
        Parameters
        ----------
        None
        Returns
        -------
        df(DataFrame): A dataframe of training data
        """
        results = self.connection.execute('select * from train_table').fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        return df

    def insert_ideal_data(self):
        """
        Inserts ideal data into the sqlite database
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        self.ideal_data.to_sql('ideal_table',self.connection,if_exists='append',index=False)

    def get_ideal_data(self):
        """
        Gets ideal data from the sqlite database
        Parameters
        ----------
        None
        Returns
        -------
        df(DataFrame): A dataframe of ideal data
        """
        results = self.connection.execute('select * from ideal_table').fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        return df

    def create_assignment_table(self):
        """
        Creates assignment_table into the sqlite database
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        self.metadata = db.MetaData()
        self.assignment_table = db.Table('assignment_table',self.metadata,
        db.Column('x',db.Float(),nullable='False'),
        db.Column('y',db.Float(),nullable='False'),
        db.Column('deviation',db.Float(),nullable='False'),
        db.Column('ideal_function',db.Float(),nullable='True')
        )
        self.metadata.create_all(self.engine)

    def get_assigned_data(self):
        """
        Gets ideal data into the sqlite database
        Parameters
        ----------
        None
        Returns
        -------
        df(DataFrame): A dataframe of assigned test data
        """
        results = self.connection.execute(db.select([self.assignment_table])).fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        return df

    def close_connection(self):
        """
        Close an sqlite database connection
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        self.connection.close()

class IdealProcessor(DataStore):
    """ A class to represent a IdealProcessor
    ...

    Attributes
    ----------
    train_y_data : DataFrame
        DataFrame that keeps ys train data
    actual_y_data : DataFrame
        DataFrame that keeps ys ideal functions data
    ideal_columns : array
        an array of columns in actual_y_data
    train_columns : array
        an array of columns in train_y_data

    Methods
    -------
    least_square(train_y,ideal_y):
        Calculates the least squares.
    compare_list_square():
        Loops through the columns of ideal functions and train data the Calculates the least_squares
    get_ideal_function(results,maximum_deviations):
        Gets the ideal functions for the training data.
    print_ideal_functions(ideal_functions):
        Prints ideal functions for the training data.
    assign_test(deal_deviations):
        Prints ideal functions for the training data.
    """

    def __init__(self, train_data,ideal_data,test_data):
        """
        Constructs all the necessary attributes for the ideal IdealProcessor object.
        Parameters
        -----------
        train_data: DataFrame
        ideal_data: DataFrame
        test_data: DataFrame
        """
        super().__init__(train_data,ideal_data,test_data)
        self.train_y_data = self.train_data.drop(columns=['x'])
        self.actual_y_data  = self.ideal_data.drop(columns=['x'])
        self.ideal_columns = self.actual_y_data.columns
        self.train_columns = self.train_y_data.columns
        self.create_assignment_table()

    def least_square(self,train_y,ideal_y):
        """
        Computes least_square given the train_y and ideal y.

        Parameters
        ----------
        train_y : array, required
            array of train y values
        ideal_y : array, required
            array of ideal y values

        Returns
        -------
        least_sq(float) : Least square between train y and ideal y
        maximum_deviation(float) : maximum_deviation deviation trainy and ideal y
        """
        array_1 = np.array(train_y)
        array_2 = np.array(ideal_y)
        deviations = array_2 - array_1 #find the deviations between arrays
        least_sq = np.sum(np.power(deviations,2)) #calculate least_square
        return least_sq,np.max(np.abs(deviations))

    def compare_list_square(self):
        """
        Compares least_squares .

        Parameters
        ----------
        None

        Returns
        -------
        results(dictionary) : Least squares for each train column against ideal columns
        maximum_deviations(dictionary) : maximum_deviations for each train column against ideal columns
        """
        results = []
        maximum_deviations = {}
        #loop through train columns
        for i in self.train_columns:
            least_sq = {}
            max_deviations = {}
            #loop through ideal columns
            for k in self.ideal_columns:
                l_square,max_dev = self.least_square(self.train_y_data[i],self.actual_y_data[k])
                least_sq[k] = l_square
                max_deviations[k] = max_dev
            results.append({i:least_sq})
            maximum_deviations[i] = max_deviations
        return results,maximum_deviations

    def get_ideal_function(self,results,maximum_deviations):
        """
        Gets ideal functions for the train data.

        Parameters
        ----------
        results : dictionary, required
            a dictionary of least squares
        maximum_deviations : dictionary, required
            a dictionary of maximum_deviations

        Returns
        -------
        ideal_functions(dictionary) : Ideal functions for train data
        ideal_deviations(dictionary) : maximum deviations for the selected ideal functions
        """
        ideal_functions = {}
        ideal_deviations = {}
        #loop through the least_squares
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

    def print_ideal_functions(self,ideal_functions):
        """
        Prints ideal function for each train data

        Parameters
        ----------
        ideal_functions : dictionary, required
            a dictionary of ideal functions of each train function
        Returns
        -------
        None
        """
        for key,value in ideal_functions.items():
            print('The ideal function for {} is {}'.format(key,value))

    def assign_test(self,ideal_deviations):
        """
        Assigns an ideal function to the test data.

        Parameters
        ----------
        ideal_deviations : dictionary, required
            a dictionary of ideal functions and its maximum_deviation against the train function
        Returns
        -------
        None
        """
        #ideal columns and x value
        ideal_columns = list(ideal_deviations.keys())
        columns = ['x'] + ideal_columns
        ideal_func = self.ideal_data[columns]

        for index, row in self.test_data.iterrows():

            #find the row with the x value of the test
            ideal_row = ideal_func.loc[ideal_func['x'] == row['x']]
            #loop through the ideal columns
            assigned = {}
            ideal_values = {}
            #check the assignment condition
            for col in ideal_columns:
                dev = abs(row['y'] - ideal_row[col].values[0])
                max_dev = ideal_deviations[col]*math.sqrt(2)
                if dev <= max_dev:
                    assigned[col]=dev
                    ideal_values[col]=ideal_row[col].values[0]
            #test if its an empty dictionary
            ideal_val = None
            ideal_dev = None
            if len(assigned) == 0:
                pass
            else:
                assigned_func = min(assigned, key=lambda k: assigned[k]) #get the key with the minimum deviation
                ideal_val = ideal_values[assigned_func]
                ideal_dev = assigned[assigned_func]
            #save the outcome to the database
            query = db.insert(self.assignment_table).values(x=row['x'],y=row['y'],deviation=ideal_dev,ideal_function=ideal_val)
            ResultProxy = self.connection.execute(query)


def main():
    #create an instance of an IdealProcessor
    ideal_processor = IdealProcessor(pd.read_csv('train.csv'),pd.read_csv('ideal.csv'),pd.read_csv('test.csv'))
    #visualize the train data
    ideal_processor.visualize_train_data()
    #save the datasets to their respective table using sqlalchemy
    ideal_processor.insert_train_data()
    ideal_processor.insert_ideal_data()
    #compare the least_square and their maximum_deviations
    results,maximum_deviations = ideal_processor.compare_list_square()
    #get the ideal functions and ideal deviations
    ideal_functions,ideal_deviations = ideal_processor.get_ideal_function(results,maximum_deviations)
    ideal_processor.print_ideal_functions(ideal_functions)

    #visualize the ideal functions selected
    columns = ['x']+list(ideal_functions.values())
    ideal_processor.visualize_ideal_functions(columns)

    #assign the test data
    ideal_processor.assign_test(ideal_deviations)

    #print the assigned,train,ideal functions data saved
    print(ideal_processor.get_ideal_data())
    print(ideal_processor.get_train_data())
    print(ideal_processor.get_assigned_data())
    #close db connection
    ideal_processor.close_connection()
if __name__ == '__main__':
    main()

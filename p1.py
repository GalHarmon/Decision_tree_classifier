import copy
import math
import pandas as pd
from pptree import *
import numpy as np
import random
import calendar
from datetime import datetime
from scipy.stats import chi2


class treeObject:
    def __init__(self, atribute, optionStr, father, ans, myExample, ifLeave):
        if ans == '':
            self.name = atribute
        else:
            self.name = str(ans) + ', ' + atribute
        self.children = []  # list of children trees
        self.optionStr = optionStr
        self.parent = father  # the tree that this tree came from
        self.atribute = atribute
        self.ans = ans
        self.myExample = myExample
        self.ifLeave = ifLeave

    def __str__(self):
        t = self.name
        return t

    def proning(self):
        for child in self.children:#for every child is not a leave
            if not child.ifLeave:
                child.proning()
        ifCut = chi2Test(self.myExample, self.parent.myExample, self)
        if not ifCut: #we need to cut
            if self.atribute == 'busy' or self.atribute == 'notbusy':
                self.parent.atribute = self.atribute
                print(self.parent.atribute)
                self.parent.children.remove(self)

def chi2Test(ex, exFather, subTree):
    busyFather = len(exFather[exFather['Rented Bike Count'] == 'busy'])
    notBusyFather = len(exFather[exFather['Rented Bike Count'] == 'not busy'])
    freeRange = busyFather + notBusyFather - 1#from the furmola
    crit = chi2.ppf(q = 0.05, df = freeRange)#use in package thet calc the value from the chi test table
    delta = 0
    for ans in subTree.children:
        newData = ex[ex[subTree.atribute] == ans.ans]
        if not newData.empty:#calc all the parameters for the delta value
            newBusy = len(newData[newData['Rented Bike Count'] == 'busy'])
            NewNotBusy = len(newData[newData['Rented Bike Count'] == 'not busy'])
            Pk = busyFather*(newBusy+NewNotBusy)/(busyFather+notBusyFather)
            Nk = notBusyFather*(newBusy+NewNotBusy)/(busyFather+notBusyFather)
            delta = delta + ((newBusy-Pk)**2/Pk) + ((NewNotBusy-Nk)**2/Nk)
    if delta < crit:#if relevan, what false is need to cut
        return False
    else:
        return True

def build_tree(ratio):
    df, attributes = bucketMyData()#function that return normal dataframe and a dictionary of the attributes
    msk = np.random.rand(len(df)) < ratio
    train = df[msk]#split the data to train and test by the ratio
    test = df[~msk]
    fective = treeObject('exxxx', '', '', '', train, False)
    myTree = decisionTreeLearning(train, attributes, fective, '', '')#call the recursive function
    myTree.proning()
    print_tree(myTree, "children", "name", horizontal=True)#print the tree
    error = checkError(myTree, test)
    print('The error rate is: ' + str(error))



def checkError(tree, test):
    totalErrors = 0
    list2D = []
    for row in test.iterrows():#for every row in the dataframe
        if type(row) == tuple:
            list = row[1].values.tolist()
            list2D = build2Dlist(list)#change to 2D array according to the data
        totalErrors = totalErrors + calcSpecificError(tree, list2D)
    return totalErrors/len(test)#calc the average

def build2Dlist(list):#build the array
    temp = []
    temp.append(['Rented Bike Count', list[0]])
    temp.append(['Hour', list[1]])
    temp.append(['Temperature', list[2]])
    temp.append(['Humidity', list[3]])
    temp.append(['Wind', list[4]])
    temp.append(['Visibility', list[5]])
    temp.append(['DewPointTemperature', list[6]])
    temp.append(['SolarRadiation', list[7]])
    temp.append(['Rainfall', list[8]])
    temp.append(['Snowfall', list[9]])
    temp.append(['Season', list[10]])
    temp.append(['Holiday', list[11]])
    temp.append(['FuncDay', list[12]])
    temp.append(['Quarter', list[13]])
    temp.append(['dayInWeek', list[14]])
    return temp


def calcSpecificError(tree, list):#calc the error for every row, if wrong return 1 else return 0
    if tree.ans == 'busy' or tree.ans == 'not busy':
        if list[0][1] == tree.ans:
            return 0
        else:
            return 1
    elif tree.atribute == 'busy' or tree.atribute == 'not busy':
        if list[0][1] == tree.atribute:
            return 0
        else:
            return 1
    else:
        for x in range(len(list)):#moving al the tree way
            if list[x][0] == tree.atribute:
                for child in tree.children:
                    if child.ans == '':
                        if list[0][1] == tree.atribute:
                            return 0
                        else:
                            return 1
                    elif list[x][1] == child.ans:
                        return calcSpecificError(child, list)

def findMegority(ex, par, ans, data):#find the majority of the data in the examples
    busyNum = len(ex.loc[ex['Rented Bike Count'] == 'busy'])
    notBusyNum = len(ex.loc[ex['Rented Bike Count'] == 'not busy'])
    if busyNum > notBusyNum:
        return treeObject('busy', [], par, ans, data, True)
    else:
        return treeObject('not busy', [], par, ans, data, True)

def decisionTreeLearning(dataTrain, attributes, cameFrom, parentex, ans):
    tree = treeObject('', '', '', '', dataTrain, None)
    if len(dataTrain) == 0:#stop condition - no more data
        tree = findMegority(parentex, cameFrom, ans, dataTrain)
        return tree
    elif len(dataTrain) == len(dataTrain.loc[dataTrain['Rented Bike Count'] == 'not busy']):#stop condition - all the data with the same answer
        tree = treeObject('not busy', '', cameFrom, ans, dataTrain, True)
        return tree
    elif len(dataTrain) == len(dataTrain.loc[dataTrain['Rented Bike Count'] == 'busy']):#stop condition - all the data with the same answer
        tree = treeObject('busy', '', cameFrom, ans, dataTrain, True)
        return tree
    elif len(attributes) == 0:#stop condition - no more attribiuts to check
        tree = findMegority(dataTrain, cameFrom, ans, dataTrain)
        return tree
    else:
        A = IMPORTANCE(attributes, dataTrain)#return the attrubute wuth the minimum value
        if A != '':
            tree = treeObject(A, attributes[A], cameFrom, ans, dataTrain, False)#create new branch
            for value in attributes[A]:
                atr = copy.deepcopy(attributes)
                atr.pop(A)#take off the relevant attribute
                exs = dataTrain.loc[dataTrain[A] == value]#set the new example
                subtree = decisionTreeLearning(exs, atr, tree, dataTrain, value)
                tree.children.append(subtree)#add that subtree as a children to his father
        else:
            if len(dataTrain) == 0:  # #stop condition - no more data
                tree = findMegority(parentex, cameFrom, ans, dataTrain)
                return tree
            elif len(dataTrain) == len(dataTrain.loc[dataTrain['Rented Bike Count'] == 'not busy']):
                tree = treeObject('not busy', '', cameFrom, ans, dataTrain, True)
                return tree
            elif len(dataTrain) == len(dataTrain.loc[dataTrain['Rented Bike Count'] == 'busy']):
                tree = treeObject('busy', '', cameFrom, ans, dataTrain, True)
                return tree
            else:
                tree = findMegority(dataTrain, cameFrom, ans, dataTrain)
                return tree
    return tree

def IMPORTANCE(attributes, data):
    max = 0
    entropyAttributesName = ''
    for key in attributes:#move on all the relevant attributes that left to check
        if key != 'Rented Bike Count':
            Entropy = calcEntropy(key, data) #for every attribute we want the max value for minimize the total value
            if Entropy > max:
                max = Entropy
                entropyAttributesName = key
    return entropyAttributesName

def calcEntropy(attribute, data):#calc the entropy per option in every attribute
    values = data[attribute].unique()#collect al the option in the field of the specific attribute
    totalEntropy = 0
    for option in values:
        tempData = data.loc[data[attribute] == option]
        countsFalse = len(tempData.loc[tempData['Rented Bike Count'] == 'not busy'])
        countsTrue = len(tempData.loc[tempData['Rented Bike Count'] == 'busy'])
        totalOption = countsFalse+countsTrue
        pFalse = countsFalse/totalOption
        pTrue = countsTrue/totalOption
        entropy = calcAllEntropy(pFalse, pTrue)
        totalEntropy = totalEntropy + (totalOption/len(data))*entropy
    return totalEntropy

def calcAllEntropy(pFalse, pTrue):
    if pFalse == 0 and pTrue != 0:#to ignore not valid step in math
        return -pTrue*math.log(pTrue, 2)#via the furmola
    elif pTrue == 0 and pFalse != 0:
        return -pFalse*math.log(pFalse, 2)
    elif pTrue == 0 and pFalse == 0:
        return 0
    else:
        return -pFalse*math.log(pFalse, 2)-pTrue*math.log(pTrue, 2)

def bucketMyData():#change the value in the dataframe from continual to "drilldown"
    attributes = {}  # array that every element contain string of the attributes name and array of the values
    df = pd.read_csv("/Users/galharmon/Desktop/SeoulBikeData.csv", encoding='unicode_escape')
    df.columns = ['Date', 'Rented Bike Count', 'Hour', 'Temperature', 'Humidity', 'Wind', 'Visibility','DewPointTemperature', 'SolarRadiation', 'Rainfall', 'Snowfall', 'Season', 'Holiday', 'FuncDay']
    df['Quarter'] = pd.DatetimeIndex(df['Date']).month
    df['Quarter'] = df['Quarter'].apply(lambda h: 'Q1' if 3 <= h else ('Q2' if h <= 6 else ('Q3' if h <= 9 else 'Q4')))
    attributes.update({'Quarter': ['Q1', 'Q2', 'Q3', 'Q4']})
    df['dayInWeek'] = pd.DatetimeIndex(df['Date']).dayofweek
    df = df.drop(columns=['Date'])
    df['dayInWeek'] = df['dayInWeek'].apply(lambda h: 'weekend' if 3 <= h else 'week')
    attributes.update({'dayInWeek': ['weekend', 'week']})
    df['Rented Bike Count'] = np.where(df['Rented Bike Count'] >= 650, 'busy', 'not busy')
    attributes.update({'Rented Bike Count': ['busy', 'not busy']})
    df['Hour'] = df['Hour'].apply(lambda h: 'between 0-6' if 0 <= h <= 6 else ('between 7-21' if 7 <= h <= 21 else 'between 22-23'))
    attributes.update({'Hour': ['between 0-6', 'between 7-21', 'between 22-23']})
    df['Temperature'] = df['Temperature'].apply(lambda t: 'less than 0 C' if t <= 0 else ('between 0-15 C' if t <= 15 else 'above to 15 C'))
    attributes.update({'Temperature': ['less than 0 C', 'between 0-15 C', 'above to 15 C']})
    df['Humidity'] = np.where(df['Humidity'] >= 50, 'less than 50%', 'above to 50%')
    attributes.update({'Humidity': ['less than 50%', 'above to 50%']})
    df['Wind'] = np.where(df['Wind'] >= 5, 'above to 5(m/s)', 'less than 5(m/s)')
    attributes.update({'Wind': ['less than 5(m/s)', 'above to 5(m/s)']})
    df['Visibility'] = df['Visibility'].apply(lambda v: 'more than 1300 (10m)' if v >= 1300 else ('between 666 (10m) to 1300 (10m)' if v >= 666 else 'less than 666(10m)'))
    attributes.update({'Visibility': ['more than 1300 (10m)', 'between 666 (10m) to 1300 (10m)', 'less than 666(10m)']})
    df['DewPointTemperature'] = np.where(df['DewPointTemperature'] >= 5, 'more than 5 C', 'less than 5 C')
    attributes.update({'DewPointTemperature': ['more than 5 C', 'less than 5 C']})
    df['SolarRadiation'] = np.where(df['SolarRadiation'] >= 1.5, 'more than 1.5 (MJ/m2)', 'less than 1.5 (MJ/m2)')
    attributes.update({'SolarRadiation': ['more than 1.5 (MJ/m2)', 'less than 1.5 (MJ/m2)']})
    df['Rainfall'] = np.where(df['Rainfall'] <= 0, 'less than 0 (mm)', 'more than 0 (mm)')
    attributes.update({'Rainfall': ['less than 0 (mm)', 'more than 0 (mm)']})
    df['Snowfall'] = np.where(df['Snowfall'] <= 0, 'less than 0 (cm)', 'more than 0 (cm)')
    attributes.update({'Snowfall': ['less than 0 (cm)', 'more than 0 (cm)']})
    attributes.update({'Season': ['Winter', 'Autumn', 'Spring', 'Summer']})
    attributes.update({'Holiday': ['No Holiday', 'Holiday']})
    attributes.update({'FuncDay': ['Yes', 'No']})
    return df, attributes

def is_busy(row_input):
    df, attributes = bucketMyData()  # function that return normal dataframe and a dictionary of the attributes
    fective = treeObject('exxxx', '', '', '', df, False)
    myTree = decisionTreeLearning(df, attributes, fective, '', '')  # call the recursive function
    list = build2Darray(row_input)
    ans = ifBusy(myTree, list)
    if ans == 'busy':
        print(1)
    else:
        print(0)

def build2Darray(row_input):#build array for the input from the user according to the condition
    temp = []
    if row_input[1] <= 6:
        temp.append(['Hour', 'between 0-6'])
    elif row_input[1] <= 21:
        temp.append(['Hour', 'between 7-21'])
    else:
        temp.append(['Hour', 'between 22-23'])
    if row_input[2] <= 0:
        temp.append(['Temperature', 'less than 0 C'])
    elif row_input[2] <= 16:
        temp.append(['Temperature', 'between 0-15 C'])
    else:
        temp.append(['Temperature', 'above to 15 C'])
    if row_input[3] >= 50:
        temp.append(['Humidity', 'less than 50%'])
    else:
        temp.append(['Humidity', 'above to 50%'])
    if row_input[4] >= 5:
        temp.append(['Wind', 'above to 5(m/s)'])
    else:
        temp.append(['Wind', 'less than 5(m/s)'])
    if row_input[5] >= 1300:
        temp.append(['Visibility', 'more than 1300 (10m)'])
    elif row_input[5] >= 666:
        temp.append(['Visibility', 'between 666 (10m) to 1300 (10m)'])
    else:
        temp.append(['Visibility', 'less than 666(10m)'])
    if row_input[6] >= 5:
        temp.append(['DewPointTemperature', 'more than 5 C'])
    else:
        temp.append(['DewPointTemperature', 'less than 5 C'])
    if row_input[7] >= 1.5:
        temp.append(['SolarRadiation', 'more than 1.5 (MJ/m2)'])
    else:
        temp.append(['SolarRadiation', 'less than 1.5 (MJ/m2)'])
    if row_input[8] <= 0:
        temp.append(['Rainfall', 'less than 0 (mm)'])
    else:
        temp.append(['Rainfall', 'more than 0 (mm)'])

    if row_input[9] <= 0:
        temp.append(['Snowfall', 'less than 0 (cm)'])
    else:
        temp.append(['Snowfall', 'more than 0 (cm)'])
    temp.append(['Season', row_input[10]])
    temp.append(['Holiday', row_input[11]])
    temp.append(['FuncDay', row_input[12]])
    date = pd.to_datetime(row_input[0], format="%d/%m/%Y")#change the format of the date to be able to take the month number and the day of the week
    if date.month <= 3:
        temp.append(['Quarter', 'Q1'])
    elif date.month <= 6:
        temp.append(['Quarter', 'Q2'])
    elif date.month <= 9:
        temp.append(['Quarter', 'Q3'])
    else:
        temp.append(['Quarter', 'Q4'])
    if date.weekday() >= 3:
        temp.append(['dayInWeek', 'weekend'])
    else:
        temp.append(['dayInWeek', 'week'])
    return temp

def ifBusy(tree, list):
    if tree.ans == 'busy' or tree.ans == 'not busy':
        return tree.ans
    elif tree.atribute == 'busy' or tree.atribute == 'not busy':
        return tree.atribute
    else:
        for child in tree.children:
            for option in list:
                if option[1] == child.ans:
                    if child.ans == 'busy' or child.ans == 'not busy':
                        return child.ans
                    elif child.atribute == 'busy' or child.atribute == 'not busy':
                        return child.atribute
                    else:
                        return ifBusy(child, list)

def tree_error(k):
    df, attributes = bucketMyData()#function that return normal dataframe and a dictionary of the attributes
    totErrors = 0
    fective = treeObject('exxxx', '', '', '', df, False)
    temp = np.array_split(df, k)
    test = pd.DataFrame([])
    train = None
    counter = 0
    for x in range(k):
        for y in range(k):
            if x == y:
                test = temp[y]
            elif counter == 0:
                train = temp[y]
                counter = counter + 1
            else:
                train = pd.concat([train, temp[y]], axis=0)
        myTree = decisionTreeLearning(train, attributes, fective, '', '')  # call the recursive function
        totErrors = totErrors + checkError(myTree, test)
    print('The average error rate is: ' + str(totErrors/k))



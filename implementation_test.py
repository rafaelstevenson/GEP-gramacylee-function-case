import numpy as np
import pandas as pd
from GEP import GeneExpressionProgramming


df = pd.read_excel('../testing_datasets/forrester.xls')
print(df.head())

class MinMaxScaler:
    def fit_transform(self, arr):
        self.max = max(arr)
        self.min = min(arr)
        arr = arr-self.min/(self.max-self.min)
        return arr
    def transform(self, arr):
        arr = arr-self.min/(self.max-self.min)
        return arr

#scaler_x = MinMaxScaler()
#scaler_y = MinMaxScaler()
#df['AoA'] = scaler_x.fit_transform(df['AoA'])
#df['Cd'] = scaler_y.fit_transform(df['Cd'])


func_set = ['+','-','*','/','(sin)','(X2)']

term_set = ['a', '?']
#term_set = ['a']
const_range = [1,15]
operator_probabilities = {
    "Mutation":0.1, "Inversion":0.1, "IS Transposition":0.1,
    "RIS Transposition":0.1, "One-point Recombination":0.3,
    "Two-point Recombination":0.3
}

head_length = 13
population_size = 300
generations = 200
fitness_func = 'mse'

GEPProcess = GeneExpressionProgramming(head_length,func_set,term_set,const_range,operator_probabilities)
GEPProcess.RunGEP(df['input1'],df['output'],population_size,generations,fitness_func)
GEPProcess.VisualizeResults()
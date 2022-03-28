import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score

one_arity_func = ['(sqrt)', '(sin)', '(exp)', '(ln)', '(inv)', '(gau)', '(X2)']
two_arity_func = ['+', '-', '*', '/']

def ChromToET(chromosome):
    '''Take a string of chromosome change it to
    a dictionary of expression tree{row:element_on_row}'''
    expr_tree = {0: [chromosome[0]]}

    i = 1
    start_counter = 1
    while True:

        take_next = 0
        terminal_row = True

        for element in expr_tree[i - 1]:
            if element in two_arity_func:
                terminal_row = False
                take_next += 2
            elif element in one_arity_func:
                terminal_row = False
                take_next += 1

        if terminal_row==True:
            break

        last_counter = start_counter + take_next
        next_elem = chromosome[start_counter:last_counter]
        expr_tree[i] = next_elem
        start_counter = last_counter

        i += 1

    return expr_tree


def EvaluateET(chromosome, variable_dict, const_list):
    '''Take chromosome and terminal variable dictionary{'symbol':value}
    and perform calculation from the chromosome->ET->calculation->prediction'''

    # Change string to list in each row of ET and change variables to sample value
    expr_tree = ChromToET(chromosome)
    el_dc = 0
    for i in range(len(expr_tree)):  # iterate rows
        el = 0
        for element in expr_tree[i]:  # iterate elements in a row
            if element in variable_dict.keys():
                expr_tree[i][el] = str(variable_dict[element])
            elif element =='?':
                expr_tree[i][el] = str(const_list[el_dc])
                el_dc += 1
            el += 1


    def operate_two_arity(representation, a, b):
        a = float(a)
        b = float(b)

        if representation=='+':
            result = a + b
        elif representation=='-':
            result = a - b
        elif representation=='*':
            result = a * b
        elif representation=='/':
            if b==0:
                b = 1e-6
            result = a / b

        return str(result)

    def operate_one_arity(representation, a):
        a = float(a)

        if representation=='(sqrt)':
            if a >= 0:
                result = a ** 0.5
            else:
                result = (abs(a)) ** 0.5
            # result = math.sqrt(a)

        elif representation=='(sin)':
            result = math.sin(a)
        elif representation=='(exp)':
            try:
                result = math.exp(a)
            except:
                result = 1e6
        elif representation=='(ln)':
            if a==0:
                a = 1e-6
            elif a < 0:
                a = abs(a)
            result = math.log(a, math.e)
        elif representation=='(inv)':
            if a==0:
                a = 1e-6
            result = 1 / a
        elif representation=='(gau)':
            result = np.random.normal(1)
        elif representation=='(X2)':
            result = a ** 2

        return str(result)

    for row in range(len(expr_tree) - 2, -1, -1):  # iterate rows from second last row to root
        i = 0
        for element in expr_tree[row]:
            if element in two_arity_func:
                a = expr_tree[row + 1][0]
                b = expr_tree[row + 1][1]

                result = operate_two_arity(element, a, b)
                # buang 2 elemen pertama di row+1 dan replace elemen pertama di row
                expr_tree[row + 1] = expr_tree[row + 1][2:]
                expr_tree[row][i] = result

            elif element in one_arity_func:
                a = expr_tree[row + 1][0]
                result = operate_one_arity(element, a)
                expr_tree[row + 1] = expr_tree[row + 1][1:]
                expr_tree[row][i] = result

            i += 1

    prediction = float(expr_tree[0][0])
    return prediction


# load dataset to get ground truth
df = pd.read_excel('../testing_datasets/forrester.xls')
y_true = df['output']
x = df['input1']

# declare the chromosome and the variable dictionary
chromosome = ['*', '(X2)', '(sin)', '*', '-', '+', '(X2)', '(X2)', '*', '-', '(X2)', '?', '(sin)', '?', 'a', 'a', 'a', 'a', '?', '?', 'a', 'a', 'a', 'a', 'a', '?', '?', '0', '9', '3', '4', '7', '5', '3', '10', '4', '2', '4', '3', '3', '4']
term_set = ['a']
const_list = [1.9907733343148943, 11.454457419645728, 10.220522956304762, 2.5186616881399333, 2.678209163314668, 10.394952859605853, 6.198823321013686, 8.37558978215862, 10.600249168174287, 7.502779881454574, 2.6961058577195915, 5.254872366367976, 6.875852467120374, 1.0653453543681537]

y_true = df['output']

# Perform Prediction
y_pred = []
for i in range(len(pd.DataFrame(x))):
    variable_dict = {}
    nth_input = 0
    for term in term_set:
        variable_dict[term] = pd.DataFrame(x).iloc[i, nth_input]
        nth_input += 1
    prediction = EvaluateET(chromosome, variable_dict, const_list)
    y_pred.append(prediction)

y_pred = np.array(y_pred)

# get metric of predictions
print(f'Prediction Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred)}')
print(f'Prediction R-squared Score (R2): {r2_score(y_true, y_pred)}')

plt.plot(x, y_pred, label='Prediction', ls='--',color='k')
#plt.scatter(x, y_pred ,label='Prediction')
#plt.scatter(x, y_true, label='Ground Truth')
plt.plot(x,y_true,label='Ground Truth',color='r')
plt.title('Prediction vs Ground Truth')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Prediction vs Ground Truth.png')
plt.show()

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
df = pd.read_excel('../testing_datasets/gramacylee.xls')
y_true = df['output']
x = df[['input1','input2']]

# declare the chromosome and the variable dictionary
chromosome = ['/', 'a', '+', '(exp)', '(exp)', '*', '(X2)', 'b', 'b', 'a', 'a', 'b', '/', 'a', 'b', 'a', 'a', 'b', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'b', 'a', '4', '8', '4', '1', '5', '0', '13', '9', '13', '1', '11', '6', '4', '10']
term_set = ['a','b']
const_list = [9.085438328853577, 12.541568931269392, 14.228939030815598, 1.7243040260042173, 9.927060931656424, 13.910065740856533, 11.923836084853418, 12.466478431804356, 10.316798354700671, 11.526828984322051, 9.37978289606355, 3.3828761393003153, 13.466390125971392, 1.095020280549908]

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


# # plt.imshow(x,y_true)
# import plotly.graph_objects as go
# import numpy as np
#
# fig1 = go.Figure(data=[go.Scatter3d(x=x['input1'], y=x['input2'], z=y_true,
#                                    mode='markers')])
# fig2 = go.Figure(data=[go.Scatter3d(x=x['input1'], y=x['input2'], z=y_pred,
#                                    mode='markers')])
# fig3 = go.Figure(data=fig1.data + fig2.data)
# fig3.show()
#
# fig4 = go.Figure(data=[go.Scatter3d(x=x['input1'], y=x['input2'], z=abs(y_pred-y_true),
#                                    mode='markers')])
# fig4.show()



# #plt.scatter(x, y_pred ,label='Prediction')
# #plt.scatter(x, y_true, label='Ground Truth')
# plt.scatter(x['input1'],y_true, label='true')
# plt.scatter(x['input1'],y_pred, label='pred')
# #plt.imshow(x,y_true)
# plt.title('Prediction vs Ground Truth')
# plt.xlabel('a')
# plt.ylabel('b')
# plt.legend()
# #plt.savefig('Prediction vs Ground Truth.png')
# plt.show()
#
#
# plt.scatter(x['input2'],y_true, label='true')
# plt.scatter(x['input2'],y_pred, label='pred')
# #plt.imshow(x,y_true)
# plt.title('Prediction vs Ground Truth')
# plt.xlabel('a')
# plt.ylabel('b')
# plt.legend()
# #plt.savefig('Prediction vs Ground Truth.png')
# plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# from matplotlib import cm
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_trisurf(x['input1'], x['input2'], y_true, cmap=cm.jet, linewidth=0)
# fig.colorbar(surf)
# ax.xaxis.set_major_locator(MaxNLocator(5))
# ax.yaxis.set_major_locator(MaxNLocator(6))
# ax.zaxis.set_major_locator(MaxNLocator(5))
# fig.tight_layout()
# plt.show()

ax = plt.axes(projection='3d')
ax.plot_trisurf(x['input1'], x['input2'], y_true,cmap='seismic', edgecolor='None')
plt.title('Ground Truth')
plt.savefig('Ground Truth.png')
plt.show()
bx = plt.axes(projection='3d')
bx.plot_trisurf(x['input1'], x['input2'], y_pred,cmap='seismic', edgecolor='None')
plt.title('Prediction')
plt.savefig('Prediction.png')
plt.show()
cx = plt.axes(projection='3d')
cx.plot_trisurf(x['input1'], x['input2'], abs(y_pred-y_true),cmap='seismic', edgecolor='None')
plt.title('Mean Absolute Error')
plt.savefig('error.png')
plt.show()

plt.tricontourf(x['input1'],x['input2'],y_true, cmap='jet',levels=100)
plt.title('Ground Truth')
plt.colorbar()
plt.show()
plt.tricontourf(x['input1'],x['input2'],y_pred, cmap='jet',levels=100)
plt.title('Prediction')
plt.colorbar()
plt.show()
plt.tricontourf(x['input1'],x['input2'],abs(y_true-y_pred), cmap='jet',levels=100)
plt.title('Mean Absolute Error')
plt.colorbar()
plt.show()

plt.tricontour(x['input1'],x['input2'],y_true, cmap='jet',levels=100)
plt.colorbar()
plt.show()
plt.tricontour(x['input1'],x['input2'],y_pred, cmap='jet',levels=100)
plt.colorbar()
plt.show()
plt.tricontour(x['input1'],x['input2'],abs(y_true-y_pred), cmap='jet',levels=100)
plt.colorbar()
plt.show()
import csv
import numpy as np
csvName = "dataset/Unsheltered Adult VISPDAT Data.csv"

input_columns = [1, 3, 5, 6, 8, 9, 10, 11, 12, 15, 16, 18, 20, 21, 23, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 39, 40, 42, 43, 44, 45, 48, 49, 50, 52, 55, 56, 58, 72, 73, 74]
output_columns = [59]

input_data = []
output_data = []

with open(csvName, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == '': break
        if row[58] in ['unknown', 'pending', 'deceased', 'incarcerated']: continue

        input_data.append([])
        output_data.append([])
        for i, element in enumerate(row):
            if i in input_columns: input_data[-1].append(element)
            if i in output_columns: output_data[-1].append(element)

print('Headings:')
print(input_data.pop(0))
print(output_data.pop(0))
print('')
input_data = np.array(input_data)
output_data = np.array(output_data)

print('Data Shape:')
print(input_data.shape)
print(output_data.shape)
print('')

# Cleaning data
prego_column = input_data[:, 24]
prego_column[prego_column == ''] = 'n'

pet_column = input_data[:, 40]
pet_column[pet_column == ''] = 'n'

lgbtq_column = input_data[:, 39]
lgbtq_column[lgbtq_column == ''] = 'n'

print('First 5 Rows:')
print(input_data[:5])
print('')

# Making everything numerical
input_data[input_data == 'n'] = 0# no
input_data[input_data == 'N'] = 0# no
input_data[input_data == 'y'] = 1# yes
input_data[input_data == 'Y'] = 1# yes

input_data[input_data == 'm'] = 2# male
input_data[input_data == 't'] = 1# trans
input_data[input_data == 'f'] = 0# female

current_living = input_data[:, 1]
current_living[current_living == 'unsheltered'] = 1
current_living[current_living == 'car'] = 0

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer = ColumnTransformer([('onehot', OneHotEncoder(), [36, 37])], remainder = 'passthrough')
input_data = transformer.fit_transform(input_data).astype('float32')

print('First Row, numerical:')
print(input_data[0])
print('')

output_data[output_data == 'n'] = 0
output_data[output_data == 'y'] = 1
output_data = output_data.astype('float32')
print('First 5 Rows, output:')
print(output_data[:5])
print('')

# Making everything between 0 and 1
input_data = input_data/input_data.max(axis = 0)
print('Final input, shape = {}'.format(input_data.shape))
print(input_data[:5])
print('')
print('Final output, shape = {}'.format(output_data.shape))
print(output_data[:5])
print('')

np.save('dataset/inputs.npy', input_data)
np.save('dataset/outputs.npy', output_data)


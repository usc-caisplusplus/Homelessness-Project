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

headings = input_data.pop(0)
weights = np.load('logreg_coeff.npy')[0]

print(headings.pop(36) + ' and ' + headings.pop(36) + (' account for %.4f' % np.abs(weights[:20]).sum()))

weights = weights[20:]
for i in range(len(headings)):
    print(('%.4f     :   ' % weights[i]) + headings[i])

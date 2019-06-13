

import csv
import operator
import random

with open('another_data.csv', 'r') as file:
    reader = csv.DictReader(file)
    datas = [row for row in reader]


# print(datas)
random.shuffle(datas)
n = len(datas)//3

test_set = datas[0:n]
train_set = datas[n:]


def distance(d1, d2):
    sum = 0

    for key in('radius', 'texture', 'perimeter', 'area', 'smoothness',
               'compactness', 'symmetry', 'fractal_dimension'):
        sum += (float(d1[key])-float(d2[key]))**2

    return sum**0.5


K = 5


def knn(data):
    res = [
        {"result": train['diagnosis_result'], "distance": distance(data, train)}
        for train in train_set
    ]
    res = sorted(res, key=lambda item: item['distance'])
    print(res)

    class_count = {}
    for i in range(K):
        vote_label = res[i]["result"]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    print("sorted_class_count:", sorted_class_count)
    return sorted_class_count[0]


if __name__ == '__main__':
    test_class = knn(test_set[0])
    print(test_class[0])


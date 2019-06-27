import operator
import numpy as np
import random
from matplotlib import pyplot as plt

data_train = []
data_test = []


def creat_data():
    for i in range(1, 5):
        x = random.random() * 20
        y = random.random() * 20
        for j in range(10):
            x_shift = random.random() * 5
            y_shift = random.random() * 5
            data_train.append([x+x_shift, y+y_shift])
    for i in range(1, 5):
        x = random.random() * 20
        y = random.random() * 20
        data_test.append([x, y])


def prepare_data():
    creat_data()
    group = np.array(data_train)
    labels = []
    for i in range(4):
        for j in range(10):
            labels.append(chr(ord('A')+i))
    return group, labels


def knn(x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    extend_x = np.tile(x, (data_set_size, 1))
    pre_distance = (extend_x-data_set)**2
    se_distance = pre_distance.sum(axis=1)
    distances = se_distance**0.5
    sorted_dist = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


k = 5
group, labels = prepare_data()
result = knn(data_test[0], group, labels, k)
print(result)


point_dict = {
    0: 'o',
    1: 'v',
    2: 'H',
    3: 'D',
    4: '*'
}
color_dict = {
    0: 'r',
    1: 'g',
    2: 'b',
    3: 'c',
    4: 'k'
}
plt.figure(dpi=80)
ax = plt.subplot(1, 1, 1)
for i in range(4):
    for j in range(10):
        plt.sca(ax)
        plt.scatter(data_train[i*10+j][0], data_train[i*10+j][1], c=color_dict[i], marker=point_dict[i])
plt.sca(ax)
plt.scatter(data_test[0][0], data_test[0][1], c=color_dict[4], marker=point_dict[4])
plt.show()


import random

import pandas as pd

import os

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

import Utils
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')  # to suppress some matplotlib deprecation warnings

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

# Any results you write to the current directory are saved as output.

input_path = "/Users/blue/Machine Learning"
train_simplified_path = input_path + "/train_simplified/"

train_simplified = os.listdir(train_simplified_path)
number_of_processes = os.cpu_count()
print("Number of processes:", number_of_processes)


def get_lambda(value):
    return lambda x: value


def read_file(file_name, label, value, rows=100):
    data = pd.read_csv(file_name, index_col='key_id', nrows=rows)
    data['word'] = data['word'].replace(label, value, regex=True)
    return data


# print(train_simplified)
def get_small_sample(dir_listing, sample_size=5, rows=100, random_sample=False):
    sample_files = None
    if random_sample:
        sample_files = random.sample(dir_listing, sample_size)
    else:
        sample_files = dir_listing[:sample_size]

    print("Samples in Account :", sample_files)
    mapping = dict()
    reverse_mapping = dict()

    data_set = []
    for i, sample in enumerate(sample_files):
        ex_name_raw = sample.split('.')[0]
        ex_name = ex_name_raw.replace(' ', '_')
        mapping[ex_name] = i
        reverse_mapping[i] = ex_name
        data_set.append(read_file(train_simplified_path + sample, ex_name_raw, i, rows))
    return data_set, mapping, reverse_mapping


def split_data(list_data_frames):
    training_frames = []
    testing_frames = []
    for df in list_data_frames:
        train_df, test_df = train_test_split(df, test_size=0.1, shuffle=False)
        training_frames.append(train_df)
        testing_frames.append(test_df)
    return pd.concat(training_frames), pd.concat(testing_frames)


class KNN:
    def __init__(self, name="classfier", n=3):
        self.name = name
        self.neigh = KNeighborsClassifier(n_neighbors=n)

    def fit(self, draw, words):
        self.neigh = self.neigh.fit(draw, words)
        return 0

    def predict(self, draw):
        return self.neigh.predict(draw)


no_of_classes = 10
no_of_rows = 1000
scaling_factor = [30, 40, 50, 60, 70, 80, 100]

small_data_set, y_mapping, rev_y_mapping = get_small_sample(train_simplified, sample_size=no_of_classes,
                                                            rows=no_of_rows)
train_set, test_set = split_data(small_data_set)

name_to_number = lambda x: y_mapping[x]
number_to_name = lambda x: rev_y_mapping[x]

x_train, y_train = Utils.image_generator_xd(train_set, 64)
x_test, y_test = Utils.image_generator_xd(test_set, 64)
print("Data prepared")

knn = KNN(n=3)
print(x_train.shape)
print(y_train.shape)

knn.fit(x_train, y_train)
print("Trained KNN", knn)
y_pred_knn = knn.predict(x_test)
ccc = 0
for pred, act in zip(y_pred_knn, y_test):
    if pred == act:
        ccc += 1

acc_knn = accuracy_score(y_test, y_pred_knn)
print('KNN accuracy: ', acc_knn)
print('KNN accuracy: ', ccc / len(y_test))

print("---Starting Random Forest----")
# Base RFC model
rfc = RandomForestClassifier(random_state=1)
rfc.fit(x_train, y_train)
print(rfc)
y_pred_rfc = rfc.predict(x_test)
acc_rfc = accuracy_score(y_test, y_pred_rfc)
print('Random forest accuracy: ', acc_rfc)

print("---Finished Random Forest----")

print("SVM with linear kernal")
lsvc = LinearSVC(random_state=1)
lsvc.fit(x_train, y_train)
print(lsvc)
y_pred_lsvc = lsvc.predict(x_test)
acc_lsvc = accuracy_score(y_test, y_pred_lsvc)
print('Linear SVC accuracy: ', acc_lsvc)

print("SVM with RBF Kernal")
svc = SVC(kernel='rbf', random_state=1)
svc.fit(x_train, y_train)
print(svc)
y_pred_svc = svc.predict(x_test)
acc_svc = accuracy_score(y_test, y_pred_svc)
print('Gaussian Radial Basis Function SVC Accuracy: ', acc_svc)

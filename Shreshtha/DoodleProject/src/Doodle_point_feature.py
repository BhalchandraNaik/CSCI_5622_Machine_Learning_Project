# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import random
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import copy
import os

import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')  # to suppress some matplotlib deprecation warnings

import ast
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from multiprocessing import Pool

# Any results you write to the current directory are saved as output.

input_path = "/Users/blue/Machine Learning"
train_simplified_path = input_path + "/train_simplified/"
train_simplified = os.listdir(train_simplified_path)
number_of_processes = os.cpu_count()
print("Number of processes:", number_of_processes)


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


def fit_in_dimention(vector, dimention):
    l = len(vector)
    vect = np.zeros(dimention)
    if l < dimention:
        # padd it
        for i, v in enumerate(vector):
            vect[i] = v
    else:
        # scale it down
        # sum first k and place on the first place
        sum_of_window = 0
        index = 0
        k = np.ceil(l / dimention)
        remainder = l % dimention
        mod = k - 1
        for i, v in enumerate(vector):
            if i % k == mod:
                vect[index] = int(sum_of_window / k)
                sum_of_window = 0
                index += 1
                if (index == dimention):
                    break
            else:
                sum_of_window += v
    return vect


def flat(cord, arr):
    for p in arr:
        cord.append(p)


def preprocess(img):
    dimention = 100
    img_arr = ast.literal_eval(img)
    xc = []
    yc = []
    for stroke in img_arr:
        flat(xc, stroke[0])
        flat(yc, stroke[1])
    # flat_series = flatten(img)
    #         print("Flat series",flat_series)
    feature_vecter = fit_in_dimention(xc + yc, dimention)
    #         print("Feature", feature_vecter)
    return feature_vecter


def add_to_dict(_dict, elem):
    if elem in _dict:
        _dict[elem] += 1
    else:
        _dict[elem] = 1


class KNN:
    def __init__(self, name="classfier", n=3):
        self.name = name
        self.neigh = KNeighborsClassifier(n_neighbors=n)

    def fit(self, draw, words):
        self.neigh = self.neigh.fit(draw, words)
        return 0

    def predict(self, draw):
        return self.neigh.predict(draw)


def test_knn(train, test, clf, scaling_factor):
    words = train["word"]
    with Pool(number_of_processes) as pool:
        draw = pool.map(preprocess, train["drawing"])
        clf.fit(draw, words.tolist())

        ans = dict()
        actual = dict()
        words = test["word"]
        # draw = preprocess(test["drawing"], scaling_factor)
        draw_test = pool.map(preprocess, test["drawing"])

        for d, w in zip(draw_test, words):
            dd = np.array([d])
            prediction = clf.predict(dd)
            add_to_dict(actual, w)
            if w == prediction[0]:
                if w in ans:
                    ans[w] += 1
                else:
                    ans[w] = 1
            else:
                if w not in ans:
                    ans[w] = 0
        # print(ans)
        # print(actual)
        return sum(ans.values()) / sum(actual.values())
    # print("accuracy", )


no_of_classes = 10
no_of_rows = 1000
scaling_factor = [30, 40, 50, 60, 70, 80, 100]

small_data_set, y_mapping, rev_y_mapping = get_small_sample(train_simplified, sample_size=no_of_classes,
                                                            rows=no_of_rows)
train_set, test_set = split_data(small_data_set)


def prepare_data(raw_data):
    words = raw_data["word"]
    with Pool(number_of_processes) as pool:
        x_valid = pool.map(preprocess, raw_data["drawing"])
        y_valid = raw_data["word"].tolist()

        return x_valid, y_valid


x_train, y_train = prepare_data(train_set)
x_test, y_test = prepare_data(test_set)

knn = KNN(n=3)
print(len(x_train))
print(len(y_train))

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

print("---Grid search on random forest---")
parameters = {'n_estimators': np.arange(10, 150, 10)}
rfc = RandomForestClassifier(random_state=1, n_jobs=-1)
rfc_gs = GridSearchCV(rfc, parameters, n_jobs=-1)
rfc_gs.fit(x_train, y_train)

results_rfc_gs = pd.DataFrame(rfc_gs.cv_results_)
results_rfc_gs.sort_values('mean_test_score', ascending=False)
results_rfc_gs.plot('param_n_estimators', 'mean_test_score')

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

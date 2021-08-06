import numpy as np
import csv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from metric.correlation_coe import pearson_r
import time
import argparse
import sys

parser = argparse.ArgumentParser(description='KRR machine learning model')
parser.add_argument('--root1', type=str, help='The path where you define id_prop.csv file')
parser.add_argument('--root2', type=str, help='The path where you save npy file')
parser.add_argument('--random_state', type=int, help='the way to split your dataset', default=0)
args = parser.parse_args(sys.argv[1:])

with open(args.root1) as f:
    reader = csv.reader(f)
    magmom_data = [row for row in reader]

y_dict = {}
for i in range(len(magmom_data)):
    y_dict[magmom_data[i][0]] = float(magmom_data[i][1])

ofm_atom = np.load(args.root2, allow_pickle=True).item()
for idx, ids in enumerate(list(y_dict.keys())):
    fea_ofm = ofm_atom[ids]

    if idx == 0:
        X_ofm = fea_ofm
        y = y_dict[ids]
    else:
        X_ofm = np.vstack((X_ofm, fea_ofm))
        y = np.vstack((y, y_dict[ids]))

y = np.squeeze(y)
# -----------------------------Preprocessing----------------------------------------------

X_ofm_train, X_ofm_test, y_train, y_test = train_test_split(X_ofm, y, test_size=0.2, random_state=args.random_state)

X_ofm_train = StandardScaler().fit_transform(X_ofm_train)
X_ofm_test = StandardScaler().fit_transform(X_ofm_test)

# ------------------------- KRR -------------------------------------------

# I see alpha in KRR usually takes 1 to 1e-3

scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

# This is the first GS with low resolution

param_range1 = np.logspace(-10, 0, 11)
param_range2 = np.logspace(-10, 0, 11)


param_grid = {'kernel': ['rbf', 'laplacian'],
              'alpha': param_range1,
              'gamma': param_range2
              }


# This is the second GS with high resolution

# param_range1 = np.logspace(-4, -2, 11)
# param_range2 = np.logspace(-4, -2, 11)

"""
param_grid = {'kernel': ['rbf', 'laplacian'],
              'alpha': param_range1,
              'gamma': param_range2
              }
"""

gs = GridSearchCV(estimator=KernelRidge(),
                  param_grid=param_grid,
                  scoring=scoring,
                  refit='neg_mean_absolute_error',
                  cv=10,
                  n_jobs=1)

t0 = time.time()
gs = gs.fit(X_ofm_train, y_train)
krr_fit = time.time() - t0
print("Traing time is : ", krr_fit, 's')
print("--------- This is the best score(neg_mae) ---------")
print(gs.best_score_)
print("--------- This is the best parameter ---------")
print(gs.best_params_)

best_dict = {'best_score': gs.best_score_, 'best_params': gs.best_params_}
np.save("best_dict_1st_8_6_" + str(args.random_state) + "_.npy")
# np.save("best_dict_2nd_8_6_" + str(args.random_state) + "_.npy")

results = gs.cv_results_
np.save('cv_gs_1st_8_6_' + str(args.random_state) + '_.npy', results)
# np.save('cv_gs_2nd_8_6_' + str(args.random_state) + '_.npy', results)

clf = gs.best_estimator_

y_pred = clf.predict(X_ofm_test)

y_results_dict = {'y_pred': y_pred, 'y_true': y_test}
np.save('y_results_dict_1st_8_6_' + str(args.random_state) + '_.npy', y_results_dict)
# np.save('y_results_dict_1st_8_6_' + str(args.random_state) + '_.npy', y_results_dict)

mae_score = mean_absolute_error(y_test, y_pred)
mse_score = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)
pcc_score = pearson_r(y_test, y_pred)

print("This is the test mae score: ", mae_score)
print("This is the test mse score: ", mse_score)
print("This is the test r2 score: ", r2_score)
print("This is the test PCC score: ", pcc_score)

# clf.fit(X_train, y_train)
# note that we do not need to refit the classifier
# because this is done automatically via refit=True.

# print("-----------This is the test accuracy(r2?) -------------")
pint('Test accuracy: %.3f' % clf.score(X_ofm_test, y_test))

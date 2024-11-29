import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pydotplus
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix, roc_curve, auc
import warnings
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings('ignore')
import joblib
import os
import shutil
from glob import glob
from openpyxl import load_workbook
from utils import create_dir,mycopydir,mycopyfile,show_importance,shap_analysis
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Font set for Chinese characters
plt.rcParams['font.size'] = 12  # Font size
plt.rcParams['axes.unicode_minus'] = False

data_excel = 'predict.xlsx'
data_name = 'GDS'  
label_name = 'GDS'  # Name of the dataset label
sheet_name = f'{data_name}_data'
data = pd.read_excel(data_excel, sheet_name=sheet_name)
data = data.fillna(0)

feature_names = ['cC', 'cO', 'cN', 'cF', 'cP', 'cS', 'OCr', 'FCr', 'FOr', 'PCr', 'NCr',
                 'SCr', 'NFr', 'POr', 'NOr', 'SOr', 'PFr', 'SFr', 'SPr', 'NSr', 'FNS/Or',
                 'FP/Or', 'FNS/FPr']

data_features_part = data[feature_names]
epoch_num = 100
StrK = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)


n_splits=3

ls1 = ['fold' + str(i) for i in range(n_splits)]

fold_dict = dict(zip(ls1, [0 for i in range(len(ls1))]))



train =False

X = data_features_part
y = data[label_name]
from sklearn.preprocessing import KBinsDiscretizer



def discretize_labels_equal_bins(labels, n_classes):
    """
    Convert continuous labels into discrete labels with approximately equal numbers of instances in each class.

    Parameters:
    - labels: pandas Series, containing the original continuous labels.
    - n_classes: int, the number of classes to discretize into.

    Returns:
    - labels_discretized: numpy array, the discretized labels.
    """
    # Calculate quantiles to ensure a relatively even number of labels in each quantile range
    quantiles = [i / n_classes for i in range(n_classes + 1)]
    bin_edges = labels.quantile(quantiles).values
    # Create KBinsDiscretizer object, using 'quantile' strategy
    est = KBinsDiscretizer(n_bins=n_classes, encode='ordinal', strategy='quantile')
    print('Bin edges:', bin_edges.tolist())

    # Discretize the labels
    labels_discretized = est.fit_transform(labels.values.reshape(-1, 1))

    return labels_discretized.flatten()


# Try different numbers of classes (3, 5, 7)
for n_classes in [3, 5, 7]:
    # Discretize labels

    y_discretized = discretize_labels_equal_bins(y, n_classes)
    print(y_discretized)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_discretized, test_size=0.2, random_state=12)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)


    print(f'Number of Classes: {n_classes}, Accuracy: {accuracy:.4f}')
    y_label=pd.Series(y_discretized)
    print(y_label.value_counts())


#
# # Dataset table
if train == True:
    clf = RandomForestClassifier(criterion='gini', n_estimators=30, max_features='log2', max_depth=5,
                                 max_samples=None)
    for n_classes in [3, 5, 7]:
        # Create related folders
        main_path = f'{data_name}_{n_classes}_classification_predict'
        create_dir(main_path, True)
        for i in range(1000):
            if not os.path.exists(main_path + '/' + main_path + '1'):
                main_path = main_path + '/' + main_path + '1'
                create_dir(main_path, True)
                break
            else:
                if not os.path.exists(main_path + '/' + main_path + str(i)):
                    main_path = main_path + '/' + main_path + str(i)
                    create_dir(main_path, True)
                    break
                else:
                    continue

        create_dir(main_path + '/report')
        create_dir(main_path + '/model')
        create_dir(main_path + '/confusion_matrix')
        create_dir(main_path + '/Feature importance of different models')
        # Discretize labels
        print(f'Now performing {n_classes} classification')
        y_discretized = discretize_labels_equal_bins(y, n_classes)

        for i, (train_index, test_index) in enumerate(StrK.split(X, y_discretized)):
            # pass
            print(f'-------------Fold {i} cross-validation-----------')
            fold_dict['fold' + str(i)] = train_index
            f1_score_ls = []
            x_train = X.iloc[train_index]
            y_train = y_discretized[train_index]

            x_test = X.iloc[test_index]
            y_test = y_discretized[test_index]

            for n in range(epoch_num):
                f1_score_sum = 0
                # print(f'Training for the {n}th time')
                clf.fit(x_train, y_train)

                test_predict = clf.predict(x_test)
                f1_score = metrics.f1_score(y_test, test_predict, average='weighted')
                report = metrics.classification_report(y_test, test_predict)
                joblib.dump(clf, main_path + '/model(temp)/' + f'model{i}.pkl')
                # print('Saving model file')

                # Print the report
                file = open(main_path + f'/report(temp)/report{i}.txt', 'w')
                file.write(report)

                if len(f1_score_ls) == 0 or f1_score > max(f1_score_ls):
                    print(f'Fold {i}, Training for the {n}th time, f1_score mean', f1_score)
                    f1_score_ls.append(f1_score)
                    mycopyfile(main_path + f'/model(temp)/model{i}.pkl', main_path + '/model')
                    mycopyfile(main_path + f'/report(temp)/report{i}.txt', main_path + '/report')
                    file = open(main_path + f'/report/report{i}.txt', 'w')
                    print(f'Fold {i}, Training for the {n}th time, report\n', report)
                    file.write(report)


                if f1_score >= 0.95:
                    break

    # plt.cla()
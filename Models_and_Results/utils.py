Here is the code with all comments and Chinese content translated into English:

```python
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pydotplus
from sklearn import metrics
# import graphviz
import os
import warnings

warnings.filterwarnings('ignore')
import joblib
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix, roc_curve, auc
import shap
from openpyxl import load_workbook

# Plotting font settings
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'Times New Roman']  # Chinese font set
plt.rcParams['font.size'] = 12  # Font size
plt.rcParams['axes.unicode_minus'] = False

os.environ['PATH'] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'
test_ratio = 0.3


def mycopyfile(srcfile, dstpath):  # Copy function
    if not os.path.isfile(srcfile):
        print("%s does not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # Separate file name and path
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # Create path
        # print(dstpath+fname)
        shutil.copy(srcfile, dstpath + '/' + fname)  # Copy file
        # print ("copy %s -> %s"%(srcfile, dstpath + fname))


def mycopydir(srcdir, dstdir):
    for dirpath, dirnames, filenames in os.walk(srcdir):
        # print('dirpath=',dirpath)
        # print('dirnames=',dirnames)
        # print('file_names=',filenames)
        for file in filenames:
            path_file = os.path.join(dirpath, file)
            mycopyfile(path_file, dstdir)


def create_dir(save_path, is_mainpath=False):
    # For main_path, if it does not exist, create it
    if is_mainpath:
        if os.path.exists(save_path):
            print(f'Folder {save_path} already exists')
        else:
            print(f'Creating {save_path} folder')
            os.mkdir(save_path)
    # For non-main_path, create a backup folder for each main_path
    else:
        if os.path.exists(save_path):
            print(f'Folder {save_path} already exists')
        else:
            print(f'Creating {save_path} folder')
            os.mkdir(save_path)

        if os.path.exists(save_path + '(temp)'):
            print(f'Folder {save_path} already exists')
        else:
            print(f'Creating {save_path} folder')
            os.mkdir(save_path + '(temp)')


def create_empty_ls(num):
    if not isinstance(num, float) and not isinstance(num, int):
        print('Array length is not an integer')
        return
    else:
        ls = list(np.zeros(num))
        for i in range(num):
            ls[i] = []
        return ls


# def cross_validation_k_fold(dataset,fold_num):
#     """
#     dataset should be a list of datasets, fold_num is the number of folds
#     """
#     data_num = len(dataset)
#     data_ls = [i for i in range(data_num)]
#     fold_length = data_num // fold_num
#     ls = []  # ls is the length of each dataset division
#     for i in range(fold_num):
#         if i != fold_num - 1:
#             ls.append(fold_length)
#         else:
#             ls.append(len(dataset) - (fold_num - 1) * fold_length)
#     return ls


# Plot the model's confusion matrix
def draw_confusion():
    pass


def Regression(model_name, data_features_part, data_target, label_name='Predicted Variable', test_ratio=test_ratio, epoch_num=100,
               save_path='.',
               RF_module=RandomForestRegressor(criterion='squared_error', n_estimators=30, max_features='log2',
                                               max_samples=None)):
    '''
    data_features is feature information
    data_targets is target value information
    label_name is the input table file
    '''
    n_splits = 10
    StrK = StratifiedKFold(n_splits=n_splits, shuffle=True)
    data_features_part = data_features_part.fillna(0)
    data_target = data_target.fillna(0)
    pred_acc_ls = []
    # mae_ls=[]
    for epoch in range(epoch_num):

        x_train, x_test, y_train, y_test = train_test_split(data_features_part, data_target, test_size=test_ratio,
                                                            random_state=None)
        if model_name == 'rf':
            reg_mod = RandomForestRegressor(criterion='squared_error', n_estimators=30, max_features='log2',
                                            max_samples=None)
            reg_mod.fit(x_train, y_train)
        else:
            print('model should be ''rf''')

        test_predict = reg_mod.predict(x_test)
        mae = mean_absolute_error(test_predict, y_test)
        mse = mean_squared_error(test_predict, y_test)
        train_acc = reg_mod.score(x_train, y_train)
        test_acc = reg_mod.score(x_test, y_test)

        report = dict()
        if len(pred_acc_ls) == 0 or test_acc > max(pred_acc_ls):
            # if len(pred_acc_ls)==0 or (test_acc>=max(pred_acc_ls) and mae>=mae_ls[-1]):
            # print(len(pred_acc_ls),test_acc,max(pred_acc_ls))
            print(f'——————————{label_name} Prediction Status——————————')
            # report=metrics.classification_report(y_test,test_predict)
            print('accuracy of training data =', train_acc)
            print('accuracy of test data =', test_acc)
            sqrt_mse = np.sqrt(mse)
            print('mae_score', mae)
            print('mse_score', mse)
            print('sqrt_mse', sqrt_mse)
            # cross_val_score(test_pre)
            joblib.dump(reg_mod, save_path + '/' + f'{label_name}.pkl')
            file = open(save_path + f'/report{label_name}.txt', 'w')
            report = {'train_acc_{label_name}': train_acc, 'test_acc_{label_name}': test_acc, 'mae_score': mae,
                      'mse_score': mse}
            file.write(str(report))
        pred_acc_ls.append(test_acc)
        # mae_ls.append(mae)

    # Plot prediction graph
    reg_mod = joblib.load(save_path + '/' + f'{label_name}.pkl')
    predictions = reg_mod.predict(x_test)
    plt.figure(figsize=(10, 5), dpi=300)
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label='True_Values')
    plt.plot(x_ax, predictions, label='Predicted Values')
    plt.title('True Values Vs Predicted Values(RF_model)')
    plt.xlabel('Sample Number')
    plt.ylabel(f'{label_name}_predictions')
    plt.legend()
    plt.savefig(save_path + f'/{model_name}predictions_{label_name}.jpg')
    plt.show()
    print(f'{label_name} prediction completed')
    print(f'Model saved at {save_path}/{label_name}.pkl')


def show_importance(model, feature_names, label_name='Predicted Value', save_path='.', pic_index=0, save_model=False):
    feature_importances = model.feature_importances_
    feature_names = list(feature_names)
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    # Visualize feature importance
    fig, ax = plt.subplots(figsize=(10, 6))  # fig is the figure object, ax is the axis object
    ax.barh(feature_importances['feature'], feature_importances['importance'], color=colors)
    ax.invert_yaxis()  # Invert y-axis, making the largest feature at the top
    ax.set_xlabel('Feature Importance', fontsize=12)  # Graph's x-label
    ax.set_title(f'{label_name} Feature Importance Visualization', fontsize=16)

    print('feature_importance_name:\n', feature_importances['feature'])
    print('feature_importance_value:\n', feature_importances['importance'])

    for i, v in enumerate(feature_importances['importance']):
        ax.text(v + 0.005, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=10)
        # Set graph style
        ax.spines['top'].set_visible(False)  # Remove top border
        ax.spines['right'].set_visible(False)  # Remove right border
        ax.spines['left'].set_linewidth(0.5)  # Left border thickness
        ax.spines['bottom'].set_linewidth(0.5)  # Bottom border thickness
        ax.tick_params(width=0.5)
        ax.set_facecolor('white')  # Background color white
        ax.grid(False)  # Turn off internal grid lines
        # Save graph
    if save_model:
        plt.savefig(save_path + f'/{label_name}_feature_importance{pic_index}.jpg', dpi=1000, bbox_inches='tight')
        print(f'Feature image saved at {save_path}/{label_name}_feature_importance{pic_index}.jpg')
        plt.show()



def shap_analysis(model, x_test, label_name='Predicted Value', element_name='FNS/Or', save_path='.', save_model=False):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    # shap.initjs()
    shap_values = explainer.shap_values(x_test)

    shap.summary_plot(shap_values, x_test, plot_type='bar', show=False)
    plt.tight_layout()
    if save_model:
        plt.savefig(save_path + f'/{label_name}_summary_sample_bar_chart.jpg', dpi=1000, bbox_inches='tight')
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values, x_test, show=False)
    plt.tight_layout()
    if save_model:
        plt.savefig(save_path + f'/{label_name}_summary_sample_plot.jpg', dpi=1000, bbox_inches='tight')
    plt.show()

    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[0, :], x_test.iloc[0, :])
    if save_model:
        plt.savefig(save_path + '\\Local Interpretability.jpg', dpi=1000, bbox_inches='tight')
    plt.show()

    plt.figure()
    # print(len(x_test))
    element_name = 'FNS/Or'  # 'cC','cO','cN','cF','cP','cS','OCr','FCr','FOr','PCr','NCr','SCr','NFr','POr','NOr','SOr','PFr','SFr','SPr','NSr','FNS/Or','FP/Or','FNS/FPr'
    shap.dependence_plot(element_name, shap_values, x_test, interaction_index=None, show=False)
    if save_model:
        plt.savefig(save_path + '\\Local Interpretability.jpg', dpi=1000, bbox_inches='tight')
    plt.show()

    plt.figure()
    # ex = shap.Explanation(shap_values[0], explainer.expected_value, x_test.iloc[0])
    # shap.plots.waterfall(ex)
    explainer2 = shap.Explainer(model, x_test)
    shap_values = explainer2(x_test)
    shap.plots.waterfall(shap_values[0])
    plt.rcParams['axes.unicode_minus'] = False  # Default to using Unicode minus sign, set to display characters normally, such as displaying the minus sign correctly
    plt.show()
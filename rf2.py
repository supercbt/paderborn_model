from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import json
import joblib

class Result:
    precision = 0
    recall = 0
    accuracy = 0
    rocX = []
    rocY = []
    featureImportances = []
params = {}
# params['n_estimators'] = 90
# params['max_depth'] = 9 # 8
# params['max_features'] = 5
# params['min_samples_split'] = 60 # 6
# params['min_samples_leaf'] = 40 # 3
# params['traindata_path'] = ''
# params['testdata_path'] = ''

params['n_estimators'] = 20
params['max_depth'] = 8 # 8
params['max_features'] = 'auto'
params['min_samples_split'] = 6 # 6
params['min_samples_leaf'] = 3 # 3
params['traindata_path'] = ''
params['testdata_path'] = ''

argvs = sys.argv
try:
    for i in range(len(argvs)):
        if i < 1 or i == 3:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])

    data_csv = pd.read_csv(params['testdata_path'])
    data_csv['labels'] = 0
    data_csv.to_csv(params['testdata_path'], index=False, sep=',')

    train = np.array(pd.read_csv(params['traindata_path']))
    train_y = train[:, -1]
    train_x = train[:, :-1]

    test = np.array(pd.read_csv(params['testdata_path']))
    test_y = test[:, -1]
    test_x = test[:, :-1]

    clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                             max_features=params['max_features'],
                             max_depth=params['max_depth'],
                             min_samples_split=params['min_samples_split'],
                             min_samples_leaf=params['min_samples_leaf']).fit(train_x, train_y)

    joblib.dump(clf, 'rf2_model_1.model')

    predict = clf.predict(test_x)

    pr_title = predict

    wrtocsv = pd.DataFrame(predict)
    wrtocsv.to_csv('finallist_test_rf2_tetette111.csv', index=False, header=False)
    df = pd.read_csv('finallist_test_rf2_tetette111.csv',header=None,names=['label'])
    df.to_csv('finallist_test_rf2_tetette111.csv',index=False)

    # yt = [*map(round, test_y.tolist())]
    # xt = [*map(round, test_x.tolist())]
    # pt = [*map(round, predict.tolist())]


    precision = precision_score([*map(round, test_y)],[*map(round, predict)],average='macro')
    print(precision)
    recall = recall_score([*map(round, test_y)],[*map(round, predict)],average='macro')
    print(recall)
    accuracy = accuracy_score([*map(round, test_y)],[*map(round, predict)])
    print(accuracy)
    res = {}
    res['precision'] = precision
    res['recall'] = recall
    res['accuracy'] = accuracy
    res['fMeasure'] = f1_score(test_y, predict,average='macro')
    print(res['fMeasure'])
    # res['rocArea'] = '0.9999999999'
    # res['featureImportances'] = ['0.1','0.2','0.3']
    # print(res['featureImportances'])
    print(json.dumps(res))
except Exception as e:
    print(e)

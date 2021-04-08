import sklearn.tree as tree
from sklearn.feature_selection import RFECV
import pandas as pd
import sklearn.tree as tree
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_squared_log_error

data = pd.read_csv('train_X.csv')

outcome = np.asarray(pd.read_csv('train_y.csv')).flatten()

kf = KFold(n_splits=5)

data_arrary = np.array(data)
outcome_array = np.array(outcome)

rmsle = 0
best_rmsle = 0
i = 1
for train,test in kf.split(data,outcome):
    model = tree.DecisionTreeRegressor()
    rfe = RFECV(estimator=model)
    
    scaler = preprocessing.MinMaxScaler().fit(data_arrary[train])
    X_train_scaled = scaler.transform(data_arrary[train])
    X_test_scaled = scaler.transform(data_arrary[test])
    
    rfe.fit(X_train_scaled,outcome_array[train])
    y_pred = rfe.predict(X_test_scaled)

    rmsle_inner = np.sqrt(mean_squared_log_error(outcome_array[test], y_pred))
    print(str(i)+"th fold: CV RMSLE = "+str(rmsle_inner))

    rmsle += rmsle_inner

    if rmsle_inner > best_rmsle:
        best_rmsle = rmsle_inner
        selection_result = rfe.get_support()
    
pickle.dump(selection_result,open("DTRFE_CV.pkl","wb"))
print("average rmsle = "+str(rmsle/10))

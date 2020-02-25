from sklearn.ensemble import RandomForestClassifier
from transfer_model import TransferModel
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import KFold


rf = RandomForestClassifier(n_estimators=100,
                        criterion="gini",
                        max_depth=None,
                        min_samples_split=2, 
                        min_samples_leaf=1, 
                        min_weight_fraction_leaf=0., 
                        max_features="auto", 
                        max_leaf_nodes=None, 
                        min_impurity_decrease=0., 
                        min_impurity_split=None, 
                        bootstrap=True, 
                        oob_score=False, 
                        n_jobs=-1, 
                        random_state=None, 
                        verbose=0, 
                        warm_start=False, 
                        class_weight=None)

def cross_val(X_train, y_train, k, model):
    kf = KFold(n_splits=k, shuffle = True, random_state=0)
    accs = []
    prec = []
    recall = []
    for train, test in kf.split(X_train):
        X_tr, X_test = X_train[train], X_train[test]
        y_tr, y_test = y_train[train], y_train[test]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred,average = 'macro'))
        recall.append(recall_score(y_test, y_pred, average = 'macro'))
    return [np.mean(accs), np.mean(prec), np.mean(recall)]

if __name__ == "__main__":
    train_loc = 'data/Train'
    holdout_loc = 'data/Holdout'
    test_loc = 'data/Test'
    transfer = TransferModel('transfer', (100,100,3), 3, 10)
    transfer.fit(train_loc,test_loc,holdout_loc)
    
    train_labels, train_feats = transfer.train_labels.reshape(-1,1), transfer.train_features 
    test_labels, test_feats = transfer.test_labels.reshape(-1,1), transfer.test_features
    holdout_labels, holdout_feats = transfer.holdout_labels, transfer.holdout_features

    train_df = np.vstack([train_feats, test_feats])
    train_labels = np.vstack([train_labels, test_labels]).reshape(-1,)


    # rf.fit(train_df, train_labels)
    # scoring = [accuracy_score(), precision_score('macro'), recall_score('macro')]
    scores = cross_val(train_df, train_labels, 10, rf)
    print('Mean Accuracy: {0}'.format(scores[0]))
    print('Mean Precision: {0}'.format(scores[1]))
    print('Mean Recall: {0}'.format(scores[2]))
    acc = rf.score(holdout_feats, holdout_labels)
    print('Holdout Accuracy: {0}'.format(acc))



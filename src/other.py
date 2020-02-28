
import pickle
import pandas as pd 
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import KFold
from numpy.random import seed
import os
import itertools
from transfer_model import TransferModel
from eda import bar_chart


train_loc = 'data/Train'
holdout_loc = 'data/Holdout'
test_loc = 'data/Test'
transfer = TransferModel('transfer', (250,250,3), 3, 10)
transfer.fit(train_loc,test_loc,holdout_loc)

train_labels, train_feats = transfer.train_labels.reshape(-1,1), transfer.train_features 
test_labels, test_feats = transfer.test_labels, transfer.test_features
holdout_labels, holdout_feats = transfer.holdout_labels.reshape(-1,1), transfer.holdout_features
class_names = transfer.class_names

train_df = np.vstack([train_feats, holdout_feats])
train_labels = np.vstack([train_labels, holdout_labels]).reshape(-1,)

rf_model = pickle.load(open('models/randomforest.pkl', 'rb'))
gdbc_model = pickle.load(open('models/gradientboost.pkl', 'rb'))


#Random Forest
y_pred = rf_model.predict(test_feats)
pred_prob = rf_model.predict_proba(test_feats)
acc = accuracy_score(test_labels, y_pred)
p = precision_score(test_labels, y_pred, average='macro')
r = recall_score(test_labels, y_pred, average = 'macro')
f_score = f1_score(test_labels, y_pred, average = 'macro')
print('Holdout Accuracy: {0}'.format(acc))
print('Holdout Precision: {0}'.format(p))
print('Holdout Recall: {0}'.format(r))
print('Holdout F1 Score: {0}'.format(f_score))
cm = confusion_matrix(test_labels, y_pred)
print('Confusion Matrix: \n {}'.format(cm))
predictions = np.array(test_labels == y_pred)
misclass = np.where(predictions == False)[0]


#Gradient Boost
y_pred_gdbc = gdbc_model.predict(test_feats)
pred_prob_gbdc = rf_model.predict_proba(test_feats)
acc = accuracy_score(test_labels, y_pred_gdbc)
p = precision_score(test_labels, y_pred_gdbc, average='macro')
r = recall_score(test_labels, y_pred_gdbc, average = 'macro')
f_score = f1_score(test_labels, y_pred_gdbc, average = 'macro')
print('Holdout Gradient Boosting Accuracy: {0}'.format(acc))
print('Holdout Gradient Boosting Precision: {0}'.format(p))
print('Holdout Gradient Boosting Recall: {0}'.format(r))
print('Holdout Gradient Boosting F1 Score: {0}'.format(f_score))
cm_gdbc = confusion_matrix(test_labels, y_pred_gdbc)
print('Confusion Matrix: \n {}'.format(cm))


predictions_gdbc = np.array(test_labels == y_pred_gdbc)
misclass_gdbc = np.where(predictions == False)[0]

home = os.getcwd()
imgs = []
for i in class_names:
    os.chdir(os.path.abspath(test_loc + '/' + i))
    files = os.listdir()
    imgs.append(files)
    os.chdir(home)

images = np.array(list(itertools.chain.from_iterable(imgs)))[misclass]
wrong_class = test_labels[misclass]
wrong_pred = y_pred[misclass]
img_class = pd.DataFrame(np.vstack((images, wrong_class)).T, columns = ['image', 'pred_class', 'actual_class'])
plot_img = img_class.loc[[24,25,58],].reset_index(drop=True)
plot_probs = pred_prob[misclass][[24,25,58]]
labels = ['Degas', 'Picasso', 'Van Gogh']





fig, axs = plt.subplots(1,3)
for i, ax in enumerate(axs.flatten()):
    file = class_names[int(plot_img.loc[i,2])]
    path_img = os.path.join(home, test_loc, file, plot_img.loc[i,0])
    img = skimage.io.imread(path_img)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(labels[int(plot_img.loc[i,2])]) 
    
plt.savefig('img/misclassified_imgs.png', bbox_inches='tight')
plt.tight_layout()

fig, axs = plt.subplots(1,3)
for i, ax in enumerate(axs.flatten()):
    xtickLocations = np.arange(len(labels))
    data = plot_probs[i]
    ax.bar(xtickLocations, data , width = .5)
    ax.set_xticks(xtickLocations)
    ax.axis('off')
    for i, p in enumerate(data):
        ax.annotate(f'{p*100:0.1f}%', (p + 0.005, i))
plt.savefig('img/misclassified_probs.png', bbox_inches='tight')
plt.tight_layout()


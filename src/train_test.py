import boto3
import os 
import numpy as np
boto3_connection = boto3.resource('s3')

home = os.path.abspath('resized')

train = ['Edgar_Degas', 'Pablo_Picasso', 'Vincent_Van_Gogh']
test_dest = os.path.abspath('resized/Test')
train_dest = os.path.abspath('resized/Train')
holdout = os.path.abspath('resized/Holdout')

os.mkdir(test_dest)
os.mkdir(train_dest)
os.mkdir(holdout)

for i in train:

    os.mkdir(os.path.join(test_dest, i))
    os.mkdir(os.path.join(train_dest,i))
    os.mkdir(os.path.join(holdout, i))
    os.chdir(os.path.join(home, i))
    files = os.listdir()
    holdout_num = np.round(len(files)*.1)
    test_num = np.round(len(files)*.2)
    train_num = len(files) - (holdout_num + test_num)
    os.chdir(home)
    for idx in np.arange(len(files)):
        if idx < holdout_num:
            os.rename(os.path.join(home, i, files[idx]), os.path.join(holdout, i, files[idx]))
        elif idx < (holdout_num + test_num):
            os.rename(os.path.join(home, i, files[idx]), os.path.join(test_dest, i, files[idx]))
        else:
            os.rename(os.path.join(home, i, files[idx]), os.path.join(train_dest, i, files[idx]))

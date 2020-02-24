import os 
import numpy as np
from shutil import copytree, copy


home = os.path.abspath('data')

train = ['Edgar_Degas', 'Pablo_Picasso', 'Vincent_van_Gogh']
balanced = ['balanced_Edgar_Degas', 'balanced_Pablo_Picasso', 'balanced_Vincent_Van_Gogh']
test_dest = os.path.abspath('data/Test')
train_dest = os.path.abspath('data/Train')
holdout = os.path.abspath('data/Holdout')
try:
    os.mkdir(test_dest)
    os.mkdir(train_dest)
    os.mkdir(holdout)
except:
    pass



def class_balance(artists, home):
    '''
    Get the number of images for each artist change the number of images to the 
    artist with the least photos.
    '''
    os.chdir(home)
    num_pics = []
    for i in artists:
        os.chdir(os.path.abspath(i))
        files = os.listdir()
        num_pics.append(len(files))
        os.chdir(home)

    num = np.min(num_pics)
    for i in artists:
        try:
            os.mkdir(os.path.abspath('balanced_' + i))
        except:
            continue
        os.chdir(os.path.abspath(i))
        files = os.listdir()
        short_list = files[:num]
        for j in np.arange(len(short_list)):
            copy(os.path.join(os.path.abspath(short_list[j])), 
                        os.path.join(home, 'balanced_' + i, short_list[j]))
        os.chdir(home)

def train_test(artists, home):
    '''
    Split each artists into a train, test and holdout data
    '''

    for i in artists:
        try:
            os.mkdir(os.path.join(test_dest, i))
            os.mkdir(os.path.join(train_dest,i))
            os.mkdir(os.path.join(holdout, i))
        except:
            continue
        os.chdir(os.path.join(home, i))
        files = os.listdir()
        holdout_num = np.round(len(files)*.1)
        test_num = np.round(len(files)*.2)
        train_num = len(files) - (holdout_num + test_num)
        os.chdir(home)
        for idx in np.arange(len(files)):
            if idx < holdout_num:
                copy(os.path.join(home, i, files[idx]), os.path.join(holdout, i, files[idx]))
            elif idx < (holdout_num + test_num):
                copy(os.path.join(home, i, files[idx]), os.path.join(test_dest, i, files[idx]))
            else:
                copy(os.path.join(home, i, files[idx]), os.path.join(train_dest, i, files[idx]))




if __name__ == "__main__":
    class_balance(train, home)
    train_test(balanced, home)
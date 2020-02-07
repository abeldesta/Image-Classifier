import boto3
import os 
import pandas as pd 
import numpy as np 


artists = pd.read_csv('data/artists.csv')

home = os.path.abspath('data')


# os.chdir(home)
# files = os.listdir()
# artists = [x.replace(' ', '_') for x in artists['name']]
# for artist in artists:
#     art = [x for x in files if x.startswith(artist)]
#     os.mkdir(os.path.join(home, artist))
#     for i in np.arange(len(art)):
#         os.rename(os.path.join(home,art[i]), os.path.join(home, artist, art[i]))
#     os.chdir(home)

# artist = 'Albrecht_Du╠êrer'
# art = [x for x in files if x.startswith(artist)]
# os.mkdir(os.path.join(home, artist))
# for i in np.arange(len(art)):
#     os.rename(os.path.join(home,art[i]), os.path.join(home, artist, art[i]))
# os.chdir(home)



# artist = 'Albrecht_Dürer'
# art = [x for x in files if x.startswith(artist)]
# for i in np.arange(len(art)):
#     os.rename(os.path.join(home,art[i]), os.path.join(home, artist, art[i]))
# os.chdir(home)

num = 439
classes = ['Edgar_Degas', 'Pablo_Picasso', 'Vincent_Van_Gogh']
for i in classes:
    os.mkdir(os.path.abspath('new_' + i))
    os.chdir(os.path.abspath(i))
    files = os.listdir()
    short_list = files[:439]
    for j in np.arange(len(short_list)):
        os.rename(os.path.join(os.path.abspath(short_list[j])), os.path.join(home, 'new_' + i, short_list[j]))
    os.chdir(home)
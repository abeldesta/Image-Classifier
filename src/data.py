import os 
import pandas as pd 
import numpy as np 


artists = pd.read_csv('data/artists.csv')

home = os.path.abspath('data')


os.chdir(home)
files = os.listdir()
artists = [x.replace(' ', '_') for x in artists['name']]
for artist in artists:
    art = [x for x in files if x.startswith(artist)]
    try:
        os.mkdir(os.path.join(home, artist))
        for i in np.arange(len(art)):
            os.rename(os.path.join(home,art[i]), os.path.join(home, artist, art[i]))
        os.chdir(home)
    except:
        continue

artist = 'Albrecht_Du╠êrer'
art = [x for x in files if x.startswith(artist)]
try:
        
    os.mkdir(os.path.join(home, artist))
    for i in np.arange(len(art)):
        os.rename(os.path.join(home,art[i]), os.path.join(home, artist, art[i]))
    os.chdir(home)
except:
    pass

try:
    artist = 'Albrecht_Dürer'
    art = [x for x in files if x.startswith(artist)]
    for i in np.arange(len(art)):
        os.rename(os.path.join(home,art[i]), os.path.join(home, artist, art[i]))
    os.chdir(home)
except:
    pass


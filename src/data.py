import boto3
import os 
import pandas as pd 
import numpy as np 


artists = pd.read_csv('data/artists.csv')

home = '/Users/abeldesta531/dsi/repos/Capstone-2/data'


os.chdir(home)
files = os.listdir()
artists = [x.replace(' ', '_') for x in artists['name']]
for artist in artists:
    art = [x for x in files if x.startswith(artist)]
    os.mkdir(home + '/' + artist)
    for i in np.arange(len(art)):
        os.rename(home +'/'+ art[i], '{0}/{1}/{2}'.format(home,artist, art[i]))
    os.chdir(home)





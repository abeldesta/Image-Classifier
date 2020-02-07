import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

artists = pd.read_csv('data/artists.csv')

if __name__ == "__main__":
    print(artists.head())
    paint = artists.sort_values('paintings')[::-1]
    labels = paint.name
    xtickLocations = np.arange(len(labels))
    counts = paint.paintings 
    width = 0.3

    fig, ax = plt.subplots(1,1, figsize = (9, 6))
    ax.bar(xtickLocations, counts, width)
    ax.set_xticks(xtickLocations)
    ax.set_xticklabels(labels)
    ax.set_title('Number of Paintings Per Artist')
    ax.set_xlabel('Artist')
    ax.set_ylabel('Image Count')
    plt.xticks(rotation= 85)
    plt.tight_layout()
    plt.savefig('img/paintings.png')


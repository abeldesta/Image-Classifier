import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

artists = pd.read_csv('data/artists.csv')

def bar_chart(labels, data, fig_name, xlabel = None, ylabel = None, title = None):
    fig, ax = plt.subplots(1,1, figsize = (9, 6))
    xtickLocations = np.arange(len(labels))
    ax.bar(xtickLocations, data, width = .5)
    ax.set_xticks(xtickLocations)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation= 85)
    plt.tight_layout()
    plt.savefig('img/{0}'.format(fig_name))

if __name__ == "__main__":
    print(artists.head())
    paint = artists.sort_values('paintings')[::-1]
    labels = paint.name
    xtickLocations = np.arange(len(labels))
    counts = paint.paintings 
    width = 0.3

    bar_chart(labels, counts, 'paintings', 'Artist', 'Image Count', 
                    'Number of Paintings Per Artist')

    nation = artists.nationality.value_counts()
    labels = nation.index 
    data = nation.values 
    bar_chart(labels, data, 'nationality', 'Nationality', 'Number of Artist', 'Frequency of Nationality')

    # fig, ax = plt.subplots(1,1, figsize = (9, 6))
    # ax.bar(xtickLocations, counts, width)
    # ax.set_xticks(xtickLocations)
    # ax.set_xticklabels(labels)
    # ax.set_title('Number of Paintings Per Artist')
    # ax.set_xlabel('Artist')
    # ax.set_ylabel('Image Count')
    # plt.xticks(rotation= 85)
    # plt.tight_layout()
    # plt.savefig('img/paintings.png')



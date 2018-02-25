from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def visualize_features(features, labels, model_name):
    color_map = {0: 'black', 1: 'gray', 2: 'silver', 3: 'rosybrown', 4: 'red', 5: 'sienna',
            6: 'gold',7: 'olivedrab', 8: 'darkgreen', 9: 'blue'}
    colors = np.array([color_map[l] for l in labels])
    tsne_features = TSNE(n_components=2).fit_transform(features)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=colors)
    color_legend = [mpatches.Patch(color=color_map[l], label=l) for l in color_map]
    plt.legend(handles=color_legend)
    plt.show()
    plt.savefig('{}_tsne.pdf'.format(model_name))

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y

plt.figure(figsize=(8, 6))
for label in pca_df['Target'].unique():
    plt.scatter(pca_df[pca_df['Target'] == label]['PC1'],
                pca_df[pca_df['Target'] == label]['PC2'],
                label=iris.target_names[label])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Iris Dataset (2D Visualization)')
plt.legend()
plt.grid(True)
plt.show()

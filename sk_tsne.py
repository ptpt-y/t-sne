# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=6) # 只选取数字0~5进行可视化
X, y = digits.data, digits.target
n_samples, n_features = X.shape

'''显示原始数据'''
n = 30  # 每行30个数字，每列30个数字
img = np.zeros((10 * n, 10 * n))
for i in range(n):
    ix = 10 * i + 1
    for j in range(n):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()

'''t-SNE'''
print('Computing t-SNE embedding')
tsne = manifold.TSNE(n_components=2, init='pca', verbose=1, random_state=501)
t0 = time()
X_tsne = tsne.fit_transform(X)
print("Org data dimension is {}. Embedded data dimension is {}"\
        .format(X.shape[-1], X_tsne.shape[-1]))

'''可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 坐标归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
title = 't-SNE embedding of the digits (time %.2fs)' % (time() - t0)
plt.title(title)
plt.show()


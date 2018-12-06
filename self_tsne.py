@PengTingyu_201692392
from time import time
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def neg_squared_euc_dists(X):
    '''
    # 参数:
        X: [N,D]的矩阵
    # 返回值:
        -D：[N,N]的矩阵, -D[i,j]表示X_i & X_j 之间的欧式距离的平方的负数
    '''
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return -D


def softmax(X, diag_zero=True):
    '''
    softmax 计算 exp()/∑exp()
    '''
    # 归一化
    e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
    # 设置对角线值为0
    if diag_zero:
        np.fill_diagonal(e_x, 0.)
    e_x = e_x + 1e-8  # 数值稳定
    return e_x / e_x.sum(axis=1).reshape([-1, 1])


def calc_prob_matrix(distances, sigmas=None):
    '''
    将距离矩阵转化为概率矩阵
    # 参数：
        distances：距离矩阵
        sigmas:高斯分布的方差
    '''
    if sigmas is not None:
        two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
        return softmax(distances / two_sig_sq)
    else:
        return softmax(distances)


def p_conditional_to_joint(P):
    '''
    将高维中的距离变为对称的
    # 参数：
        P: 条件概率矩阵
    '''
    return (P + P.T) / (2. * P.shape[0])


def p_joint(X, target_perplexity):
    '''
    根据给的原数据X和困惑度，计算最后高维中的联合概率分布矩阵    
    # 参数        
        X: 原数据矩阵    
    # 返回值:        
        P: 高维中的联合概率分布矩阵.
    '''
    distances = neg_squared_euc_dists(X)
    # 找到最优的高斯分布方差sigma
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    p_conditional = calc_prob_matrix(distances, sigmas)
    P = p_conditional_to_joint(p_conditional)
    return P


def binary_search(eval_fn, target, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.):
    '''
    二分查找
    # 参数：
        eval_fn: 优化的函数
        target: 目标输出值
        tol: Float, guess < tol时停止搜索
        max_iter: int, 最大迭代次数
        lower: Float, 搜索下界
        upper: Float, 搜索上界
    # Returns:
        Float, 优化函数的最优输入值
    '''
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess


def calc_perplexity(prob_matrix):
    '''
    计算困惑度
    # 参数：
        prob_matrix: 困惑度分布
    '''
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity


def perplexity(distances, sigmas):
    '''
    根据距离矩阵和高斯方差计算困惑度
    '''
    return calc_perplexity(calc_prob_matrix(distances, sigmas))


def find_optimal_sigmas(distances, target_perplexity):
    '''
    给距离矩阵的每一行找一个能产生最优困惑度分布的sigma[].
    '''
    sigmas = [] 
    # 距离矩阵的每一行代表一个数据点
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma:perplexity(distances[i:i+1, :], np.array(sigma))
        # 执行二分查找
        correct_sigma = binary_search(eval_fn, target_perplexity)
        sigmas.append(correct_sigma)
    return np.array(sigmas)


def estimate_tsne(X, y, P, rng, num_iters, q_fn, grad_fn, learning_rate, momentum):
    '''
    模拟函数
    # 参数：
        X: 原数据矩阵
        y: labels
        P: 高维中的联合分布矩阵
        rng: np.random.RandomState().
        num_iters: 迭代次数
        q_fn: 获得Y和低维空间中的联合概率分布矩阵Q
    # 返回值:
        Y: X在低维空间中的分布.
    '''
    # Y随机初始化
    Y = rng.normal(0., 0.0001, [X.shape[0], 2])

    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    # 梯度下降
    for i in range(num_iters):
        Q, distances = q_fn(Y)
        grads = grad_fn(P, Q, Y, distances)
        # 更新Y
        Y = Y - learning_rate * grads
        if momentum:  
            Y += momentum * (Y_m1 - Y_m2)
            # 更新
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

    return Y


def q_tsne(Y):
    '''
    根据t-SNE低维空间的分布Y，计算低维空间的联合概率分布矩阵
    '''
    distances = neg_squared_euc_dists(Y)
    inv_distances = np.power(1. - distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances / np.sum(inv_distances), inv_distances


def tsne_grad(P, Q, Y, inv_distances):
    '''
    t-SNE的梯度
    '''
    pq_diff = P - Q
    pq_expanded = np.expand_dims(pq_diff, 2)
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

    distances_expanded = np.expand_dims(inv_distances, 2)
    y_diffs_wt = y_diffs * distances_expanded
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
    return grad


def plot_embedding(data, label, title):
    '''
    可视化嵌入空间
    '''
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


# 参数
PERPLEXITY = 20 
SEED = 1  # 随机种子
MOMENTUM = 0.9 
LEARNING_RATE = 10. 
NUM_ITERS = 500  

def main():
    rng = np.random.RandomState(SEED)
    # 加载数据  
    digits = datasets.load_digits(n_class=6) # 只选取数字0~5进行可视化
    X, y = digits.data, digits.target
    P = p_joint(X, PERPLEXITY)

    # t-SNE
    t0 = time()
    Y = estimate_tsne(X, y, P, rng, num_iters=NUM_ITERS, q_fn=q_tsne, grad_fn=tsne_grad,\
            learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    # 可视化
    fig = plot_embedding(Y, y,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)

if __name__ == '__main__':
    main()
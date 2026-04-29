import numpy as np


def cov1para(x, shrink=-1):
    """
    参数
    x：类数组，形状（t，n）
    n个变量的t个观测值。每一列都是一个资产/变量。
    收缩：浮动，默认值-1
    如果省略或设置为-1，则计算最佳收缩强度。
    否则，直接使用此固定收缩强度。
    返回
    σ：ndarray，形状（n，n）
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (t, n)")

    t, n = x.shape
    if t <= 0 or n <= 0:
        raise ValueError("x must have positive dimensions")

    # 每列去均值
    meanx = np.mean(x, axis=0)
    x = x - meanx

    # 样本标准差
    sample = (x.T @ x) / t

    # 收缩目标：方差相等，协方差为零
    meanvar = np.mean(np.diag(sample))
    prior = meanvar * np.eye(n)

    if shrink == -1:
        y = x ** 2
        phi_mat = (y.T @ y) / t - sample ** 2
        phi = np.sum(phi_mat)

        gamma = np.linalg.norm(sample - prior, ord="fro") ** 2
        kappa = phi / gamma
        shrinkage = max(0.0, min(1.0, kappa / t))
    else:
        shrinkage = float(shrink)

    sigma = shrinkage * prior + (1.0 - shrinkage) * sample
    return sigma, shrinkage



def my_stationary_bootstrap(data, B, w, rng=None):
    """
    使用 Politis-Romano 的 stationary bootstrap 方法对平稳相关序列进行重采样。

    参数
    data : (t, k) 输入时间序列。k 允许 1 或 2；
        但该函数会返回两个重采样矩阵，因此实际使用建议传入两列数据
    B : int，bootstrap 次数。
    w : int，平均区块长度，新的区块起点概率为 p=1/w。
    rng : numpy.random.Generator, 可选
        随机数生成器；不传则使用 numpy 默认随机数生成器。

    返回
    bsdata1 : ndarray, 形状 (t, B) 第一列序列的 bootstrap 重采样结果
    bsdata2 : ndarray, 形状 (t, B) 第二列序列的 bootstrap 重采样结果
    indices : ndarray, 形状 (t, B) 1-based 索引矩阵
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("DATA must be a 2D array")

    t, k = x.shape
    if k > 2:
        raise ValueError("DATA must be two column vectors")
    if t < 2:
        raise ValueError("DATA must have at least 2 observations.")
    if not np.isscalar(w) or w < 1 or int(w) != w:
        raise ValueError("W must be a positive scalar integer")
    if not np.isscalar(B) or B < 1 or int(B) != B:
        raise ValueError("B must be a positive scalar integer")

    # 将参数转为整数，以防传入浮点数但语义上是整数的值
    w = int(w)
    B = int(B)

    if rng is None:
        rng = np.random.default_rng()

    # 新区块起点概率 p=1/w。
    p = 1.0 / w

    # 采用 1-based 索引
    indices = np.zeros((t, B), dtype=np.int64)

    # 第一行随机起点：ceil(t*rand)。
    indices[0, :] = np.ceil(t * rng.random(B)).astype(np.int64)

    # select=True 表示该位置新开区块，随机重置起点。
    select = rng.random((t, B)) < p
    num_select = int(np.sum(select))
    if num_select > 0:
        indices[select] = np.ceil(rng.random(num_select) * t).astype(np.int64)

    # 非新区块位置沿用上一期索引并 +1。
    for i in range(1, t):
        stay_mask = ~select[i, :]
        indices[i, stay_mask] = indices[i - 1, stay_mask] + 1

    # 通过拼接两倍长度实现“环绕”
    data1 = np.concatenate([x[:, 0], x[:, 0]], axis=0)

    # 由于原 Matlab 代码会访问第二列，这里在单列输入时显式报错，避免静默偏差。
    if k < 2:
        raise ValueError("DATA must contain 2 columns for this implementation")
    data2 = np.concatenate([x[:, 1], x[:, 1]], axis=0)

    # Matlab 索引从 1 开始；Python 取值时需转换为 0-based。
    idx0 = indices - 1
    bsdata1 = data1[idx0]
    bsdata2 = data2[idx0]

    return bsdata1, bsdata2, indices


def fivefold_cv(T):
    """
    生成五折交叉验证的样本索引
    参数
    T : 总样本长度。

    返回
    five_samples 长度为 5 的列表，每个元素是 1-based 索引数组。
        前四折长度均为 floor(T/5)，最后一折包含剩余样本。
    """
    if not np.isscalar(T) or T < 5 or int(T) != T:
        raise ValueError("T must be a scalar integer and T >= 5")

    T = int(T)
    Q = T // 5

    # 使用 1-based 索引
    fold1 = np.arange(1, Q + 1, dtype=np.int64)
    fold2 = np.arange(Q + 1, 2 * Q + 1, dtype=np.int64)
    fold3 = np.arange(2 * Q + 1, 3 * Q + 1, dtype=np.int64)
    fold4 = np.arange(3 * Q + 1, 4 * Q + 1, dtype=np.int64)
    fold5 = np.arange(4 * Q + 1, T + 1, dtype=np.int64)

    return [fold1, fold2, fold3, fold4, fold5]


def optimal_shrinkage(Data, gam, lambda_, kappa, B=1000, rng=None):
    """
    使用 bootstrap + 五折交叉验证估计最优收缩强度。
    Data : (T, N) 收益率矩阵
    gam : 风险厌恶系数。
    lambda_ : 稳健性惩罚系数
    kappa : 组合权重的收缩参数。
    B : 默认 1000 bootstrap 次数。
    rng : numpy.random.Generator, 可选

    返回
    Krobust : 最大化稳健目标 mean(U)-lambda*std(U) 对应的收缩强度。
    Kmean : 最大化 mean(U) 对应的收缩强度。
    """
    x = np.asarray(Data, dtype=float)
    if x.ndim != 2:
        raise ValueError("Data must be a 2D array with shape (T, N)")

    T, N = x.shape
    if T < 5:
        raise ValueError("Data must contain at least 5 observations")
    if not np.isscalar(B) or B < 1 or int(B) != B:
        raise ValueError("B must be a positive scalar integer")

    B = int(B)
    if rng is None:
        rng = np.random.default_rng()

    # 候选网格为 0:0.05:1。
    shv = np.arange(0.0, 1.0000001, 0.05)
    e = np.ones((N, 1))
    I = np.eye(N)

    # 五折划分使用 1-based 索引，后续取样时再转为 0-based。
    five_samples = fivefold_cv(T)

    # 先生成 1-based 再转 0-based
    bootstrap_1b = np.ceil(rng.random((T, B)) * T).astype(np.int64)
    bootstrap_0b = bootstrap_1b - 1

    CER = np.zeros((len(shv), B), dtype=float)

    for k_idx, sh in enumerate(shv):
        for b in range(B):
            rets_list = []
            data_b = x[bootstrap_0b[:, b], :]

            for s in range(5):
                sample_out_1b = five_samples[s]
                sample_out_0b = sample_out_1b - 1

                sample_in_0b = np.setdiff1d(np.arange(T, dtype=np.int64), sample_out_0b)

                data_est = data_b[sample_in_0b, :]
                data_oos = data_b[sample_out_0b, :]

                # 使用 1/T 归一化（bias=True）。
                S = np.cov(data_est, rowvar=False, bias=True)
                Sigma = sh * np.mean(np.diag(S)) * I + (1.0 - sh) * S
                mu = np.mean(data_est, axis=0, keepdims=True).T

                # Sigma\\I 等价于 inv(Sigma)，数值上更稳健地使用 solve。
                Sigmainv = np.linalg.solve(Sigma, I)
                wg = (Sigmainv @ e) / (e.T @ (Sigmainv @ e))
                Bm = Sigmainv @ (I - e @ wg.T)
                wz = Bm @ mu
                wmv = wg + (kappa / gam) * wz

                rets_list.append(data_oos @ wmv)

            rets = np.vstack(rets_list).reshape(-1)
            # 默认是样本方差（n-1）。
            CER[k_idx, b] = np.mean(rets) - (gam / 2.0) * np.var(rets, ddof=1)


    robust_measure = np.mean(CER, axis=1) - lambda_ * np.std(CER, axis=1, ddof=1)
    oos_u_mean = np.mean(CER, axis=1)

    # 若并列最大，取第一个。
    k_robust_idx = int(np.argmax(robust_measure))
    k_mean_idx = int(np.argmax(oos_u_mean))

    Krobust = float(shv[k_robust_idx])
    Kmean = float(shv[k_mean_idx])
    return Krobust, Kmean


def optimalShrinkage(Data, gam, lambda_, kappa, B=1000, rng=None):
    return optimal_shrinkage(Data, gam, lambda_, kappa, B=B, rng=rng)


def _ecdf_values(x):
    """
    计算经验分布函数离散点。

    F : 每个唯一取值处的累计概率
    xs : 排序后的唯一取值
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    # 忽略 NaN
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([np.nan]), np.array([np.nan])

    xs, counts = np.unique(np.sort(x), return_counts=True)
    F = np.cumsum(counts) / x.size
    return F, xs


def dgnu_statistics_politis_romano(xr, gam, B=1000, w=5, rng=None):
    """
    使用 stationary bootstrap 计算 CER 与 Sharpe ratio 比较的双侧 p 值。
    参数
    xr : (T, 2)，两个策略的收益序列。
    gam : 风险厌恶系数。
    B : 默认 1000，bootstrap 次数。
    w : 默认 5，平均区块长度。
    rng : numpy.random.Generator, 随机数生成器。

    返回
    pvalCER : CER 差异的双侧 p 值。
    pvalSR : Sharpe ratio 差异的双侧 p 值。
    """
    x = np.asarray(xr, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("xr must be a 2D array with shape (T, 2)")

    bsdata1, bsdata2, _ = my_stationary_bootstrap(x, B=B, w=w, rng=rng)

    # 按列（跨时间）计算，方差 ddof=1。
    CER1 = np.mean(bsdata1, axis=0) - (gam / 2.0) * np.var(bsdata1, axis=0, ddof=1)
    CER1[np.isnan(CER1)] = 0.0
    CER2 = np.mean(bsdata2, axis=0) - (gam / 2.0) * np.var(bsdata2, axis=0, ddof=1)
    CER2[np.isnan(CER2)] = 0.0

    delta_cer = CER1 - CER2
    F, xs = _ecdf_values(delta_cer)
    j = int(np.nanargmin(np.abs(xs)))
    if np.median(delta_cer) < 0:
        pvalCER = 2.0 * (1.0 - F[j])
    else:
        pvalCER = 2.0 * F[j]

    # Sharpe ratio：mean/std，std 使用 ddof=1。
    delta_sr = (
        np.mean(bsdata1, axis=0) / np.std(bsdata1, axis=0, ddof=1)
        - np.mean(bsdata2, axis=0) / np.std(bsdata2, axis=0, ddof=1)
    )
    F, xs = _ecdf_values(delta_sr)
    j = int(np.nanargmin(np.abs(xs)))
    if np.median(delta_sr) < 0:
        pvalSR = 2.0 * (1.0 - F[j])
    else:
        pvalSR = 2.0 * F[j]

    # 双侧 p 值理论上不超过 1，这里做截断提高数值稳健性。
    pvalCER = float(np.clip(pvalCER, 0.0, 1.0))
    pvalSR = float(np.clip(pvalSR, 0.0, 1.0))
    return pvalCER, pvalSR


def dgnu_StatisticsPolitisRomano(xr, gam, B=1000, w=5, rng=None):
    return dgnu_statistics_politis_romano(xr, gam, B=B, w=w, rng=rng)

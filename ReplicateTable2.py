from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
from scipy.special import beta, betainc

from Functions import cov1para, optimalShrinkage, dgnu_StatisticsPolitisRomano


# ============================================================
# Table 2 复现脚本
# 默认参数：lambda=2, gam=3, T=120, cbps=20
#
# 说明：该脚本计算量非常大。可选环境变量：
# TABLE2_MAX_WINDOWS: 每个数据集只跑前多少个滚动窗口（默认全部）
# TABLE2_OPT_B: 覆盖 optimalShrinkage 中 bootstrap 次数（默认 1000）
# TABLE2_PVAL_B: 覆盖 p-value 计算中的 bootstrap 次数（默认 1000）
# TABLE2_SEED: 随机种子（主要影响 numpy 全局随机）
# TABLE2_ONLY_FIRST_DATASET: 默认 1，仅计算第一个数据集（10MOM）
# TABLE2_WORKERS: 并行进程数，默认 cpu_count()-1
# ============================================================

base_dir = Path(__file__).resolve().parent
project_dir = base_dir.parent

lambda_ = 2.0
gam = 3.0
T = 120
cbps = 20.0

seed = int(os.getenv("TABLE2_SEED", "20260424"))
max_windows_env = os.getenv("TABLE2_MAX_WINDOWS", "")
opt_b_env = os.getenv("TABLE2_OPT_B", "")
pval_b_env = os.getenv("TABLE2_PVAL_B", "")
only_first_env = os.getenv("TABLE2_ONLY_FIRST_DATASET", "1")
workers_env = os.getenv("TABLE2_WORKERS", "")

max_windows = int(max_windows_env) if max_windows_env.strip() else None
opt_B = int(opt_b_env) if opt_b_env.strip() else None
pval_B = int(pval_b_env) if pval_b_env.strip() else None
only_first_dataset = only_first_env.strip() != "0"

if workers_env.strip():
    num_workers = max(1, int(workers_env))
else:
    num_workers = max(1, (cpu_count() or 2) - 1)

np.random.seed(seed)


def psi2_adjustment(raw_psi2, T, N):
    a = (N - 1) / 2
    b = (T - N + 1) / 2
    frac = raw_psi2 / (1 + raw_psi2)
    num = 2 * (raw_psi2 ** ((N - 1) / 2)) * ((1 + raw_psi2) ** (-(T - 2) / 2))
    den = T * beta(a, b) * betainc(a, b, frac)
    return ((T - N - 1) * raw_psi2 - (N - 1)) / T + num / den


def transaction_return(gross_ret, turnover, cbps):
    return (1 + gross_ret) * (1 - (cbps / 10000.0) * turnover) - 1


def compute_turnover(current_w, prev_w, prev_returns):
    # wplus 不做再归一化
    wplus = prev_w * (1 + prev_returns)
    return np.sum(np.abs(current_w - wplus))


def kappa_r_from_closed_form(psi2hat, varghat, T, N, gam, lambda_, KappaE, dk=0.001):
    C = (2 * T * psi2hat + N - 1) * (
        N**4 + N**3 * T - 3 * N**3 - 4 * N**2 * T**2 + 22 * N**2 * T - 31 * N**2
        + N * T**3 - 7 * N * T**2 + 13 * N * T - 5 * N + T**4 - 12 * T**3
        + 53 * T**2 - 100 * T + 70
    ) + T**2 * psi2hat**2 * (
        N**3 + 2 * N**2 * T - 6 * N**2 - 7 * N * T**2 + 40 * N * T - 53 * N
        + 4 * T**3 - 34 * T**2 + 88 * T - 70
    )

    a1 = (1 / (gam**2)) * (
        (T**2 * (T - 2) * C)
        / (2 * (T - N - 7) * (T - N - 5) * (T - N - 3) ** 2 * (T - N - 2) * (T - N - 1) ** 2 * (T - N) ** 2)
    )
    a2 = -(2 * psi2hat / (gam**2)) * (
        (T**2 * (T - 2) * (T + N + 2 * T * psi2hat - 3))
        / ((T - N - 5) * (T - N - 3) * (T - N - 1) ** 2 * (T - N))
    )
    a3 = (T * psi2hat / (gam**2)) * (
        (2 * (N + 1) + T * (T - N - 3 + 2 * psi2hat * (T - N)))
        / ((T - N - 3) * (T - N - 1) ** 2 * (T - N))
    ) + varghat * (
        (T * (T - 2) * (T + N - 3) * (T * psi2hat + N - 1))
        / ((T - N - 5) * (T - N - 3) * (T - N - 1) ** 2 * (T - N))
    )
    a4 = -2 * varghat * psi2hat * ((T * (T - 2)) / ((T - N - 3) * (T - N - 1) ** 2))

    h = psi2hat + (N - 1) / T
    VUgmv = (psi2hat * varghat) / (T - N - 1) + (gam**2 / 2) * (varghat**2) * (((N - 1) * (T - 2)) / (((T - N - 1) ** 2) * (T - N - 3)))

    kappavec = np.arange(0, max(KappaE, 0.0) + dk / 2, dk)
    mean_risk = np.zeros_like(kappavec)

    for i, kappa in enumerate(kappavec):
        oos_u_mean = (1 / gam) * (T / (T - N - 1)) * (kappa * psi2hat - ((kappa**2) / 2) * h * ((T * (T - 2)) / ((T - N) * (T - N - 3))))
        oos_u_var = VUgmv + a1 * kappa**4 + a2 * kappa**3 + a3 * kappa**2 + a4 * kappa
        mean_risk[i] = oos_u_mean - lambda_ * np.sqrt(oos_u_var)

    idx = int(np.argmax(mean_risk))
    return idx * dk


def _window_worker(args):
    j, Xis, Xoos, N, T, gam, lambda_, opt_B = args

    one = np.ones((N, 1))
    eye = np.eye(N)

    muhat = np.mean(Xis, axis=0).reshape(-1, 1)
    sigmahat = np.cov(Xis, rowvar=False, bias=True)
    invsigmahat = np.linalg.inv(sigmahat)

    wghat = (invsigmahat @ one / np.sum(invsigmahat @ one)).reshape(-1, 1)
    Bhat = invsigmahat @ (eye - one @ wghat.T)
    wzhat = Bhat @ muhat

    raw_psi2 = (wzhat.T @ muhat).item()
    varghat = (wghat.T @ sigmahat @ wghat).item()

    psi2hat = psi2_adjustment(raw_psi2, T, N)

    wR = one / N
    tauR = ((wR.T @ sigmahat @ wR).item() - varghat) / varghat
    kappaFM = min(((N - 3) / (T - N + 2)) * (1.0 / tauR), 1.0)
    wFM = kappaFM * wR + (1 - kappaFM) * wghat
    varghat = (wFM.T @ sigmahat @ wFM).item()

    KappaE = (((T - N) * (T - N - 3)) / (T * (T - 2))) * (psi2hat / (psi2hat + (N - 1) / T))
    KappaR = kappa_r_from_closed_form(psi2hat, varghat, T, N, gam, lambda_, KappaE, dk=0.001)

    sigmaLW, _ = cov1para(Xis)
    invsigmaLW = np.linalg.inv(sigmaLW)
    wgLW = invsigmaLW @ one / np.sum(invsigmaLW @ one)
    BLW = invsigmaLW @ (eye - one @ wgLW.T)
    wzLW = BLW @ muhat

    # EW
    wEW = np.ones(N) / N
    EWgross = wEW @ Xoos

    # RTR
    muhatplus = np.maximum(muhat, 0.0)
    sigmahatdiag = np.diag(np.diag(sigmahat))
    invsigmahatdiag = np.linalg.inv(sigmahatdiag)
    wRTR = (invsigmahatdiag @ muhatplus).reshape(-1)
    wRTR = wRTR / np.sum(wRTR)
    RTRgross = wRTR @ Xoos

    # SGMV
    wSGMV = wgLW.reshape(-1)
    SGMVgross = wSGMV @ Xoos

    # SMV
    wSMV = (wgLW + (1 / gam) * wzLW).reshape(-1)
    SMVgross = wSMV @ Xoos

    # KappaE
    if opt_B is None:
        _, deltaE = optimalShrinkage(Xis, gam, lambda_, KappaE)
    else:
        _, deltaE = optimalShrinkage(Xis, gam, lambda_, KappaE, B=opt_B)

    sigmaLWrob_E = deltaE * np.mean(np.diag(sigmahat)) * eye + (1 - deltaE) * sigmahat
    invsigmaLWrob_E = np.linalg.inv(sigmaLWrob_E)
    wgLWrob_E = invsigmaLWrob_E @ one / np.sum(invsigmaLWrob_E @ one)
    BLWrob_E = invsigmaLWrob_E @ (eye - one @ wgLWrob_E.T)
    wzLWrob_E = BLWrob_E @ muhat
    wKappaE = (wgLWrob_E + (KappaE / gam) * wzLWrob_E).reshape(-1)
    KappaEgross = wKappaE @ Xoos

    # KappaR
    if opt_B is None:
        deltaR, _ = optimalShrinkage(Xis, gam, lambda_, KappaR)
    else:
        deltaR, _ = optimalShrinkage(Xis, gam, lambda_, KappaR, B=opt_B)

    sigmaLWrob_R = deltaR * np.mean(np.diag(sigmahat)) * eye + (1 - deltaR) * sigmahat
    invsigmaLWrob_R = np.linalg.inv(sigmaLWrob_R)
    wgLWrob_R = invsigmaLWrob_R @ one / np.sum(invsigmaLWrob_R @ one)
    BLWrob_R = invsigmaLWrob_R @ (eye - one @ wgLWrob_R.T)
    wzLWrob_R = BLWrob_R @ muhat
    wKappaR = (wgLWrob_R + (KappaR / gam) * wzLWrob_R).reshape(-1)
    KappaRgross = wKappaR @ Xoos

    return {
        "j": j,
        "wEW": wEW,
        "wRTR": wRTR,
        "wSGMV": wSGMV,
        "wSMV": wSMV,
        "wKappaE": wKappaE,
        "wKappaR": wKappaR,
        "EWgross": EWgross,
        "RTRgross": RTRgross,
        "SGMVgross": SGMVgross,
        "SMVgross": SMVgross,
        "KappaEgross": KappaEgross,
        "KappaRgross": KappaRgross,
        "KappaE": KappaE,
        "KappaR": KappaR,
    }


def _pval_worker(args):
    pair, gam, pval_B = args
    if pval_B is None:
        return dgnu_StatisticsPolitisRomano(pair, gam)
    return dgnu_StatisticsPolitisRomano(pair, gam, B=pval_B)


def run_one_dataset(X, dname):
    N = X.shape[1]
    Ttotal = X.shape[0]
    NumberWindow = Ttotal - T

    if max_windows is not None:
        NumberWindow = min(NumberWindow, max_windows)

    # 策略收益序列
    EWgross = np.zeros(NumberWindow)
    EWnet = np.zeros(NumberWindow)
    wEW = np.zeros((NumberWindow, N))

    RTRgross = np.zeros(NumberWindow)
    RTRnet = np.zeros(NumberWindow)
    wRTR = np.zeros((NumberWindow, N))

    SGMVgross = np.zeros(NumberWindow)
    SGMVnet = np.zeros(NumberWindow)
    wSGMV = np.zeros((NumberWindow, N))

    SMVgross = np.zeros(NumberWindow)
    SMVnet = np.zeros(NumberWindow)
    wSMV = np.zeros((NumberWindow, N))

    KappaEgross = np.zeros(NumberWindow)
    KappaEnet = np.zeros(NumberWindow)
    wKappaE = np.zeros((NumberWindow, N))
    KappaEvec = np.zeros(NumberWindow)

    KappaRgross = np.zeros(NumberWindow)
    KappaRnet = np.zeros(NumberWindow)
    wKappaR = np.zeros((NumberWindow, N))
    KappaRvec = np.zeros(NumberWindow)

    TurnoverEW = np.zeros(NumberWindow)
    TurnoverRTR = np.zeros(NumberWindow)
    TurnoverSGMV = np.zeros(NumberWindow)
    TurnoverSMV = np.zeros(NumberWindow)
    TurnoverKappaE = np.zeros(NumberWindow)
    TurnoverKappaR = np.zeros(NumberWindow)

    tasks = []
    for j in range(NumberWindow):
        Xis = X[j : j + T, :]
        Xoos = X[j + T, :]
        tasks.append((j, Xis, Xoos, N, T, gam, lambda_, opt_B))

    # 窗口级并行：每个窗口独立计算权重与 gross return
    chunksize = max(1, NumberWindow // max(1, num_workers * 4))
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_window_worker, tasks, chunksize=chunksize))

    results.sort(key=lambda z: z["j"])

    for r in results:
        j = r["j"]
        wEW[j, :] = r["wEW"]
        wRTR[j, :] = r["wRTR"]
        wSGMV[j, :] = r["wSGMV"]
        wSMV[j, :] = r["wSMV"]
        wKappaE[j, :] = r["wKappaE"]
        wKappaR[j, :] = r["wKappaR"]

        EWgross[j] = r["EWgross"]
        RTRgross[j] = r["RTRgross"]
        SGMVgross[j] = r["SGMVgross"]
        SMVgross[j] = r["SMVgross"]
        KappaEgross[j] = r["KappaEgross"]
        KappaRgross[j] = r["KappaRgross"]

        KappaEvec[j] = r["KappaE"]
        KappaRvec[j] = r["KappaR"]

        if (j + 1) % 20 == 0 or (j + 1) == NumberWindow:
            print(f"  {dname}: 窗口 {j+1}/{NumberWindow} 完成")

    # 由权重序列回填换手与净收益
    for j in range(NumberWindow):
        if j > 0:
            prev_returns = X[T + j - 1, :]
            TurnoverEW[j] = compute_turnover(wEW[j, :], wEW[j - 1, :], prev_returns)
            TurnoverRTR[j] = compute_turnover(wRTR[j, :], wRTR[j - 1, :], prev_returns)
            TurnoverSGMV[j] = compute_turnover(wSGMV[j, :], wSGMV[j - 1, :], prev_returns)
            TurnoverSMV[j] = compute_turnover(wSMV[j, :], wSMV[j - 1, :], prev_returns)
            TurnoverKappaE[j] = compute_turnover(wKappaE[j, :], wKappaE[j - 1, :], prev_returns)
            TurnoverKappaR[j] = compute_turnover(wKappaR[j, :], wKappaR[j - 1, :], prev_returns)

        EWnet[j] = transaction_return(EWgross[j], TurnoverEW[j], cbps)
        RTRnet[j] = transaction_return(RTRgross[j], TurnoverRTR[j], cbps)
        SGMVnet[j] = transaction_return(SGMVgross[j], TurnoverSGMV[j], cbps)
        SMVnet[j] = transaction_return(SMVgross[j], TurnoverSMV[j], cbps)
        KappaEnet[j] = transaction_return(KappaEgross[j], TurnoverKappaE[j], cbps)
        KappaRnet[j] = transaction_return(KappaRgross[j], TurnoverKappaR[j], cbps)

    OOSGrossCERvec = np.array([
        np.mean(EWgross) - (gam / 2) * np.var(EWgross, ddof=1),
        np.mean(RTRgross) - (gam / 2) * np.var(RTRgross, ddof=1),
        np.mean(SGMVgross) - (gam / 2) * np.var(SGMVgross, ddof=1),
        np.mean(SMVgross) - (gam / 2) * np.var(SMVgross, ddof=1),
        np.mean(KappaEgross) - (gam / 2) * np.var(KappaEgross, ddof=1),
        np.mean(KappaRgross) - (gam / 2) * np.var(KappaRgross, ddof=1),
    ])

    OOSNetCERvec = np.array([
        np.mean(EWnet) - (gam / 2) * np.var(EWnet, ddof=1),
        np.mean(RTRnet) - (gam / 2) * np.var(RTRnet, ddof=1),
        np.mean(SGMVnet) - (gam / 2) * np.var(SGMVnet, ddof=1),
        np.mean(SMVnet) - (gam / 2) * np.var(SMVnet, ddof=1),
        np.mean(KappaEnet) - (gam / 2) * np.var(KappaEnet, ddof=1),
        np.mean(KappaRnet) - (gam / 2) * np.var(KappaRnet, ddof=1),
    ])

    OOSGrossSRvec = np.array([
        np.mean(EWgross) / np.std(EWgross, ddof=1),
        np.mean(RTRgross) / np.std(RTRgross, ddof=1),
        np.mean(SGMVgross) / np.std(SGMVgross, ddof=1),
        np.mean(SMVgross) / np.std(SMVgross, ddof=1),
        np.mean(KappaEgross) / np.std(KappaEgross, ddof=1),
        np.mean(KappaRgross) / np.std(KappaRgross, ddof=1),
    ])

    # 保持 Matlab 原脚本这一处口径
    OOSNetSRvec = np.array([
        np.mean(EWnet) / np.std(EWnet, ddof=1),
        np.mean(RTRnet) / np.std(RTRnet, ddof=1),
        np.mean(SGMVnet) / np.std(SGMVnet, ddof=1),
        np.mean(SMVnet) / np.std(SMVnet, ddof=1),
        np.mean(KappaEnet) / np.std(KappaEnet, ddof=1),
        np.mean(KappaRnet) / np.std(KappaRnet, ddof=1),
    ])

    AverageKappa = np.array([np.nan, np.nan, 0.0, 1.0, np.mean(KappaEvec), np.mean(KappaRvec)])

    Xpval = np.column_stack([EWgross, EWnet, KappaEgross, KappaEnet, KappaRgross, KappaRnet])
    pvaluesEW = np.zeros(4)
    pvaluesKappaE = np.zeros(4)

    if Xpval.shape[0] < 2:
        pvaluesEW[:] = np.nan
        pvaluesKappaE[:] = np.nan
    else:
        pairs = [
            np.column_stack([Xpval[:, 0], Xpval[:, 4]]),
            np.column_stack([Xpval[:, 1], Xpval[:, 5]]),
            np.column_stack([Xpval[:, 2], Xpval[:, 4]]),
            np.column_stack([Xpval[:, 3], Xpval[:, 5]]),
        ]

        with ProcessPoolExecutor(max_workers=min(num_workers, 4)) as ex:
            pouts = list(ex.map(_pval_worker, [(pair, gam, pval_B) for pair in pairs]))

        pvaluesEW[0], pvaluesEW[2] = pouts[0]
        pvaluesEW[1], pvaluesEW[3] = pouts[1]
        pvaluesKappaE[0], pvaluesKappaE[2] = pouts[2]
        pvaluesKappaE[1], pvaluesKappaE[3] = pouts[3]

    Results = np.vstack([
        OOSGrossCERvec * 12,
        OOSNetCERvec * 12,
        OOSGrossSRvec * np.sqrt(12),
        OOSNetSRvec * np.sqrt(12),
        AverageKappa,
    ])

    return Results, pvaluesEW / 2.0, pvaluesKappaE / 2.0


def fmt(x, digits=4):
    if np.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def build_outputs(Table2Performance, Table2pvaluesEW, Table2pvaluesKappaE, datasets):
    out_file = base_dir / "Table2_results.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("Table2Performance\n")
        f.write(np.array2string(Table2Performance, precision=6, suppress_small=False))
        f.write("\n\nTable2pvaluesEW\n")
        f.write(np.array2string(Table2pvaluesEW, precision=6, suppress_small=False))
        f.write("\n\nTable2pvaluesKappaE\n")
        f.write(np.array2string(Table2pvaluesKappaE, precision=6, suppress_small=False))
        f.write("\n")

    strategies = ["EW", "RTR", "SGMV", "SMV", "$\\hat\\kappa_E$", "$\\hat\\kappa_R$"]
    row_labels = ["Gross CER", "Net CER", "Gross SR", "Net SR", "Average $\\hat\\kappa$"]
    pval_row_labels = ["Gross CER", "Net CER", "Gross SR", "Net SR"]

    tex_file = base_dir / "Table2_results.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        def wl(line=""):
            f.write(line + "\n")

        wl("% Auto-generated by ReplicateTable2.py")
        wl("% Table 2 performance and one-sided p-values")
        wl()

        wl("\\begin{table}[!htbp]")
        wl("\\centering")
        wl("\\caption{Table 2 Performance (Python replication)}")
        wl("\\label{tab:table2_performance_python}")
        wl("\\begin{tabular}{llrrrrrr}")
        wl("\\hline")
        wl("Dataset & Metric & " + " & ".join(strategies) + " \\\\")
        wl("\\hline")

        for d_idx, (dname, _) in enumerate(datasets):
            block = Table2Performance[d_idx * 5 : (d_idx + 1) * 5, :]
            for r_idx in range(5):
                ds_name = dname if r_idx == 0 else ""
                vals = " & ".join(fmt(v, 4) for v in block[r_idx, :])
                wl(f"{ds_name} & {row_labels[r_idx]} & {vals} \\\\")
            wl("\\hline")

        wl("\\end{tabular}")
        wl("\\end{table}")
        wl()

        wl("\\begin{table}[!htbp]")
        wl("\\centering")
        wl("\\caption{One-sided p-values: $\\kappa_R$ relative to EW}")
        wl("\\label{tab:table2_pvalues_ew_python}")
        wl("\\begin{tabular}{llr}")
        wl("\\hline")
        wl("Dataset & Metric & p-value \\\\")
        wl("\\hline")

        for d_idx, (dname, _) in enumerate(datasets):
            block = Table2pvaluesEW[d_idx * 4 : (d_idx + 1) * 4, 0]
            for r_idx in range(4):
                ds_name = dname if r_idx == 0 else ""
                wl(f"{ds_name} & {pval_row_labels[r_idx]} & {fmt(block[r_idx], 4)} \\\\")
            wl("\\hline")

        wl("\\end{tabular}")
        wl("\\end{table}")
        wl()

        wl("\\begin{table}[!htbp]")
        wl("\\centering")
        wl("\\caption{One-sided p-values: $\\kappa_R$ relative to $\\kappa_E$}")
        wl("\\label{tab:table2_pvalues_kappae_python}")
        wl("\\begin{tabular}{llr}")
        wl("\\hline")
        wl("Dataset & Metric & p-value \\\\")
        wl("\\hline")

        for d_idx, (dname, _) in enumerate(datasets):
            block = Table2pvaluesKappaE[d_idx * 4 : (d_idx + 1) * 4, 0]
            for r_idx in range(4):
                ds_name = dname if r_idx == 0 else ""
                wl(f"{ds_name} & {pval_row_labels[r_idx]} & {fmt(block[r_idx], 4)} \\\\")
            wl("\\hline")

        wl("\\end{tabular}")
        wl("\\end{table}")

    return out_file, tex_file


def main():
    datasets = [
        ("10MOM", "Dataset10MOM.txt"),
        ("25SBTM", "Dataset25SBTM.txt"),
        ("25OPINV", "Dataset25OPINV.txt"),
        ("49IND", "Dataset49IND.txt"),
        ("16LTANOM", "Dataset16LTANOM.txt"),
        ("46ANOM", "Dataset46ANOM.txt"),
    ]

    if only_first_dataset:
        datasets = datasets[:1]

    results_blocks = []
    pval_ew_blocks = []
    pval_ke_blocks = []

    for dname, dfile in datasets:
        print(f"\n开始处理数据集: {dname}")
        X = np.loadtxt(project_dir / dfile, delimiter=",")

        Results, pvalEW, pvalKE = run_one_dataset(X, dname)
        results_blocks.append(Results)
        pval_ew_blocks.append(pvalEW.reshape(-1, 1))
        pval_ke_blocks.append(pvalKE.reshape(-1, 1))

    Table2Performance = np.vstack(results_blocks)
    Table2pvaluesEW = np.vstack(pval_ew_blocks)
    Table2pvaluesKappaE = np.vstack(pval_ke_blocks)

    out_file, tex_file = build_outputs(Table2Performance, Table2pvaluesEW, Table2pvaluesKappaE, datasets)

    print(f"结果已写入: {out_file}")
    print(f"LaTeX 已写入: {tex_file}")

    return {
        "Table2Performance": Table2Performance,
        "Table2pvaluesEW": Table2pvaluesEW,
        "Table2pvaluesKappaE": Table2pvaluesKappaE,
    }


if __name__ == "__main__":
    main()

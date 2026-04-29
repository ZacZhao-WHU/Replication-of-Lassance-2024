from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
from scipy.special import beta, betainc


# ============================================================
# Table 1 复现脚本
# 1) 仅计算 10MOM
# 2) 多进程并行
# 3) 输出 txt 与 LaTeX
#
# 备注：默认 M=100000（与论文一致，耗时较长）
# 若想先快速验证，可把 M 改小，例如 2000。
# ============================================================

base_dir = Path(__file__).resolve().parent
project_dir = base_dir.parent

DATASET_NAME = "10MOM"
DATASET_FILE = "Dataset10MOM.txt"

lambda_ = 2.0
gam = 3.0
M = 100000
T_list = [120, 180, 240]
seed = 20260425

num_workers = max(1, (cpu_count() or 2) - 1)


def psi2_adjustment(raw_psi2, T, N):
    a = (N - 1) / 2
    b = (T - N + 1) / 2
    frac = raw_psi2 / (1 + raw_psi2)
    num = 2 * (raw_psi2 ** ((N - 1) / 2)) * ((1 + raw_psi2) ** (-(T - 2) / 2))
    den = T * beta(a, b) * betainc(a, b, frac)
    return ((T - N - 1) * raw_psi2 - (N - 1)) / T + num / den


def compute_kappa_e_r(muhat, sigmahat, T, N, gam, lambda_, dk):
    one = np.ones((N, 1))
    eye = np.eye(N)

    invsigmahat = np.linalg.inv(sigmahat)
    wghat = invsigmahat @ one / np.sum(invsigmahat @ one)
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

    kappavec = np.arange(0.0, max(KappaE, 0.0) + dk / 2, dk)
    mean_risk = np.zeros_like(kappavec)

    for i, kappa in enumerate(kappavec):
        oos_u_mean = (1 / gam) * (T / (T - N - 1)) * (kappa * psi2hat - ((kappa**2) / 2) * h * ((T * (T - 2)) / ((T - N) * (T - N - 3))))
        oos_u_var = VUgmv + a1 * kappa**4 + a2 * kappa**3 + a3 * kappa**2 + a4 * kappa
        mean_risk[i] = oos_u_mean - lambda_ * np.sqrt(oos_u_var)

    idx = int(np.argmax(mean_risk))
    KappaR = idx * dk
    wKappaE = wghat + (KappaE / gam) * wzhat
    wKappaR = wghat + (KappaR / gam) * wzhat
    return wKappaE, wKappaR


def _gaussian_chunk_worker(args):
    seed_chunk, T, mu, sigma, N, gam, lambda_ = args
    utility_e = []
    utility_r = []
    for s in seed_chunk:
        rng = np.random.default_rng(int(s))
        Xnorm = rng.multivariate_normal(mu.reshape(-1), sigma, size=T)
        muhat = np.mean(Xnorm, axis=0).reshape(-1, 1)
        sigmahat = np.cov(Xnorm, rowvar=False, bias=True)
        wKappaE, wKappaR = compute_kappa_e_r(muhat, sigmahat, T, N, gam, lambda_, dk=0.001)
        utility_e.append((wKappaE.T @ mu - (gam / 2) * (wKappaE.T @ sigma @ wKappaE)).item())
        utility_r.append((wKappaR.T @ mu - (gam / 2) * (wKappaR.T @ sigma @ wKappaR)).item())
    return np.array(utility_e), np.array(utility_r)


def _bootstrap_chunk_worker(args):
    seed_chunk, T, X, N, gam, lambda_ = args
    utility_e = []
    utility_r = []
    for s in seed_chunk:
        rng = np.random.default_rng(int(s))
        idx = rng.integers(0, X.shape[0], size=2 * T)
        Xrand = X[idx, :]
        Xis = Xrand[:T, :]
        Xoos = Xrand[T:2 * T, :]

        muoos = np.mean(Xoos, axis=0).reshape(-1, 1)
        sigmaoos = ((T - 1) / T) * np.cov(Xoos, rowvar=False, bias=False)
        muhat = np.mean(Xis, axis=0).reshape(-1, 1)
        sigmahat = np.cov(Xis, rowvar=False, bias=False)

        wKappaE, wKappaR = compute_kappa_e_r(muhat, sigmahat, T, N, gam, lambda_, dk=0.0001)
        utility_e.append((wKappaE.T @ muoos - (gam / 2) * (wKappaE.T @ sigmaoos @ wKappaE)).item())
        utility_r.append((wKappaR.T @ muoos - (gam / 2) * (wKappaR.T @ sigmaoos @ wKappaR)).item())
    return np.array(utility_e), np.array(utility_r)


def split_into_chunks(array, chunk_size):
    return [array[i : i + chunk_size] for i in range(0, len(array), chunk_size)]


def run_parallel_simulation(mode, T, M, X, mu, sigma, N, gam, lambda_, seed, workers):
    seeds = np.random.SeedSequence(seed).generate_state(M, dtype=np.uint64)
    chunk_size = max(100, M // max(1, workers * 4))
    seed_chunks = split_into_chunks(seeds, chunk_size)

    tasks = []
    if mode == "gaussian":
        for chunk in seed_chunks:
            tasks.append((chunk, T, mu, sigma, N, gam, lambda_))
        worker = _gaussian_chunk_worker
    else:
        for chunk in seed_chunks:
            tasks.append((chunk, T, X, N, gam, lambda_))
        worker = _bootstrap_chunk_worker

    utility_e_all = []
    utility_r_all = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for ue, ur in ex.map(worker, tasks):
            utility_e_all.append(ue)
            utility_r_all.append(ur)

    utility_e = np.concatenate(utility_e_all)
    utility_r = np.concatenate(utility_r_all)
    return utility_e, utility_r


def summarize_results(utility_e, utility_r, lambda_):
    return np.array([
        np.mean(utility_e),
        np.mean(utility_r),
        np.std(utility_e, ddof=1),
        np.std(utility_r, ddof=1),
        np.mean(utility_e) - lambda_ * np.std(utility_e, ddof=1),
        np.mean(utility_r) - lambda_ * np.std(utility_r, ddof=1),
    ])


def build_outputs(results_gaussian, results_boot, M, workers):
    txt_file = base_dir / "Table1_results.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("Table 1 replication results (10MOM only)\n")
        f.write(f"lambda={lambda_}, gamma={gam}, M={M}, workers={workers}, seed={seed}\n\n")

        f.write("Gaussian (rows: KappaE, KappaR; cols: T120,T180,T240,Std120,Std180,Std240,MR120,MR180,MR240)\n")
        f.write(np.array2string(results_gaussian, precision=6, suppress_small=False))
        f.write("\n\n")

        f.write("Bootstrap (rows: KappaE, KappaR; cols: T120,T180,T240,Std120,Std180,Std240,MR120,MR180,MR240)\n")
        f.write(np.array2string(results_boot, precision=6, suppress_small=False))
        f.write("\n")

    def pct2(x):
        return f"{100.0 * x:.2f}\\%"

    tex_file = base_dir / "Table1_results.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by ReplicateTable1.py\n")
        f.write("% Need package: \\usepackage{booktabs,multirow}\n")
        f.write("\\begin{table}[!htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{llccccccccc}\n")
        f.write("\\toprule\n")
        f.write("& & \\multicolumn{3}{c}{OOSU mean} & \\multicolumn{3}{c}{OOSU risk} & \\multicolumn{3}{c}{Mean-risk OOSU} \\\\ \n")
        f.write("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}\n")
        f.write("& $T$ & 120 & 180 & 240 & 120 & 180 & 240 & 120 & 180 & 240 \\\\ \n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{11}{l}{\\textbf{复现 Panel A: Gaussian data}} \\\\ \n")
        f.write("\\multirow{2}{*}{$10MOM$}\n")
        f.write(
            "& $\\hat{\\kappa}_E^*$ & "
            + " & ".join(pct2(v) for v in results_gaussian[0, :])
            + " \\\\ \n"
        )
        f.write(
            "& $\\hat{\\kappa}_R^*$ & "
            + " & ".join(pct2(v) for v in results_gaussian[1, :])
            + " \\\\ \n"
        )
        f.write("\\midrule\n")
        f.write("\\multicolumn{11}{l}{\\textbf{复现 Panel B: Bootstrapped data}} \\\\ \n")
        f.write("\\multirow{2}{*}{$10MOM$}\n")
        f.write(
            "& $\\hat{\\kappa}_E^*$ & "
            + " & ".join(pct2(v) for v in results_boot[0, :])
            + " \\\\ \n"
        )
        f.write(
            "& $\\hat{\\kappa}_R^*$ & "
            + " & ".join(pct2(v) for v in results_boot[1, :])
            + " \\\\ \n"
        )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return txt_file, tex_file


def run_table1(M_value=M, workers=num_workers):
    print(f"开始处理数据集: {DATASET_NAME}")
    X = np.loadtxt(project_dir / DATASET_FILE, delimiter=",")

    N = X.shape[1]
    mu = np.mean(X, axis=0).reshape(-1, 1)
    sigma = np.cov(X, rowvar=False, bias=False)

    results_gaussian = np.zeros((2, 9), dtype=float)
    results_boot = np.zeros((2, 9), dtype=float)

    for t_idx, T in enumerate(T_list):
        ue, ur = run_parallel_simulation(
            mode="gaussian",
            T=T,
            M=M_value,
            X=X,
            mu=mu,
            sigma=sigma,
            N=N,
            gam=gam,
            lambda_=lambda_,
            seed=seed + 1000 + T,
            workers=workers,
        )
        s = summarize_results(ue, ur, lambda_)
        results_gaussian[0, t_idx] = s[0]
        results_gaussian[1, t_idx] = s[1]
        results_gaussian[0, t_idx + 3] = s[2]
        results_gaussian[1, t_idx + 3] = s[3]
        results_gaussian[0, t_idx + 6] = s[4]
        results_gaussian[1, t_idx + 6] = s[5]
        print(f"  Gaussian T={T} 完成")

    for t_idx, T in enumerate(T_list):
        ue, ur = run_parallel_simulation(
            mode="bootstrap",
            T=T,
            M=M_value,
            X=X,
            mu=mu,
            sigma=sigma,
            N=N,
            gam=gam,
            lambda_=lambda_,
            seed=seed + 2000 + T,
            workers=workers,
        )
        s = summarize_results(ue, ur, lambda_)
        results_boot[0, t_idx] = s[0]
        results_boot[1, t_idx] = s[1]
        results_boot[0, t_idx + 3] = s[2]
        results_boot[1, t_idx + 3] = s[3]
        results_boot[0, t_idx + 6] = s[4]
        results_boot[1, t_idx + 6] = s[5]
        print(f"  Bootstrap T={T} 完成")

    txt_file, tex_file = build_outputs(results_gaussian, results_boot, M_value, workers)
    print("\nTable 1 复现完成")
    print(f"结果已写入: {txt_file}")
    print(f"LaTeX 已写入: {tex_file}")

    return {
        "Gaussian": results_gaussian,
        "Bootstrap": results_boot,
    }


def main():
    run_table1()


if __name__ == "__main__":
    main()

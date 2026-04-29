from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =========================
# 读取数据
# =========================
base_dir = Path(__file__).resolve().parent
project_dir = base_dir.parent

数据文件 = project_dir / "Dataset25SBTM.txt"
X = np.loadtxt(数据文件, delimiter=",")


# =========================
# 基础参数
# =========================
N0 = 25
T0 = 120
gam = 3

mu = np.mean(X, axis=0).reshape(-1, 1)
sigma = np.cov(X, rowvar=False, bias=False)
invsigma = np.linalg.inv(sigma)
one = np.ones((N0, 1))

wg = invsigma @ one / np.sum(invsigma @ one)
varg = (wg.T @ sigma @ wg).item()
B = invsigma @ (np.eye(N0) - one @ wg.T)
psi2 = (mu.T @ B @ mu).item()


# =========================
# Figure 2: T 的影响
# =========================
Tvec = np.arange(100, 1001)
StdUgmvT = np.zeros(len(Tvec))
StdUsmvT = np.zeros(len(Tvec))

for j, T in enumerate(Tvec):
    T = float(T)
    Nf = float(N0)

    StdUgmvT[j] = np.sqrt(
        (psi2 * varg) / (T - Nf - 1)
        + (gam**2 / 2) * (varg**2) * (Nf - 1) * (T - 2) / ((T - Nf - 1) ** 2 * (T - Nf - 3))
    )

    C = (
        (2 * T * psi2 + Nf - 1)
        * (
            Nf**4
            + Nf**3 * T
            - 3 * Nf**3
            - 4 * Nf**2 * T**2
            + 22 * Nf**2 * T
            - 31 * Nf**2
            + Nf * T**3
            - 7 * Nf * T**2
            + 13 * Nf * T
            - 5 * Nf
            + T**4
            - 12 * T**3
            + 53 * T**2
            - 100 * T
            + 70
        )
        + T**2
        * psi2**2
        * (Nf**3 + 2 * Nf**2 * T - 6 * Nf**2 - 7 * Nf * T**2 + 40 * Nf * T - 53 * Nf + 4 * T**3 - 34 * T**2 + 88 * T - 70)
    )

    a1 = (1 / (gam**2)) * (
        (T**2 * (T - 2) * C)
        / (2 * (T - Nf - 7) * (T - Nf - 5) * (T - Nf - 3) ** 2 * (T - Nf - 2) * (T - Nf - 1) ** 2 * (T - Nf) ** 2)
    )
    a2 = -(2 * psi2 / (gam**2)) * (
        (T**2 * (T - 2) * (T + Nf + 2 * T * psi2 - 3)) / ((T - Nf - 5) * (T - Nf - 3) * (T - Nf - 1) ** 2 * (T - Nf))
    )
    a3 = (
        (T * psi2 / (gam**2))
        * ((2 * (Nf + 1) + T * (T - Nf - 3 + 2 * psi2 * (T - Nf))) / ((T - Nf - 3) * (T - Nf - 1) ** 2 * (T - Nf)))
        + varg * ((T * (T - 2) * (T + Nf - 3) * (T * psi2 + Nf - 1)) / ((T - Nf - 5) * (T - Nf - 3) * (T - Nf - 1) ** 2 * (T - Nf)))
    )
    a4 = -2 * varg * psi2 * ((T * (T - 2)) / ((T - Nf - 3) * (T - Nf - 1) ** 2))

    StdUsmvT[j] = np.sqrt(StdUgmvT[j] ** 2 + a1 + a2 + a3 + a4)


# =========================
# Figure 2: N 的影响
# =========================
T = 120
Nvec = np.arange(2, 101)
StdUgmvN = np.zeros(len(Nvec))
StdUsmvN = np.zeros(len(Nvec))

for j, N in enumerate(Nvec):
    T_f = float(T)
    N = float(N)

    StdUgmvN[j] = np.sqrt(
        (psi2 * varg) / (T_f - N - 1)
        + (gam**2 / 2) * (varg**2) * (N - 1) * (T_f - 2) / ((T_f - N - 1) ** 2 * (T_f - N - 3))
    )

    C = (
        (2 * T_f * psi2 + N - 1)
        * (
            N**4
            + N**3 * T_f
            - 3 * N**3
            - 4 * N**2 * T_f**2
            + 22 * N**2 * T_f
            - 31 * N**2
            + N * T_f**3
            - 7 * N * T_f**2
            + 13 * N * T_f
            - 5 * N
            + T_f**4
            - 12 * T_f**3
            + 53 * T_f**2
            - 100 * T_f
            + 70
        )
        + T_f**2
        * psi2**2
        * (N**3 + 2 * N**2 * T_f - 6 * N**2 - 7 * N * T_f**2 + 40 * N * T_f - 53 * N + 4 * T_f**3 - 34 * T_f**2 + 88 * T_f - 70)
    )

    a1 = (1 / (gam**2)) * (
        (T_f**2 * (T_f - 2) * C)
        / (2 * (T_f - N - 7) * (T_f - N - 5) * (T_f - N - 3) ** 2 * (T_f - N - 2) * (T_f - N - 1) ** 2 * (T_f - N) ** 2)
    )
    a2 = -(2 * psi2 / (gam**2)) * (
        (T_f**2 * (T_f - 2) * (T_f + N + 2 * T_f * psi2 - 3)) / ((T_f - N - 5) * (T_f - N - 3) * (T_f - N - 1) ** 2 * (T_f - N))
    )
    a3 = (
        (T_f * psi2 / (gam**2))
        * ((2 * (N + 1) + T_f * (T_f - N - 3 + 2 * psi2 * (T_f - N))) / ((T_f - N - 3) * (T_f - N - 1) ** 2 * (T_f - N)))
        + varg * ((T_f * (T_f - 2) * (T_f + N - 3) * (T_f * psi2 + N - 1)) / ((T_f - N - 5) * (T_f - N - 3) * (T_f - N - 1) ** 2 * (T_f - N)))
    )
    a4 = -2 * varg * psi2 * ((T_f * (T_f - 2)) / ((T_f - N - 3) * (T_f - N - 1) ** 2))

    StdUsmvN[j] = np.sqrt(StdUgmvN[j] ** 2 + a1 + a2 + a3 + a4)


# =========================
# 作图
# =========================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.plot(Tvec, StdUgmvT, "--b", linewidth=1)
ax.plot(Tvec, StdUsmvT, "r", linewidth=1)
ax.set_xlim(100, 1000)
ax.set_ylim(0, 0.05)
ax.set_xticks([100, 250, 500, 750, 1000])
ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
ax.set_title(r"Effect of $T$", fontsize=14)
ax.legend([r"SGMV ($\kappa=0$)", r"SMV ($\kappa=1$)"], fontsize=11, loc="best")
ax.tick_params(labelsize=11)

ax = axes[0, 1]
ax.plot(Nvec, np.log10(StdUgmvN), "--b", linewidth=1)
ax.plot(Nvec, np.log10(StdUsmvN), "r", linewidth=1)
ax.set_xlim(2, 100)
ax.set_ylim(-3, 2)
ax.set_xticks([2, 20, 40, 60, 80, 100])
ax.set_yticks([-3, -2, -1, 0, 1, 2])
ax.set_title(r"Effect of $N$ ($\log_{10}$ scale)", fontsize=14)
ax.tick_params(labelsize=11)

# 第三、第四个子图在原 Matlab 代码中留空，这里保持空白以匹配版式
axes[1, 0].axis("off")
axes[1, 1].axis("off")

fig.tight_layout()

figure_path = base_dir / "Figure2.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")

print("Figure 2 已生成")
print(f"图片路径: {figure_path}")

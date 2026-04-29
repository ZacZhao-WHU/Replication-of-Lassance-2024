from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# 读取数据
base_dir = Path(__file__).resolve().parent
project_dir = base_dir.parent

data_file = project_dir / "Dataset25SBTM.txt"
X = np.loadtxt(data_file, delimiter=",")



# 参数设定
N = 25
T = 120
gam = 3
lambda_ = 2

mu = np.mean(X, axis=0).reshape(-1, 1)
sigma = np.cov(X, rowvar=False, bias=False)
invsigma = np.linalg.inv(sigma)
one = np.ones((N, 1))

wg = invsigma @ one / np.sum(invsigma @ one)
mug = (wg.T @ mu).item()
varg = (wg.T @ sigma @ wg).item()
B = invsigma @ (np.eye(N) - one @ wg.T)
psi2 = (mu.T @ B @ mu).item()


# 解析式
EUgmv = mug - (gam / 2) * varg * (T - 2) / (T - N - 1)

EUwz = lambda k: (1 / gam) * (T / (T - N - 1)) * (
    k * psi2
    - ((k**2) / 2) * (psi2 + (N - 1) / T) * T * (T - 2) / ((T - N) * (T - N - 3))
)

kappaE = (T - N) * (T - N - 3) * (psi2 / (psi2 + (N - 1) / T)) / (T * (T - 2))

VUgmv = (psi2 * varg) / (T - N - 1) + (gam**2 / 2) * (varg**2) * (N - 1) * (T - 2) / (
    (T - N - 1) ** 2 * (T - N - 3)
)

C = (
    (2 * T * psi2 + N - 1)
    * (
        N**4
        + N**3 * T
        - 3 * N**3
        - 4 * N**2 * T**2
        + 22 * N**2 * T
        - 31 * N**2
        + N * T**3
        - 7 * N * T**2
        + 13 * N * T
        - 5 * N
        + T**4
        - 12 * T**3
        + 53 * T**2
        - 100 * T
        + 70
    )
    + T**2
    * psi2**2
    * (N**3 + 2 * N**2 * T - 6 * N**2 - 7 * N * T**2 + 40 * N * T - 53 * N + 4 * T**3 - 34 * T**2 + 88 * T - 70)
)

a1 = (1 / (gam**2)) * (
    (T**2 * (T - 2) * C)
    / (2 * (T - N - 7) * (T - N - 5) * (T - N - 3) ** 2 * (T - N - 2) * (T - N - 1) ** 2 * (T - N) ** 2)
)
a2 = -(2 * psi2 / (gam**2)) * (
    (T**2 * (T - 2) * (T + N + 2 * T * psi2 - 3)) / ((T - N - 5) * (T - N - 3) * (T - N - 1) ** 2 * (T - N))
)
a3 = (
    (T * psi2 / (gam**2))
    * ((2 * (N + 1) + T * (T - N - 3 + 2 * psi2 * (T - N))) / ((T - N - 3) * (T - N - 1) ** 2 * (T - N)))
    + varg * ((T * (T - 2) * (T + N - 3) * (T * psi2 + N - 1)) / ((T - N - 5) * (T - N - 3) * (T - N - 1) ** 2 * (T - N)))
)
a4 = -2 * varg * psi2 * ((T * (T - 2)) / ((T - N - 3) * (T - N - 1) ** 2))


# 选择 kappa_V 与 kappa_R
dk = 0.0001
kappavec = np.arange(0, 1 + dk / 2, dk)
MeanRiskOOSU = np.zeros_like(kappavec)
VU = np.zeros_like(kappavec)

for j, kappa in enumerate(kappavec):
    OOSUMean = EUgmv + EUwz(kappa)
    VU[j] = VUgmv + a1 * kappa**4 + a2 * kappa**3 + a3 * kappa**2 + a4 * kappa
    MeanRiskOOSU[j] = OOSUMean - lambda_ * np.sqrt(VU[j])

index_R = int(np.argmax(MeanRiskOOSU))
kappaR = index_R * dk
index_V = int(np.argmin(VU))
kappaV = index_V * dk


# 构造有效前沿
kappavec1 = np.arange(0, kappaV + dk / 2, dk)
EU1 = np.zeros_like(kappavec1)
StdU1 = np.zeros_like(kappavec1)
for j, kappa in enumerate(kappavec1):
    EU1[j] = EUgmv + EUwz(kappa)
    StdU1[j] = np.sqrt(VUgmv + a1 * kappa**4 + a2 * kappa**3 + a3 * kappa**2 + a4 * kappa)

kappavec2 = np.arange(kappaV, kappaE + dk / 2, dk)
EU2 = np.zeros_like(kappavec2)
StdU2 = np.zeros_like(kappavec2)
for j, kappa in enumerate(kappavec2):
    EU2[j] = EUgmv + EUwz(kappa)
    StdU2[j] = np.sqrt(VUgmv + a1 * kappa**4 + a2 * kappa**3 + a3 * kappa**2 + a4 * kappa)

kappavec3 = np.arange(kappaE, 1 + dk / 2, dk)
EU3 = np.zeros_like(kappavec3)
StdU3 = np.zeros_like(kappavec3)
for j, kappa in enumerate(kappavec3):
    EU3[j] = EUgmv + EUwz(kappa)
    StdU3[j] = np.sqrt(VUgmv + a1 * kappa**4 + a2 * kappa**3 + a3 * kappa**2 + a4 * kappa)

EUkappaV = EUgmv + EUwz(kappaV)
StdUkappaV = np.sqrt(VUgmv + a1 * kappaV**4 + a2 * kappaV**3 + a3 * kappaV**2 + a4 * kappaV)
EUkappaR = EUgmv + EUwz(kappaR)
StdUkappaR = np.sqrt(VUgmv + a1 * kappaR**4 + a2 * kappaR**3 + a3 * kappaR**2 + a4 * kappaR)
EUkappaE = EUgmv + EUwz(kappaE)
StdUkappaE = np.sqrt(VUgmv + a1 * kappaE**4 + a2 * kappaE**3 + a3 * kappaE**2 + a4 * kappaE)



# 画图
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(StdU1, EU1, "--k", linewidth=0.5)
ax.plot(StdU2, EU2, "b", linewidth=1.5)
ax.plot(StdU3, EU3, "--k", linewidth=0.5)
ax.plot(np.sqrt(VUgmv), EUgmv, marker="x", markersize=12, markeredgecolor="k", linewidth=1.5, color="k")
ax.plot(StdUkappaV, EUkappaV, marker="x", markersize=12, markeredgecolor="r", linewidth=1.5, color="r")
ax.plot(StdUkappaR, EUkappaR, marker="x", markersize=12, markeredgecolor="r", linewidth=1.5, color="r")
ax.plot(StdUkappaE, EUkappaE, marker="x", markersize=12, markeredgecolor="r", linewidth=1.5, color="r")

ax.set_xlim(0.00085, 0.0035)
ax.set_ylim(0.0028, 0.006)
ax.tick_params(labelsize=12)

# 文字标注
ax.text(np.sqrt(VUgmv) + 0.00002, EUgmv - 0.00002, r'$\kappa=0$ (GMV)', fontsize=16)
ax.text(StdUkappaV + 0.00002, EUkappaV - 0.00002, r'$\kappa_V^\star=0.0147$', fontsize=16)
ax.text(StdUkappaR + 0.00002, EUkappaR - 0.00005, r'$\kappa_R^\star=0.0886$', fontsize=16)
ax.text(StdUkappaE - 0.0001, EUkappaE - 0.0001, r'$\kappa_E^\star=0.147$', fontsize=16)
ax.text(0.003, 0.0038, r'$\kappa=1$ (MV)', fontsize=16)

# 箭头注释
ax.annotate("",xy=(0.885, 0.26),xytext=(0.83, 0.34),xycoords="figure fraction", 
            textcoords="figure fraction",arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),)

ax.set_xlabel(r'样本外效用风险', fontsize=16)
ax.set_ylabel(r'样本外效用均值', fontsize=16)

fig.tight_layout()

figure_path = base_dir / "Figure1.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")

print(f"kappaV = {kappaV:.4f}")
print(f"kappaR = {kappaR:.4f}")
print(f"kappaE = {kappaE:.4f}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

np.random.seed(42)
os.makedirs("/Users/ayushi/Downloads/outputs", exist_ok=True)


df = pd.read_csv("AzureLLMInferenceTrace_conv.csv",
                 parse_dates=["TIMESTAMP"])
df = df.sort_values("TIMESTAMP").reset_index(drop=True)
df["hour"] = df["TIMESTAMP"].dt.hour

print(f"Rows: {len(df):,}")
print(f"Time range: {df.TIMESTAMP.min()} → {df.TIMESTAMP.max()}")
print(f"ContextTokens: mean={df.ContextTokens.mean():.0f}, "
      f"median={df.ContextTokens.median():.0f}, "
      f"p95={df.ContextTokens.quantile(0.95):.0f}, max={df.ContextTokens.max()}")
print(f"GeneratedTokens: mean={df.GeneratedTokens.mean():.0f}, "
      f"median={df.GeneratedTokens.median():.0f}, "
      f"p95={df.GeneratedTokens.quantile(0.95):.0f}, max={df.GeneratedTokens.max()}")
print()


BINS   = [0, 500, 1200, 3000, np.inf]
LABELS = ["S1·Short", "S2·Medium", "S3·Long", "S4·XLong"]
N_ST   = len(LABELS)
COLORS = ["#27AE60", "#F39C12", "#E74C3C", "#8E44AD"]

df["state"]     = pd.cut(df["ContextTokens"], bins=BINS, labels=LABELS, right=False)
df["state_idx"] = df["state"].cat.codes

counts = df["state"].value_counts().sort_index()
print("State distribution:")
for s, c in counts.items():
    print(f"  {s:12s}: {c:6,}  ({c/len(df)*100:.1f}%)")
print()

count_mat = np.zeros((N_ST, N_ST), dtype=int)
sv = df["state_idx"].values
for a, b in zip(sv[:-1], sv[1:]):
    count_mat[a][b] += 1

row_sums = count_mat.sum(axis=1, keepdims=True)
P = np.divide(count_mat, row_sums,
              out=np.zeros_like(count_mat, dtype=float),
              where=row_sums != 0)

P_df = pd.DataFrame(P, index=LABELS, columns=LABELS)
print("Transition Probability Matrix P:")
print(P_df.round(4))
print()


eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi  = np.real(eigenvectors[:, idx])
pi /= pi.sum()

print("Stationary Distribution :")
for label, prob in zip(LABELS, pi):
    print(f"  {label:12s}: {prob:.4f}  ({prob*100:.1f}%)")
print()

def mean_first_passage(P, pi):
    n  = P.shape[0]
    PI = np.outer(np.ones(n), pi)
    Z  = np.linalg.inv(np.eye(n) - P + PI)
    M  = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i][j] = (Z[j][j] - Z[i][j]) / pi[j] if i != j else 1.0 / pi[j]
    return M

MFPT    = mean_first_passage(P, pi)
MFPT_df = pd.DataFrame(MFPT, index=LABELS, columns=LABELS)
print("Mean First Passage Times (requests):")
print(MFPT_df.round(2))
print()


steps   = [1, 3, 5, 10, 20, 50]
forecasts = {}
P_pow = np.eye(N_ST)
sit = iter(steps); nxt = next(sit)
for k in range(1, max(steps)+1):
    P_pow = P_pow @ P
    if k == nxt:
        forecasts[k] = P_pow[0].copy()
        try: nxt = next(sit)
        except StopIteration: break

ms_df = pd.DataFrame(forecasts, index=LABELS).T
ms_df.index.name = "Step"
print("Multi-step forecast from S1·Short:")
print(ms_df.round(4))
print()


corr = df[["ContextTokens","GeneratedTokens"]].corr().iloc[0,1]
print(f"Pearson correlation (Context vs Generated tokens): {corr:.4f}")
print()
BASE = "/Users/ayushi/Downloads/outputs/"


fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(P_df, annot=True, fmt=".3f", cmap="Blues",
            linewidths=.6, ax=ax, cbar_kws={"label": "Probability"},
            annot_kws={"size": 11})
ax.set_title("KV-Cache State Transition Matrix\n(Azure LLM 2023 : conv)", fontsize=12)
ax.set_xlabel("Next Request State"); ax.set_ylabel("Current Request State")
plt.tight_layout()
plt.savefig(BASE+"03_transition_heatmap.png", dpi=150)
plt.close()


fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(MFPT_df, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=.6, ax=ax, cbar_kws={"label": "Requests"},
            annot_kws={"size": 11})
ax.set_title("Mean First Passage Times (requests)\nAzure LLM : conv service", fontsize=12)
ax.set_xlabel("Target State"); ax.set_ylabel("Origin State")
plt.tight_layout()
plt.savefig(BASE+"05_mfpt_heatmap.png", dpi=150)
plt.close()

]
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(steps)); w = 0.18
for k, (lbl, col) in enumerate(zip(LABELS, COLORS)):
    vals = [forecasts[s][k]*100 for s in steps]
    ax.bar(x + k*w, vals, w, label=lbl, color=col, edgecolor="white")
    ax.axhline(pi[k]*100, color=col, linestyle=":", linewidth=1.2, alpha=0.7)
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels([f"t+{s}" for s in steps])
ax.set_ylabel("Probability (%)")
ax.set_title("Multi-step State Forecast (starting from S1·Short)\nDotted lines = stationary values", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(BASE+"06_multistep_forecast.png", dpi=150)
plt.close()


hourly = df.groupby(["hour","state"]).size().unstack(fill_value=0)
hourly_pct = hourly.div(hourly.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(hourly_pct.T, cmap="Blues", ax=ax, linewidths=0.4,
            cbar_kws={"label": "% of requests"}, fmt=".0f", annot=True,
            annot_kws={"size": 8})
ax.set_title("Hourly KV-Cache State Distribution : Azure LLM conv (2023-11-16)", fontsize=11)
ax.set_xlabel("Hour of Day (UTC)"); ax.set_ylabel("KV-Cache State")
plt.tight_layout()
plt.savefig(BASE+"07_hourly_state_heatmap.png", dpi=150)
plt.close()


fig, ax = plt.subplots(figsize=(7, 5))
for lbl, col in zip(LABELS, COLORS):
    sub = df[df["state"]==lbl]
    ax.scatter(sub["ContextTokens"], sub["GeneratedTokens"],
               alpha=0.15, s=8, color=col, label=lbl, rasterized=True)
ax.set_xlabel("Context Tokens"); ax.set_ylabel("Generated Tokens")
ax.set_title(f"Context vs Generated Tokens (ρ = {corr:.3f})", fontsize=12)
ax.legend(fontsize=9, markerscale=3)
ax.set_xlim(0, 8000); ax.set_ylim(0, 1000)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(BASE+"08_context_vs_generated.png", dpi=150)
plt.close()

print("All 8 figures saved to outputs/")
print()


print("=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"1. Dominant state       : {LABELS[pi.argmax()]} ({pi.max()*100:.1f}% of all requests)")
print(f"2. Highest self-persist : {LABELS[P.diagonal().argmax()]} "
      f"(P={P.diagonal().max():.3f})")
print(f"3. MFPT to S4·XLong     : {MFPT[0,3]:.1f} reqs from Short, "
      f"{MFPT[1,3]:.1f} from Medium")
print(f"4. Convergence step     : system reaches stationarity by ~t+{steps[next(i for i,s in enumerate(steps) if all(abs(forecasts[s][j]-pi[j])<0.01 for j in range(N_ST)) )]} requests")
print(f"5. Token correlation    : {corr:.3f} (context & output weakly correlated)")

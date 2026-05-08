import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
N_BOOT = 1000

root_folder = os.path.join('..', 'outputs', 'run_20260507_144936')

# === Classification (stratified bootstrap) ===
df_cls = pd.read_csv(os.path.join(root_folder, 'classification_results.csv'))
y_true = (df_cls['ground_truth'] == 'melanoma').astype(int).values
y_pred = (df_cls['predicted'] == 'melanoma').astype(int).values
conf   = df_cls['confidence'].values

idx_mm = np.where(y_true == 1)[0]
idx_nv = np.where(y_true == 0)[0]

def metrics(yt, yp, c):
    tp = ((yt==1)&(yp==1)).sum()
    fn = ((yt==1)&(yp==0)).sum()
    fp = ((yt==0)&(yp==1)).sum()
    tn = ((yt==0)&(yp==0)).sum()
    acc  = (tp+tn) / len(yt)
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    prec = tp / max(tp+fp, 1)
    f1   = 2*prec*sens / max(prec+sens, 1e-9)
    # AUC via ranking on confidence (positive = melanoma)
    score = np.where(yp==1, c, 1-c)
    pos = score[yt==1]; neg = score[yt==0]
    auc = np.mean(pos[:,None] > neg[None,:]) + 0.5*np.mean(pos[:,None]==neg[None,:])
    return acc, sens, spec, prec, f1, auc

boot = []
for _ in range(N_BOOT):
    s_mm = RNG.choice(idx_mm, size=len(idx_mm), replace=True)
    s_nv = RNG.choice(idx_nv, size=len(idx_nv), replace=True)
    s = np.concatenate([s_mm, s_nv])
    boot.append(metrics(y_true[s], y_pred[s], conf[s]))
boot = np.array(boot)
ci = np.percentile(boot, [2.5, 97.5], axis=0)
print('CLS CI95:', dict(zip(['acc','sens','spec','prec','f1','auc'], ci.T.tolist())))

# === Qualitative + NLG (non-stratified bootstrap) ===
def boot_ci(values, n=N_BOOT, rng=np.random.default_rng(42)):
    vals = np.array(values)
    boot = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return np.percentile(boot, [2.5, 97.5])

df_q = pd.read_csv(os.path.join(root_folder, 'qualitative_scores.csv'))
for col in ['completeness','clinical_consistency','specificity',
            'gradcam_integration','actionability','disclaimer_present',
            'total','percentage']:
    print(col, boot_ci(df_q[col]))

df_n = pd.read_csv(os.path.join(root_folder, 'nlg_metrics.csv'))
for col in df_n.columns[1:]:
    print(col, boot_ci(df_n[col]))
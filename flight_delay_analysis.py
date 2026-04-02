import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

# =======================
# Configuration
# =======================
DATA_PATH  = r"C:\Users\Sanjali\Desktop\Flight\flight_data_2024.csv"
OUTPUT_DIR = r"C:\Users\Sanjali\Desktop\Flight\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG      = "#f8f9fa"
ACCENT  = "#e63946"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
})

# =======================
# Load Dataset
# =======================
print(f"[INFO] Loading dataset from:\n  {DATA_PATH}\n")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"[INFO] Loaded {len(df):,} rows x {df.shape[1]} columns")

# =======================
# Feature Engineering
# =======================
print("[INFO] Engineering features ...")

df['total_delay'] = (
    df['taxi_out'].fillna(0)
    + df['weather_delay'].fillna(0)
    + df['late_aircraft_delay'].fillna(0)
)
df['dep_hour']   = (df['dep_time'].fillna(0) // 100).clip(0, 23).astype(int)
df['is_delayed'] = (df['total_delay'] > 15).astype(int)
df['day_period'] = pd.cut(
    df['dep_hour'],
    bins=[-1, 5, 11, 17, 20, 23],
    labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"]
).astype(str)

FEATURES = [
    'dep_hour', 'day_of_week', 'month', 'distance',
    'taxi_out', 'air_time', 'cancelled',
    'weather_delay', 'late_aircraft_delay'
]
TARGET = 'is_delayed'

ml_df = df[FEATURES + [TARGET]].dropna()
print(f"[INFO] Clean ML dataset : {len(ml_df):,} rows")
print(f"[INFO] Delayed flights  : {df['is_delayed'].sum():,} ({df['is_delayed'].mean()*100:.1f}%)\n")

# =======================
# Train / Test Split
# =======================
X = ml_df[FEATURES]
y = ml_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =======================
# Logistic Regression
# =======================
print("[INFO] Training Logistic Regression ...")
lr = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
lr.fit(X_train_sc, y_train)
lr_pred  = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
lr_acc   = accuracy_score(y_test, lr_pred)
print(f"  Accuracy : {lr_acc*100:.2f}%")

# =======================
# Random Forest
# =======================
print("[INFO] Training Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=150, max_depth=12,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_acc   = accuracy_score(y_test, rf_pred)
print(f"  Accuracy : {rf_acc*100:.2f}%\n")

# Feature importance & coefficients dataframes
fi = pd.DataFrame({
    'feature':    FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance')

coef_df = pd.DataFrame({
    'feature': FEATURES,
    'coef':    lr.coef_[0]
}).sort_values('coef')

# Pre-compute ROC / AUC
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

# =======================
# Analysis aggregations
# =======================
top_airports = (
    df.groupby('origin')['total_delay'].mean()
    .reset_index()
    .rename(columns={'total_delay': 'avg_delay'})
    .sort_values('avg_delay', ascending=False)
    .head(15)
)

hour_delay = (
    df.groupby('dep_hour')['total_delay'].mean()
    .reset_index()
    .rename(columns={'total_delay': 'avg_delay'})
)
peak = hour_delay.loc[hour_delay['avg_delay'].idxmax()]

day_labels   = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}
heatmap_data = (
    df.groupby(['day_of_week', 'dep_hour'])['total_delay']
    .mean()
    .unstack(fill_value=0)
)
heatmap_data.index = [day_labels.get(i, str(i)) for i in heatmap_data.index]

period_delay = (
    df.groupby('day_period')['total_delay'].mean()
    .reset_index()
    .sort_values('total_delay', ascending=False)
)

# ================================================================
# CHART 1 ── ML Results (2 × 3)       →  02_ml_results.png
# ================================================================
print("[INFO] Generating 02_ml_results.png ...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
fig.suptitle("Machine Learning Results — Flight Delay Prediction",
             fontsize=16, fontweight='bold', y=1.01)

# ── Confusion Matrix – Logistic Regression ──────────────────────
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt=',d', cmap='Blues',
            xticklabels=['On-Time', 'Delayed'],
            yticklabels=['On-Time', 'Delayed'],
            ax=axes[0, 0], linewidths=1, linecolor='white',
            annot_kws={'size': 13, 'weight': 'bold'})
axes[0, 0].set_title(
    f'Logistic Regression\nConfusion Matrix (Acc≈{lr_acc*100:.1f}%)',
    fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# ── Confusion Matrix – Random Forest ────────────────────────────
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt=',d', cmap='Greens',
            xticklabels=['On-Time', 'Delayed'],
            yticklabels=['On-Time', 'Delayed'],
            ax=axes[0, 1], linewidths=1, linecolor='white',
            annot_kws={'size': 13, 'weight': 'bold'})
axes[0, 1].set_title(
    f'Random Forest\nConfusion Matrix (Acc≈{rf_acc*100:.1f}%)',
    fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# ── ROC Curves ──────────────────────────────────────────────────
axes[0, 2].plot(fpr_lr, tpr_lr,
                label=f'Logistic Reg  (AUC={auc_lr:.2f})',
                color='#3498db', lw=2)
axes[0, 2].plot(fpr_rf, tpr_rf,
                label=f'Random Forest (AUC={auc_rf:.2f})',
                color='#e67e22', lw=2)
axes[0, 2].plot([0, 1], [0, 1], '--', color='grey', lw=1)
axes[0, 2].set_title('ROC Curves — Both Models',
                     fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].legend(fontsize=9)
axes[0, 2].grid(alpha=0.3, linestyle='--')

# ── Feature Importance – Random Forest ──────────────────────────
sns.barplot(x='importance', y='feature', data=fi,
            ax=axes[1, 0], palette='summer')
axes[1, 0].set_title('Feature Importance (Random Forest)',
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].set_ylabel('')
for patch, val in zip(axes[1, 0].patches, fi['importance']):
    axes[1, 0].text(patch.get_width() + 0.005,
                    patch.get_y() + patch.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8)

# ── Logistic Regression Coefficients ────────────────────────────
bar_colors = ['#e74c3c' if c > 0 else '#27ae60' for c in coef_df['coef']]
sns.barplot(x='coef', y='feature', data=coef_df,
            ax=axes[1, 1], palette=bar_colors)
axes[1, 1].axvline(0, color='black', lw=0.8)
axes[1, 1].set_title(
    'Logistic Regression Coefficients\n(+ve = More Delay Risk)',
    fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Coefficient Value')
axes[1, 1].set_ylabel('')

# ── Model Performance Comparison ────────────────────────────────
scores = [lr_acc, rf_acc]
bars_perf = axes[1, 2].bar(
    ['Logistic Regression', 'Random Forest'], scores,
    color=['#3498db', '#27ae60'], edgecolor='white', linewidth=1.5
)
for b, v in zip(bars_perf, scores):
    axes[1, 2].text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.005,
                    f'{v:.3f}', ha='center',
                    fontsize=11, fontweight='bold')
axes[1, 2].set_title('Model Performance Comparison',
                     fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_ylim(0, 1.12)
axes[1, 2].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
out_ml = os.path.join(OUTPUT_DIR, '02_ml_results.png')
plt.savefig(out_ml, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  [✓] Saved → {out_ml}")

# ================================================================
# CHART 2 ── Delay Analysis (4-row grid)  →  01_delay_analysis_charts.png
# ================================================================
print("[INFO] Generating 01_delay_analysis_charts.png ...")

fig2 = plt.figure(figsize=(22, 28), facecolor=BG)
fig2.suptitle("Flight Delay Analysis 2024",
              fontsize=22, fontweight='bold', y=0.98)
gs = fig2.add_gridspec(4, 2, hspace=0.55, wspace=0.35)

# ── Delay by Hour (Line Chart, full width) ───────────────────────
ax1 = fig2.add_subplot(gs[0, :])
cmap_line = plt.cm.RdYlGn_r
norm      = plt.Normalize(hour_delay['avg_delay'].min(),
                          hour_delay['avg_delay'].max())
for i in range(len(hour_delay) - 1):
    ax1.plot(
        hour_delay['dep_hour'].iloc[i:i+2],
        hour_delay['avg_delay'].iloc[i:i+2],
        color=cmap_line(norm(hour_delay['avg_delay'].iloc[i])),
        linewidth=3
    )
ax1.fill_between(hour_delay['dep_hour'], hour_delay['avg_delay'],
                 alpha=0.15, color='#0f4c81')
sc = ax1.scatter(hour_delay['dep_hour'], hour_delay['avg_delay'],
                 c=hour_delay['avg_delay'], cmap='RdYlGn_r',
                 s=80, zorder=5, edgecolors='white', linewidths=1)
plt.colorbar(sc, ax=ax1, label='Avg Delay (min)', fraction=0.02)
ax1.annotate(
    f"  Peak: {int(peak['dep_hour']):02d}:00\n  {peak['avg_delay']:.1f} min",
    xy=(peak['dep_hour'], peak['avg_delay']),
    xytext=(peak['dep_hour'] + 1.5, peak['avg_delay'] + 1.5),
    fontsize=9, color=ACCENT, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.5)
)
ax1.set_title('Average Departure Delay by Hour of Day',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Avg Delay (minutes)')
ax1.set_xticks(range(0, 24))
ax1.set_xticklabels([f'{h:02d}:00' for h in range(24)],
                    rotation=45, fontsize=8)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# ── Top 15 Airports Bar Chart (full width) ───────────────────────
ax2 = fig2.add_subplot(gs[1, :])
top15       = top_airports.sort_values('avg_delay')
colors_bar  = plt.cm.OrRd(np.linspace(0.3, 0.9, len(top15)))
bars2       = ax2.barh(top15['origin'], top15['avg_delay'],
                       color=colors_bar, edgecolor='white', height=0.7)
for bar, val in zip(bars2, top15['avg_delay']):
    ax2.text(bar.get_width() + 0.3,
             bar.get_y() + bar.get_height() / 2,
             f'{val:.1f}m', va='center', fontsize=9, fontweight='bold')
ax2.set_title('Top 15 Airports by Average Flight Delay',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Average Delay (minutes)')
ax2.set_ylabel('Airport Code')
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# ── Heatmap Day × Hour (full width) ──────────────────────────────
ax3 = fig2.add_subplot(gs[2, :])
sns.heatmap(heatmap_data, ax=ax3, cmap='YlOrRd',
            linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'Avg Delay (min)', 'shrink': 0.8})
ax3.set_title(
    'Heatmap: Day of Week × Hour of Day → Average Delay',
    fontsize=14, fontweight='bold')
ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Day of Week')
hour_ticks = list(range(0, 24, 2))
ax3.set_xticks([x + 0.5 for x in hour_ticks])
ax3.set_xticklabels([f'{h:02d}:00' for h in hour_ticks],
                    rotation=45, fontsize=8)

# ── Delay by Period Pie ───────────────────────────────────────────
ax4 = fig2.add_subplot(gs[3, 0])
period_order  = ['Morning', 'Afternoon', 'Evening', 'Late Night', 'Night']
period_subset = period_delay[period_delay['day_period'].isin(period_order)]
wedge_colors  = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#3498db']
wedges, texts, autotexts = ax4.pie(
    period_subset['total_delay'],
    labels=period_subset['day_period'],
    autopct='%1.1f%%',
    colors=wedge_colors[:len(period_subset)],
    startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')
ax4.set_title('Average Delay Distribution\nby Day Period',
              fontsize=13, fontweight='bold')

# ── Flight Status Breakdown ───────────────────────────────────────
ax5 = fig2.add_subplot(gs[3, 1])
labels5  = ['On-Time\n(≤15 min)', 'Delayed\n(>15 min)', 'Cancelled']
values5  = [
    (df['is_delayed'] == 0).sum(),
    df['is_delayed'].sum(),
    int(df['cancelled'].sum())
]
bcolors5 = ['#27ae60', '#e67e22', '#c0392b']
bars5 = ax5.bar(labels5, values5, color=bcolors5,
                edgecolor='white', linewidth=2, width=0.6)
for b, v in zip(bars5, values5):
    ax5.text(b.get_x() + b.get_width() / 2,
             b.get_height() + 2000,
             f'{v:,}\n({v / len(df) * 100:.1f}%)',
             ha='center', fontsize=9, fontweight='bold')
ax5.set_title('Flight Status Breakdown',
              fontsize=13, fontweight='bold')
ax5.set_ylabel('Number of Flights')
ax5.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax5.grid(axis='y', alpha=0.3, linestyle='--')

out_delay = os.path.join(OUTPUT_DIR, '01_delay_analysis_charts.png')
plt.savefig(out_delay, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  [✓] Saved → {out_delay}")

# =======================
# Final Summary
# =======================
print(f"""
╔══════════════════════════════════════════════════════════════╗
║           FLIGHT DELAY ANALYSIS 2024 — COMPLETE             ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset           │ {len(df):>10,} flights                  ║
║  Delayed (>15 min) │ {df['is_delayed'].sum():>10,} ({df['is_delayed'].mean()*100:.1f}%)              ║
║  Most Delayed Arpt │ {top_airports.iloc[0]['origin']} ({top_airports.iloc[0]['avg_delay']:.1f} min avg)              ║
║  Peak Delay Hour   │ {int(peak['dep_hour']):02d}:00 ({peak['avg_delay']:.1f} min avg)              ║
╠══════════════════════════════════════════════════════════════╣
║  Logistic Regression  Accuracy={lr_acc*100:.2f}%  AUC={auc_lr:.3f}       ║
║  Random Forest        Accuracy={rf_acc*100:.2f}%  AUC={auc_rf:.3f}       ║
╠══════════════════════════════════════════════════════════════╣
║  SAVED TO: {OUTPUT_DIR:<50s}║
║    01_delay_analysis_charts.png                              ║
║    02_ml_results.png                                         ║
╚══════════════════════════════════════════════════════════════╝
""")
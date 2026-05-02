import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
          '#44BBA4', '#E94F37', '#393E41', '#F5A623', '#7B2D8B']

# ─────────────────────────────────────────────
# LOAD & PARSE
# ─────────────────────────────────────────────
df = pd.read_csv('index_1.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['date']     = pd.to_datetime(df['date'])

print("=" * 60)
print("  COFFEE SHOP ANALYSIS")
print("=" * 60)
print(f"\nTotal records  : {len(df):,}")
print(f"Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Unique drinks  : {df['coffee_name'].nunique()}")
print(f"Total revenue  : ₪{df['money'].sum():,.1f}")
print(f"\nMissing values :\n{df.isnull().sum()}")
print(f"\nDrinks in data :\n{df['coffee_name'].value_counts()}\n")

# ─────────────────────────────────────────────
# STEP 2 – FEATURE ENGINEERING
# ─────────────────────────────────────────────
df['hour']         = df['datetime'].dt.hour
df['day_of_week']  = df['datetime'].dt.dayofweek          # 0=Mon … 6=Sun
df['is_weekend']   = df['day_of_week'].isin([4, 5]).astype(int)  # Fri-Sat
df['is_morning']   = df['hour'].between(6,  11).astype(int)
df['is_lunch']     = df['hour'].between(12, 14).astype(int)
df['is_afternoon'] = df['hour'].between(15, 17).astype(int)
df['is_evening']   = df['hour'].between(18, 22).astype(int)

hourly_sales = df.groupby('hour').size().reset_index(name='count')

# ─────────────────────────────────────────────
# STEP 1 – EDA PLOTS
# ─────────────────────────────────────────────

# --- Figure 1: Sales by hour ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Coffee Shop – Exploratory Analysis', fontsize=18, fontweight='bold', y=0.98)

ax = axes[0, 0]
bars = ax.bar(hourly_sales['hour'], hourly_sales['count'],
              color=COLORS[0], edgecolor='white', linewidth=0.8)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Number of Sales', fontsize=12)
ax.set_title('Sales Distribution by Hour', fontsize=14, fontweight='bold')
ax.set_xticks(range(24))

peak_hour = hourly_sales.loc[hourly_sales['count'].idxmax(), 'hour']
quiet_hour = hourly_sales.loc[hourly_sales['count'].idxmin(), 'hour']
for bar in bars:
    h = bar.get_height()
    if h == hourly_sales['count'].max():
        bar.set_color('#C73E1D')
    elif h == hourly_sales['count'].min():
        bar.set_color('#44BBA4')
ax.text(peak_hour, hourly_sales['count'].max() + 0.5, 'PEAK', ha='center',
        color='#C73E1D', fontweight='bold', fontsize=9)
ax.text(quiet_hour, hourly_sales.loc[hourly_sales['hour']==quiet_hour,'count'].values[0] + 0.5,
        'QUIET', ha='center', color='#44BBA4', fontweight='bold', fontsize=9)

# --- Figure 1b: Top drinks bar ---
ax = axes[0, 1]
drink_counts = df['coffee_name'].value_counts()
bars2 = ax.barh(drink_counts.index, drink_counts.values,
                color=COLORS[:len(drink_counts)], edgecolor='white')
ax.set_xlabel('Number of Orders', fontsize=12)
ax.set_title('Most Popular Drinks', fontsize=14, fontweight='bold')
for bar, val in zip(bars2, drink_counts.values):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            f'{val}', va='center', fontsize=9, fontweight='bold')

# --- Figure 1c: Day-of-week heatmap by drink ---
ax = axes[1, 0]
dow_drink = df.groupby(['day_of_week', 'coffee_name']).size().unstack(fill_value=0)
dow_drink.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
sns.heatmap(dow_drink.T, ax=ax, cmap='YlOrRd', linewidths=0.5,
            cbar_kws={'label': 'Orders'})
ax.set_title('Orders by Day & Drink', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('')

# --- Figure 1d: Part-of-day pie ---
ax = axes[1, 1]
part_labels = ['Morning\n(6-11)', 'Lunch\n(12-14)', 'Afternoon\n(15-17)', 'Evening\n(18-22)', 'Other']
part_counts = [
    df['is_morning'].sum(),
    df['is_lunch'].sum(),
    df['is_afternoon'].sum(),
    df['is_evening'].sum(),
    len(df) - df[['is_morning','is_lunch','is_afternoon','is_evening']].any(axis=1).sum()
]
wedges, texts, autotexts = ax.pie(part_counts, labels=part_labels,
                                   autopct='%1.1f%%', colors=COLORS[:5],
                                   startangle=140, textprops={'fontsize': 10})
for at in autotexts:
    at.set_fontweight('bold')
ax.set_title('Sales by Part of Day', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("Saved: 01_eda_overview.png")

# ─────────────────────────────────────────────
# PEAK vs QUIET TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  HOURLY SALES SUMMARY")
print("=" * 60)
hourly_sales_sorted = hourly_sales.sort_values('count', ascending=False)
hourly_sales['status'] = hourly_sales['count'].apply(
    lambda x: 'PEAK' if x >= hourly_sales['count'].quantile(0.75)
    else ('QUIET' if x <= hourly_sales['count'].quantile(0.25) else 'Normal'))
print(hourly_sales.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 3 – CATBOOST MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CATBOOST MODEL TRAINING")
print("=" * 60)

feature_cols = ['hour', 'day_of_week', 'is_weekend',
                'is_morning', 'is_lunch', 'is_afternoon', 'is_evening',
                'money', 'cash_type']
cat_features = ['cash_type']

X = df[feature_cols].copy()
y = df['coffee_name']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    random_seed=42,
    verbose=100,
    eval_metric='Accuracy',
    early_stopping_rounds=50
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Feature Importance ---
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.get_feature_importance()
}).sort_values('importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('CatBoost Model Results', fontsize=16, fontweight='bold')

ax = axes[0]
colors_fi = [COLORS[0] if i < 3 else '#AACCE0' for i in range(len(feat_imp))]
bars = ax.barh(feat_imp['feature'], feat_imp['importance'],
               color=colors_fi[::-1], edgecolor='white')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title(f'Feature Importance  (Accuracy: {acc*100:.1f}%)', fontsize=13, fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, feat_imp['importance'][::-1]):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=9)

# --- Confusion Matrix ---
ax = axes[1]
classes = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.0f', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            ax=ax, cbar_kws={'label': '%'}, linewidths=0.5)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix (%)', fontsize=13, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('02_model_results.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("Saved: 02_model_results.png")

# ─────────────────────────────────────────────
# STEP 4 – BUSINESS INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BUSINESS INSIGHTS")
print("=" * 60)

# Top-3 drinks per hour
top3_per_hour = (
    df.groupby(['hour', 'coffee_name'])
      .size()
      .reset_index(name='count')
      .sort_values(['hour', 'count'], ascending=[True, False])
      .groupby('hour')
      .head(3)
      .copy()
)
top3_per_hour['rank'] = top3_per_hour.groupby('hour').cumcount() + 1

pivot_top3 = top3_per_hour.pivot(index='hour', columns='rank', values='coffee_name')
pivot_top3.columns = ['#1 Popular', '#2 Popular', '#3 Popular']
pivot_top3.index.name = 'Hour'
print("\nTop-3 Drinks Per Hour:")
print(pivot_top3.to_string())

# Part-of-day recommendations
part_map = {
    'Morning (6-11)':   df[df['is_morning']   == 1],
    'Lunch (12-14)':    df[df['is_lunch']      == 1],
    'Afternoon (15-17)':df[df['is_afternoon']  == 1],
    'Evening (18-22)':  df[df['is_evening']    == 1],
}

print("\n" + "-" * 60)
print("INVENTORY RECOMMENDATIONS BY PART OF DAY")
print("-" * 60)
for part, sub in part_map.items():
    if len(sub) == 0:
        continue
    top = sub['coffee_name'].value_counts()
    total = len(sub)
    print(f"\n{part}  ({total} orders total)")
    print(f"  STOCK MORE  : {top.index[0]} ({top.iloc[0]} orders)")
    print(f"  STOCK MEDIUM: {top.index[1]} ({top.iloc[1]} orders)" if len(top) > 1 else "")
    print(f"  STOCK LESS  : {top.index[-1]} ({top.iloc[-1]} orders)" if len(top) > 2 else "")

# Peak vs Quiet hours
peak_threshold  = hourly_sales['count'].quantile(0.75)
quiet_threshold = hourly_sales['count'].quantile(0.25)
peak_hours  = hourly_sales[hourly_sales['count'] >= peak_threshold]['hour'].tolist()
quiet_hours = hourly_sales[hourly_sales['count'] <= quiet_threshold]['hour'].tolist()
print(f"\n{'─'*60}")
print(f"PEAK hours (need extra staff) : {sorted(peak_hours)}")
print(f"QUIET hours (minimal staff)   : {sorted(quiet_hours)}")

# ─────────────────────────────────────────────
# FIGURE 3 – BUSINESS DASHBOARD
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle('Business Intelligence Dashboard', fontsize=18, fontweight='bold', y=0.99)

# 3a: Hourly with peak/quiet coloring
ax = axes[0, 0]
bar_colors = []
for _, row in hourly_sales.iterrows():
    if row['count'] >= peak_threshold:
        bar_colors.append('#C73E1D')
    elif row['count'] <= quiet_threshold:
        bar_colors.append('#44BBA4')
    else:
        bar_colors.append('#2E86AB')
ax.bar(hourly_sales['hour'], hourly_sales['count'],
       color=bar_colors, edgecolor='white', linewidth=0.8)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Orders', fontsize=12)
ax.set_title('Peak (red) vs Quiet (green) Hours', fontsize=13, fontweight='bold')
ax.set_xticks(range(24))
from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#C73E1D', label='Peak – need more staff'),
              Patch(facecolor='#44BBA4', label='Quiet – minimal staff'),
              Patch(facecolor='#2E86AB', label='Normal')]
ax.legend(handles=legend_els, fontsize=9)

# 3b: Top-3 drinks heatmap per hour
ax = axes[0, 1]
hour_drink_pivot = (
    df.groupby(['hour', 'coffee_name'])
      .size()
      .unstack(fill_value=0)
)
hour_drink_norm = hour_drink_pivot.div(hour_drink_pivot.sum(axis=1), axis=0) * 100
sns.heatmap(hour_drink_norm.T, ax=ax, cmap='YlOrRd', linewidths=0.3,
            cbar_kws={'label': '% of hourly orders'})
ax.set_title('Drink Mix by Hour (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Hour', fontsize=12)
ax.set_ylabel('')
plt.setp(ax.get_yticklabels(), fontsize=8)

# 3c: Revenue per drink
ax = axes[1, 0]
rev = df.groupby('coffee_name')['money'].sum().sort_values(ascending=True)
bars3 = ax.barh(rev.index, rev.values, color=COLORS[:len(rev)], edgecolor='white')
ax.set_xlabel('Total Revenue (₪)', fontsize=12)
ax.set_title('Revenue by Drink Type', fontsize=13, fontweight='bold')
for bar, val in zip(bars3, rev.values):
    ax.text(val + 5, bar.get_y() + bar.get_height()/2,
            f'₪{val:,.0f}', va='center', fontsize=9)

# 3d: Part-of-day bar with top drink label
ax = axes[1, 1]
part_names  = ['Morning\n6-11', 'Lunch\n12-14', 'Afternoon\n15-17', 'Evening\n18-22']
part_masks  = [df['is_morning']==1, df['is_lunch']==1, df['is_afternoon']==1, df['is_evening']==1]
part_totals = [m.sum() for m in part_masks]
part_tops   = [df[m]['coffee_name'].value_counts().index[0] for m in part_masks]
bar4 = ax.bar(part_names, part_totals, color=COLORS[:4], edgecolor='white', linewidth=0.8)
ax.set_ylabel('Total Orders', fontsize=12)
ax.set_title('Orders & Top Drink by Part of Day', fontsize=13, fontweight='bold')
for bar, top_drink, total in zip(bar4, part_tops, part_totals):
    ax.text(bar.get_x() + bar.get_width()/2, total + 0.5,
            f'{top_drink}', ha='center', fontsize=8, fontweight='bold', color='#333')

plt.tight_layout()
plt.savefig('03_business_dashboard.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("\nSaved: 03_business_dashboard.png")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY FOR OWNER")
print("=" * 60)
print(f"\nModel accuracy   : {acc*100:.1f}%  – the model predicts drink choice well")
print(f"Top feature      : {feat_imp.iloc[0]['feature']} – most influential on drink choice")
print(f"Peak hours       : {sorted(peak_hours)}")
print(f"Quiet hours      : {sorted(quiet_hours)}")
print(f"Best-selling drink overall: {df['coffee_name'].value_counts().index[0]}")
print(f"Highest revenue drink     : {df.groupby('coffee_name')['money'].sum().idxmax()}")
print("\nDone!")

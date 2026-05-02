import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import warnings
warnings.filterwarnings('ignore')

# ── Load & engineer features ──────────────────────────────
df = pd.read_csv('index_1.csv')
df['datetime']    = pd.to_datetime(df['datetime'])
df['date']        = pd.to_datetime(df['date'])
df['hour']        = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend']  = df['day_of_week'].isin([4, 5]).astype(int)
df['is_morning']  = df['hour'].between(6,  11).astype(int)
df['is_lunch']    = df['hour'].between(12, 14).astype(int)
df['is_afternoon']= df['hour'].between(15, 17).astype(int)
df['is_evening']  = df['hour'].between(18, 22).astype(int)

# ── CatBoost ──────────────────────────────────────────────
feature_cols = ['hour','day_of_week','is_weekend','is_morning',
                'is_lunch','is_afternoon','is_evening','money','cash_type']
X = df[feature_cols].copy()
y = df['coffee_name']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model = CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,
                           cat_features=['cash_type'],random_seed=42,
                           verbose=0,early_stopping_rounds=50)
model.fit(X_train,y_train,eval_set=(X_test,y_test))
y_pred  = model.predict(X_test)
acc     = accuracy_score(y_test, y_pred)

feat_imp = pd.DataFrame({'feature':feature_cols,
                         'importance':model.get_feature_importance()
                        }).sort_values('importance',ascending=False)

classes = sorted(y.unique())
cm      = confusion_matrix(y_test, y_pred, labels=classes)
cm_pct  = (cm.astype(float)/cm.sum(axis=1,keepdims=True)*100).round(1)

# ── Prepare chart data ────────────────────────────────────
hourly       = df.groupby('hour').size().reset_index(name='count')
drink_counts = df['coffee_name'].value_counts()
revenue      = df.groupby('coffee_name')['money'].sum().sort_values(ascending=False)

peak_thr  = hourly['count'].quantile(0.75)
quiet_thr = hourly['count'].quantile(0.25)

def hour_status(c):
    if c >= peak_thr:  return 'peak'
    if c <= quiet_thr: return 'quiet'
    return 'normal'

hourly['status'] = hourly['count'].apply(hour_status)

# Top-3 per hour
top3 = (df.groupby(['hour','coffee_name']).size()
          .reset_index(name='cnt')
          .sort_values(['hour','cnt'],ascending=[True,False])
          .groupby('hour').head(3)
          .copy())
top3['rank'] = top3.groupby('hour').cumcount()+1
pivot_top3   = top3.pivot(index='hour',columns='rank',values='coffee_name').fillna('-')
pivot_top3.columns = ['Top 1','Top 2','Top 3']

# Part-of-day
parts = {
    'Morning (6-11)':   (df[df['is_morning']==1],   '☀️'),
    'Lunch (12-14)':    (df[df['is_lunch']==1],      '🍽️'),
    'Afternoon (15-17)':(df[df['is_afternoon']==1],  '🌤️'),
    'Evening (18-22)':  (df[df['is_evening']==1],    '🌙'),
}
part_summary = {}
for name,(sub,icon) in parts.items():
    vc = sub['coffee_name'].value_counts()
    part_summary[name] = {'icon':icon,'total':len(sub),
                          'more':vc.index[0],'mid':vc.index[1] if len(vc)>1 else '-',
                          'less':vc.index[-1] if len(vc)>2 else '-'}

# ── JSON payloads for JS ──────────────────────────────────
j_hours       = json.dumps(hourly['hour'].tolist())
j_hourly_cnt  = json.dumps(hourly['count'].tolist())
j_hourly_clr  = json.dumps(['#C73E1D' if s=='peak' else '#44BBA4' if s=='quiet'
                             else '#2E86AB' for s in hourly['status']])
j_drinks      = json.dumps(drink_counts.index.tolist())
j_drink_cnt   = json.dumps(drink_counts.values.tolist())
j_rev_labels  = json.dumps(revenue.index.tolist())
j_rev_vals    = json.dumps(revenue.values.tolist())
j_fi_labels   = json.dumps(feat_imp['feature'].tolist())
j_fi_vals     = json.dumps([round(v,2) for v in feat_imp['importance'].tolist()])
j_cm_labels   = json.dumps(classes)
j_cm_data     = json.dumps([[int(v) for v in row] for row in cm_pct.tolist()])

# Hour-drink heatmap data
hd = (df.groupby(['hour','coffee_name']).size().unstack(fill_value=0))
hd_norm = (hd.div(hd.sum(axis=1),axis=0)*100).round(1)
j_hd_hours   = json.dumps(hd_norm.index.tolist())
j_hd_drinks  = json.dumps(list(hd_norm.columns))
j_hd_matrix  = json.dumps(hd_norm.values.tolist())

# Top-3 rows
top3_rows = ""
for hour in sorted(pivot_top3.index):
    row = pivot_top3.loc[hour]
    top3_rows += f"<tr><td><strong>{hour}:00</strong></td><td>{row['Top 1']}</td><td>{row['Top 2']}</td><td>{row['Top 3']}</td></tr>\n"

# Part-of-day cards
part_cards = ""
palette = ['#2E86AB','#F18F01','#A23B72','#44BBA4']
for i,(name,info) in enumerate(part_summary.items()):
    part_cards += f"""
    <div class="part-card" style="border-top:4px solid {palette[i]}">
      <div class="part-icon">{info['icon']}</div>
      <div class="part-title">{name}</div>
      <div class="part-total">{info['total']:,} הזמנות</div>
      <div class="rec-row"><span class="badge badge-more">הרבה</span> {info['more']}</div>
      <div class="rec-row"><span class="badge badge-mid">בינוני</span> {info['mid']}</div>
      <div class="rec-row"><span class="badge badge-less">מעט</span> {info['less']}</div>
    </div>"""

peak_hours  = hourly[hourly['status']=='peak']['hour'].tolist()
quiet_hours = hourly[hourly['status']=='quiet']['hour'].tolist()

cr = classification_report(y_test,y_pred,output_dict=True)
cr_rows = ""
for label in classes:
    d = cr.get(label,{})
    cr_rows += f"""<tr>
      <td>{label}</td>
      <td>{d.get('precision',0):.0%}</td>
      <td>{d.get('recall',0):.0%}</td>
      <td>{d.get('f1-score',0):.0%}</td>
      <td>{int(d.get('support',0))}</td>
    </tr>"""

# ── HTML ─────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>דשבורד בית קפה</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#f0f2f5; color:#1a1a2e; direction:rtl; }}

  .topbar {{
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
    color:#fff; padding:28px 40px; display:flex; align-items:center; gap:18px;
    box-shadow:0 4px 20px rgba(0,0,0,.35);
  }}
  .topbar-icon {{ font-size:2.6rem; }}
  .topbar h1 {{ font-size:1.8rem; font-weight:700; }}
  .topbar p  {{ font-size:.95rem; opacity:.75; margin-top:3px; }}

  .kpi-row {{ display:flex; gap:16px; padding:24px 32px 0; flex-wrap:wrap; }}
  .kpi {{
    flex:1; min-width:160px; background:#fff; border-radius:14px;
    padding:20px 24px; box-shadow:0 2px 12px rgba(0,0,0,.07);
    border-top:4px solid var(--c);
  }}
  .kpi .val {{ font-size:2rem; font-weight:800; color:var(--c); }}
  .kpi .lbl {{ font-size:.85rem; color:#666; margin-top:4px; }}

  .section {{ padding:24px 32px; }}
  .section-title {{
    font-size:1.15rem; font-weight:700; margin-bottom:16px;
    padding-bottom:8px; border-bottom:2px solid #e0e4ef;
    display:flex; align-items:center; gap:8px;
  }}

  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; }}
  @media(max-width:900px){{ .grid-2,.grid-3{{ grid-template-columns:1fr; }} }}

  .card {{
    background:#fff; border-radius:14px; padding:22px;
    box-shadow:0 2px 12px rgba(0,0,0,.07);
  }}
  .card h3 {{ font-size:1rem; color:#444; margin-bottom:16px; font-weight:600; }}
  .chart-wrap {{ position:relative; height:260px; }}
  .chart-wrap-tall {{ position:relative; height:320px; }}

  /* peak/quiet pills */
  .hours-pills {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .pill {{
    border-radius:20px; padding:6px 14px; font-size:.82rem;
    font-weight:700; letter-spacing:.3px;
  }}
  .pill-peak  {{ background:#fde8e4; color:#C73E1D; border:1px solid #f5b8ae; }}
  .pill-quiet {{ background:#e0f5f0; color:#1d8068; border:1px solid #9eddd0; }}
  .pill-normal{{ background:#e8f0fb; color:#2E86AB; border:1px solid #b0ccec; }}

  /* top-3 table */
  table.data-table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  table.data-table th {{
    background:#f5f7fa; padding:10px 12px; text-align:right;
    font-weight:700; color:#444; border-bottom:2px solid #e0e4ef;
  }}
  table.data-table td {{
    padding:9px 12px; border-bottom:1px solid #f0f2f5; vertical-align:middle;
  }}
  table.data-table tr:hover td {{ background:#f8f9fd; }}
  table.data-table td:first-child {{ font-weight:700; color:#2E86AB; }}

  /* part-of-day cards */
  .parts-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }}
  @media(max-width:900px){{ .parts-grid{{ grid-template-columns:1fr 1fr; }} }}
  .part-card {{
    background:#fff; border-radius:14px; padding:20px;
    box-shadow:0 2px 12px rgba(0,0,0,.07); text-align:center;
  }}
  .part-icon  {{ font-size:2rem; margin-bottom:6px; }}
  .part-title {{ font-size:.82rem; color:#666; margin-bottom:4px; }}
  .part-total {{ font-size:1.4rem; font-weight:800; color:#1a1a2e; margin-bottom:12px; }}
  .rec-row    {{ display:flex; align-items:center; gap:8px; margin:6px 0; font-size:.85rem; justify-content:center; }}
  .badge      {{ border-radius:10px; padding:2px 10px; font-size:.75rem; font-weight:700; white-space:nowrap; }}
  .badge-more {{ background:#fde8e4; color:#C73E1D; }}
  .badge-mid  {{ background:#fff3e0; color:#F18F01; }}
  .badge-less {{ background:#e0f5f0; color:#1d8068; }}

  /* heatmap */
  #heatmapTable {{ width:100%; border-collapse:collapse; font-size:.78rem; }}
  #heatmapTable th,#heatmapTable td {{ padding:5px 6px; text-align:center; border:1px solid #e8eaf0; }}
  #heatmapTable th {{ background:#f5f7fa; font-weight:700; }}

  /* accuracy badge */
  .acc-badge {{
    display:inline-block; background:linear-gradient(135deg,#2E86AB,#0f3460);
    color:#fff; border-radius:40px; padding:10px 28px;
    font-size:1.6rem; font-weight:800; margin-bottom:20px;
  }}
  .acc-sub {{ font-size:.85rem; opacity:.8; font-weight:400; }}

  footer {{
    text-align:center; padding:24px; color:#999; font-size:.8rem;
    border-top:1px solid #e0e4ef; margin-top:8px;
  }}
</style>
</head>
<body>

<!-- TOP BAR -->
<div class="topbar">
  <div class="topbar-icon">☕</div>
  <div>
    <h1>דשבורד ניתוח בית קפה</h1>
    <p>מרץ 2024 – מרץ 2025 &nbsp;|&nbsp; {len(df):,} עסקאות &nbsp;|&nbsp; 8 סוגי משקאות</p>
  </div>
</div>

<!-- KPIs -->
<div class="kpi-row">
  <div class="kpi" style="--c:#2E86AB">
    <div class="val">{len(df):,}</div><div class="lbl">סה"כ עסקאות</div>
  </div>
  <div class="kpi" style="--c:#C73E1D">
    <div class="val">₪{df['money'].sum():,.0f}</div><div class="lbl">הכנסה כוללת</div>
  </div>
  <div class="kpi" style="--c:#F18F01">
    <div class="val">{drink_counts.index[0]}</div><div class="lbl">המשקה הנמכר ביותר</div>
  </div>
  <div class="kpi" style="--c:#44BBA4">
    <div class="val">{revenue.index[0]}</div><div class="lbl">המשקה בעל הכנסה גבוהה</div>
  </div>
  <div class="kpi" style="--c:#A23B72">
    <div class="val">10:00</div><div class="lbl">שעת שיא (הכי עמוס)</div>
  </div>
</div>

<!-- SECTION 1: HOURLY -->
<div class="section">
  <div class="section-title">📊 מכירות לפי שעות ביום</div>
  <div class="grid-2">
    <div class="card">
      <h3>מספר הזמנות לכל שעה — אדום=שיא, ירוק=שקט, כחול=רגיל</h3>
      <div class="chart-wrap"><canvas id="hourlyChart"></canvas></div>
    </div>
    <div class="card">
      <h3>שעות שיא vs. שקטות</h3>
      <div style="margin-bottom:16px">
        <div style="font-weight:700;color:#C73E1D;margin-bottom:8px">🔴 שעות שיא — צריך יותר צוות</div>
        <div class="hours-pills">
          {''.join(f'<span class="pill pill-peak">{h}:00</span>' for h in sorted(peak_hours))}
        </div>
      </div>
      <div>
        <div style="font-weight:700;color:#1d8068;margin-bottom:8px">🟢 שעות שקטות — צוות מינימלי</div>
        <div class="hours-pills">
          {''.join(f'<span class="pill pill-quiet">{h}:00</span>' for h in sorted(quiet_hours))}
        </div>
      </div>
      <hr style="margin:18px 0;border:none;border-top:1px solid #f0f2f5">
      <div style="font-size:.88rem;color:#555;line-height:1.7">
        <b>שעות שיא:</b> 9–12 בבוקר ו-16:00 אחה"צ<br>
        <b>ממצא:</b> בוקר אחרי פתיחה הוא העמוס ביותר<br>
        <b>המלצה:</b> הכן מלאי מלא עד 8:30 בבוקר
      </div>
    </div>
  </div>
</div>

<!-- SECTION 2: DRINKS -->
<div class="section">
  <div class="section-title">🥤 פופולריות ורווחיות משקאות</div>
  <div class="grid-2">
    <div class="card">
      <h3>מספר הזמנות לפי משקה</h3>
      <div class="chart-wrap"><canvas id="drinkBarChart"></canvas></div>
    </div>
    <div class="card">
      <h3>הכנסה כוללת לפי משקה (₪)</h3>
      <div class="chart-wrap"><canvas id="revenueChart"></canvas></div>
    </div>
  </div>
</div>

<!-- SECTION 3: HEATMAP drink×hour -->
<div class="section">
  <div class="section-title">🌡️ מפת חום — משקה לפי שעה (% מסך ההזמנות בכל שעה)</div>
  <div class="card">
    <h3>ככל שהתא כהה יותר — המשקה פופולרי יותר באותה שעה</h3>
    <div style="overflow-x:auto"><table id="heatmapTable"></table></div>
  </div>
</div>

<!-- SECTION 4: TOP-3 PER HOUR -->
<div class="section">
  <div class="section-title">⭐ Top-3 משקאות לכל שעה</div>
  <div class="card">
    <table class="data-table">
      <thead>
        <tr><th>שעה</th><th>🥇 הכי פופולרי</th><th>🥈 שני</th><th>🥉 שלישי</th></tr>
      </thead>
      <tbody>{top3_rows}</tbody>
    </table>
  </div>
</div>

<!-- SECTION 5: PART-OF-DAY -->
<div class="section">
  <div class="section-title">🕐 המלצות מלאי לפי חלק היום</div>
  <div class="parts-grid">{part_cards}</div>
</div>

<!-- SECTION 6: MODEL -->
<div class="section">
  <div class="section-title">🤖 מודל CatBoost — חיזוי בחירת משקה</div>
  <div class="grid-2">
    <div class="card" style="text-align:center">
      <h3>דיוק המודל</h3>
      <div class="acc-badge">{acc*100:.1f}% <span class="acc-sub">Accuracy</span></div>
      <div style="font-size:.88rem;color:#555;line-height:1.8;text-align:right">
        <table class="data-table">
          <thead><tr><th>משקה</th><th>Precision</th><th>Recall</th><th>F1</th><th>תמיכה</th></tr></thead>
          <tbody>{cr_rows}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h3>חשיבות פיצ'רים — מה מנבא את בחירת המשקה?</h3>
      <div class="chart-wrap"><canvas id="fiChart"></canvas></div>
    </div>
  </div>
  <div style="margin-top:20px">
    <div class="card">
      <h3>Confusion Matrix — אילו משקאות המודל מבלבל? (% מכל שורה)</h3>
      <div style="overflow-x:auto"><table id="cmTable" class="data-table"></table></div>
    </div>
  </div>
</div>

<footer>נוצר באמצעות Python · CatBoost · Chart.js</footer>

<script>
// ─── DATA ────────────────────────────────────────────────
const hours      = {j_hours};
const hourlyCnt  = {j_hourly_cnt};
const hourlyClr  = {j_hourly_clr};
const drinks     = {j_drinks};
const drinkCnt   = {j_drink_cnt};
const revLabels  = {j_rev_labels};
const revVals    = {j_rev_vals};
const fiLabels   = {j_fi_labels};
const fiVals     = {j_fi_vals};
const cmLabels   = {j_cm_labels};
const cmData     = {j_cm_data};
const hdHours    = {j_hd_hours};
const hdDrinks   = {j_hd_drinks};
const hdMatrix   = {j_hd_matrix};

const PALETTE = ['#2E86AB','#A23B72','#F18F01','#C73E1D','#3B1F2B',
                 '#44BBA4','#E94F37','#F5A623'];

// ─── Hourly bar ──────────────────────────────────────────
new Chart(document.getElementById('hourlyChart'), {{
  type:'bar',
  data:{{ labels:hours.map(h=>h+':00'), datasets:[{{
    data:hourlyCnt, backgroundColor:hourlyClr,
    borderRadius:6, borderSkipped:false
  }}]}},
  options:{{ plugins:{{legend:{{display:false}}}},
    scales:{{ y:{{beginAtZero:true}}, x:{{grid:{{display:false}}}} }},
    responsive:true, maintainAspectRatio:false }}
}});

// ─── Drink bar ──────────────────────────────────────────
new Chart(document.getElementById('drinkBarChart'), {{
  type:'bar',
  data:{{ labels:drinks, datasets:[{{ data:drinkCnt,
    backgroundColor:PALETTE, borderRadius:6, borderSkipped:false }}]}},
  options:{{ indexAxis:'y', plugins:{{legend:{{display:false}}}},
    scales:{{ x:{{beginAtZero:true}}, y:{{grid:{{display:false}}}} }},
    responsive:true, maintainAspectRatio:false }}
}});

// ─── Revenue doughnut ────────────────────────────────────
new Chart(document.getElementById('revenueChart'), {{
  type:'doughnut',
  data:{{ labels:revLabels, datasets:[{{
    data:revVals, backgroundColor:PALETTE,
    borderWidth:2, borderColor:'#fff'
  }}]}},
  options:{{ plugins:{{ legend:{{ position:'right', labels:{{ font:{{size:11}} }} }} }},
    responsive:true, maintainAspectRatio:false }}
}});

// ─── Feature Importance bar ──────────────────────────────
new Chart(document.getElementById('fiChart'), {{
  type:'bar',
  data:{{ labels:fiLabels, datasets:[{{ data:fiVals,
    backgroundColor: fiVals.map((_,i)=>i===0?'#C73E1D':i===1?'#F18F01':'#2E86AB'),
    borderRadius:6, borderSkipped:false }}]}},
  options:{{ indexAxis:'y', plugins:{{legend:{{display:false}}}},
    scales:{{ x:{{beginAtZero:true}}, y:{{grid:{{display:false}}}} }},
    responsive:true, maintainAspectRatio:false }}
}});

// ─── Heatmap table ────────────────────────────────────────
(function(){{
  const tbl = document.getElementById('heatmapTable');
  // header: drinks
  let hdr = '<thead><tr><th>שעה</th>';
  hdDrinks.forEach(d=>{{ hdr+=`<th>${{d}}</th>`; }});
  hdr += '</tr></thead>';
  // body
  let body = '<tbody>';
  hdHours.forEach((h,ri)=>{{
    body += `<tr><td><b>${{h}}:00</b></td>`;
    hdMatrix[ri].forEach(v=>{{
      const intensity = Math.round((v/100)*220);
      const bg = `rgb(${{255-Math.round(intensity*0.4)}},${{255-Math.round(intensity*0.15)}},${{255-intensity}})`;
      const fg = v>55?'#fff':'#333';
      body += `<td style="background:${{bg}};color:${{fg}};font-weight:${{v>40?700:400}}">${{v>0?v.toFixed(0)+'%':''}}</td>`;
    }});
    body += '</tr>';
  }});
  body += '</tbody>';
  tbl.innerHTML = hdr+body;
}})();

// ─── Confusion Matrix table ───────────────────────────────
(function(){{
  const tbl = document.getElementById('cmTable');
  let hdr = '<thead><tr><th>בפועל \\ חיזוי</th>';
  cmLabels.forEach(l=>{{ hdr+=`<th>${{l}}</th>`; }});
  hdr += '</tr></thead>';
  let body = '<tbody>';
  cmData.forEach((row,ri)=>{{
    body += `<tr><td><b>${{cmLabels[ri]}}</b></td>`;
    row.forEach((v,ci)=>{{
      const isMain = ri===ci;
      const bg = isMain
        ? `rgba(46,134,171,${{0.2+v/100*0.7}})`
        : v>20?`rgba(199,62,29,${{v/100*0.5}})`:'transparent';
      const fw = isMain?700:400;
      body += `<td style="background:${{bg}};font-weight:${{fw}}">${{v>0?v+'%':''}}</td>`;
    }});
    body += '</tr>';
  }});
  body += '</tbody>';
  tbl.innerHTML = hdr+body;
}})();
</script>
</body>
</html>"""

with open('coffee_dashboard.html','w',encoding='utf-8') as f:
    f.write(html)
print("Done! → coffee_dashboard.html")

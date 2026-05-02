# Coffee Shop Analysis ☕

ניתוח מכירות של בית קפה לשנת 2024–2025, כולל EDA, הנדסת פיצ'רים, מודל CatBoost ודשבורד אינטראקטיבי.

## מה יש כאן?

| קובץ | תיאור |
|------|-------|
| `index_1.csv` | נתוני המכירות הגולמיים |
| `coffee_analysis.py` | ניתוח ראשוני + אימון מודל + גרפים (PNG) |
| `generate_dashboard.py` | יצירת דשבורד HTML אינטראקטיבי |
| `coffee_dashboard.html` | הדשבורד המוכן לצפייה בדפדפן |

## איך להריץ

```bash
pip install pandas numpy matplotlib seaborn catboost scikit-learn

python coffee_analysis.py        # ניתוח + גרפים
python generate_dashboard.py     # דשבורד HTML
```

## מה הנתונים מכילים?

| עמודה | תיאור |
|-------|-------|
| `datetime` | תאריך ושעת העסקה |
| `cash_type` | סוג תשלום (card/cash) |
| `money` | מחיר |
| `coffee_name` | סוג המשקה |

## תוצאות

- **3,636 עסקאות** בין מרץ 2024 למרץ 2025
- **8 סוגי משקאות** — הנמכר ביותר: Americano with Milk
- **שעות שיא:** 9:00–12:00 ו-16:00
- **דיוק מודל CatBoost:** 58.8%

### המלצות מלאי לפי שעות

| חלק היום | הכן הרבה | הכן מעט |
|-----------|----------|---------|
| בוקר 6–11 | Americano with Milk | Espresso |
| צהריים 12–14 | Americano with Milk | Hot Chocolate |
| אחה"צ 15–17 | Latte | Espresso |
| ערב 18–22 | Latte | Espresso |

## טכנולוגיות

`Python` · `Pandas` · `CatBoost` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Chart.js`

# DataPulse - E-Commerce Analytics Platform

<div align="center">
  <h3>📊 Production-Grade Analytics for E-Commerce</h3>
  <p>Upload → Analyze → Visualize → Decide</p>
</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📁 **Multi-Dataset** | Upload and manage multiple CSV/Excel files |
| 🔗 **Smart Column Mapping** | Auto-detect + manual override |
| 📊 **Live Dashboard** | Real-time KPIs, charts, trends |
| 🤖 **AI Analyst** | Ask questions in plain English |
| 📈 **Dynamic Reports** | Generate detailed analysis reports |
| 🔴 **Live Mode** | Auto-refresh dashboard every 5 seconds |
| 💰 **INR Support** | Indian Rupee with Lakhs/Crores formatting |
| 🔐 **Business Rules** | Accurate calculations (no hallucinations) |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open browser at http://localhost:8501
```

---

## 📊 Dashboard Sections

### Key Metrics
- **Total Revenue** - Sum of delivered orders only
- **Average Order Value** - Mean of delivered orders
- **Total Orders** - All orders count
- **RTO Rate** - Correct formula: RTO / (Delivered + RTO)

### Charts
- Revenue Trend (daily/weekly/monthly)
- Orders by Status (pie)
- Top Products (bar)
- Payment Methods (bar)
- COD vs Prepaid Analysis
- RTO by Payment Method
- RTO by City
- Category Performance (treemap)

### AI Analyst Queries
- "What is my total revenue?"
- "Show RTO rate by payment method"
- "Top 10 products"
- "COD vs Prepaid comparison"
- "Revenue by category"

---

## 🔐 Business Rules (Hardcoded)

```python
# Revenue = Delivered orders only
revenue = df[df['status'] == 'Delivered']['amount'].sum()

# AOV = Delivered orders only  
aov = df[df['status'] == 'Delivered']['amount'].mean()

# RTO Rate = RTO / (Delivered + RTO) × 100
shipped = df[df['status'].isin(['Delivered', 'RTO'])]
rto_rate = len(shipped[shipped['status'] == 'RTO']) / len(shipped) * 100
```

---

## 📁 Data Requirements

### Required Columns
| Field | Description |
|-------|-------------|
| Order ID | Unique identifier |
| Order Amount | Total value |
| Order Status | Delivered, RTO, Cancelled |

### Optional Columns
| Field | Description |
|-------|-------------|
| Order Date | For trend analysis |
| Payment Method | COD, UPI, Card |
| Customer Name | For customer analysis |
| Category | Product category |
| City | For location analysis |

---

## 🌐 Deploy to Streamlit Cloud (FREE)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → Deploy
4. Get URL: `https://your-app.streamlit.app`

---

## 📄 License

MIT License - Use freely for your projects.

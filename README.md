# E-Commerce Analytics Dashboard (Streamlit)

A complete, production-ready analytics dashboard built with Streamlit.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open in browser
# http://localhost:8501
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📁 **CSV Upload** | Drag & drop CSV/Excel files |
| 🔗 **Column Mapping** | Map your columns to standard fields |
| 🧹 **Auto-Cleaning** | Remove duplicates, test orders, standardize data |
| 📊 **Live Dashboard** | KPIs, charts, trends - all from your data |
| 🤖 **AI Chat** | Ask questions in plain English |
| 📈 **Dynamic Visuals** | Charts generated based on queries |
| 💰 **INR Currency** | Indian Rupee formatting (₹, Lakhs, Crores) |

## 📊 Dashboard Sections

### KPIs
- Total Revenue (delivered orders only)
- Average Order Value
- Total Orders
- RTO Rate (correct formula!)

### Charts
- Revenue Trend (monthly)
- Orders by Status (pie)
- Top Products (bar)
- COD vs Prepaid comparison
- RTO by Payment Method

### AI Chat
Ask questions like:
- "What is my total revenue?"
- "Show RTO rate by payment method"
- "Top 10 products"
- "COD vs Prepaid comparison"
- "Revenue by category"

## 🔐 Business Rules (Built-In)

```
✅ Revenue = SUM(amount) WHERE status = 'delivered'
✅ AOV = AVG(amount) WHERE status = 'delivered'
✅ RTO Rate = RTO / (Delivered + RTO) × 100
```

These rules are **hardcoded** - the AI cannot hallucinate wrong calculations.

## 🌐 Deploy to Streamlit Cloud (Free!)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/ecommerce-dashboard.git
git push -u origin main
```

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repo
4. Select `app.py` as main file
5. Click "Deploy"

**That's it!** You'll get a free URL like:
`https://your-app.streamlit.app`

## 📁 Project Structure

```
streamlit_dashboard/
├── app.py              # Main application (everything in one file!)
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎯 Why Streamlit for MVP?

| Advantage | Explanation |
|-----------|-------------|
| **Speed** | Built entire dashboard in 1 file |
| **Python Native** | No separate frontend/backend |
| **Free Hosting** | Streamlit Cloud is free |
| **Real-time** | Auto-refresh on data changes |
| **Easy Demo** | Just share the URL |

## 📱 Screenshots

### Dashboard
![Dashboard](https://via.placeholder.com/800x400?text=Dashboard+Screenshot)

### AI Chat
![AI Chat](https://via.placeholder.com/800x400?text=AI+Chat+Screenshot)

### Column Mapping
![Mapping](https://via.placeholder.com/800x400?text=Column+Mapping+Screenshot)

## 🔧 Customization

### Add New Query Type

1. Add pattern to `parse_user_query()`:
```python
patterns = {
    'my_new_query': [r'pattern1', r'pattern2'],
    ...
}
```

2. Add method to `QueryEngine`:
```python
def _query_my_new_query(self) -> Tuple[Any, str, str]:
    # Your logic here
    return result, "Explanation", "SQL equivalent"
```

### Change Currency

Edit the `format_inr()` function:
```python
def format_inr(value: float) -> str:
    # Change to USD, EUR, etc.
    ...
```

## 🤝 Contributing

1. Fork the repo
2. Create a branch
3. Make changes
4. Submit PR

## 📄 License

MIT License - Use freely for your projects!

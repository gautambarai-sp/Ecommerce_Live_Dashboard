"""
E-Commerce Analytics Dashboard - Streamlit App
==============================================
A complete analytics platform with:
- CSV Upload & Auto-cleaning
- Column Mapping
- AI Chatbot with deterministic queries
- Dynamic visualizations
- Real-time updates

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any, Optional, Tuple
import json

# Page config
st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
    }
    .chat-user {
        background: #3b82f6;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-assistant {
        background: #f1f5f9;
        color: #1e293b;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
    }
    .sql-box {
        background: #1e293b;
        color: #e2e8f0;
        padding: 12px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.8rem;
        margin-top: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'mappings_confirmed' not in st.session_state:
    st.session_state.mappings_confirmed = False

# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def detect_column_type(series: pd.Series, col_name: str) -> str:
    """Detect the type of a column based on name and content."""
    col_lower = col_name.lower()
    
    # Pattern-based detection
    patterns = {
        'order_id': [r'order[_\s]?id', r'order[_\s]?no', r'^id$', r'invoice'],
        'total_amount': [r'total', r'amount', r'value', r'price', r'revenue', r'grand'],
        'order_status': [r'status', r'state', r'delivery'],
        'order_date': [r'date', r'created', r'placed', r'time'],
        'payment_method': [r'payment', r'pay.*method', r'pay.*mode'],
        'customer_name': [r'customer.*name', r'name', r'buyer'],
        'customer_email': [r'email', r'mail'],
        'city': [r'city', r'town'],
        'state': [r'state', r'region', r'province'],
        'product_name': [r'product', r'item', r'sku.*name'],
        'category': [r'category', r'type', r'department'],
        'quantity': [r'qty', r'quantity', r'units'],
        'courier': [r'courier', r'carrier', r'shipping.*partner'],
    }
    
    for field_type, field_patterns in patterns.items():
        for pattern in field_patterns:
            if re.search(pattern, col_lower):
                return field_type
    
    return 'unknown'


def clean_dataframe(df: pd.DataFrame, mappings: Dict[str, str]) -> pd.DataFrame:
    """Clean and standardize the dataframe based on mappings."""
    df = df.copy()
    
    # Status standardization
    status_col = mappings.get('order_status')
    if status_col and status_col in df.columns:
        status_map = {
            'delivered': 'delivered', 'completed': 'delivered', 'complete': 'delivered',
            'fulfilled': 'delivered', 'success': 'delivered',
            'cancelled': 'cancelled', 'canceled': 'cancelled', 'cancel': 'cancelled',
            'refunded': 'cancelled',
            'rto': 'rto', 'returned': 'rto', 'return': 'rto', 'undelivered': 'rto',
            'return to origin': 'rto', 'failed delivery': 'rto',
            'processing': 'processing', 'pending': 'processing', 'confirmed': 'processing',
            'shipped': 'in_transit', 'in transit': 'in_transit', 'dispatched': 'in_transit',
        }
        df[status_col] = df[status_col].astype(str).str.lower().str.strip()
        df[status_col] = df[status_col].map(lambda x: status_map.get(x, x))
    
    # Payment method standardization
    payment_col = mappings.get('payment_method')
    if payment_col and payment_col in df.columns:
        payment_map = {
            'cod': 'COD', 'cash on delivery': 'COD', 'cash': 'COD',
            'upi': 'UPI', 'gpay': 'UPI', 'phonepe': 'UPI', 'paytm': 'UPI',
            'credit card': 'Card', 'debit card': 'Card', 'card': 'Card',
            'net banking': 'Net Banking', 'netbanking': 'Net Banking',
            'prepaid': 'Prepaid', 'online': 'Prepaid',
        }
        df[payment_col] = df[payment_col].astype(str).str.lower().str.strip()
        df[payment_col] = df[payment_col].map(lambda x: payment_map.get(x, x.title()))
    
    # Amount cleaning
    amount_col = mappings.get('total_amount')
    if amount_col and amount_col in df.columns:
        df[amount_col] = df[amount_col].astype(str).str.replace(r'[₹$,Rs\.\s]', '', regex=True)
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
    
    # Date cleaning
    date_col = mappings.get('order_date')
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    
    # Remove duplicates and invalid rows
    df = df.drop_duplicates()
    
    # Remove test orders
    for col in [mappings.get('customer_email'), mappings.get('customer_name')]:
        if col and col in df.columns:
            test_patterns = ['test', 'demo', 'sample', 'dummy']
            mask = ~df[col].astype(str).str.lower().str.contains('|'.join(test_patterns), na=False)
            df = df[mask]
    
    return df


# =============================================================================
# QUERY ENGINE (Deterministic - No Hallucination)
# =============================================================================

class QueryEngine:
    """Deterministic query engine for e-commerce analytics."""
    
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, str]):
        self.df = df
        self.mappings = mappings
    
    def get_column(self, field: str) -> Optional[str]:
        """Get actual column name for a standard field."""
        return self.mappings.get(field)
    
    def execute(self, query_type: str, **kwargs) -> Tuple[Any, str, str]:
        """
        Execute a query and return (result, explanation, sql_equivalent).
        """
        method_name = f"_query_{query_type}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(**kwargs)
        return None, "Unknown query type", ""
    
    def _query_total_revenue(self) -> Tuple[float, str, str]:
        """Total revenue from delivered orders only."""
        amount_col = self.get_column('total_amount')
        status_col = self.get_column('order_status')
        
        if not amount_col or not status_col:
            return 0, "Missing required columns", ""
        
        delivered = self.df[self.df[status_col] == 'delivered']
        revenue = delivered[amount_col].sum()
        
        sql = f"""SELECT SUM({amount_col}) as revenue
FROM orders
WHERE {status_col} = 'delivered'
-- Revenue only counts DELIVERED orders"""
        
        return revenue, f"Total revenue from {len(delivered):,} delivered orders", sql
    
    def _query_aov(self) -> Tuple[float, str, str]:
        """Average order value from delivered orders only."""
        amount_col = self.get_column('total_amount')
        status_col = self.get_column('order_status')
        
        if not amount_col or not status_col:
            return 0, "Missing required columns", ""
        
        delivered = self.df[self.df[status_col] == 'delivered']
        aov = delivered[amount_col].mean()
        
        sql = f"""SELECT AVG({amount_col}) as aov
FROM orders
WHERE {status_col} = 'delivered'
-- AOV only from DELIVERED orders"""
        
        return aov, f"Average order value from {len(delivered):,} delivered orders", sql
    
    def _query_order_count(self) -> Tuple[int, str, str]:
        """Total order count."""
        return len(self.df), f"Total orders in dataset", "SELECT COUNT(*) FROM orders"
    
    def _query_rto_rate(self) -> Tuple[Dict, str, str]:
        """RTO rate with CORRECT denominator (delivered + rto only)."""
        status_col = self.get_column('order_status')
        
        if not status_col:
            return {}, "Missing status column", ""
        
        shipped = self.df[self.df[status_col].isin(['delivered', 'rto'])]
        rto_orders = len(shipped[shipped[status_col] == 'rto'])
        total_shipped = len(shipped)
        
        rate = (rto_orders / total_shipped * 100) if total_shipped > 0 else 0
        
        sql = f"""SELECT 
    COUNT(CASE WHEN {status_col} = 'rto' THEN 1 END) as rto_orders,
    COUNT(*) as shipped_orders,
    COUNT(CASE WHEN {status_col} = 'rto' THEN 1 END) * 100.0 / COUNT(*) as rto_rate
FROM orders
WHERE {status_col} IN ('delivered', 'rto')
-- IMPORTANT: Denominator is shipped orders only (delivered + rto)"""
        
        return {
            'rto_orders': rto_orders,
            'shipped_orders': total_shipped,
            'rto_rate': rate
        }, f"RTO rate based on {total_shipped:,} shipped orders", sql
    
    def _query_status_breakdown(self) -> Tuple[pd.DataFrame, str, str]:
        """Orders by status."""
        status_col = self.get_column('order_status')
        
        if not status_col:
            return pd.DataFrame(), "Missing status column", ""
        
        breakdown = self.df.groupby(status_col).size().reset_index(name='count')
        breakdown['percentage'] = (breakdown['count'] / len(self.df) * 100).round(2)
        
        sql = f"""SELECT {status_col}, COUNT(*) as count,
       COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
FROM orders
GROUP BY {status_col}"""
        
        return breakdown, "Orders breakdown by status", sql
    
    def _query_revenue_by_category(self) -> Tuple[pd.DataFrame, str, str]:
        """Revenue by category (delivered only)."""
        amount_col = self.get_column('total_amount')
        status_col = self.get_column('order_status')
        category_col = self.get_column('category')
        
        if not all([amount_col, status_col, category_col]):
            return pd.DataFrame(), "Missing required columns", ""
        
        delivered = self.df[self.df[status_col] == 'delivered']
        by_category = delivered.groupby(category_col)[amount_col].sum().reset_index()
        by_category.columns = ['category', 'revenue']
        by_category = by_category.sort_values('revenue', ascending=False)
        
        sql = f"""SELECT {category_col}, SUM({amount_col}) as revenue
FROM orders
WHERE {status_col} = 'delivered'
GROUP BY {category_col}
ORDER BY revenue DESC"""
        
        return by_category, "Revenue by category (delivered orders only)", sql
    
    def _query_top_products(self, limit: int = 10) -> Tuple[pd.DataFrame, str, str]:
        """Top products by revenue."""
        amount_col = self.get_column('total_amount')
        status_col = self.get_column('order_status')
        product_col = self.get_column('product_name')
        
        if not all([amount_col, status_col, product_col]):
            return pd.DataFrame(), "Missing required columns", ""
        
        delivered = self.df[self.df[status_col] == 'delivered']
        top = delivered.groupby(product_col).agg({
            amount_col: 'sum'
        }).reset_index()
        top.columns = ['product', 'revenue']
        top = top.sort_values('revenue', ascending=False).head(limit)
        
        sql = f"""SELECT {product_col}, SUM({amount_col}) as revenue
FROM orders
WHERE {status_col} = 'delivered'
GROUP BY {product_col}
ORDER BY revenue DESC
LIMIT {limit}"""
        
        return top, f"Top {limit} products by revenue", sql
    
    def _query_cod_vs_prepaid(self) -> Tuple[pd.DataFrame, str, str]:
        """Compare COD vs Prepaid."""
        amount_col = self.get_column('total_amount')
        status_col = self.get_column('order_status')
        payment_col = self.get_column('payment_method')
        
        if not all([amount_col, status_col, payment_col]):
            return pd.DataFrame(), "Missing required columns", ""
        
        delivered = self.df[self.df[status_col] == 'delivered']
        
        # Group into COD vs Prepaid
        delivered['payment_type'] = delivered[payment_col].apply(
            lambda x: 'COD' if x == 'COD' else 'Prepaid'
        )
        
        comparison = delivered.groupby('payment_type').agg({
            amount_col: ['sum', 'mean', 'count']
        }).reset_index()
        comparison.columns = ['payment_type', 'revenue', 'aov', 'orders']
        
        sql = f"""SELECT 
    CASE WHEN {payment_col} = 'COD' THEN 'COD' ELSE 'Prepaid' END as payment_type,
    SUM({amount_col}) as revenue,
    AVG({amount_col}) as aov,
    COUNT(*) as orders
FROM orders
WHERE {status_col} = 'delivered'
GROUP BY payment_type"""
        
        return comparison, "COD vs Prepaid comparison (delivered orders)", sql
    
    def _query_rto_by_payment(self) -> Tuple[pd.DataFrame, str, str]:
        """RTO rate by payment method."""
        status_col = self.get_column('order_status')
        payment_col = self.get_column('payment_method')
        
        if not all([status_col, payment_col]):
            return pd.DataFrame(), "Missing required columns", ""
        
        shipped = self.df[self.df[status_col].isin(['delivered', 'rto'])]
        shipped['payment_type'] = shipped[payment_col].apply(
            lambda x: 'COD' if x == 'COD' else 'Prepaid'
        )
        
        rto_analysis = shipped.groupby('payment_type').apply(
            lambda x: pd.Series({
                'total_shipped': len(x),
                'rto_orders': len(x[x[status_col] == 'rto']),
                'rto_rate': len(x[x[status_col] == 'rto']) / len(x) * 100 if len(x) > 0 else 0
            })
        ).reset_index()
        
        sql = f"""SELECT 
    CASE WHEN {payment_col} = 'COD' THEN 'COD' ELSE 'Prepaid' END as payment_type,
    COUNT(*) as total_shipped,
    COUNT(CASE WHEN {status_col} = 'rto' THEN 1 END) as rto_orders,
    COUNT(CASE WHEN {status_col} = 'rto' THEN 1 END) * 100.0 / COUNT(*) as rto_rate
FROM orders
WHERE {status_col} IN ('delivered', 'rto')
GROUP BY payment_type"""
        
        return rto_analysis, "RTO rate by payment method", sql
    
    def _query_customer_count(self) -> Tuple[int, str, str]:
        """Unique customer count."""
        email_col = self.get_column('customer_email')
        name_col = self.get_column('customer_name')
        
        col = email_col or name_col
        if not col:
            return 0, "Missing customer identifier column", ""
        
        unique = self.df[col].nunique()
        
        sql = f"SELECT COUNT(DISTINCT {col}) as customers FROM orders"
        
        return unique, f"Unique customers based on {col}", sql
    
    def _query_top_customers(self, limit: int = 10) -> Tuple[pd.DataFrame, str, str]:
        """Top customers by spending."""
        amount_col = self.get_column('total_amount')
        status_col = self.get_column('order_status')
        customer_col = self.get_column('customer_name') or self.get_column('customer_email')
        
        if not all([amount_col, status_col, customer_col]):
            return pd.DataFrame(), "Missing required columns", ""
        
        delivered = self.df[self.df[status_col] == 'delivered']
        top = delivered.groupby(customer_col).agg({
            amount_col: 'sum'
        }).reset_index()
        top.columns = ['customer', 'total_spent']
        top = top.sort_values('total_spent', ascending=False).head(limit)
        
        sql = f"""SELECT {customer_col}, SUM({amount_col}) as total_spent
FROM orders
WHERE {status_col} = 'delivered'
GROUP BY {customer_col}
ORDER BY total_spent DESC
LIMIT {limit}"""
        
        return top, f"Top {limit} customers by spending", sql
    
    def _query_revenue_trend(self) -> Tuple[pd.DataFrame, str, str]:
        """Revenue trend over time."""
        amount_col = self.get_column('total_amount')
        status_col = self.get_column('order_status')
        date_col = self.get_column('order_date')
        
        if not all([amount_col, status_col, date_col]):
            return pd.DataFrame(), "Missing required columns", ""
        
        delivered = self.df[self.df[status_col] == 'delivered'].copy()
        delivered['month'] = delivered[date_col].dt.to_period('M').astype(str)
        
        trend = delivered.groupby('month').agg({
            amount_col: 'sum'
        }).reset_index()
        trend.columns = ['month', 'revenue']
        
        sql = f"""SELECT DATE_TRUNC('month', {date_col}) as month, SUM({amount_col}) as revenue
FROM orders
WHERE {status_col} = 'delivered'
GROUP BY month
ORDER BY month"""
        
        return trend, "Monthly revenue trend", sql
    
    def _query_city_breakdown(self) -> Tuple[pd.DataFrame, str, str]:
        """Revenue by city."""
        amount_col = self.get_column('total_amount')
        status_col = self.get_column('order_status')
        city_col = self.get_column('city')
        
        if not all([amount_col, status_col, city_col]):
            return pd.DataFrame(), "Missing required columns", ""
        
        delivered = self.df[self.df[status_col] == 'delivered']
        by_city = delivered.groupby(city_col).agg({
            amount_col: ['sum', 'count']
        }).reset_index()
        by_city.columns = ['city', 'revenue', 'orders']
        by_city = by_city.sort_values('revenue', ascending=False).head(15)
        
        sql = f"""SELECT {city_col}, SUM({amount_col}) as revenue, COUNT(*) as orders
FROM orders
WHERE {status_col} = 'delivered'
GROUP BY {city_col}
ORDER BY revenue DESC
LIMIT 15"""
        
        return by_city, "Top 15 cities by revenue", sql


def parse_user_query(query: str) -> Tuple[str, Dict]:
    """Parse user's natural language query into query type and params."""
    query_lower = query.lower()
    
    # Query patterns
    patterns = {
        'total_revenue': [r'total revenue', r'revenue$', r'how much.*made', r'total sales', r'gmv'],
        'aov': [r'average order value', r'\baov\b', r'avg.*order', r'average.*order'],
        'order_count': [r'how many orders', r'total orders', r'order count'],
        'rto_rate': [r'rto rate', r'return.*rate', r'rto percent', r'delivery fail'],
        'status_breakdown': [r'status breakdown', r'orders by status', r'status wise'],
        'revenue_by_category': [r'revenue by category', r'category.*revenue', r'category wise'],
        'top_products': [r'top.*product', r'best.*product', r'best sell'],
        'cod_vs_prepaid': [r'cod.*prepaid', r'prepaid.*cod', r'cod.*vs', r'compare.*payment'],
        'rto_by_payment': [r'rto.*payment', r'rto.*cod', r'cod.*rto'],
        'customer_count': [r'how many customer', r'customer count', r'unique customer'],
        'top_customers': [r'top.*customer', r'best.*customer'],
        'revenue_trend': [r'revenue trend', r'revenue.*time', r'monthly revenue', r'revenue.*month'],
        'city_breakdown': [r'city', r'revenue.*city', r'city.*wise'],
    }
    
    for query_type, type_patterns in patterns.items():
        for pattern in type_patterns:
            if re.search(pattern, query_lower):
                return query_type, {}
    
    # Default to summary
    if any(word in query_lower for word in ['summary', 'overview', 'dashboard', 'kpi']):
        return 'summary', {}
    
    return 'unknown', {}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def format_inr(value: float) -> str:
    """Format value in Indian Rupees."""
    if value >= 10000000:
        return f"₹{value/10000000:.2f} Cr"
    elif value >= 100000:
        return f"₹{value/100000:.2f} L"
    elif value >= 1000:
        return f"₹{value/1000:.1f}K"
    else:
        return f"₹{value:,.2f}"


def create_kpi_chart(value: float, title: str, prefix: str = "₹") -> go.Figure:
    """Create a simple KPI indicator."""
    fig = go.Figure(go.Indicator(
        mode="number",
        value=value,
        number={'prefix': prefix, 'valueformat': ',.0f'},
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e293b'}
    )
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/analytics.png", width=60)
        st.title("E-Commerce Analytics")
        st.caption("AI-Powered Dashboard")
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "📁 Upload Data", "🤖 AI Chat", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Dataset info
        if st.session_state.cleaned_df is not None:
            st.success(f"✓ Data loaded")
            st.caption(f"{len(st.session_state.cleaned_df):,} rows")
            if st.session_state.mappings_confirmed:
                st.caption(f"{len(st.session_state.column_mappings)} columns mapped")
        else:
            st.warning("No data loaded")
            st.caption("Upload a CSV to start")
    
    # Main content based on navigation
    if page == "📁 Upload Data":
        render_upload_page()
    elif page == "🤖 AI Chat":
        render_chat_page()
    elif page == "⚙️ Settings":
        render_settings_page()
    else:
        render_dashboard_page()


def render_upload_page():
    """Render the data upload page."""
    st.markdown('<p class="main-header">📁 Upload Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your e-commerce data to get started</p>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Drop your CSV or Excel file here",
        type=['csv', 'xlsx', 'xls'],
        help="Max 50MB, up to 50,000 rows"
    )
    
    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            
            # Show preview
            st.success(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
            
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Column mapping section
            st.subheader("🔗 Map Your Columns")
            st.caption("Tell us which columns match our standard fields. This ensures accurate analytics.")
            
            # Auto-detect columns
            detected = {}
            for col in df.columns:
                col_type = detect_column_type(df[col], col)
                if col_type != 'unknown':
                    detected[col_type] = col
            
            # Required fields
            st.markdown("**Required Fields** (must be mapped)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                order_id_col = st.selectbox(
                    "Order ID *",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('order_id', '')) + 1 if detected.get('order_id') in df.columns else 0,
                    help="Unique identifier for each order"
                )
            
            with col2:
                amount_col = st.selectbox(
                    "Order Amount *",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('total_amount', '')) + 1 if detected.get('total_amount') in df.columns else 0,
                    help="Total order value"
                )
            
            with col3:
                status_col = st.selectbox(
                    "Order Status *",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('order_status', '')) + 1 if detected.get('order_status') in df.columns else 0,
                    help="Delivery status (Delivered, Cancelled, RTO, etc.)"
                )
            
            # Optional fields
            st.markdown("**Optional Fields** (for deeper insights)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                date_col = st.selectbox(
                    "Order Date",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('order_date', '')) + 1 if detected.get('order_date') in df.columns else 0
                )
            
            with col2:
                payment_col = st.selectbox(
                    "Payment Method",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('payment_method', '')) + 1 if detected.get('payment_method') in df.columns else 0
                )
            
            with col3:
                customer_col = st.selectbox(
                    "Customer Name",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('customer_name', '')) + 1 if detected.get('customer_name') in df.columns else 0
                )
            
            with col4:
                category_col = st.selectbox(
                    "Category",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('category', '')) + 1 if detected.get('category') in df.columns else 0
                )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                city_col = st.selectbox(
                    "City",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('city', '')) + 1 if detected.get('city') in df.columns else 0
                )
            
            with col2:
                product_col = st.selectbox(
                    "Product Name",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('product_name', '')) + 1 if detected.get('product_name') in df.columns else 0
                )
            
            with col3:
                email_col = st.selectbox(
                    "Customer Email",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('customer_email', '')) + 1 if detected.get('customer_email') in df.columns else 0
                )
            
            with col4:
                courier_col = st.selectbox(
                    "Courier",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected.get('courier', '')) + 1 if detected.get('courier') in df.columns else 0
                )
            
            # Validate and save
            if st.button("✓ Confirm & Clean Data", type="primary", use_container_width=True):
                # Validate required fields
                if not order_id_col or not amount_col or not status_col:
                    st.error("Please map all required fields (Order ID, Order Amount, Order Status)")
                else:
                    # Build mappings
                    mappings = {
                        'order_id': order_id_col,
                        'total_amount': amount_col,
                        'order_status': status_col,
                    }
                    if date_col: mappings['order_date'] = date_col
                    if payment_col: mappings['payment_method'] = payment_col
                    if customer_col: mappings['customer_name'] = customer_col
                    if category_col: mappings['category'] = category_col
                    if city_col: mappings['city'] = city_col
                    if product_col: mappings['product_name'] = product_col
                    if email_col: mappings['customer_email'] = email_col
                    if courier_col: mappings['courier'] = courier_col
                    
                    # Clean data
                    with st.spinner("Cleaning data..."):
                        cleaned = clean_dataframe(df, mappings)
                        
                        st.session_state.cleaned_df = cleaned
                        st.session_state.column_mappings = mappings
                        st.session_state.mappings_confirmed = True
                    
                    # Show cleaning report
                    removed = len(df) - len(cleaned)
                    st.success(f"""
                    ✓ Data cleaned successfully!
                    - Original rows: {len(df):,}
                    - Cleaned rows: {len(cleaned):,}
                    - Removed: {removed:,} rows (duplicates, test orders, invalid)
                    """)
                    
                    st.balloons()
                    st.info("Go to **Dashboard** to see your analytics!")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def render_dashboard_page():
    """Render the main dashboard."""
    st.markdown('<p class="main-header">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time analytics from your data</p>', unsafe_allow_html=True)
    
    if st.session_state.cleaned_df is None or not st.session_state.mappings_confirmed:
        st.warning("⚠️ No data loaded. Please upload a CSV file first.")
        if st.button("Go to Upload Page"):
            st.rerun()
        return
    
    df = st.session_state.cleaned_df
    mappings = st.session_state.column_mappings
    engine = QueryEngine(df, mappings)
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        revenue, _, _ = engine.execute('total_revenue')
        st.metric(
            label="Total Revenue",
            value=format_inr(revenue or 0),
            delta="Delivered orders only",
            delta_color="off"
        )
    
    with col2:
        aov, _, _ = engine.execute('aov')
        st.metric(
            label="Average Order Value",
            value=format_inr(aov or 0),
            delta="Delivered orders only",
            delta_color="off"
        )
    
    with col3:
        orders, _, _ = engine.execute('order_count')
        st.metric(
            label="Total Orders",
            value=f"{orders:,}",
            delta="All orders"
        )
    
    with col4:
        rto_data, _, _ = engine.execute('rto_rate')
        rto_rate = rto_data.get('rto_rate', 0) if isinstance(rto_data, dict) else 0
        st.metric(
            label="RTO Rate",
            value=f"{rto_rate:.2f}%",
            delta=f"{rto_data.get('rto_orders', 0):,} returns" if isinstance(rto_data, dict) else "",
            delta_color="inverse"
        )
    
    st.divider()
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Trend")
        trend_df, _, _ = engine.execute('revenue_trend')
        if isinstance(trend_df, pd.DataFrame) and not trend_df.empty:
            fig = px.area(
                trend_df, x='month', y='revenue',
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Revenue (₹)",
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date column required for trend analysis")
    
    with col2:
        st.subheader("🥧 Orders by Status")
        status_df, _, _ = engine.execute('status_breakdown')
        if isinstance(status_df, pd.DataFrame) and not status_df.empty:
            fig = px.pie(
                status_df, values='count', names=mappings.get('order_status', 'status'),
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No status data available")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Products")
        top_products, _, _ = engine.execute('top_products', limit=10)
        if isinstance(top_products, pd.DataFrame) and not top_products.empty:
            fig = px.bar(
                top_products.head(10), x='revenue', y='product',
                orientation='h',
                color_discrete_sequence=['#10b981']
            )
            fig.update_layout(
                xaxis_title="Revenue (₹)",
                yaxis_title="",
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product column required")
    
    with col2:
        st.subheader("💳 COD vs Prepaid")
        comparison, _, _ = engine.execute('cod_vs_prepaid')
        if isinstance(comparison, pd.DataFrame) and not comparison.empty:
            fig = px.bar(
                comparison, x='payment_type', y='revenue',
                color='payment_type',
                color_discrete_map={'COD': '#f59e0b', 'Prepaid': '#3b82f6'}
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Revenue (₹)",
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Payment method column required")
    
    # RTO Analysis
    st.subheader("🔄 RTO Analysis by Payment Method")
    rto_payment, _, _ = engine.execute('rto_by_payment')
    if isinstance(rto_payment, pd.DataFrame) and not rto_payment.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                rto_payment, x='payment_type', y='rto_rate',
                color='payment_type',
                color_discrete_map={'COD': '#ef4444', 'Prepaid': '#22c55e'}
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="RTO Rate (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Key Insight:**")
            if len(rto_payment) > 1:
                cod_rto = rto_payment[rto_payment['payment_type'] == 'COD']['rto_rate'].values
                prepaid_rto = rto_payment[rto_payment['payment_type'] == 'Prepaid']['rto_rate'].values
                if len(cod_rto) > 0 and len(prepaid_rto) > 0:
                    diff = cod_rto[0] - prepaid_rto[0]
                    if diff > 0:
                        st.warning(f"COD has {diff:.1f}% higher RTO than Prepaid")
                    else:
                        st.success(f"Prepaid has {abs(diff):.1f}% higher RTO than COD")
    else:
        st.info("Payment method and status columns required for RTO analysis")


def render_chat_page():
    """Render the AI chat page."""
    st.markdown('<p class="main-header">🤖 AI Chat</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your data in plain English</p>', unsafe_allow_html=True)
    
    if st.session_state.cleaned_df is None or not st.session_state.mappings_confirmed:
        st.warning("⚠️ No data loaded. Please upload a CSV file first.")
        return
    
    df = st.session_state.cleaned_df
    mappings = st.session_state.column_mappings
    engine = QueryEngine(df, mappings)
    
    # Suggested questions
    st.markdown("**Try asking:**")
    suggestions = [
        "What is my total revenue?",
        "What is my RTO rate?",
        "Show COD vs Prepaid comparison",
        "Top 10 products",
        "Revenue by category",
        "RTO rate by payment method"
    ]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': suggestion
                })
                # Process query
                query_type, params = parse_user_query(suggestion)
                if query_type != 'unknown':
                    result, explanation, sql = engine.execute(query_type, **params)
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': explanation,
                        'result': result,
                        'sql': sql
                    })
                else:
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': "I couldn't understand that question. Try asking about revenue, orders, RTO rate, or products."
                    })
                st.rerun()
    
    st.divider()
    
    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
            
            if msg['role'] == 'assistant' and 'result' in msg:
                result = msg['result']
                
                # Display result based on type
                if isinstance(result, pd.DataFrame) and not result.empty:
                    st.dataframe(result, use_container_width=True)
                    
                    # Visualize if possible
                    if len(result.columns) >= 2:
                        num_cols = result.select_dtypes(include=['number']).columns
                        cat_cols = result.select_dtypes(exclude=['number']).columns
                        
                        if len(num_cols) > 0 and len(cat_cols) > 0:
                            fig = px.bar(result, x=cat_cols[0], y=num_cols[0])
                            st.plotly_chart(fig, use_container_width=True)
                
                elif isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            st.metric(key.replace('_', ' ').title(), 
                                     format_inr(value) if 'revenue' in key or 'amount' in key else f"{value:,.2f}")
                
                elif isinstance(result, (int, float)):
                    st.metric("Result", format_inr(result) if result > 100 else f"{result:,.2f}")
                
                # Show SQL
                if msg.get('sql'):
                    with st.expander("🔍 View SQL Query"):
                        st.code(msg['sql'], language='sql')
    
    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        
        # Process query
        query_type, params = parse_user_query(prompt)
        
        if query_type != 'unknown':
            result, explanation, sql = engine.execute(query_type, **params)
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': explanation,
                'result': result,
                'sql': sql
            })
        else:
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': """I couldn't understand that question. Here are some things I can help with:

• **Revenue**: "What is my total revenue?"
• **AOV**: "What is my average order value?"
• **RTO**: "What is my RTO rate?" or "RTO by payment method"
• **Products**: "Top 10 products"
• **Categories**: "Revenue by category"
• **Payments**: "COD vs Prepaid comparison"
• **Customers**: "Top customers" or "Customer count"
• **Cities**: "Revenue by city"

Try one of these!"""
            })
        
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


def render_settings_page():
    """Render the settings page."""
    st.markdown('<p class="main-header">⚙️ Settings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Configure your dashboard</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Column Mappings", "About"])
    
    with tab1:
        if st.session_state.column_mappings:
            st.subheader("Current Column Mappings")
            
            for field, column in st.session_state.column_mappings.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{field.replace('_', ' ').title()}**")
                with col2:
                    st.code(column)
            
            if st.button("🔄 Re-upload Data"):
                st.session_state.df = None
                st.session_state.cleaned_df = None
                st.session_state.column_mappings = {}
                st.session_state.mappings_confirmed = False
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No column mappings configured. Upload data first.")
    
    with tab2:
        st.subheader("About This Dashboard")
        st.markdown("""
        **E-Commerce Analytics Platform**
        
        Built with:
        - 🐍 Python + Streamlit
        - 📊 Plotly for visualizations
        - 🧠 Deterministic Query Engine (no hallucinations!)
        
        **Key Features:**
        - Auto-detect and clean e-commerce data
        - Accurate business metrics (Revenue = Delivered only)
        - Correct RTO calculation (proper denominator)
        - AI chat with SQL transparency
        
        **Business Rules:**
        - Revenue only counts delivered orders
        - AOV only from delivered orders
        - RTO Rate = RTO / (Delivered + RTO) × 100
        """)


if __name__ == "__main__":
    main()

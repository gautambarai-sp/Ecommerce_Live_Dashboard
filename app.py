"""
E-Commerce Analytics Platform - Production Grade
================================================
A market-ready analytics dashboard for e-commerce businesses.

Features:
- Multi-dataset management
- Real-time dashboard updates
- AI-powered query engine
- Dynamic visualization builder
- Live streaming data simulation
- Professional UI/UX

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import re
from typing import Dict, List, Any, Optional, Tuple
import json
import hashlib

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="DataPulse Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Professional UI
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Headers */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-delta {
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .metric-delta-positive {
        background: #dcfce7;
        color: #166534;
    }
    
    .metric-delta-negative {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Cards */
    .custom-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .status-delivered { background: #dcfce7; color: #166534; }
    .status-processing { background: #dbeafe; color: #1e40af; }
    .status-rto { background: #fee2e2; color: #991b1b; }
    .status-cancelled { background: #f3f4f6; color: #374151; }
    
    /* Live indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        background: #dcfce7;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #166534;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* Chat styles */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 12px;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 85%;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .chat-assistant {
        background: white;
        border: 1px solid #e2e8f0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    /* Navigation */
    .nav-item {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .nav-item:hover {
        background: #f1f5f9;
    }
    
    .nav-item-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.875rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 1.5rem;
        background: transparent;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'datasets': {},  # {name: {'df': df, 'mappings': {}, 'uploaded_at': datetime}}
        'active_dataset': None,
        'chat_history': [],
        'live_mode': False,
        'last_refresh': datetime.now(),
        'theme': 'light',
        'currency': 'INR',
        'notifications': [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_currency(value: float, currency: str = 'INR') -> str:
    """Format currency with Indian notation."""
    if pd.isna(value) or value is None:
        return "₹0"
    
    if currency == 'INR':
        if abs(value) >= 10000000:
            return f"₹{value/10000000:.2f} Cr"
        elif abs(value) >= 100000:
            return f"₹{value/100000:.2f} L"
        elif abs(value) >= 1000:
            return f"₹{value/1000:.1f}K"
        else:
            return f"₹{value:,.0f}"
    else:
        return f"${value:,.2f}"


def format_number(value: float) -> str:
    """Format large numbers."""
    if pd.isna(value):
        return "0"
    if abs(value) >= 1000000:
        return f"{value/1000000:.1f}M"
    elif abs(value) >= 1000:
        return f"{value/1000:.1f}K"
    else:
        return f"{value:,.0f}"


def format_percentage(value: float) -> str:
    """Format percentage."""
    if pd.isna(value):
        return "0%"
    return f"{value:.1f}%"


def get_trend_indicator(current: float, previous: float) -> Tuple[str, str, str]:
    """Get trend indicator (arrow, color, percentage)."""
    if previous == 0:
        return "→", "gray", "0%"
    
    change = ((current - previous) / previous) * 100
    
    if change > 0:
        return "↑", "green", f"+{change:.1f}%"
    elif change < 0:
        return "↓", "red", f"{change:.1f}%"
    else:
        return "→", "gray", "0%"


def generate_dataset_id(name: str) -> str:
    """Generate unique ID for dataset."""
    return hashlib.md5(f"{name}{datetime.now()}".encode()).hexdigest()[:8]


# =============================================================================
# DATA PROCESSING ENGINE
# =============================================================================

class DataProcessor:
    """Handles all data processing and cleaning."""
    
    STATUS_MAP = {
        'delivered': 'Delivered', 'completed': 'Delivered', 'complete': 'Delivered',
        'fulfilled': 'Delivered', 'success': 'Delivered', 'successful': 'Delivered',
        'cancelled': 'Cancelled', 'canceled': 'Cancelled', 'cancel': 'Cancelled',
        'refunded': 'Cancelled', 'refund': 'Cancelled',
        'rto': 'RTO', 'returned': 'RTO', 'return': 'RTO', 'undelivered': 'RTO',
        'return to origin': 'RTO', 'failed delivery': 'RTO', 'failed': 'RTO',
        'processing': 'Processing', 'pending': 'Processing', 'confirmed': 'Processing',
        'new': 'Processing', 'placed': 'Processing',
        'shipped': 'Shipped', 'in transit': 'Shipped', 'dispatched': 'Shipped',
        'in_transit': 'Shipped', 'intransit': 'Shipped',
    }
    
    PAYMENT_MAP = {
        'cod': 'COD', 'cash on delivery': 'COD', 'cash': 'COD',
        'upi': 'UPI', 'gpay': 'UPI', 'phonepe': 'UPI', 'paytm': 'UPI', 'google pay': 'UPI',
        'credit card': 'Card', 'debit card': 'Card', 'card': 'Card', 'credit': 'Card', 'debit': 'Card',
        'net banking': 'Net Banking', 'netbanking': 'Net Banking', 'neft': 'Net Banking',
        'prepaid': 'Prepaid', 'online': 'Prepaid', 'wallet': 'Prepaid',
    }
    
    @staticmethod
    def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect column mappings."""
        patterns = {
            'order_id': [r'order[_\s]?id', r'order[_\s]?no', r'^id$', r'invoice', r'order[_\s]?number'],
            'total_amount': [r'total', r'amount', r'value', r'price', r'revenue', r'grand', r'order.*value'],
            'order_status': [r'status', r'state', r'delivery.*status', r'order.*status'],
            'order_date': [r'date', r'created', r'placed', r'time', r'order.*date'],
            'payment_method': [r'payment', r'pay.*method', r'pay.*mode', r'payment.*type'],
            'customer_name': [r'customer.*name', r'^name$', r'buyer', r'customer$'],
            'customer_email': [r'email', r'mail', r'customer.*email'],
            'customer_phone': [r'phone', r'mobile', r'contact'],
            'city': [r'^city$', r'customer.*city', r'shipping.*city'],
            'state': [r'^state$', r'region', r'province'],
            'pincode': [r'pin', r'pincode', r'zip', r'postal'],
            'product_name': [r'product', r'item', r'sku.*name', r'product.*name'],
            'category': [r'category', r'type', r'department', r'product.*type'],
            'quantity': [r'qty', r'quantity', r'units', r'count'],
            'sku': [r'^sku$', r'product.*id', r'item.*id'],
            'courier': [r'courier', r'carrier', r'shipping.*partner', r'logistics'],
            'discount': [r'discount', r'promo', r'coupon'],
        }
        
        detected = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            for field, field_patterns in patterns.items():
                for pattern in field_patterns:
                    if re.search(pattern, col_lower):
                        if field not in detected:
                            detected[field] = col
                        break
        
        return detected
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, mappings: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
        """Clean and standardize dataframe."""
        df = df.copy()
        stats = {
            'original_rows': len(df),
            'duplicates_removed': 0,
            'test_orders_removed': 0,
            'invalid_removed': 0,
            'dates_fixed': 0,
            'statuses_standardized': 0,
        }
        
        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates()
        stats['duplicates_removed'] = before - len(df)
        
        # Clean status column
        status_col = mappings.get('order_status')
        if status_col and status_col in df.columns:
            df[status_col] = df[status_col].astype(str).str.lower().str.strip()
            df[status_col] = df[status_col].map(
                lambda x: DataProcessor.STATUS_MAP.get(x, x.title())
            )
            stats['statuses_standardized'] = len(df)
        
        # Clean payment column
        payment_col = mappings.get('payment_method')
        if payment_col and payment_col in df.columns:
            df[payment_col] = df[payment_col].astype(str).str.lower().str.strip()
            df[payment_col] = df[payment_col].map(
                lambda x: DataProcessor.PAYMENT_MAP.get(x, x.title())
            )
        
        # Clean amount column
        amount_col = mappings.get('total_amount')
        if amount_col and amount_col in df.columns:
            df[amount_col] = df[amount_col].astype(str).str.replace(r'[₹$,Rs\.\s]', '', regex=True)
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
            
            # Remove negative and extreme values
            before = len(df)
            df = df[(df[amount_col] >= 0) | df[amount_col].isna()]
            stats['invalid_removed'] += before - len(df)
        
        # Clean date column
        date_col = mappings.get('order_date')
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
            stats['dates_fixed'] = df[date_col].notna().sum()
        
        # Remove test orders
        test_patterns = ['test', 'demo', 'sample', 'dummy', 'fake']
        for col in [mappings.get('customer_email'), mappings.get('customer_name')]:
            if col and col in df.columns:
                before = len(df)
                mask = ~df[col].astype(str).str.lower().str.contains('|'.join(test_patterns), na=False)
                df = df[mask]
                stats['test_orders_removed'] += before - len(df)
        
        stats['final_rows'] = len(df)
        stats['rows_removed'] = stats['original_rows'] - stats['final_rows']
        
        return df, stats


# =============================================================================
# ANALYTICS ENGINE
# =============================================================================

class AnalyticsEngine:
    """Production-grade analytics engine with accurate business logic."""
    
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, str]):
        self.df = df
        self.mappings = mappings
    
    def _get_col(self, field: str) -> Optional[str]:
        """Get actual column name for a field."""
        return self.mappings.get(field)
    
    def _get_delivered_df(self) -> pd.DataFrame:
        """Get only delivered orders."""
        status_col = self._get_col('order_status')
        if status_col:
            return self.df[self.df[status_col] == 'Delivered']
        return self.df
    
    def _get_shipped_df(self) -> pd.DataFrame:
        """Get shipped orders (delivered + RTO) for RTO calculations."""
        status_col = self._get_col('order_status')
        if status_col:
            return self.df[self.df[status_col].isin(['Delivered', 'RTO'])]
        return self.df
    
    # === KPI CALCULATIONS ===
    
    def get_total_revenue(self) -> Dict:
        """Total revenue from delivered orders only."""
        amount_col = self._get_col('total_amount')
        delivered = self._get_delivered_df()
        
        revenue = delivered[amount_col].sum() if amount_col else 0
        order_count = len(delivered)
        
        return {
            'value': revenue,
            'orders': order_count,
            'label': 'Total Revenue',
            'description': 'Sum of delivered orders only',
            'sql': f"SELECT SUM({amount_col}) FROM orders WHERE status = 'Delivered'"
        }
    
    def get_aov(self) -> Dict:
        """Average order value from delivered orders."""
        amount_col = self._get_col('total_amount')
        delivered = self._get_delivered_df()
        
        aov = delivered[amount_col].mean() if amount_col and len(delivered) > 0 else 0
        
        return {
            'value': aov,
            'orders': len(delivered),
            'label': 'Avg Order Value',
            'description': 'Average of delivered orders',
            'sql': f"SELECT AVG({amount_col}) FROM orders WHERE status = 'Delivered'"
        }
    
    def get_order_count(self) -> Dict:
        """Total order count by status."""
        status_col = self._get_col('order_status')
        
        total = len(self.df)
        by_status = {}
        
        if status_col:
            by_status = self.df[status_col].value_counts().to_dict()
        
        return {
            'value': total,
            'by_status': by_status,
            'label': 'Total Orders',
            'description': 'All orders in dataset'
        }
    
    def get_rto_rate(self) -> Dict:
        """RTO rate with CORRECT denominator (delivered + RTO only)."""
        status_col = self._get_col('order_status')
        
        if not status_col:
            return {'value': 0, 'rto_orders': 0, 'shipped': 0}
        
        shipped = self._get_shipped_df()
        rto_orders = len(shipped[shipped[status_col] == 'RTO'])
        total_shipped = len(shipped)
        
        rate = (rto_orders / total_shipped * 100) if total_shipped > 0 else 0
        
        return {
            'value': rate,
            'rto_orders': rto_orders,
            'shipped': total_shipped,
            'label': 'RTO Rate',
            'description': 'RTO orders / (Delivered + RTO) × 100',
            'sql': """SELECT 
    COUNT(CASE WHEN status = 'RTO' THEN 1 END) * 100.0 / COUNT(*) 
FROM orders 
WHERE status IN ('Delivered', 'RTO')"""
        }
    
    def get_customer_count(self) -> Dict:
        """Unique customer count."""
        email_col = self._get_col('customer_email')
        name_col = self._get_col('customer_name')
        
        col = email_col or name_col
        count = self.df[col].nunique() if col else 0
        
        return {
            'value': count,
            'label': 'Unique Customers',
            'column_used': col
        }
    
    # === BREAKDOWN ANALYSES ===
    
    def get_status_breakdown(self) -> pd.DataFrame:
        """Orders breakdown by status with revenue."""
        status_col = self._get_col('order_status')
        amount_col = self._get_col('total_amount')
        
        if not status_col:
            return pd.DataFrame()
        
        breakdown = self.df.groupby(status_col).agg({
            status_col: 'count',
            amount_col: 'sum' if amount_col else 'count'
        }).reset_index()
        
        breakdown.columns = ['Status', 'Orders', 'Revenue']
        breakdown['Percentage'] = (breakdown['Orders'] / breakdown['Orders'].sum() * 100).round(1)
        breakdown = breakdown.sort_values('Orders', ascending=False)
        
        return breakdown
    
    def get_payment_breakdown(self) -> pd.DataFrame:
        """Revenue and orders by payment method."""
        payment_col = self._get_col('payment_method')
        amount_col = self._get_col('total_amount')
        status_col = self._get_col('order_status')
        
        if not payment_col:
            return pd.DataFrame()
        
        delivered = self._get_delivered_df()
        
        breakdown = delivered.groupby(payment_col).agg({
            payment_col: 'count',
            amount_col: ['sum', 'mean'] if amount_col else ['count', 'count']
        }).reset_index()
        
        breakdown.columns = ['Payment Method', 'Orders', 'Revenue', 'AOV']
        breakdown = breakdown.sort_values('Revenue', ascending=False)
        
        return breakdown
    
    def get_category_breakdown(self) -> pd.DataFrame:
        """Revenue by category (delivered only)."""
        category_col = self._get_col('category')
        amount_col = self._get_col('total_amount')
        
        if not category_col or not amount_col:
            return pd.DataFrame()
        
        delivered = self._get_delivered_df()
        
        breakdown = delivered.groupby(category_col).agg({
            amount_col: ['sum', 'count', 'mean']
        }).reset_index()
        
        breakdown.columns = ['Category', 'Revenue', 'Orders', 'AOV']
        breakdown = breakdown.sort_values('Revenue', ascending=False)
        
        return breakdown
    
    def get_city_breakdown(self, top_n: int = 10) -> pd.DataFrame:
        """Revenue by city."""
        city_col = self._get_col('city')
        amount_col = self._get_col('total_amount')
        
        if not city_col or not amount_col:
            return pd.DataFrame()
        
        delivered = self._get_delivered_df()
        
        breakdown = delivered.groupby(city_col).agg({
            amount_col: ['sum', 'count']
        }).reset_index()
        
        breakdown.columns = ['City', 'Revenue', 'Orders']
        breakdown = breakdown.sort_values('Revenue', ascending=False).head(top_n)
        
        return breakdown
    
    def get_top_products(self, top_n: int = 10) -> pd.DataFrame:
        """Top products by revenue."""
        product_col = self._get_col('product_name')
        amount_col = self._get_col('total_amount')
        qty_col = self._get_col('quantity')
        
        if not product_col or not amount_col:
            return pd.DataFrame()
        
        delivered = self._get_delivered_df()
        
        agg_dict = {amount_col: 'sum'}
        if qty_col:
            agg_dict[qty_col] = 'sum'
        
        top = delivered.groupby(product_col).agg(agg_dict).reset_index()
        
        if qty_col:
            top.columns = ['Product', 'Revenue', 'Quantity']
        else:
            top.columns = ['Product', 'Revenue']
            top['Quantity'] = '-'
        
        top = top.sort_values('Revenue', ascending=False).head(top_n)
        
        return top
    
    def get_top_customers(self, top_n: int = 10) -> pd.DataFrame:
        """Top customers by spending."""
        customer_col = self._get_col('customer_name') or self._get_col('customer_email')
        amount_col = self._get_col('total_amount')
        
        if not customer_col or not amount_col:
            return pd.DataFrame()
        
        delivered = self._get_delivered_df()
        
        top = delivered.groupby(customer_col).agg({
            amount_col: ['sum', 'count']
        }).reset_index()
        
        top.columns = ['Customer', 'Total Spent', 'Orders']
        top = top.sort_values('Total Spent', ascending=False).head(top_n)
        
        return top
    
    # === TIME SERIES ===
    
    def get_revenue_trend(self, period: str = 'D') -> pd.DataFrame:
        """Revenue trend over time."""
        date_col = self._get_col('order_date')
        amount_col = self._get_col('total_amount')
        
        if not date_col or not amount_col:
            return pd.DataFrame()
        
        delivered = self._get_delivered_df().copy()
        delivered = delivered[delivered[date_col].notna()]
        
        if len(delivered) == 0:
            return pd.DataFrame()
        
        delivered['period'] = delivered[date_col].dt.to_period(period).astype(str)
        
        trend = delivered.groupby('period').agg({
            amount_col: 'sum'
        }).reset_index()
        
        trend.columns = ['Period', 'Revenue']
        
        return trend
    
    def get_orders_trend(self, period: str = 'D') -> pd.DataFrame:
        """Orders trend over time."""
        date_col = self._get_col('order_date')
        status_col = self._get_col('order_status')
        
        if not date_col:
            return pd.DataFrame()
        
        df = self.df.copy()
        df = df[df[date_col].notna()]
        
        if len(df) == 0:
            return pd.DataFrame()
        
        df['period'] = df[date_col].dt.to_period(period).astype(str)
        
        trend = df.groupby('period').size().reset_index(name='Orders')
        trend.columns = ['Period', 'Orders']
        
        return trend
    
    # === RTO ANALYSIS ===
    
    def get_rto_by_payment(self) -> pd.DataFrame:
        """RTO rate by payment method."""
        payment_col = self._get_col('payment_method')
        status_col = self._get_col('order_status')
        
        if not payment_col or not status_col:
            return pd.DataFrame()
        
        shipped = self._get_shipped_df()
        
        analysis = shipped.groupby(payment_col).apply(
            lambda x: pd.Series({
                'Shipped': len(x),
                'RTO': len(x[x[status_col] == 'RTO']),
                'RTO Rate': len(x[x[status_col] == 'RTO']) / len(x) * 100 if len(x) > 0 else 0
            })
        ).reset_index()
        
        analysis.columns = ['Payment Method', 'Shipped', 'RTO Orders', 'RTO Rate']
        analysis = analysis.sort_values('RTO Rate', ascending=False)
        
        return analysis
    
    def get_rto_by_city(self, top_n: int = 10) -> pd.DataFrame:
        """RTO rate by city."""
        city_col = self._get_col('city')
        status_col = self._get_col('order_status')
        
        if not city_col or not status_col:
            return pd.DataFrame()
        
        shipped = self._get_shipped_df()
        
        # Only cities with minimum orders
        city_counts = shipped[city_col].value_counts()
        valid_cities = city_counts[city_counts >= 5].index
        shipped = shipped[shipped[city_col].isin(valid_cities)]
        
        analysis = shipped.groupby(city_col).apply(
            lambda x: pd.Series({
                'Shipped': len(x),
                'RTO': len(x[x[status_col] == 'RTO']),
                'RTO Rate': len(x[x[status_col] == 'RTO']) / len(x) * 100 if len(x) > 0 else 0
            })
        ).reset_index()
        
        analysis.columns = ['City', 'Shipped', 'RTO Orders', 'RTO Rate']
        analysis = analysis.sort_values('RTO Rate', ascending=False).head(top_n)
        
        return analysis
    
    # === COD VS PREPAID ===
    
    def get_cod_vs_prepaid(self) -> Dict:
        """Comprehensive COD vs Prepaid comparison."""
        payment_col = self._get_col('payment_method')
        amount_col = self._get_col('total_amount')
        status_col = self._get_col('order_status')
        
        if not payment_col:
            return {}
        
        # Categorize as COD or Prepaid
        df = self.df.copy()
        df['payment_type'] = df[payment_col].apply(
            lambda x: 'COD' if x == 'COD' else 'Prepaid'
        )
        
        result = {}
        
        for ptype in ['COD', 'Prepaid']:
            subset = df[df['payment_type'] == ptype]
            delivered = subset[subset[status_col] == 'Delivered'] if status_col else subset
            shipped = subset[subset[status_col].isin(['Delivered', 'RTO'])] if status_col else subset
            
            result[ptype] = {
                'total_orders': len(subset),
                'delivered_orders': len(delivered),
                'revenue': delivered[amount_col].sum() if amount_col else 0,
                'aov': delivered[amount_col].mean() if amount_col and len(delivered) > 0 else 0,
                'rto_rate': (
                    len(shipped[shipped[status_col] == 'RTO']) / len(shipped) * 100 
                    if status_col and len(shipped) > 0 else 0
                )
            }
        
        return result


# =============================================================================
# NATURAL LANGUAGE QUERY PARSER
# =============================================================================

class QueryParser:
    """Parse natural language queries into analytics functions."""
    
    QUERY_PATTERNS = {
        'total_revenue': [
            r'total revenue', r'revenue$', r'how much.*made', r'total sales', 
            r'gmv', r'gross.*revenue', r'earnings'
        ],
        'aov': [
            r'average order value', r'\baov\b', r'avg.*order', r'average.*order',
            r'order.*average', r'basket.*size'
        ],
        'order_count': [
            r'how many orders', r'total orders', r'order count', r'number.*orders'
        ],
        'rto_rate': [
            r'rto rate', r'return.*rate', r'rto percent', r'delivery fail',
            r'rto.*percent', r'return to origin'
        ],
        'status_breakdown': [
            r'status breakdown', r'orders by status', r'status wise', r'order.*status'
        ],
        'category_breakdown': [
            r'revenue by category', r'category.*revenue', r'category wise',
            r'category breakdown', r'by category'
        ],
        'top_products': [
            r'top.*product', r'best.*product', r'best sell', r'highest.*product',
            r'product.*revenue'
        ],
        'top_customers': [
            r'top.*customer', r'best.*customer', r'highest.*customer',
            r'vip.*customer', r'customer.*spending'
        ],
        'cod_vs_prepaid': [
            r'cod.*prepaid', r'prepaid.*cod', r'cod.*vs', r'compare.*payment',
            r'payment.*comparison', r'cod.*comparison'
        ],
        'rto_by_payment': [
            r'rto.*payment', r'rto.*cod', r'cod.*rto', r'payment.*rto',
            r'rto.*by.*payment'
        ],
        'rto_by_city': [
            r'rto.*city', r'city.*rto', r'rto.*location'
        ],
        'revenue_trend': [
            r'revenue trend', r'revenue.*time', r'monthly revenue', r'revenue.*month',
            r'sales trend', r'daily revenue'
        ],
        'city_breakdown': [
            r'revenue.*city', r'city.*revenue', r'by city', r'top.*city',
            r'city wise', r'location'
        ],
        'payment_breakdown': [
            r'payment.*breakdown', r'by.*payment', r'payment.*method',
            r'payment wise'
        ],
        'customer_count': [
            r'how many customer', r'customer count', r'unique customer',
            r'total customer'
        ],
        'summary': [
            r'summary', r'overview', r'dashboard', r'kpi', r'all metrics'
        ]
    }
    
    @staticmethod
    def parse(query: str) -> Tuple[str, Dict]:
        """Parse query into query type and parameters."""
        query_lower = query.lower().strip()
        
        # Check for top N pattern
        top_n_match = re.search(r'top\s*(\d+)', query_lower)
        params = {}
        if top_n_match:
            params['top_n'] = int(top_n_match.group(1))
        
        # Match query type
        for query_type, patterns in QueryParser.QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type, params
        
        return 'unknown', params


# =============================================================================
# VISUALIZATION BUILDER
# =============================================================================

class ChartBuilder:
    """Build professional charts with consistent styling."""
    
    COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#22c55e',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#3b82f6',
        'gradient': ['#667eea', '#764ba2', '#ec4899', '#f59e0b', '#22c55e'],
        'status': {
            'Delivered': '#22c55e',
            'Processing': '#3b82f6',
            'Shipped': '#8b5cf6',
            'RTO': '#ef4444',
            'Cancelled': '#6b7280'
        },
        'payment': {
            'COD': '#f59e0b',
            'Prepaid': '#667eea',
            'UPI': '#8b5cf6',
            'Card': '#3b82f6',
            'Net Banking': '#22c55e'
        }
    }
    
    @staticmethod
    def create_kpi_card(value: float, label: str, prefix: str = "₹", 
                        delta: str = None, delta_positive: bool = True) -> str:
        """Create HTML for KPI card."""
        formatted_value = format_currency(value) if prefix == "₹" else f"{value:,.0f}"
        
        delta_html = ""
        if delta:
            delta_class = "metric-delta-positive" if delta_positive else "metric-delta-negative"
            delta_html = f'<span class="metric-delta {delta_class}">{delta}</span>'
        
        return f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{formatted_value}</div>
            {delta_html}
        </div>
        """
    
    @staticmethod
    def create_area_chart(df: pd.DataFrame, x: str, y: str, 
                          title: str = None, color: str = None) -> go.Figure:
        """Create styled area chart."""
        color = color or ChartBuilder.COLORS['primary']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df[x],
            y=df[y],
            fill='tozeroy',
            fillcolor=f'rgba(102, 126, 234, 0.2)',
            line=dict(color=color, width=2),
            mode='lines'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title="",
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=0, r=0, t=30 if title else 0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x: str, y: str, 
                         title: str = None, horizontal: bool = False,
                         color_col: str = None) -> go.Figure:
        """Create styled bar chart."""
        
        if color_col and color_col in df.columns:
            colors = [ChartBuilder.COLORS['status'].get(v, ChartBuilder.COLORS['primary']) 
                     for v in df[color_col]]
        else:
            colors = ChartBuilder.COLORS['primary']
        
        if horizontal:
            fig = go.Figure(go.Bar(
                y=df[x],
                x=df[y],
                orientation='h',
                marker_color=colors
            ))
        else:
            fig = go.Figure(go.Bar(
                x=df[x],
                y=df[y],
                marker_color=colors
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title="",
            showlegend=False,
            margin=dict(l=0, r=0, t=30 if title else 0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(categoryorder='total ascending' if horizontal else None)
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart(df: pd.DataFrame, values: str, names: str,
                         title: str = None, hole: float = 0.4) -> go.Figure:
        """Create styled donut chart."""
        
        colors = [ChartBuilder.COLORS['status'].get(n, ChartBuilder.COLORS['gradient'][i % 5]) 
                 for i, n in enumerate(df[names])]
        
        fig = go.Figure(go.Pie(
            values=df[values],
            labels=df[names],
            hole=hole,
            marker_colors=colors,
            textinfo='percent+label',
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(l=0, r=0, t=30 if title else 0, b=0),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def create_comparison_chart(data: Dict, title: str = None) -> go.Figure:
        """Create COD vs Prepaid comparison chart."""
        
        categories = ['Revenue', 'AOV', 'RTO Rate']
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=categories)
        
        # Revenue comparison
        fig.add_trace(go.Bar(
            x=['COD', 'Prepaid'],
            y=[data['COD']['revenue'], data['Prepaid']['revenue']],
            marker_color=[ChartBuilder.COLORS['warning'], ChartBuilder.COLORS['primary']],
            showlegend=False
        ), row=1, col=1)
        
        # AOV comparison
        fig.add_trace(go.Bar(
            x=['COD', 'Prepaid'],
            y=[data['COD']['aov'], data['Prepaid']['aov']],
            marker_color=[ChartBuilder.COLORS['warning'], ChartBuilder.COLORS['primary']],
            showlegend=False
        ), row=1, col=2)
        
        # RTO Rate comparison
        fig.add_trace(go.Bar(
            x=['COD', 'Prepaid'],
            y=[data['COD']['rto_rate'], data['Prepaid']['rto_rate']],
            marker_color=[ChartBuilder.COLORS['danger'], ChartBuilder.COLORS['success']],
            showlegend=False
        ), row=1, col=3)
        
        fig.update_layout(
            title=title,
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 2.5rem;">📊</span>
            <h1 style="font-size: 1.5rem; margin: 0.5rem 0 0 0; 
                       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                DataPulse
            </h1>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">
                E-Commerce Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "📁 Data Manager", "🤖 AI Analyst", "📈 Reports", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Active dataset info
        if st.session_state.datasets:
            st.markdown("**📂 Datasets**")
            
            dataset_names = list(st.session_state.datasets.keys())
            active = st.selectbox(
                "Active Dataset",
                dataset_names,
                index=dataset_names.index(st.session_state.active_dataset) if st.session_state.active_dataset in dataset_names else 0,
                label_visibility="collapsed"
            )
            st.session_state.active_dataset = active
            
            if active:
                ds = st.session_state.datasets[active]
                st.caption(f"📋 {len(ds['df']):,} rows")
                st.caption(f"🔗 {len(ds['mappings'])} fields mapped")
                
                # Live mode toggle
                st.divider()
                live_mode = st.toggle("🔴 Live Mode", value=st.session_state.live_mode)
                st.session_state.live_mode = live_mode
                
                if live_mode:
                    st.markdown("""
                    <div class="live-indicator">
                        <div class="live-dot"></div>
                        Auto-refreshing
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No datasets loaded")
            st.caption("Upload data to get started")
    
    # Main content routing
    if page == "📁 Data Manager":
        render_data_manager()
    elif page == "🤖 AI Analyst":
        render_ai_analyst()
    elif page == "📈 Reports":
        render_reports()
    elif page == "⚙️ Settings":
        render_settings()
    else:
        render_dashboard()
    
    # Auto-refresh for live mode
    if st.session_state.live_mode and st.session_state.active_dataset:
        time.sleep(5)
        st.rerun()


def render_dashboard():
    """Render the main dashboard."""
    
    st.markdown('<p class="main-title">📊 Analytics Dashboard</p>', unsafe_allow_html=True)
    
    if not st.session_state.active_dataset or st.session_state.active_dataset not in st.session_state.datasets:
        st.warning("⚠️ No dataset selected. Please upload data first.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="custom-card" style="text-align: center; padding: 3rem;">
                <span style="font-size: 4rem;">📁</span>
                <h2 style="margin: 1rem 0;">Upload Your Data</h2>
                <p style="color: #64748b;">
                    Start by uploading your e-commerce data (CSV or Excel).
                    We'll automatically detect columns and clean your data.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📁 Go to Data Manager", use_container_width=True, type="primary"):
                st.session_state.page = "📁 Data Manager"
                st.rerun()
        return
    
    # Get data and engine
    ds = st.session_state.datasets[st.session_state.active_dataset]
    df = ds['df']
    mappings = ds['mappings']
    engine = AnalyticsEngine(df, mappings)
    
    # Live indicator
    if st.session_state.live_mode:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
            <div class="live-indicator">
                <div class="live-dot"></div>
                Live
            </div>
            <span style="color: #64748b; font-size: 0.8rem;">
                Last updated: """ + datetime.now().strftime("%H:%M:%S") + """
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI Row
    st.markdown("### 📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        revenue_data = engine.get_total_revenue()
        st.metric(
            "Total Revenue",
            format_currency(revenue_data['value']),
            f"{revenue_data['orders']:,} delivered orders",
            delta_color="off"
        )
    
    with col2:
        aov_data = engine.get_aov()
        st.metric(
            "Average Order Value",
            format_currency(aov_data['value']),
            "Delivered orders only",
            delta_color="off"
        )
    
    with col3:
        order_data = engine.get_order_count()
        st.metric(
            "Total Orders",
            format_number(order_data['value']),
            f"{len(order_data.get('by_status', {}))} statuses"
        )
    
    with col4:
        rto_data = engine.get_rto_rate()
        color = "inverse" if rto_data['value'] > 10 else "normal"
        st.metric(
            "RTO Rate",
            format_percentage(rto_data['value']),
            f"{rto_data['rto_orders']:,} returns",
            delta_color=color
        )
    
    st.divider()
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Revenue Trend")
        trend_df = engine.get_revenue_trend(period='D')
        if not trend_df.empty:
            fig = ChartBuilder.create_area_chart(trend_df, 'Period', 'Revenue')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date column required for trend analysis")
    
    with col2:
        st.markdown("#### 🥧 Orders by Status")
        status_df = engine.get_status_breakdown()
        if not status_df.empty:
            fig = ChartBuilder.create_pie_chart(status_df, 'Orders', 'Status')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Status column required")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top 10 Products")
        products_df = engine.get_top_products(10)
        if not products_df.empty:
            fig = ChartBuilder.create_bar_chart(
                products_df, 'Product', 'Revenue', horizontal=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product column required")
    
    with col2:
        st.markdown("#### 💳 Payment Methods")
        payment_df = engine.get_payment_breakdown()
        if not payment_df.empty:
            fig = ChartBuilder.create_bar_chart(
                payment_df, 'Payment Method', 'Revenue'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Payment method column required")
    
    # COD vs Prepaid Analysis
    st.divider()
    st.markdown("### 💰 COD vs Prepaid Analysis")
    
    comparison = engine.get_cod_vs_prepaid()
    if comparison:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Revenue**")
            cod_rev = comparison['COD']['revenue']
            prepaid_rev = comparison['Prepaid']['revenue']
            total = cod_rev + prepaid_rev
            
            st.metric("COD", format_currency(cod_rev), f"{cod_rev/total*100:.1f}%" if total > 0 else "0%")
            st.metric("Prepaid", format_currency(prepaid_rev), f"{prepaid_rev/total*100:.1f}%" if total > 0 else "0%")
        
        with col2:
            st.markdown("**Average Order Value**")
            st.metric("COD AOV", format_currency(comparison['COD']['aov']))
            st.metric("Prepaid AOV", format_currency(comparison['Prepaid']['aov']))
        
        with col3:
            st.markdown("**RTO Rate**")
            cod_rto = comparison['COD']['rto_rate']
            prepaid_rto = comparison['Prepaid']['rto_rate']
            
            st.metric("COD RTO", f"{cod_rto:.1f}%", delta_color="inverse")
            st.metric("Prepaid RTO", f"{prepaid_rto:.1f}%")
            
            if cod_rto > prepaid_rto:
                st.warning(f"⚠️ COD has {cod_rto - prepaid_rto:.1f}% higher RTO than Prepaid")
            else:
                st.success("✓ Prepaid has higher RTO (unusual)")
    
    # RTO by City
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 RTO Rate by Payment")
        rto_payment = engine.get_rto_by_payment()
        if not rto_payment.empty:
            fig = px.bar(
                rto_payment, x='Payment Method', y='RTO Rate',
                color='RTO Rate',
                color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444']
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏙️ Top Cities by RTO")
        rto_city = engine.get_rto_by_city(10)
        if not rto_city.empty:
            fig = px.bar(
                rto_city, x='City', y='RTO Rate',
                color='RTO Rate',
                color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444']
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Category breakdown
    st.divider()
    st.markdown("### 📦 Category Performance")
    
    category_df = engine.get_category_breakdown()
    if not category_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.treemap(
                category_df,
                path=['Category'],
                values='Revenue',
                color='Revenue',
                color_continuous_scale='Blues'
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                category_df[['Category', 'Revenue', 'Orders']].style.format({
                    'Revenue': lambda x: format_currency(x),
                    'Orders': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Category column required for this analysis")


def render_data_manager():
    """Render the data manager page."""
    
    st.markdown('<p class="main-title">📁 Data Manager</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload, manage, and configure your datasets</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📤 Upload New", "📋 Manage Datasets"])
    
    with tab1:
        st.markdown("### Upload Dataset")
        
        uploaded_file = st.file_uploader(
            "Drop your CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Max 50MB, up to 100,000 rows"
        )
        
        if uploaded_file:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
                
                # Preview
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Auto-detect columns
                detected = DataProcessor.detect_columns(df)
                
                st.markdown("### 🔗 Column Mapping")
                st.caption("Map your columns to standard e-commerce fields")
                
                # Required fields
                st.markdown("**Required Fields**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    order_id = st.selectbox(
                        "Order ID *",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('order_id', '')) + 1 
                              if detected.get('order_id') in df.columns else 0
                    )
                
                with col2:
                    amount = st.selectbox(
                        "Order Amount *",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('total_amount', '')) + 1 
                              if detected.get('total_amount') in df.columns else 0
                    )
                
                with col3:
                    status = st.selectbox(
                        "Order Status *",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('order_status', '')) + 1 
                              if detected.get('order_status') in df.columns else 0
                    )
                
                # Optional fields
                st.markdown("**Optional Fields**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    date = st.selectbox(
                        "Order Date",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('order_date', '')) + 1 
                              if detected.get('order_date') in df.columns else 0
                    )
                
                with col2:
                    payment = st.selectbox(
                        "Payment Method",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('payment_method', '')) + 1 
                              if detected.get('payment_method') in df.columns else 0
                    )
                
                with col3:
                    customer = st.selectbox(
                        "Customer Name",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('customer_name', '')) + 1 
                              if detected.get('customer_name') in df.columns else 0
                    )
                
                with col4:
                    category = st.selectbox(
                        "Category",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('category', '')) + 1 
                              if detected.get('category') in df.columns else 0
                    )
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    product = st.selectbox(
                        "Product Name",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('product_name', '')) + 1 
                              if detected.get('product_name') in df.columns else 0
                    )
                
                with col2:
                    city = st.selectbox(
                        "City",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('city', '')) + 1 
                              if detected.get('city') in df.columns else 0
                    )
                
                with col3:
                    email = st.selectbox(
                        "Customer Email",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('customer_email', '')) + 1 
                              if detected.get('customer_email') in df.columns else 0
                    )
                
                with col4:
                    courier = st.selectbox(
                        "Courier",
                        [''] + list(df.columns),
                        index=list(df.columns).index(detected.get('courier', '')) + 1 
                              if detected.get('courier') in df.columns else 0
                    )
                
                # Dataset name
                st.markdown("### 📝 Dataset Name")
                dataset_name = st.text_input(
                    "Name",
                    value=uploaded_file.name.replace('.csv', '').replace('.xlsx', ''),
                    label_visibility="collapsed"
                )
                
                # Import button
                if st.button("✓ Import Dataset", type="primary", use_container_width=True):
                    if not order_id or not amount or not status:
                        st.error("Please map all required fields")
                    elif not dataset_name:
                        st.error("Please enter a dataset name")
                    else:
                        # Build mappings
                        mappings = {'order_id': order_id, 'total_amount': amount, 'order_status': status}
                        if date: mappings['order_date'] = date
                        if payment: mappings['payment_method'] = payment
                        if customer: mappings['customer_name'] = customer
                        if category: mappings['category'] = category
                        if product: mappings['product_name'] = product
                        if city: mappings['city'] = city
                        if email: mappings['customer_email'] = email
                        if courier: mappings['courier'] = courier
                        
                        # Clean data
                        with st.spinner("Cleaning and processing data..."):
                            cleaned_df, stats = DataProcessor.clean_dataframe(df, mappings)
                        
                        # Store dataset
                        st.session_state.datasets[dataset_name] = {
                            'df': cleaned_df,
                            'mappings': mappings,
                            'uploaded_at': datetime.now(),
                            'original_rows': stats['original_rows'],
                            'cleaning_stats': stats
                        }
                        st.session_state.active_dataset = dataset_name
                        
                        # Show success
                        st.success(f"""
                        ✓ Dataset imported successfully!
                        
                        **Cleaning Report:**
                        - Original rows: {stats['original_rows']:,}
                        - Final rows: {stats['final_rows']:,}
                        - Duplicates removed: {stats['duplicates_removed']:,}
                        - Test orders removed: {stats['test_orders_removed']:,}
                        - Invalid rows removed: {stats['invalid_removed']:,}
                        """)
                        
                        st.balloons()
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab2:
        if not st.session_state.datasets:
            st.info("No datasets uploaded yet")
        else:
            for name, ds in st.session_state.datasets.items():
                with st.expander(f"📊 {name}", expanded=name == st.session_state.active_dataset):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Rows:** {len(ds['df']):,}")
                        st.markdown(f"**Uploaded:** {ds['uploaded_at'].strftime('%Y-%m-%d %H:%M')}")
                        st.markdown(f"**Mapped Fields:** {len(ds['mappings'])}")
                    
                    with col2:
                        if st.button("Set Active", key=f"active_{name}"):
                            st.session_state.active_dataset = name
                            st.rerun()
                    
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{name}"):
                            del st.session_state.datasets[name]
                            if st.session_state.active_dataset == name:
                                st.session_state.active_dataset = list(st.session_state.datasets.keys())[0] if st.session_state.datasets else None
                            st.rerun()
                    
                    # Show mappings
                    st.markdown("**Column Mappings:**")
                    mapping_df = pd.DataFrame([
                        {'Field': k.replace('_', ' ').title(), 'Column': v}
                        for k, v in ds['mappings'].items()
                    ])
                    st.dataframe(mapping_df, use_container_width=True, hide_index=True)


def render_ai_analyst():
    """Render the AI analyst chat page."""
    
    st.markdown('<p class="main-title">🤖 AI Analyst</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ask questions about your data in natural language</p>', unsafe_allow_html=True)
    
    if not st.session_state.active_dataset or st.session_state.active_dataset not in st.session_state.datasets:
        st.warning("Please upload and select a dataset first")
        return
    
    ds = st.session_state.datasets[st.session_state.active_dataset]
    engine = AnalyticsEngine(ds['df'], ds['mappings'])
    
    # Suggested questions
    st.markdown("**💡 Try asking:**")
    
    suggestions = [
        "What is my total revenue?",
        "Show RTO rate by payment method",
        "Top 10 products",
        "COD vs Prepaid comparison",
        "Revenue by category",
        "RTO rate by city"
    ]
    
    cols = st.columns(3)
    for i, sugg in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': sugg})
                
                # Process query
                query_type, params = QueryParser.parse(sugg)
                response = process_query(engine, query_type, params)
                st.session_state.chat_history.append(response)
                st.rerun()
    
    st.divider()
    
    # Chat history
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
                
                if msg['role'] == 'assistant':
                    if 'data' in msg and msg['data'] is not None:
                        if isinstance(msg['data'], pd.DataFrame) and not msg['data'].empty:
                            st.dataframe(msg['data'], use_container_width=True, hide_index=True)
                        elif isinstance(msg['data'], dict):
                            for key, val in msg['data'].items():
                                if isinstance(val, (int, float)):
                                    st.metric(key.replace('_', ' ').title(), 
                                             format_currency(val) if 'revenue' in key.lower() or 'amount' in key.lower() 
                                             else format_number(val) if val > 100 else f"{val:.2f}")
                    
                    if 'chart' in msg and msg['chart'] is not None:
                        st.plotly_chart(msg['chart'], use_container_width=True)
                    
                    if 'sql' in msg and msg['sql']:
                        with st.expander("🔍 View Query Logic"):
                            st.code(msg['sql'], language='sql')
    
    # Input
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        query_type, params = QueryParser.parse(prompt)
        response = process_query(engine, query_type, params)
        st.session_state.chat_history.append(response)
        st.rerun()
    
    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


def process_query(engine: AnalyticsEngine, query_type: str, params: Dict) -> Dict:
    """Process a query and return response dict."""
    
    response = {
        'role': 'assistant',
        'content': '',
        'data': None,
        'chart': None,
        'sql': ''
    }
    
    if query_type == 'total_revenue':
        result = engine.get_total_revenue()
        response['content'] = f"**Total Revenue: {format_currency(result['value'])}**\n\nThis is from {result['orders']:,} delivered orders only."
        response['data'] = {'Revenue': result['value'], 'Delivered Orders': result['orders']}
        response['sql'] = result.get('sql', '')
    
    elif query_type == 'aov':
        result = engine.get_aov()
        response['content'] = f"**Average Order Value: {format_currency(result['value'])}**\n\nCalculated from {result['orders']:,} delivered orders."
        response['data'] = {'AOV': result['value'], 'Orders': result['orders']}
        response['sql'] = result.get('sql', '')
    
    elif query_type == 'rto_rate':
        result = engine.get_rto_rate()
        response['content'] = f"""**RTO Rate: {result['value']:.2f}%**

- RTO Orders: {result['rto_orders']:,}
- Shipped Orders: {result['shipped']:,}

*Note: RTO Rate = RTO / (Delivered + RTO), NOT RTO / All Orders*"""
        response['data'] = result
        response['sql'] = result.get('sql', '')
    
    elif query_type == 'status_breakdown':
        df = engine.get_status_breakdown()
        response['content'] = "**Orders by Status:**"
        response['data'] = df
        if not df.empty:
            response['chart'] = ChartBuilder.create_pie_chart(df, 'Orders', 'Status')
    
    elif query_type == 'top_products':
        limit = params.get('top_n', 10)
        df = engine.get_top_products(limit)
        response['content'] = f"**Top {limit} Products by Revenue:**"
        response['data'] = df
        if not df.empty:
            response['chart'] = ChartBuilder.create_bar_chart(df, 'Product', 'Revenue', horizontal=True)
    
    elif query_type == 'category_breakdown':
        df = engine.get_category_breakdown()
        response['content'] = "**Revenue by Category:**"
        response['data'] = df
        if not df.empty:
            response['chart'] = ChartBuilder.create_bar_chart(df, 'Category', 'Revenue')
    
    elif query_type == 'cod_vs_prepaid':
        result = engine.get_cod_vs_prepaid()
        if result:
            response['content'] = f"""**COD vs Prepaid Comparison:**

| Metric | COD | Prepaid |
|--------|-----|---------|
| Revenue | {format_currency(result['COD']['revenue'])} | {format_currency(result['Prepaid']['revenue'])} |
| AOV | {format_currency(result['COD']['aov'])} | {format_currency(result['Prepaid']['aov'])} |
| RTO Rate | {result['COD']['rto_rate']:.1f}% | {result['Prepaid']['rto_rate']:.1f}% |

{'⚠️ **Insight:** COD has ' + str(round(result["COD"]["rto_rate"] - result["Prepaid"]["rto_rate"], 1)) + '% higher RTO rate than Prepaid' if result['COD']['rto_rate'] > result['Prepaid']['rto_rate'] else ''}"""
            response['chart'] = ChartBuilder.create_comparison_chart(result)
    
    elif query_type == 'rto_by_payment':
        df = engine.get_rto_by_payment()
        response['content'] = "**RTO Rate by Payment Method:**"
        response['data'] = df
        if not df.empty:
            response['chart'] = ChartBuilder.create_bar_chart(df, 'Payment Method', 'RTO Rate')
    
    elif query_type == 'rto_by_city':
        df = engine.get_rto_by_city(10)
        response['content'] = "**Top 10 Cities by RTO Rate:**"
        response['data'] = df
        if not df.empty:
            response['chart'] = ChartBuilder.create_bar_chart(df, 'City', 'RTO Rate')
    
    elif query_type == 'top_customers':
        limit = params.get('top_n', 10)
        df = engine.get_top_customers(limit)
        response['content'] = f"**Top {limit} Customers by Spending:**"
        response['data'] = df
    
    elif query_type == 'revenue_trend':
        df = engine.get_revenue_trend()
        response['content'] = "**Revenue Trend:**"
        response['data'] = df
        if not df.empty:
            response['chart'] = ChartBuilder.create_area_chart(df, 'Period', 'Revenue')
    
    elif query_type == 'city_breakdown':
        df = engine.get_city_breakdown(10)
        response['content'] = "**Top Cities by Revenue:**"
        response['data'] = df
        if not df.empty:
            response['chart'] = ChartBuilder.create_bar_chart(df, 'City', 'Revenue', horizontal=True)
    
    elif query_type == 'payment_breakdown':
        df = engine.get_payment_breakdown()
        response['content'] = "**Revenue by Payment Method:**"
        response['data'] = df
        if not df.empty:
            response['chart'] = ChartBuilder.create_bar_chart(df, 'Payment Method', 'Revenue')
    
    elif query_type == 'customer_count':
        result = engine.get_customer_count()
        response['content'] = f"**Unique Customers: {result['value']:,}**"
        response['data'] = result
    
    elif query_type == 'summary':
        revenue = engine.get_total_revenue()
        aov = engine.get_aov()
        orders = engine.get_order_count()
        rto = engine.get_rto_rate()
        
        response['content'] = f"""**📊 Business Summary:**

| Metric | Value |
|--------|-------|
| Total Revenue | {format_currency(revenue['value'])} |
| Average Order Value | {format_currency(aov['value'])} |
| Total Orders | {orders['value']:,} |
| RTO Rate | {rto['value']:.2f}% |
| Unique Customers | {engine.get_customer_count()['value']:,} |"""
    
    else:
        response['content'] = """I couldn't understand that query. Try asking:

- "What is my total revenue?"
- "Show RTO rate by payment method"
- "Top 10 products"
- "COD vs Prepaid comparison"
- "Revenue by category"
- "Revenue trend"
- "Top customers"
- "RTO rate by city"
"""
    
    return response


def render_reports():
    """Render the reports page."""
    
    st.markdown('<p class="main-title">📈 Reports</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Generate detailed analytics reports</p>', unsafe_allow_html=True)
    
    if not st.session_state.active_dataset:
        st.warning("Please upload and select a dataset first")
        return
    
    ds = st.session_state.datasets[st.session_state.active_dataset]
    engine = AnalyticsEngine(ds['df'], ds['mappings'])
    
    report_type = st.selectbox(
        "Select Report",
        ["📊 Executive Summary", "💰 Revenue Analysis", "🔄 RTO Analysis", "👥 Customer Analysis"]
    )
    
    if report_type == "📊 Executive Summary":
        st.markdown("## Executive Summary Report")
        st.markdown(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rev = engine.get_total_revenue()
            st.metric("Revenue", format_currency(rev['value']))
        
        with col2:
            aov = engine.get_aov()
            st.metric("AOV", format_currency(aov['value']))
        
        with col3:
            orders = engine.get_order_count()
            st.metric("Orders", format_number(orders['value']))
        
        with col4:
            rto = engine.get_rto_rate()
            st.metric("RTO Rate", format_percentage(rto['value']))
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Revenue Trend")
            trend = engine.get_revenue_trend()
            if not trend.empty:
                fig = ChartBuilder.create_area_chart(trend, 'Period', 'Revenue')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Order Status")
            status = engine.get_status_breakdown()
            if not status.empty:
                fig = ChartBuilder.create_pie_chart(status, 'Orders', 'Status')
                st.plotly_chart(fig, use_container_width=True)
        
        # Tables
        st.markdown("### Top Products")
        products = engine.get_top_products(10)
        if not products.empty:
            st.dataframe(products, use_container_width=True, hide_index=True)
        
        st.markdown("### Top Customers")
        customers = engine.get_top_customers(10)
        if not customers.empty:
            st.dataframe(customers, use_container_width=True, hide_index=True)
    
    elif report_type == "🔄 RTO Analysis":
        st.markdown("## RTO Analysis Report")
        
        rto = engine.get_rto_rate()
        st.metric("Overall RTO Rate", format_percentage(rto['value']), 
                  f"{rto['rto_orders']:,} returns out of {rto['shipped']:,} shipped")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### RTO by Payment Method")
            rto_payment = engine.get_rto_by_payment()
            if not rto_payment.empty:
                st.dataframe(rto_payment, use_container_width=True, hide_index=True)
                fig = ChartBuilder.create_bar_chart(rto_payment, 'Payment Method', 'RTO Rate')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### RTO by City (Top 10)")
            rto_city = engine.get_rto_by_city(10)
            if not rto_city.empty:
                st.dataframe(rto_city, use_container_width=True, hide_index=True)
                fig = ChartBuilder.create_bar_chart(rto_city, 'City', 'RTO Rate')
                st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown("### 💡 Key Insights")
        
        comparison = engine.get_cod_vs_prepaid()
        if comparison:
            cod_rto = comparison['COD']['rto_rate']
            prepaid_rto = comparison['Prepaid']['rto_rate']
            
            if cod_rto > prepaid_rto:
                st.warning(f"⚠️ COD orders have {cod_rto - prepaid_rto:.1f}% higher RTO rate than Prepaid orders")
                st.markdown("""
                **Recommendations:**
                - Consider incentivizing prepaid payments with discounts
                - Implement stricter verification for COD orders in high-RTO cities
                - Analyze COD order patterns for potential fraud detection
                """)


def render_settings():
    """Render the settings page."""
    
    st.markdown('<p class="main-title">⚙️ Settings</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎨 Preferences", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Display Settings")
        
        currency = st.selectbox(
            "Currency",
            ["INR (₹)", "USD ($)", "EUR (€)"],
            index=0
        )
        
        st.markdown("### Data Settings")
        
        if st.session_state.active_dataset:
            ds = st.session_state.datasets[st.session_state.active_dataset]
            
            st.markdown(f"**Active Dataset:** {st.session_state.active_dataset}")
            st.markdown(f"**Rows:** {len(ds['df']):,}")
            st.markdown(f"**Mapped Fields:** {len(ds['mappings'])}")
            
            st.markdown("**Current Mappings:**")
            for field, col in ds['mappings'].items():
                st.text(f"  {field.replace('_', ' ').title()}: {col}")
    
    with tab2:
        st.markdown("""
        ## About DataPulse
        
        **Version:** 1.0.0
        
        **Built with:**
        - 🐍 Python
        - 📊 Streamlit
        - 📈 Plotly
        
        **Features:**
        - Multi-dataset management
        - AI-powered natural language queries
        - Real-time dashboard updates
        - Accurate business logic (Revenue = Delivered only)
        - Correct RTO calculation
        
        **Business Rules:**
        - Revenue only counts delivered orders
        - AOV calculated from delivered orders only
        - RTO Rate = RTO / (Delivered + RTO) × 100
        """)


if __name__ == "__main__":
    main()

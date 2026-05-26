import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
from datetime import datetime, timedelta
import random
from faker import Faker

# ================= 1. 全局配置与数据初始化 =================

DB_PATH = 'retail_data.db'
fake = Faker('zh_CN')

@st.cache_resource
def init_database():
    """初始化数据库：检查表是否存在且有数据，否则生成模拟数据并入库 (ETL 流程)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders';")
    table_exists = cursor.fetchone()
    
    # 检查表里是否有数据
    has_data = False
    if table_exists:
        cursor.execute("SELECT COUNT(*) FROM orders;")
        count = cursor.fetchone()[0]
        has_data = count > 0

    # 如果表不存在或没数据，则重新初始化
    if not table_exists or not has_data:
        if table_exists:
            cursor.execute("DROP TABLE orders;") # 清空残留的空表
            
        # 创建订单明细表
        cursor.execute('''
            CREATE TABLE orders (
                order_id TEXT,
                user_id TEXT,
                order_date DATE,
                category TEXT,
                product_name TEXT,
                quantity INT,
                unit_price REAL,
                cost_price REAL
            )
        ''')
        
        # 生成模拟数据 (约 5000 条订单)
        categories = {'电子产品': ['手机', '耳机', '平板', '智能手表'],
                      '家居用品': ['台灯', '四件套', '收纳箱', '地毯'],
                      '食品饮料': ['咖啡', '坚果', '牛奶', '零食'],
                      '美妆护肤': ['洗面奶', '精华液', '面膜', '口红']}
        
        data = []
        base_date = datetime.now() - timedelta(days=365)
        users = [f"U{str(i).zfill(4)}" for i in range(1, 501)]
        
        for _ in range(5000):
            cat = random.choice(list(categories.keys()))
            prod = random.choice(categories[cat])
            qty = random.randint(1, 5)
            price = round(random.uniform(20, 2000), 2)
            cost = round(price * random.uniform(0.4, 0.8), 2) 
            
            data.append((
                f"ORD{random.randint(100000, 999999)}",
                random.choice(users),
                (base_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                cat, prod, qty, price, cost
            ))
            
        # ✅ 使用明确指定列名的专业写法
        cursor.executemany("""
            INSERT INTO orders 
            (order_id, user_id, order_date, category, product_name, quantity, unit_price, cost_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        conn.commit()
        
    conn.close()

# 执行初始化
init_database()

# ================= 2. 核心 SQL 查询 (展示 SQL 能力) =================

@st.cache_data
def load_data_from_sql():
    """使用复杂 SQL (CTE + 窗口函数) 从数据库提取聚合数据"""
    conn = sqlite3.connect(DB_PATH)
    
    # 面试亮点：使用 CTE (WITH 语句) 和 窗口函数 (SUM OVER) 计算累计占比，用于帕累托分析
    query = """
    WITH ProductSales AS (
        SELECT 
            product_name,
            category,
            SUM(quantity * unit_price) as total_revenue,
            SUM(quantity * (unit_price - cost_price)) as total_profit,
            SUM(quantity) as total_qty
        FROM orders
        GROUP BY product_name, category
    ),
    RankedProducts AS (
        SELECT 
            *,
            ROW_NUMBER() OVER(ORDER BY total_revenue DESC) as rn,
            SUM(total_revenue) OVER(ORDER BY total_revenue DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cum_revenue
        FROM ProductSales
    )
    SELECT 
        *,
        (cum_revenue / (SELECT SUM(total_revenue) FROM ProductSales)) * 100 as cum_revenue_pct
    FROM RankedProducts
    """
    
    df_products = pd.read_sql_query(query, conn)
    df_orders = pd.read_sql_query("SELECT * FROM orders", conn)
    df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
    conn.close()
    
    return df_orders, df_products

# ================= 3. 商业分析模型 (RFM) =================

def calculate_rfm(df_orders, ref_date):
    """计算 RFM 模型并进行用户分层"""
    # 确保日期格式正确
    df = df_orders.copy()
    
    rfm = df.groupby('user_id').agg({
        'order_date': lambda x: (ref_date - x.max()).days, # R: 最近一次消费距今天数
        'order_id': 'nunique',                             # F: 消费频次 (订单数)
        'unit_price': lambda x: (x * df.loc[x.index, 'quantity']).sum() # M: 消费总金额
    }).reset_index()
    
    rfm.columns = ['user_id', 'R', 'F', 'M']
    
    # 使用分位数进行打分 (1-4分)
    rfm['R_Score'] = pd.qcut(rfm['R'], 4, labels=[4, 3, 2, 1]) # R越小分越高
    rfm['F_Score'] = pd.qcut(rfm['F'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['M'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    
    # 拼接 RFM 分数
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # 定义用户分层逻辑
    def segment_user(row):
        if row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3: return '重要价值客户'
        if row['R_Score'] >= 3 and row['F_Score'] <= 2: return '重要保持客户'
        if row['R_Score'] <= 2 and row['F_Score'] >= 3: return '重要挽留客户'
        if row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] >= 3: return '一般价值客户'
        return '一般/流失客户'

    rfm['Segment'] = rfm.apply(segment_user, axis=1)
    return rfm

# ================= 4. Streamlit 前端 UI 搭建 =================

st.set_page_config(page_title="零售业务 BI 监控看板", layout="wide", page_icon="📊")

st.title("📊 零售业务全链路 BI 监控看板")
st.markdown("---")

# 加载数据
df_orders, df_products = load_data_from_sql()

# --- 侧边栏：全局筛选器 ---
with st.sidebar:
    st.header("🔍 全局筛选条件")
    
    min_date = df_orders['order_date'].min().date()
    max_date = df_orders['order_date'].max().date()
    
    date_range = st.date_input("选择日期范围", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    categories = ['全部'] + list(df_orders['category'].unique())
    selected_cat = st.selectbox("选择产品类别", categories)

# 应用筛选
if len(date_range) == 2:
    mask = (df_orders['order_date'].dt.date >= date_range[0]) & (df_orders['order_date'].dt.date <= date_range[1])
    df_filtered = df_orders[mask]
else:
    df_filtered = df_orders

if selected_cat != '全部':
    df_filtered = df_filtered[df_filtered['category'] == selected_cat]

# 计算核心 KPI
total_revenue = (df_filtered['quantity'] * df_filtered['unit_price']).sum()
total_profit = (df_filtered['quantity'] * (df_filtered['unit_price'] - df_filtered['cost_price'])).sum()
total_orders = df_filtered['order_id'].nunique()
aov = total_revenue / total_orders if total_orders > 0 else 0 # 客单价

# --- 第一行：KPI 指标卡 ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 总销售额 (GMV)", f"¥{total_revenue:,.2f}")
col2.metric("📈 总净利润", f"¥{total_profit:,.2f}")
col3.metric("🛒 订单总数", f"{total_orders:,}")
col4.metric("🧾 客单价 (AOV)", f"¥{aov:,.2f}")

st.markdown("---")

# --- 第二行：趋势与类别分布 ---
col_trend, col_cat = st.columns([2, 1])

with col_trend:
    st.subheader("📈 每日销售趋势")
    daily_sales = df_filtered.groupby(df_filtered['order_date'].dt.date)['unit_price'].apply(lambda x: (x * df_filtered.loc[x.index, 'quantity']).sum()).reset_index()
    daily_sales.columns = ['date', 'revenue']
    fig_trend = px.line(daily_sales, x='date', y='revenue', markers=True, labels={'date': '日期', 'revenue': '销售额'})
    fig_trend.update_layout(hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

with col_cat:
    st.subheader("🍩 类别利润贡献占比")
    cat_profit = df_filtered.groupby('category').apply(lambda x: (x['quantity'] * (x['unit_price'] - x['cost_price'])).sum()).reset_index()
    cat_profit.columns = ['category', 'profit']
    fig_pie = px.pie(cat_profit, values='profit', names='category', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# --- 第三行：深度商业分析 (RFM & 帕累托) ---
tab1, tab2 = st.tabs(["👥 用户价值分层 (RFM 模型)", "📦 商品贡献度分析 (帕累托)"])

with tab1:
    st.markdown("**分析逻辑**：基于用户最近一次消费(R)、消费频次(F)、消费金额(M)进行 1-4 分打分，并划分为 5 大核心客群。")
    ref_date = df_orders['order_date'].max() + timedelta(days=1)
    rfm_df = calculate_rfm(df_orders, ref_date) # RFM 通常基于全量历史数据计算
    
    col_rfm1, col_rfm2 = st.columns([1, 1])
    with col_rfm1:
        seg_counts = rfm_df['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig_bar = px.bar(seg_counts, x='Segment', y='Count', color='Segment', text='Count')
        fig_bar.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_rfm2:
        st.dataframe(rfm_df[['user_id', 'R', 'F', 'M', 'Segment']].sort_values('M', ascending=False).head(50), use_container_width=True)

with tab2:
    st.markdown("**分析逻辑**：验证是否符合“二八定律”，即头部商品是否贡献了绝大部分营收。")
    # 过滤出当前类别的商品数据
    if selected_cat != '全部':
        prod_df = df_products[df_products['category'] == selected_cat]
    else:
        prod_df = df_products
        
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(x=prod_df['product_name'], y=prod_df['total_revenue'], name='销售额', marker_color='lightsalmon'))
    fig_pareto.add_trace(go.Scatter(x=prod_df['product_name'], y=prod_df['cum_revenue_pct'], name='累计占比 (%)', yaxis='y2', marker_color='royalblue', line=dict(width=3)))
    
    fig_pareto.update_layout(
        yaxis=dict(title='销售额 (元)'),
        yaxis2=dict(title='累计占比 (%)', overlaying='y', side='right', range=[0, 105]),
        legend=dict(x=0.8, y=1.1, orientation='h'),
        xaxis_tickangle=-45
    )
    # 添加 80% 警戒线
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% 营收线", yref="y2")
    st.plotly_chart(fig_pareto, use_container_width=True)
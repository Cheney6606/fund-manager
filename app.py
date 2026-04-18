import streamlit as st
import pandas as pd
import requests
import json
import re
import time
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI
from supabase import create_client, Client
from datetime import datetime

# ====================== 直接写死正确参数（先跑通再说） ======================
SUPABASE_URL = "https://jwggzxbsbzvvbusjknbu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp3Z2d6eGJzYnp2dmJ1c2prbmJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY0MTY1NTcsImV4cCI6MjA5MTk5MjU1N30.tXN1hJF8B8wB9iejrFEiEpcdTyveDRky0TM4FXrjfDg"
DEEPSEEK_API_KEY = "sk-e4cfb4a5b57c429b818ad7c1115d1741"
BAIDU_OCR_API_KEY = "lm6kOFEKbl9s7yu02WulwHwf"
BAIDU_OCR_SECRET_KEY = "NYeQaq88oNs98rxXUPatTXEcreheN2Ml"
# ====================================================================================

# 初始化客户端（去掉了不兼容的参数）
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="我的基金管家", layout="wide")
st.title("📈 我的基金智能管理系统")

# ---------------------- 基础功能函数 ----------------------
def get_fund_info(fund_code):
    """获取基金实时净值、涨跌幅"""
    url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js"
    try:
        response = requests.get(url, timeout=10)
        text = response.text
        json_str = re.search(r'jsonpgz\((.*)\);', text).group(1)
        data = json.loads(json_str)
        return {
            "code": fund_code,
            "name": data.get("name", ""),
            "net_value": float(data.get("dwjz", 0)),
            "estimate_value": float(data.get("gsz", 0)),
            "estimate_change": float(data.get("gszzl", 0)),
            "update_time": data.get("jzrq", "")
        }
    except Exception as e:
        st.error(f"获取基金{fund_code}数据失败：{e}")
        return None

def get_historical_nav(fund_code, days=365):
    """获取基金历史净值"""
    url = f"http://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
    try:
        response = requests.get(url, timeout=10)
        text = response.text
        pattern = r'var Data_netWorthTrend = (\[.*?\]);'
        match = re.search(pattern, text)
        if not match:
            return None
        data = json.loads(match.group(1))
        df = pd.DataFrame(data)
        df = df[["x", "y"]].rename(columns={"x": "date", "y": "nav"})
        df["date"] = pd.to_datetime(df["date"], unit='ms')
        df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=days)]
        df = df.sort_values("date")
        return df
    except Exception as e:
        st.error(f"获取基金{fund_code}历史数据失败：{e}")
        return None

def calculate_metrics(nav_df):
    """计算基金收益、风险指标"""
    if nav_df is None or len(nav_df) < 2:
        return {}
    nav_df = nav_df.set_index("date")
    daily_returns = nav_df["nav"].pct_change().dropna()
    if len(daily_returns) < 10:
        return {}

    total_return = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1)
    years = (nav_df.index[-1] - nav_df.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_volatility = daily_returns.std() * np.sqrt(252)

    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.expanding().max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    risk_free_rate = 0.03
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0

    windows = {"1周": 5, "2周": 10, "1月": 21, "2月": 42, "3月": 63, "6月": 126}
    window_returns = {}
    for name, w in windows.items():
        if len(nav_df) >= w:
            ret = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[-w] - 1
        else:
            ret = None
        window_returns[name] = ret

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "window_returns": window_returns,
        "nav_df": nav_df
    }

def calculate_score(metrics):
    """计算基金综合评分0-100"""
    if not metrics:
        return 0, 0, 0
    window_returns = metrics["window_returns"]
    weights = {"1周": 0.25, "2周": 0.2, "1月": 0.2, "2月": 0.15, "3月": 0.1, "6月": 0.1}
    recent_score = 0
    total_weight = 0
    for w_name, weight in weights.items():
        ret = window_returns.get(w_name)
        if ret is not None:
            ret_score = min(max(ret * 100, 0), 50)
            recent_score += ret_score * weight
            total_weight += weight
    if total_weight > 0:
        recent_score = recent_score / total_weight

    sharpe_score = min(max(metrics["sharpe"] * 25, 0), 25)
    sortino_score = min(max(metrics["sortino"] * 25, 0), 25)
    volatility_score = max(25 - metrics["annual_volatility"] * 100, 0)
    drawdown_score = max(25 - abs(metrics["max_drawdown"]) * 100, 0)
    long_term_score = sharpe_score + sortino_score + volatility_score + drawdown_score

    comprehensive = recent_score * 0.6 + long_term_score * 0.4
    return comprehensive, recent_score, long_term_score

def load_strategy_config():
    """加载策略参数"""
    try:
        res = supabase.table("strategy_config").select("*").execute()
        config = {}
        for row in res.data:
            config[row["rule_name"]] = row["rule_value"]
        return config
    except:
        return {"T_SELL_THRESHOLD": 2.0}

def ai_chat(messages, funds_context):
    """DeepSeek AI对话"""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    system_prompt = f"你是专属我的基金分析师，严格基于以下持仓和分析数据，用通俗易懂的话回答问题，给出有数据支撑的建议：\n{funds_context}"
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    try:
        response = client.chat.completions.create(model="deepseek-chat", messages=full_messages, stream=False)
        return response.choices[0].message.content
    except Exception as e:
        return f"AI调用失败：{e}"

# ---------------------- 侧边栏导航 ----------------------
st.sidebar.title("功能导航")
page = st.sidebar.radio("选择页面", [
    "📊 持仓总览",
    "📋 每日操作建议",
    "📁 持仓管理",
    "⚙️ 策略参数配置",
    "🤖 AI基金分析师"
])

# ---------------------- 1. 持仓总览页面 ----------------------
if page == "📊 持仓总览":
    st.header("📊 我的持仓总览")
    try:
        res = supabase.table("portfolio").select("*").execute()
        df_portfolio = pd.DataFrame(res.data) if res.data else pd.DataFrame()

        if not df_portfolio.empty:
            total_cost = (df_portfolio["shares"] * df_portfolio["cost_price"]).sum()
            total_market_value = 0
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总投入成本", f"¥{total_cost:,.2f}")
            
            st.subheader("持仓明细（实时更新）")
            display_df = []
            for _, row in df_portfolio.iterrows():
                fund_info = get_fund_info(row["fund_code"])
                if fund_info:
                    market_value = row["shares"] * fund_info["net_value"]
                    profit = market_value - (row["shares"] * row["cost_price"])
                    profit_pct = (fund_info["net_value"] / row["cost_price"] - 1) * 100 if row["cost_price"] > 0 else 0
                    total_market_value += market_value
                    display_df.append({
                        "基金代码": row["fund_code"],
                        "基金名称": row["fund_name"],
                        "持仓类型": row["category"],
                        "持有份额": f"{row['shares']:,.2f}",
                        "成本价": f"{row['cost_price']:.4f}",
                        "最新净值": f"{fund_info['net_value']:.4f}",
                        "今日涨跌幅": f"{fund_info['estimate_change']:.2f}%",
                        "持仓市值": f"¥{market_value:,.2f}",
                        "浮动盈亏": f"¥{profit:,.2f}",
                        "盈亏比例": f"{profit_pct:.2f}%"
                    })
                else:
                    display_df.append({
                        "基金代码": row["fund_code"],
                        "基金名称": row["fund_name"],
                        "持仓类型": row["category"],
                        "持有份额": f"{row['shares']:,.2f}",
                        "成本价": f"{row['cost_price']:.4f}",
                        "最新净值": "获取失败",
                        "今日涨跌幅": "-",
                        "持仓市值": "-",
                        "浮动盈亏": "-",
                        "盈亏比例": "-"
                    })
            
            with col2:
                st.metric("当前总市值", f"¥{total_market_value:,.2f}")
            with col3:
                total_profit = total_market_value - total_cost
                st.metric("总浮动盈亏", f"¥{total_profit:,.2f}", delta=f"{(total_profit/total_cost*100):.2f}%" if total_cost>0 else "0%")
            
            st.dataframe(pd.DataFrame(display_df), use_container_width=True)
        else:
            st.info("暂无持仓数据，请先到「📁 持仓管理」页面添加你的基金持仓")
    except Exception as e:
        st.error(f"数据库连接/读取失败：{str(e)}")

# ---------------------- 2. 每日操作建议页面 ----------------------
elif page == "📋 每日操作建议":
    st.header("📋 每日操作建议")
    st.caption("自动根据你的持仓和策略纪律，生成今日操作指令")
    
    try:
        res = supabase.table("portfolio").select("*").execute()
        df_portfolio = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        
        if df_portfolio.empty:
            st.warning("请先添加持仓数据，才能生成操作建议")
        else:
            if st.button("🚀 开始生成今日操作建议", type="primary"):
                config = load_strategy_config()
                funds_data = []
                signals = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (_, row) in enumerate(df_portfolio.iterrows()):
                    code = row["fund_code"]
                    name = row["fund_name"]
                    status_text.text(f"正在分析：{name}({code})")
                    
                    fund_info = get_fund_info(code)
                    nav_df = get_historical_nav(code, 30)
                    metrics = calculate_metrics(nav_df)
                    score = calculate_score(metrics)
                    
                    if fund_info:
                        funds_data.append({
                            "code": code,
                            "name": name,
                            "info": fund_info,
                            "nav_df": nav_df,
                            "metrics": metrics,
                            "score": score[0]
                        })
                        
                        change = fund_info["estimate_change"]
                        if change >= config.get("T_SELL_THRESHOLD", 2.0):
                            signals.append({
                                "基金": f"{name}({code})",
                                "操作": "做T卖出",
                                "理由": f"单日涨幅{change:.2f}%，达到阈值，建议卖出1/3",
                                "等级": "必须执行"
                            })
                        
                        cost = row["cost_price"]
                        current_nav = fund_info["net_value"]
                        if current_nav >= cost and (current_nav - cost)/cost*100 < 1:
                            signals.append({
                                "基金": f"{name}({code})",
                                "操作": "回本减仓",
                                "理由": "已回本，建议卖出1/2",
                                "等级": "今日可执行"
                            })
                    
                    progress_bar.progress((i + 1) / len(df_portfolio))
                    time.sleep(0.3)
                
                status_text.text("分析完成！")
                progress_bar.empty()
                
                if signals:
                    st.subheader("📢 今日操作")
                    for sig in signals:
                        if sig["等级"] == "必须执行":
                            st.warning(f"⚠️ {sig['基金']}：{sig['操作']} | {sig['理由']}")
                        else:
                            st.info(f"ℹ️ {sig['基金']}：{sig['操作']} | {sig['理由']}")
                else:
                    st.success("✅ 今日无操作，持有不动")
                
                if funds_data:
                    st.subheader("近30天净值走势")
                    fig = go.Figure()
                    for fund in funds_data:
                        if fund["nav_df"] is not None:
                            norm_nav = fund["nav_df"]["nav"] / fund["nav_df"]["nav"].iloc[0] - 1
                            fig.add_trace(go.Scatter(
                                x=fund["nav_df"]["date"], y=norm_nav * 100,
                                mode='lines', name=f"{fund['name']}"
                            ))
                    fig.update_layout(title="净值涨跌幅对比", yaxis_title="%")
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"操作建议生成失败：{str(e)}")

# ---------------------- 3. 持仓管理页面 ----------------------
elif page == "📁 持仓管理":
    st.header("📁 持仓管理")
    st.caption("添加、修改、删除基金持仓")
    
    with st.expander("➕ 添加新持仓", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            new_code = st.text_input("基金代码")
        with col2:
            new_name = st.text_input("基金名称")
        with col3:
            new_category = st.selectbox("类型", ["盈利底仓", "亏损做T仓", "观察仓"])
        with col4:
            new_shares = st.number_input("份额", min_value=0.0, step=100.0)
        with col5:
            new_cost = st.number_input("成本价", min_value=0.0, step=0.0001, format="%.4f")
        new_date = st.date_input("买入日期")
        
        if st.button("添加持仓"):
            if new_code and new_name and new_shares > 0 and new_cost > 0:
                try:
                    supabase.table("portfolio").insert({
                        "fund_code": new_code,
                        "fund_name": new_name,
                        "category": new_category,
                        "shares": new_shares,
                        "cost_price": new_cost,
                        "buy_date": str(new_date)
                    }).execute()
                    st.success("✅ 添加成功")
                    st.rerun()
                except Exception as e:
                    st.error(f"添加失败：{e}")
            else:
                st.warning("请填写完整信息")
    
    st.subheader("当前持仓")
    try:
        res = supabase.table("portfolio").select("*").execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame(columns=["fund_code", "fund_name", "category", "shares", "cost_price", "buy_date"])
        
        if not df.empty:
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                column_config={
                    "fund_code": "基金代码",
                    "fund_name": "基金名称",
                    "category": "类型",
                    "shares": st.column_config.NumberColumn("份额", format="%.2f"),
                    "cost_price": st.column_config.NumberColumn("成本", format="%.4f"),
                    "buy_date": "日期"
                },
                use_container_width=True
            )
            if st.button("💾 保存修改"):
                for _, row in edited_df.iterrows():
                    supabase.table("portfolio").update({
                        "fund_name": row["fund_name"],
                        "category": row["category"],
                        "shares": row["shares"],
                        "cost_price": row["cost_price"],
                        "buy_date": row["buy_date"]
                    }).eq("fund_code", row["fund_code"]).execute()
                st.success("✅ 保存成功")
                st.rerun()
        else:
            st.info("暂无持仓")
    except Exception as e:
        st.error(f"读取持仓失败：{e}")

# ---------------------- 4. 策略参数 ----------------------
elif page == "⚙️ 策略参数配置":
    st.header("⚙️ 策略参数")
    try:
        res = supabase.table("strategy_config").select("*").execute()
        if res.data:
            df_config = pd.DataFrame(res.data)
            edited_config = st.data_editor(
                df_config[["rule_name", "rule_value", "description"]],
                column_config={
                    "rule_name": "规则",
                    "rule_value": "参数",
                    "description": "说明"
                },
                disabled=["rule_name", "description"]
            )
            if st.button("保存参数"):
                for _, row in edited_config.iterrows():
                    supabase.table("strategy_config").update({
                        "rule_value": row["rule_value"]
                    }).eq("rule_name", row["rule_name"]).execute()
                st.success("✅ 参数已更新")
                st.rerun()
    except:
        st.info("策略配置暂不可用")

# ---------------------- 5. AI分析师 ----------------------
elif page == "🤖 AI基金分析师":
    st.header("🤖 AI基金分析师")
    
    try:
        res = supabase.table("portfolio").select("*").execute()
        df_portfolio = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        context_str = "我的持仓：\n"
        
        if not df_portfolio.empty:
            for _, row in df_portfolio.iterrows():
                context_str += f"- {row['fund_name']} | 份额{row['shares']:.2f} | 成本{row['cost_price']:.4f}\n"
        
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("提问你的持仓问题"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("分析中..."):
                    response = ai_chat(st.session_state.chat_messages, context_str)
                    st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"AI功能异常：{e}")
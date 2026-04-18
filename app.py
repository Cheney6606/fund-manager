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
import base64
from PIL import Image
import io

# ====================== 请修改为你的真实 Key ======================
SUPABASE_URL = "https://jwggzxbsbzvvbusjknbu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp3Z2d6eGJzYnp2dmJ1c2prbmJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY0MTY1NTcsImV4cCI6MjA5MTk5MjU1N30.tXN1hJF8B8wB9iejrFEiEpcdTyveDRky0TM4FXrjfDg"
DEEPSEEK_API_KEY = "sk-e4cfb4a5b57c429b818ad7c1115d1741"
BAIDU_OCR_API_KEY = "lm6kOFEKbl9s7yu02WulwHwf"
BAIDU_OCR_SECRET_KEY = "NYeQaq88oNs98rxXUPatTXEcreheN2Ml"
# =================================================================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
st.set_page_config(page_title="基金管家", layout="wide")
st.title("📈 基金智能管理系统")

# ---------------------- 百度OCR识别函数 ----------------------
def get_baidu_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": BAIDU_OCR_API_KEY,
        "client_secret": BAIDU_OCR_SECRET_KEY
    }
    try:
        response = requests.post(url, params=params)
        return response.json().get("access_token")
    except:
        return None

def ocr_image(image_file):
    access_token = get_baidu_access_token()
    if not access_token:
        st.error("百度OCR token获取失败，请检查API Key")
        return ""
    img = Image.open(image_file)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    params = {"access_token": access_token}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"image": img_base64}
    try:
        response = requests.post(url, params=params, headers=headers, data=data)
        result = response.json()
        if "words_result" in result:
            words = [item["words"] for item in result["words_result"]]
            return "\n".join(words)
        else:
            st.error(f"OCR识别失败：{result}")
            return ""
    except Exception as e:
        st.error(f"OCR请求异常：{e}")
        return ""

def parse_portfolio_from_ocr(text):
    lines = text.split("\n")
    funds = []
    code_pattern = re.compile(r'\b\d{6}\b')
    number_pattern = re.compile(r'[\d,]+\.?\d*')
    current_fund = {}
    for line in lines:
        codes = code_pattern.findall(line)
        if codes:
            if current_fund:
                funds.append(current_fund)
                current_fund = {}
            current_fund["fund_code"] = codes[0]
            name_match = re.search(r'([\u4e00-\u9fa5].+?)(?=\d|$)', line)
            if name_match:
                current_fund["fund_name"] = name_match.group(1).strip()
        if "持仓" in line or "份额" in line:
            numbers = number_pattern.findall(line)
            if numbers:
                try:
                    current_fund["shares"] = float(numbers[0].replace(",", ""))
                except:
                    pass
        if "成本" in line:
            numbers = number_pattern.findall(line)
            if numbers:
                try:
                    current_fund["cost_price"] = float(numbers[0].replace(",", ""))
                except:
                    pass
    if current_fund:
        funds.append(current_fund)
    return funds

# ---------------------- 原有基金数据函数 ----------------------
def get_fund_info(fund_code):
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
    except:
        return None

def get_historical_nav(fund_code, days=365):
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
        return df.sort_values("date")
    except:
        return None

def calculate_metrics(nav_df):
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
        "annual_return": annual_return, "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown, "sharpe": sharpe, "sortino": sortino,
        "window_returns": window_returns, "nav_df": nav_df
    }

def calculate_score(metrics):
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
    try:
        res = supabase.table("strategy_config").select("*").execute()
        return {row["rule_name"]: row["rule_value"] for row in res.data} if res.data else {"T_SELL_THRESHOLD": 2.0}
    except:
        return {"T_SELL_THRESHOLD": 2.0}

def ai_chat(messages, funds_context):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    system_prompt = f"你是专属我的基金分析师，基于以下持仓数据回答问题：\n{funds_context}"
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    try:
        response = client.chat.completions.create(model="deepseek-chat", messages=full_messages, stream=False)
        return response.choices[0].message.content
    except Exception as e:
        return f"AI调用失败：{e}"

# ---------------------- 侧边栏导航 ----------------------
st.sidebar.title("功能导航")
page = st.sidebar.radio("选择页面", [
    "📊 持仓总览", "📋 每日操作建议", "📁 持仓管理", "⚙️ 策略参数配置", "🤖 AI基金分析师"
])

# ---------------------- 持仓总览 ----------------------
if page == "📊 持仓总览":
    st.header("📊 我的持仓总览")
    try:
        res = supabase.table("portfolio").select("*").execute()
        df_portfolio = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if not df_portfolio.empty:
            total_cost = (df_portfolio["shares"] * df_portfolio["cost_price"]).sum()
            total_market_value = 0
            display_data = []
            for _, row in df_portfolio.iterrows():
                info = get_fund_info(row["fund_code"])
                if info:
                    market_value = row["shares"] * info["net_value"]
                    profit = market_value - row["shares"] * row["cost_price"]
                    profit_pct = (info["net_value"] / row["cost_price"] - 1) * 100 if row["cost_price"] > 0 else 0
                    total_market_value += market_value
                    display_data.append({
                        "代码": row["fund_code"], "名称": row["fund_name"], "类型": row["category"],
                        "份额": f"{row['shares']:,.2f}", "成本价": f"{row['cost_price']:.4f}",
                        "最新净值": f"{info['net_value']:.4f}", "今日涨幅": f"{info['estimate_change']:.2f}%",
                        "市值": f"¥{market_value:,.2f}", "盈亏": f"¥{profit:,.2f}", "盈亏%": f"{profit_pct:.2f}%"
                    })
                else:
                    display_data.append({"代码": row["fund_code"], "名称": row["fund_name"], "最新净值": "获取失败"})
            col1, col2, col3 = st.columns(3)
            col1.metric("总成本", f"¥{total_cost:,.2f}")
            col2.metric("总市值", f"¥{total_market_value:,.2f}")
            col3.metric("总盈亏", f"¥{total_market_value - total_cost:,.2f}",
                       delta=f"{(total_market_value/total_cost-1)*100:.2f}%" if total_cost>0 else "0%")
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        else:
            st.info("暂无持仓数据，请前往「📁 持仓管理」添加")
    except Exception as e:
        st.error(f"数据库错误：{e}")

# ---------------------- 每日操作建议 ----------------------
elif page == "📋 每日操作建议":
    st.header("📋 每日操作建议")
    try:
        res = supabase.table("portfolio").select("*").execute()
        df_portfolio = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if df_portfolio.empty:
            st.warning("请先添加持仓")
        elif st.button("🚀 生成今日操作建议"):
            config = load_strategy_config()
            signals = []
            progress = st.progress(0)
            for i, (_, row) in enumerate(df_portfolio.iterrows()):
                info = get_fund_info(row["fund_code"])
                if info:
                    change = info["estimate_change"]
                    if change >= config.get("T_SELL_THRESHOLD", 2.0):
                        signals.append(f"⚠️ {row['fund_name']}：涨幅{change:.2f}%，建议做T卖出1/3")
                    elif info["net_value"] >= row["cost_price"] and (info["net_value"]/row["cost_price"]-1)*100 < 1:
                        signals.append(f"ℹ️ {row['fund_name']}：已回本，可考虑减仓1/2")
                progress.progress((i+1)/len(df_portfolio))
            progress.empty()
            if signals:
                for s in signals:
                    st.warning(s) if "⚠️" in s else st.info(s)
            else:
                st.success("今日无触发操作，持有不动")
    except Exception as e:
        st.error(f"生成失败：{e}")

# ---------------------- 持仓管理（含截图识别）----------------------
elif page == "📁 持仓管理":
    st.header("📁 持仓管理")

    with st.expander("📸 上传支付宝持仓截图，自动识别导入", expanded=True):
        uploaded_file = st.file_uploader("选择截图", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            with st.spinner("OCR识别中..."):
                ocr_text = ocr_image(uploaded_file)
                if ocr_text:
                    st.text_area("识别到的文字（可手动修正）", ocr_text, height=150)
                    funds = parse_portfolio_from_ocr(ocr_text)
                    if funds:
                        st.success(f"识别到 {len(funds)} 只基金")
                        df_preview = pd.DataFrame(funds)
                        st.dataframe(df_preview)
                        if st.button("✅ 确认导入到数据库"):
                            for fund in funds:
                                if "fund_code" in fund and "fund_name" in fund:
                                    fund.setdefault("category", "盈利底仓")
                                    fund.setdefault("shares", 0.0)
                                    fund.setdefault("cost_price", 1.0)
                                    fund.setdefault("buy_date", str(datetime.now().date()))
                                    supabase.table("portfolio").upsert({
                                        "fund_code": fund["fund_code"],
                                        "fund_name": fund["fund_name"],
                                        "category": fund["category"],
                                        "shares": fund["shares"],
                                        "cost_price": fund["cost_price"],
                                        "buy_date": fund["buy_date"]
                                    }, on_conflict="fund_code").execute()
                            st.success("导入完成！")
                            st.rerun()

    with st.expander("➕ 手动添加持仓"):
        col1, col2, col3 = st.columns(3)
        with col1:
            code = st.text_input("基金代码")
        with col2:
            name = st.text_input("基金名称")
        with col3:
            category = st.selectbox("类型", ["盈利底仓", "亏损做T仓", "观察仓"])
        col4, col5, col6 = st.columns(3)
        with col4:
            shares = st.number_input("份额", min_value=0.0, step=100.0)
        with col5:
            cost = st.number_input("成本价", min_value=0.0, step=0.0001, format="%.4f")
        with col6:
            buy_date = st.date_input("买入日期")
        if st.button("添加持仓"):
            if code and name:
                supabase.table("portfolio").upsert({
                    "fund_code": code, "fund_name": name, "category": category,
                    "shares": shares, "cost_price": cost, "buy_date": str(buy_date)
                }, on_conflict="fund_code").execute()
                st.success("已添加")
                st.rerun()

    st.subheader("当前持仓")
    try:
        res = supabase.table("portfolio").select("*").execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if not df.empty:
            edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="editor")
            if st.button("💾 保存修改"):
                for _, row in edited.iterrows():
                    supabase.table("portfolio").update({
                        "fund_name": row["fund_name"], "category": row["category"],
                        "shares": row["shares"], "cost_price": row["cost_price"], "buy_date": row["buy_date"]
                    }).eq("fund_code", row["fund_code"]).execute()
                st.success("已保存")
                st.rerun()
    except Exception as e:
        st.error(f"读取失败：{e}")

# ---------------------- 策略配置 ----------------------
elif page == "⚙️ 策略参数配置":
    st.header("⚙️ 策略参数")
    try:
        res = supabase.table("strategy_config").select("*").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            edited = st.data_editor(df[["rule_name", "rule_value", "description"]], disabled=["rule_name", "description"])
            if st.button("保存"):
                for _, row in edited.iterrows():
                    supabase.table("strategy_config").update({"rule_value": row["rule_value"]}).eq("rule_name", row["rule_name"]).execute()
                st.success("已更新")
                st.rerun()
    except:
        st.info("配置表暂不可用")

# ---------------------- AI对话 ----------------------
elif page == "🤖 AI基金分析师":
    st.header("🤖 AI基金分析师")
    try:
        res = supabase.table("portfolio").select("*").execute()
        context = "我的持仓：\n"
        if res.data:
            for row in res.data:
                context += f"{row['fund_name']}({row['fund_code']})，份额{row['shares']}，成本{row['cost_price']}\n"
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("问我任何问题"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    reply = ai_chat(st.session_state.messages, context)
                    st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"AI出错：{e}")
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

# ====================== 已填入真实 Key ======================
SUPABASE_URL = "https://jwggzxbsbzvvbusjknbu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp3Z2d6eGJzYnp2dmJ1c2prbmJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY0MTY1NTcsImV4cCI6MjA5MTk5MjU1N30.tXN1hJF8B8wB9iejrFEiEpcdTyveDRky0TM4FXrjfDg"
DEEPSEEK_API_KEY = "sk-e4cfb4a5b57c429b818ad7c1115d1741"
BAIDU_OCR_API_KEY = "lm6kOFEKbl9s7yu02WulwHwf"
BAIDU_OCR_SECRET_KEY = "NYeQaq88oNs98rxXUPatTXEcreheN2Ml"
# ============================================================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
st.set_page_config(page_title="基金智能管理系统", layout="wide")
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

# ---------------------- DeepSeek AI 补全函数（增强版）----------------------
def query_fund_code_by_name(name: str) -> dict:
    """根据基金名称查询代码，返回 {"code": "xxxxxx", "name": "标准名称"}"""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    prompt = f"""请查询基金名称为"{name}"的6位数字代码。只返回一个JSON：{{"code": "xxxxxx", "name": "基金全称"}}。如果找不到，code留空。不要输出任何其他文字。"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except:
        return {}

# ---------------------- 全新解析函数（针对列表页格式）----------------------
def parse_portfolio_list_from_text(text):
    """
    专为支付宝持仓列表页设计。
    格式特点：基金名称一行，下方紧随持仓金额（带逗号的数字），再下方是盈亏数据。
    返回列表，每个元素为 {"name": xxx, "market_value": xxx}
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    funds = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 判断是否为基金名称行：包含常见基金特征词，且不含纯数字或特殊符号
        if any(kw in line for kw in ["混合", "ETF", "联接", "指数", "股票", "债券", "QDII", "优选", "成长", "价值", "灵活", "稳健"]):
            # 过滤掉干扰行（如"市场解读"）
            if "市场解读" in line or "金选" in line and "基金" in line:
                i += 1
                continue
            
            name = re.sub(r'金选指数基金|定投|市场解读.*', '', line).strip()
            if len(name) < 3:
                i += 1
                continue
            
            # 找下一行的金额（通常是带逗号的大数字）
            market_value = None
            j = i + 1
            while j < len(lines) and j < i + 5:
                num_line = lines[j]
                # 匹配金额格式（如 47,271.80 或 11,307.83）
                num_match = re.search(r'([\d,]+\.\d{2})', num_line)
                if num_match:
                    try:
                        val = float(num_match.group(1).replace(",", ""))
                        if val > 100:  # 有效持仓金额
                            market_value = val
                            break
                    except:
                        pass
                j += 1
            
            if market_value:
                funds.append({
                    "name": name,
                    "market_value": market_value
                })
                i = j  # 跳过已处理的行
            else:
                i += 1
        else:
            i += 1
    
    # 去重（根据名称）
    seen = set()
    unique_funds = []
    for f in funds:
        if f["name"] not in seen:
            seen.add(f["name"])
            unique_funds.append(f)
    return unique_funds

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

# ---------------------- 持仓管理（AI解析版）----------------------
elif page == "📁 持仓管理":
    st.header("📁 持仓管理")

    with st.expander("📸 上传支付宝持仓截图，智能更新持仓", expanded=True):
        uploaded_file = st.file_uploader("选择截图", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            with st.spinner("OCR识别中..."):
                ocr_text = ocr_image(uploaded_file)
                if ocr_text:
                    # 使用AI解析
                    funds_parsed = parse_portfolio_by_ai(ocr_text)
                    if funds_parsed:
                        st.success(f"识别到 {len(funds_parsed)} 只基金")
                        
                        # 用AI补全代码
                        for fund in funds_parsed:
                            if "code" not in fund:
                                ai_result = query_fund_code_by_name(fund["name"])
                                fund["code"] = ai_result.get("code", "")
                                if ai_result.get("name"):
                                    fund["name"] = ai_result["name"]
                        
                        # 获取现有持仓
                        res_existing = supabase.table("portfolio").select("*").execute()
                        df_existing = pd.DataFrame(res_existing.data) if res_existing.data else pd.DataFrame()
                        
                        preview_data = []
                        for fund in funds_parsed:
                            code = fund.get("code", "")
                            name = fund["name"]
                            market_value = fund["market_value"]
                            existing = df_existing[df_existing["fund_code"] == code] if code and not df_existing.empty else pd.DataFrame()
                            action = "更新" if not existing.empty else ("新增" if code else "需补充代码")
                            preview_data.append({
                                "状态": action,
                                "基金代码": code if code else "⚠️ 未识别",
                                "基金名称": name,
                                "识别市值": f"¥{market_value:,.2f}",
                                "当前份额": existing.iloc[0]["shares"] if action=="更新" else "-",
                                "当前成本": existing.iloc[0]["cost_price"] if action=="更新" else "-"
                            })
                        
                        df_preview = pd.DataFrame(preview_data)
                        st.dataframe(df_preview, use_container_width=True)
                        
                        missing_codes = any(f.get("code") == "" for f in funds_parsed)
                        if missing_codes:
                            st.warning("部分基金未能识别代码，可在下方表格手动输入代码后点击更新")
                        
                        if st.button("✅ 确认更新到我的持仓", type="primary"):
                            for fund in funds_parsed:
                                code = fund.get("code")
                                name = fund["name"]
                                if not code or not name:
                                    continue
                                
                                existing = df_existing[df_existing["fund_code"] == code] if not df_existing.empty else pd.DataFrame()
                                if not existing.empty:
                                    update_dict = {
                                        "fund_code": code,
                                        "fund_name": name,
                                        "shares": existing.iloc[0]["shares"],
                                        "cost_price": existing.iloc[0]["cost_price"],
                                        "category": existing.iloc[0]["category"]
                                    }
                                else:
                                    update_dict = {
                                        "fund_code": code,
                                        "fund_name": name,
                                        "category": "盈利底仓",
                                        "shares": 0.0,
                                        "cost_price": 0.0
                                    }
                                supabase.table("portfolio").upsert(update_dict, on_conflict="fund_code").execute()
                            st.success("持仓已更新！")
                            st.rerun()
                    else:
                        st.warning("未能从截图中解析出基金信息")

    # 后续手动添加表单和当前持仓编辑保持不变...
    with st.expander("➕ 手动添加持仓"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_code = st.text_input("基金代码")
        with col2:
            new_name = st.text_input("基金名称")
        with col3:
            new_category = st.selectbox("类型", ["盈利底仓", "亏损做T仓", "观察仓"])
        col4, col5, col6 = st.columns(3)
        with col4:
            new_shares = st.number_input("份额", min_value=0.0, step=100.0)
        with col5:
            new_cost = st.number_input("成本价", min_value=0.0, step=0.0001, format="%.4f")
        with col6:
            new_date = st.date_input("买入日期")
        if st.button("添加持仓"):
            if new_code and new_name:
                supabase.table("portfolio").upsert({
                    "fund_code": new_code, "fund_name": new_name, "category": new_category,
                    "shares": new_shares, "cost_price": new_cost, "buy_date": str(new_date)
                }, on_conflict="fund_code").execute()
                st.success("已添加")
                st.rerun()

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
                        "fund_name": row["fund_name"], "category": row["category"],
                        "shares": row["shares"], "cost_price": row["cost_price"], "buy_date": row["buy_date"]
                    }).eq("fund_code", row["fund_code"]).execute()
                st.success("✅ 保存成功")
                st.rerun()
        else:
            st.info("暂无持仓")
    except Exception as e:
        st.error(f"读取持仓失败：{e}")
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
# ==================== 第一部分：导入与基础配置 ====================
import streamlit as st
import pandas as pd
import requests
import json
import re
import os
import difflib
import akshare as ak
from openai import OpenAI, Timeout
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone, time as dt_time
import base64
from PIL import Image
import io

# -------------------- 常量定义 --------------------
TZ = timezone(timedelta(hours=8))
OCR_THUMBNAIL_SIZE = (2048, 2048)
CACHE_TTL_FUND_LIST = 3600
CACHE_TTL_FUND_INFO = 300
CACHE_TTL_INDUSTRY_DAYS = 7
TIMEOUT_OCR_TOKEN = 10
TIMEOUT_OCR_REQUEST = 15
TIMEOUT_FUND_API = 10
DEEPSEEK_TIMEOUT = Timeout(60.0, connect=10.0, read=60.0)
DEFAULT_T_SELL_THRESHOLD = 2.0
SELL_SHARE_THRESHOLD = 0.01
MAX_SELL_SHARE_RATIO = 1.0
BATCH_UPSERT_SIZE = 10

# 法定节假日列表（可按需补充）
CN_HOLIDAYS = {
    "2026-01-01", "2026-01-02", "2026-01-03",
    "2026-02-12", "2026-02-13", "2026-02-14", "2026-02-15", "2026-02-16",
    "2026-04-05",
    "2026-05-01", "2026-05-02", "2026-05-03",
    "2026-06-19",
    "2026-09-27",
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
}

# 术语统一
TERMS = {
    "净值": "单位净值",
    "成本价": "成本净值",
    "成本": "成本净值",
    "涨幅": "日涨跌幅",
    "持有收益": "持仓盈亏",
    "持仓市值": "持仓金额",
    "份额": "持有份额"
}

# -------------------- Streamlit 页面配置 --------------------
st.set_page_config(
    page_title="基金智能管理系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 移动端适配 CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    @media (max-width: 768px) {
        .metric-card { padding: 12px; }
        .stApp { overflow-x: hidden; }
    }
    .positive { color: #e31b23; }
    .negative { color: #2e8b57; }
</style>
""", unsafe_allow_html=True)

# -------------------- 密钥管理（从 Streamlit Secrets 读取）--------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
BAIDU_OCR_API_KEY = st.secrets["BAIDU_OCR_API_KEY"]
BAIDU_OCR_SECRET_KEY = st.secrets["BAIDU_OCR_SECRET_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- 工具函数 --------------------
def now_cn() -> datetime:
    """返回东八区当前时间"""
    return datetime.now(TZ)

def is_trading_day(date: datetime = None) -> bool:
    """判断是否为交易日（工作日且非法定节假日）"""
    if date is None:
        date = now_cn()
    if date.weekday() >= 5:
        return False
    date_str = date.strftime("%Y-%m-%d")
    if date_str in CN_HOLIDAYS:
        return False
    return True

def is_trading_time() -> bool:
    """判断当前是否处于交易时段（09:30-15:00）"""
    if not is_trading_day():
        return False
    now = now_cn()
    current_time = now.time()
    start = dt_time(9, 30)
    end = dt_time(15, 0)
    return start <= current_time <= end

def safe_json_parse(text: str, pattern: str) -> dict | list | None:
    """安全解析 JSON，返回 None 而非抛异常"""
    try:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None
# ==================== 第二部分：AI 与 OCR ====================
def get_deepseek_client() -> OpenAI:
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        timeout=DEEPSEEK_TIMEOUT,
        max_retries=2
    )

def get_baidu_access_token() -> str | None:
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": BAIDU_OCR_API_KEY,
        "client_secret": BAIDU_OCR_SECRET_KEY
    }
    try:
        resp = requests.post(url, params=params, timeout=TIMEOUT_OCR_TOKEN)
        return resp.json().get("access_token")
    except Exception as e:
        st.error(f"获取百度OCR Token失败: {e}")
        return None

def ocr_image(image_file) -> str:
    token = get_baidu_access_token()
    if not token:
        return ""
    try:
        img = Image.open(image_file)
        img.thumbnail(OCR_THUMBNAIL_SIZE)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_base64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        st.error(f"图片处理失败: {e}")
        return ""

    for ocr_type, url in [
        ("高精度", "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"),
        ("通用", "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic")
    ]:
        try:
            resp = requests.post(
                url,
                params={"access_token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"image": img_base64},
                timeout=TIMEOUT_OCR_REQUEST
            )
            result = resp.json()
            if "words_result" in result:
                words = [item["words"] for item in result["words_result"]]
                return "\n".join(words)
        except Exception:
            continue
    st.error("OCR识别失败，请重试或检查图片清晰度")
    return ""

def parse_portfolio_by_ai(ocr_text: str) -> list:
    client = get_deepseek_client()
    prompt = f"""
你是一个严格的基金持仓信息提取工具，必须100%遵循以下规则，禁止任何自由发挥：

1. 从以下支付宝基金持仓页面的OCR文字中，提取每一只基金的【完整原始名称】和【持仓市值】。
2. 基金名称规则：必须完整保留OCR原文中的所有文字，包括括号、QDII、C、ETF等所有关键词，绝对禁止修改、缩写、替换、脑补名称。
3. 持仓市值规则：提取基金名称右侧的第一个带逗号的数字（单位元），只保留纯数字，去掉逗号和¥符号。只提取持仓总市值，忽略昨日收益、持有收益、收益率等。
4. 只提取持仓市值大于1000元的基金。

返回格式：纯JSON数组，每个元素为：{{"name": "基金完整原始名称", "market_value": 持仓市值数字}}
如果无有效基金，返回空数组[]。

OCR原始文字内容：
{ocr_text}
"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        data = safe_json_parse(content, r'\[.*\]')
        if isinstance(data, list):
            return [f for f in data if isinstance(f, dict) and f.get("market_value", 0) > 1000]
        return []
    except Exception as e:
        st.error(f"AI解析失败: {e}")
        return []

def parse_operation_by_ai(ocr_text: str) -> dict:
    client = get_deepseek_client()
    prompt = f"""
从以下支付宝基金交易截图的OCR文字中提取信息。返回纯JSON格式：
{{"fund_name": "基金完整名称", "operation": "买入或卖出", "amount": 交易金额(元), "date": "YYYY-MM-DD"}}
如果某字段无法提取，则留空字符串。
OCR文字：
{ocr_text}
"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        data = safe_json_parse(content, r'\{.*\}')
        return data if isinstance(data, dict) else {}
    except Exception as e:
        st.error(f"操作解析失败: {e}")
        return {}
# ==================== 第三部分：基金数据与匹配 ====================
def process_fund_list(raw_df: pd.DataFrame) -> pd.DataFrame:
    """标准化基金列表，添加公司、份额、币种等辅助字段"""
    def standardize(row):
        name = str(row["基金简称"])
        name = name.translate(str.maketrans('（）【】', '()[]'))
        companies = ["建信", "华夏", "广发", "易方达", "嘉实", "汇添富", "南方", "博时", "富国", "工银瑞信"]
        company = ""
        for c in companies:
            if c in name:
                company = c
                break
        share_type = "C" if "C" in name or "C类" in name else "A"
        fund_type = ""
        if "ETF联接" in name: fund_type = "ETF联接"
        elif "指数增强" in name: fund_type = "指数增强"
        elif "混合" in name: fund_type = "混合"
        elif "股票" in name: fund_type = "股票"
        currency = "美元" if "美元" in name else "人民币"
        core = re.sub(r'(混合|ETF联接|指数增强|股票|QDII|A类|C类|发起式|美元|人民币|\(|\)|\s)', '', name)
        clean = re.sub(r'[\(\)\[\]\s\-_\.，,。·]', '', name).lower()
        return pd.Series({
            "company": company, "share_type": share_type, "fund_type": fund_type,
            "currency": currency, "core_target": core, "clean_name": clean,
            "name_length": len(name)
        })
    std = raw_df.apply(standardize, axis=1)
    return pd.concat([raw_df, std], axis=1)

@st.cache_data(ttl=CACHE_TTL_FUND_LIST)
def load_full_fund_list() -> pd.DataFrame:
    csv_path = "fund_full_list.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, dtype={"基金代码": str})
            if not df.empty and "基金代码" in df.columns and "基金简称" in df.columns:
                return process_fund_list(df)
        except Exception:
            pass
    try:
        raw_df = ak.fund_name_em()
        raw_df = raw_df[["基金代码", "基金简称"]].copy()
        raw_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return process_fund_list(raw_df)
    except Exception as e:
        st.error(f"获取全市场基金列表失败: {e}")
        return pd.DataFrame()

def query_fund_code_smart(keyword: str) -> dict:
    if not keyword:
        return {}
    df = load_full_fund_list()
    if df.empty:
        return {}
    if keyword.isdigit() and len(keyword) == 6:
        row = df[df["基金代码"] == keyword]
        if not row.empty:
            r = row.iloc[0]
            return {"code": r["基金代码"], "name": r["基金简称"]}
    kw = str(keyword).translate(str.maketrans('（）【】', '()[]'))
    companies = ["建信", "华夏", "广发", "易方达", "嘉实", "汇添富", "南方", "博时", "富国", "工银瑞信"]
    company = ""
    for c in companies:
        if c in kw:
            company = c
            break
    share_type = "C" if "C" in kw or "C类" in kw else "A"
    fund_type = ""
    if "ETF联接" in kw: fund_type = "ETF联接"
    elif "指数增强" in kw: fund_type = "指数增强"
    elif "混合" in kw: fund_type = "混合"
    elif "股票" in kw: fund_type = "股票"
    currency = "美元" if "美元" in kw else "人民币"
    core = re.sub(r'(混合|ETF联接|指数增强|股票|QDII|A类|C类|发起式|美元|人民币|\(|\)|\s)', '', kw)
    clean_kw = re.sub(r'[\(\)\[\]\s\-_\.，,。·]', '', kw).lower()

    candidates = df.copy()
    if company:
        candidates = candidates[candidates["company"] == company]
    candidates = candidates[candidates["share_type"] == share_type]
    if currency == "人民币":
        candidates = candidates[candidates["currency"] == "人民币"]
    if fund_type:
        candidates = candidates[candidates["fund_type"] == fund_type]
    if core:
        candidates = candidates[candidates["core_target"].str.contains(core, na=False)]

    if not candidates.empty:
        candidates["sim"] = candidates["clean_name"].apply(
            lambda x: difflib.SequenceMatcher(None, clean_kw, x).ratio()
        )
        candidates = candidates.sort_values(["sim", "name_length"], ascending=[False, True])
        best = candidates.iloc[0]
        return {"code": best["基金代码"], "name": best["基金简称"]}
    else:
        df["sim"] = df["clean_name"].apply(lambda x: difflib.SequenceMatcher(None, clean_kw, x).ratio())
        if currency == "人民币":
            df = df[df["currency"] == "人民币"]
        best = df.sort_values("sim", ascending=False).iloc[0]
        if best["sim"] > 0.6:
            return {"code": best["基金代码"], "name": best["基金简称"]}
    return {}

@st.cache_data(ttl=CACHE_TTL_FUND_INFO, show_spinner=False)
def get_fund_info_cached(fund_code: str) -> dict | None:
    return get_fund_info(fund_code)

def get_fund_info(fund_code: str) -> dict | None:
    url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js"
    try:
        resp = requests.get(url, timeout=TIMEOUT_FUND_API)
        text = resp.text
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
    except (requests.RequestException, json.JSONDecodeError, AttributeError) as e:
        st.error(f"获取基金 {fund_code} 实时数据失败: {e}")
        return None

def get_historical_nav(fund_code: str, days: int = 365) -> pd.DataFrame | None:
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        if df.empty:
            return None
        df = df[["净值日期", "单位净值"]].rename(columns={"净值日期": "date", "单位净值": "nav"})
        df["date"] = pd.to_datetime(df["date"])
        cutoff = df["date"].max() - pd.Timedelta(days=days)
        return df[df["date"] >= cutoff].sort_values("date")
    except Exception as e:
        st.error(f"获取 {fund_code} 历史净值失败: {e}")
        return None

def get_fund_holdings(fund_code: str) -> pd.DataFrame | None:
    try:
        df = ak.fund_portfolio_holdings_em(symbol=fund_code)
        return df if not df.empty else None
    except Exception:
        return None

def get_stock_industry(stock_code: str) -> str:
    try:
        res = supabase.table("stock_industry").select("*").eq("stock_code", stock_code).execute()
        if res.data:
            row = res.data[0]
            updated = datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
            if now_cn() - updated < timedelta(days=CACHE_TTL_INDUSTRY_DAYS):
                return row["industry"]
    except Exception:
        pass
    industry = "其他"
    try:
        info = ak.stock_individual_info_em(symbol=stock_code)
        if not info.empty:
            match = info[info["item"] == "行业"]
            if not match.empty:
                industry = match["value"].values[0]
    except Exception:
        pass
    try:
        supabase.table("stock_industry").upsert({
            "stock_code": stock_code,
            "industry": industry,
            "updated_at": now_cn().isoformat()
        }, on_conflict="stock_code").execute()
    except Exception:
        pass
    return industry
# ==================== 第四部分：市场动态、策略、持仓操作 ====================
def get_market_dynamics() -> dict:
    if not is_trading_day():
        return {}
    dyn = {}
    try:
        df = ak.stock_zh_index_spot_em()
        for name in ["上证指数", "深证成指", "创业板指", "沪深300"]:
            row = df[df["名称"] == name]
            if not row.empty:
                dyn[name] = {"current": row["最新价"].values[0], "change_pct": row["涨跌幅"].values[0]}
    except Exception:
        pass
    try:
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        today = df.iloc[-1]
        dyn["北向资金"] = {"net_flow": today["value"], "date": today["date"]}
    except Exception:
        pass
    try:
        df = ak.stock_sector_spot_em(sector="申万一级")
        df = df.sort_values("涨跌幅", ascending=False)
        dyn["行业涨幅前5"] = df.head(5)[["名称", "涨跌幅"]].to_dict("records")
        dyn["行业跌幅前5"] = df.tail(5)[["名称", "涨跌幅"]].to_dict("records")
    except Exception:
        pass
    return dyn

def strategy_advisor(messages: list, context: str) -> str:
    client = get_deepseek_client()
    system_prompt = f"""你是一位专业的量化策略顾问。仅基于以下用户持仓数据和当前策略参数给出优化建议，不做市场预测、不推荐具体基金、不构成任何投资建议。
{context}"""
    full_msgs = [{"role": "system", "content": system_prompt}] + messages
    try:
        resp = client.chat.completions.create(model="deepseek-chat", messages=full_msgs)
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI调用失败: {e}"

def update_portfolio_on_buy(fund_code: str, fund_name: str, amount: float, price: float, op_date: str):
    if price <= 0:
        st.error("买入价格必须大于0")
        return
    shares = amount / price
    if shares <= SELL_SHARE_THRESHOLD:
        st.warning("买入份额过小，无需执行")
        return
    res = supabase.table("portfolio").select("*").eq("fund_code", fund_code).execute()
    if res.data:
        row = res.data[0]
        old_shares = float(row["shares"])
        old_cost = float(row["cost_price"])
        new_shares = old_shares + shares
        new_cost = ((old_shares * old_cost) + amount) / new_shares if new_shares > 0 else price
        supabase.table("portfolio").update({
            "shares": new_shares,
            "cost_price": new_cost,
            "updated_at": now_cn().isoformat()
        }).eq("fund_code", fund_code).execute()
    else:
        supabase.table("portfolio").insert({
            "fund_code": fund_code,
            "fund_name": fund_name,
            "category": "盈利底仓",
            "shares": shares,
            "cost_price": price,
            "buy_date": op_date,
            "realized_profit": 0
        }).execute()
    supabase.table("buy_batches").insert({
        "fund_code": fund_code,
        "buy_date": op_date,
        "shares": shares,
        "cost_price": price,
        "remaining_shares": shares
    }).execute()

def update_portfolio_on_sell(fund_code: str, amount: float, price: float, op_date: str):
    if price <= 0:
        st.error("卖出价格必须大于0")
        return
    res = supabase.table("portfolio").select("shares").eq("fund_code", fund_code).execute()
    if not res.data:
        st.warning(f"{fund_code} 无持仓，无法卖出")
        return
    max_sell_shares = float(res.data[0]["shares"])
    sell_shares = amount / price
    if sell_shares > max_sell_shares:
        sell_shares = max_sell_shares
        st.warning(f"卖出份额超过持仓，已按最大可卖 {max_sell_shares:.2f} 份执行")
    if sell_shares <= SELL_SHARE_THRESHOLD:
        st.warning("卖出份额过小，无需执行")
        return

    batches = supabase.table("buy_batches").select("*") \
        .eq("fund_code", fund_code) \
        .gt("remaining_shares", 0) \
        .order("buy_date", desc=False) \
        .execute()

    if not batches.data:
        st.warning(f"{fund_code} 无可用买入批次，无法执行卖出")
        return

    remaining_to_sell = sell_shares
    realized_profit = 0.0

    for batch in batches.data:
        if remaining_to_sell <= SELL_SHARE_THRESHOLD:
            break
        batch_shares = float(batch["remaining_shares"])
        batch_cost = float(batch["cost_price"])
        sell_from_batch = min(batch_shares, remaining_to_sell)
        profit_from_batch = sell_from_batch * (price - batch_cost)
        realized_profit += profit_from_batch

        new_remaining = batch_shares - sell_from_batch
        supabase.table("buy_batches").update({
            "remaining_shares": new_remaining
        }).eq("id", batch["id"]).execute()

        remaining_to_sell -= sell_from_batch

    res_port = supabase.table("portfolio").select("*").eq("fund_code", fund_code).execute()
    if res_port.data:
        row = res_port.data[0]
        old_shares = float(row["shares"])
        old_realized = float(row.get("realized_profit", 0))
        new_shares = old_shares - (sell_shares - remaining_to_sell)
        new_realized = old_realized + realized_profit
        supabase.table("portfolio").update({
            "shares": new_shares,
            "realized_profit": new_realized,
            "updated_at": now_cn().isoformat()
        }).eq("fund_code", fund_code).execute()

def load_strategy_config() -> dict:
    try:
        res = supabase.table("strategy_config").select("*").execute()
        if res.data:
            return {row["rule_name"]: float(row["rule_value"]) for row in res.data}
    except Exception:
        pass
    return {"T_SELL_THRESHOLD": DEFAULT_T_SELL_THRESHOLD}

def save_strategy_config(config: dict):
    for rule_name, rule_value in config.items():
        supabase.table("strategy_config").upsert({
            "rule_name": rule_name,
            "rule_value": rule_value,
            "updated_at": now_cn().isoformat()
        }, on_conflict="rule_name").execute()

def batch_upsert_portfolio(funds: list):
    if not funds:
        return
    data = []
    for f in funds:
        data.append({
            "fund_code": f["code"],
            "fund_name": f["name"],
            "category": f.get("category", "盈利底仓"),
            "shares": f.get("shares", 0),
            "cost_price": f.get("cost_price", 0),
            "buy_date": f.get("buy_date", now_cn().date().isoformat()),
            "realized_profit": 0
        })
    for i in range(0, len(data), BATCH_UPSERT_SIZE):
        batch = data[i:i+BATCH_UPSERT_SIZE]
        supabase.table("portfolio").upsert(batch, on_conflict="fund_code").execute()
# ==================== 第五部分：页面路由与渲染 ====================
st.sidebar.title("功能导航")
page = st.sidebar.radio("选择页面", [
    "📊 持仓总览",
    "📋 每日操作建议",
    "📁 持仓管理",
    "📎 操作记录",
    "⚙️ 策略配置",
    "🤖 策略顾问",
    "💬 AI分析师"
])

if page == "📊 持仓总览":
    st.header("📊 持仓总览")
    if is_trading_time():
        asset_label = "实时预估总资产"
        profit_label = "当日收益 (预估)"
    else:
        asset_label = "上一交易日收盘总资产"
        profit_label = "当日收益 (预估) — 非交易时段"

    try:
        res = supabase.table("portfolio").select("*").execute()
        if not res.data:
            st.info("暂无持仓数据，请前往「📁 持仓管理」添加")
        else:
            df = pd.DataFrame(res.data)
            total_cost = 0.0
            total_market_value = 0.0
            total_estimate_profit = 0.0
            total_real_profit = 0.0
            rows = []

            for _, row in df.iterrows():
                code = row["fund_code"]
                name = row["fund_name"]
                shares = float(row["shares"])
                cost = float(row["cost_price"])
                category = row["category"]

                info = get_fund_info_cached(code)
                if info:
                    nav = info["net_value"]
                    est_nav = info["estimate_value"]
                    est_change = info["estimate_change"]
                    market_val = shares * nav
                    est_profit = shares * (est_nav - nav) if est_nav > 0 else 0
                    real_profit = market_val - shares * cost
                    real_return_pct = (real_profit / (shares * cost) * 100) if shares * cost > 0 else 0

                    total_cost += shares * cost
                    total_market_value += market_val
                    total_estimate_profit += est_profit
                    total_real_profit += real_profit

                    rows.append({
                        "基金名称": name,
                        "基金代码": code,
                        "持仓分类": category,
                        "持有份额": f"{shares:,.2f}",
                        "成本净值": f"{cost:.4f}",
                        "单位净值": f"{nav:.4f}",
                        "持仓盈亏": f"{real_profit:+,.2f}",
                        "持仓收益率": f"{real_return_pct:+.2f}%",
                        "日涨跌幅": f"{est_change:+.2f}%"
                    })
                else:
                    rows.append({
                        "基金名称": name, "基金代码": code, "持仓分类": category,
                        "持有份额": f"{shares:,.2f}", "成本净值": f"{cost:.4f}",
                        "单位净值": "获取失败", "持仓盈亏": "-", "持仓收益率": "-", "日涨跌幅": "-"
                    })

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(asset_label, f"¥{total_market_value:,.2f}")
            with col2:
                st.metric(profit_label, f"¥{total_estimate_profit:+,.2f}")
            with col3:
                st.metric("持仓盈亏 (确定)", f"¥{total_real_profit:+,.2f}")
            with col4:
                total_return = (total_real_profit / total_cost * 100) if total_cost > 0 else 0
                st.metric("持仓收益率", f"{total_return:+.2f}%")

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except Exception as e:
        st.error(f"数据加载失败: {e}")

elif page == "📋 每日操作建议":
    st.header("📋 每日操作建议")
    if is_trading_day():
        with st.expander("📈 今日市场动态", expanded=True):
            dyn = get_market_dynamics()
            if dyn:
                cols = st.columns(4)
                idx_list = ["上证指数", "深证成指", "创业板指", "沪深300"]
                for i, idx in enumerate(idx_list):
                    if idx in dyn:
                        data = dyn[idx]
                        cols[i % 4].metric(idx, data["current"], f"{data['change_pct']:+.2f}%")
                if "北向资金" in dyn:
                    nf = dyn["北向资金"]
                    st.caption(f"北向资金净流入: {nf['net_flow']}亿元 ({nf['date']})")
                if "行业涨幅前5" in dyn:
                    st.caption("行业涨幅前5: " + ", ".join([f"{x['名称']} {x['涨跌幅']:+.2f}%" for x in dyn["行业涨幅前5"]]))
            else:
                st.info("市场动态数据暂不可用")
    else:
        st.info("今日非交易日，市场动态模块隐藏")

    res = supabase.table("portfolio").select("*").execute()
    if not res.data:
        st.warning("请先添加持仓")
    else:
        df = pd.DataFrame(res.data)
        if st.button("🚀 生成今日操作建议", type="primary"):
            config = load_strategy_config()
            threshold = config.get("T_SELL_THRESHOLD", DEFAULT_T_SELL_THRESHOLD)
            signals = []
            table_rows = []
            for _, row in df.iterrows():
                code = row["fund_code"]
                name = row["fund_name"]
                shares = float(row["shares"])
                cost = float(row["cost_price"])
                category = row["category"]
                info = get_fund_info_cached(code)
                if info:
                    nav = info["net_value"]
                    est_nav = info["estimate_value"]
                    est_change = info["estimate_change"]
                    est_profit = shares * (est_nav - nav) if est_nav > 0 else 0
                    direction = "无操作"
                    if category == "亏损做T仓" and est_change >= threshold:
                        direction = "做T卖出"
                    elif nav >= cost and (nav / cost - 1) * 100 < 1:
                        direction = "回本减仓"
                    table_rows.append({
                        "基金名称": name, "基金代码": code, "持有份额": f"{shares:,.2f}",
                        "成本净值": f"{cost:.4f}", "单位净值": f"{nav:.4f}",
                        "预估涨幅": f"{est_change:+.2f}%", "预估收益": f"¥{est_profit:+,.2f}",
                        "操作方向": direction
                    })
                    if direction != "无操作":
                        signals.append(f"{'⚠️' if direction == '做T卖出' else 'ℹ️'} {name}：{direction}")
            st.subheader("📊 持仓明细与预估")
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
            st.subheader("📢 今日操作")
            if signals:
                for s in signals:
                    if "⚠️" in s:
                        st.warning(s)
                    else:
                        st.info(s)
            else:
                st.success("今日无触发操作，持有不动")

elif page == "📁 持仓管理":
    st.header("📁 持仓管理")
    with st.expander("📸 上传支付宝持仓截图，智能更新持仓", expanded=True):
        uploaded = st.file_uploader("选择截图", type=["png", "jpg", "jpeg"])
        if uploaded:
            with st.spinner("OCR识别中..."):
                ocr_text = ocr_image(uploaded)
                if ocr_text:
                    funds = parse_portfolio_by_ai(ocr_text)
                    if funds:
                        st.success(f"识别到 {len(funds)} 只基金")
                        for f in funds:
                            if "code" not in f or not f["code"]:
                                matched = query_fund_code_smart(f["name"])
                                if matched:
                                    f["code"] = matched["code"]
                                    f["name"] = matched["name"]
                        preview_df = pd.DataFrame(funds)
                        edited_df = st.data_editor(preview_df, use_container_width=True, key="portfolio_preview")
                        if st.button("✅ 确认更新到我的持仓", type="primary"):
                            batch_data = []
                            for _, row in edited_df.iterrows():
                                code = row.get("code", "")
                                name = row.get("name", "")
                                if not code or not name:
                                    continue
                                info = get_fund_info_cached(code)
                                price = info["net_value"] if info else 0.0
                                market_val = float(row.get("market_value", 0))
                                shares = market_val / price if price > 0 else 0
                                batch_data.append({
                                    "code": code, "name": name, "category": "盈利底仓",
                                    "shares": shares, "cost_price": price,
                                    "buy_date": now_cn().date().isoformat()
                                })
                            batch_upsert_portfolio(batch_data)
                            st.success("持仓已更新")
                            st.rerun()
                    else:
                        st.warning("未识别到有效基金信息")

elif page == "📎 操作记录":
    st.header("📎 操作记录")
    with st.expander("📸 上传支付宝交易截图", expanded=True):
        uploaded = st.file_uploader("选择截图", type=["png", "jpg", "jpeg"], key="op_upload")
        if uploaded:
            with st.spinner("识别中..."):
                ocr_text = ocr_image(uploaded)
                if ocr_text:
                    op = parse_operation_by_ai(ocr_text)
                    if op:
                        st.success("识别成功，请核对")
                        preview = pd.DataFrame([op])
                        edited = st.data_editor(preview, use_container_width=True, key="op_preview")
                        if st.button("✅ 确认记录并更新持仓", type="primary"):
                            row = edited.iloc[0]
                            fund_name = row.get("fund_name", "")
                            operation = row.get("operation", "")
                            amount = float(row.get("amount", 0))
                            op_date = row.get("date", now_cn().date().isoformat())
                            matched = query_fund_code_smart(fund_name)
                            if not matched:
                                st.error("无法匹配基金代码，请手动输入")
                            else:
                                code = matched["code"]
                                name = matched["name"]
                                info = get_fund_info_cached(code)
                                price = info["net_value"] if info else 0.0
                                if operation == "买入":
                                    update_portfolio_on_buy(code, name, amount, price, op_date)
                                elif operation == "卖出":
                                    update_portfolio_on_sell(code, amount, price, op_date)
                                supabase.table("operations").insert({
                                    "fund_code": code, "fund_name": name,
                                    "operation_type": "BUY" if operation == "买入" else "SELL",
                                    "amount": amount, "price": price,
                                    "shares": amount / price if price > 0 else 0,
                                    "op_date": op_date
                                }).execute()
                                st.success("操作已记录，持仓已更新")
                                st.rerun()
                    else:
                        st.warning("未能解析交易信息")

elif page == "⚙️ 策略配置":
    st.header("⚙️ 策略参数配置")
    config = load_strategy_config()
    with st.form("strategy_form"):
        new_threshold = st.number_input("做T卖出触发涨幅 (%)", value=config.get("T_SELL_THRESHOLD", 2.0), step=0.1)
        if st.form_submit_button("保存"):
            config["T_SELL_THRESHOLD"] = new_threshold
            save_strategy_config(config)
            st.success("策略已更新")
            st.rerun()

elif page == "🤖 策略顾问":
    st.header("🤖 策略顾问")
    config = load_strategy_config()
    res = supabase.table("portfolio").select("*").execute()
    holdings_text = "\n".join([f"{r['fund_name']}（{r['fund_code']}）：{r['category']}，份额{r['shares']}，成本{r['cost_price']}" for r in res.data]) if res.data else "暂无持仓"
    context = f"当前策略：做T卖出阈值 {config.get('T_SELL_THRESHOLD', 2.0)}%\n持仓概况：\n{holdings_text}"
    if "advisor_messages" not in st.session_state:
        st.session_state.advisor_messages = []
    for msg in st.session_state.advisor_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("请输入您的问题，例如：我的做T阈值需要调整吗？"):
        st.session_state.advisor_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                reply = strategy_advisor(st.session_state.advisor_messages, context)
                st.markdown(reply)
        st.session_state.advisor_messages.append({"role": "assistant", "content": reply})

elif page == "💬 AI分析师":
    st.header("💬 AI基金分析师")
    res = supabase.table("portfolio").select("*").execute()
    context = "我的持仓：\n" + "\n".join([f"{r['fund_name']}（{r['fund_code']}），份额{r['shares']}，成本{r['cost_price']}" for r in res.data]) if res.data else "暂无持仓"
    client = get_deepseek_client()
    if "analyst_messages" not in st.session_state:
        st.session_state.analyst_messages = []
    for msg in st.session_state.analyst_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("问我任何关于基金的问题"):
        st.session_state.analyst_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "system", "content": f"你是基金分析助手，基于持仓数据回答。\n{context}"}] + st.session_state.analyst_messages
                    )
                    reply = resp.choices[0].message.content
                except Exception as e:
                    reply = f"AI出错: {e}"
                st.markdown(reply)
        st.session_state.analyst_messages.append({"role": "assistant", "content": reply})
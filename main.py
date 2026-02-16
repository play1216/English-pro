import streamlit as st
from openai import OpenAI
import json
import pandas as pd
import plotly.express as px

# ================= 1. 安全配置 (兼容云端与本地) =================
# 优先读取 Secrets，如果读不到则提示
if "DEEPSEEK_API_KEY" not in st.secrets:
    st.error("❌ 未找到 API Key。请在本地 .streamlit/secrets.toml 或 Streamlit Cloud 后台配置。")
    st.stop()

# 统一读取 Key
MY_KEY = str(st.secrets["DEEPSEEK_API_KEY"]).strip()

# 初始化客户端
try:
    client = OpenAI(
        api_key=MY_KEY,
        base_url="https://api.deepseek.com"
    )
except Exception as e:
    st.error(f"❌ 客户端初始化失败: {e}")
    st.stop()

# ================= 2. 页面设置 =================
st.set_page_config(page_title="高中英语作文 AI 精批", layout="wide", page_icon="📝")

# 标题与侧边栏（可以在侧边栏加点说明，显得更专业）
st.title("📝 高中英语作文 AI 精批系统")
st.markdown("---")

with st.sidebar:
    st.header("关于系统")
    st.info("采用 DeepSeek-V3 引擎，专为高中英语作文评分标准定制。")
    st.warning("⚠️ 提示：请确保作文为纯英文，字数建议在 80-200 词之间。")

# ================= 3. 主界面布局 =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✍️ 提交作文")
    user_text = st.text_area("在此粘贴你的作文内容...", height=450, placeholder="Once upon a time...")
    start_btn = st.button("🚀 开始 AI 老师批改", type="primary", use_container_width=True)

# ================= 4. 核心逻辑 =================
if start_btn:
    if not user_text:
        st.warning("请输入作文内容后再提交。")
    else:
        with col2:
            with st.spinner("AI 老师正在认真阅卷并查阅词典..."):
                try:
                    # 调用 DeepSeek API
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一位高中英语老师。分析作文并返回严格的JSON格式，包含score(total, grammar, vocabulary, logic, structure), comment, suggestions(original, improved, reason)。"},
                            {"role": "user", "content": user_text}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    # 解析结果
                    result = json.loads(response.choices[0].message.content)
                    
                    # --- A. 展示分数卡片 ---
                    st.success("✅ 批改完成！")
                    s = result.get('score', {})
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🏆 预估总分", f"{s.get('total', 0)}/25")
                    c2.metric("📝 语法分", s.get('grammar', 0))
                    c3.metric("📖 词汇分", s.get('vocabulary', 0))

                    # --- B. 雷达图分析 ---
                    st.subheader("📊 维度分析")
                    try:
                        # 准备雷达图数据
                        categories = ['语法', '词汇', '逻辑', '结构']
                        scores = [
                            s.get('grammar', 0), 
                            s.get('vocabulary', 0), 
                            s.get('logic', 0), 
                            s.get('structure', 0)
                        ]
                        # 为了闭合图形，需要重复第一个点
                        df = pd.DataFrame(dict(r=scores + [scores[0]], theta=categories + [categories[0]]))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        fig.update_traces(fill='toself') # 填充颜色，更美观
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as radar_err:
                        st.error(f"图表生成失败: {radar_err}")

                    # --- C. 名师点评 ---
                    st.subheader("👨‍🏫 名师点评")
                    st.info(result.get('comment', '暂无总体点评'))

                    # --- D. 提分建议 ---
                    st.subheader("✨ 逐句精修")
                    for item in result.get('suggestions', []):
                        with st.expander(f"❌ 原文: {item.get('original')}"):
                            st.success(f"✅ 建议: {item.get('improved')}")
                            st.caption(f"💡 提分点: {item.get('reason')}")

                except Exception as e:
                    st.error(f"批改过程中出错: {e}")
                    st.write("原始响应内容：", response.choices[0].message.content if 'response' in locals() else "无")

import streamlit as st
from openai import OpenAI
import json
import pandas as pd
import plotly.express as px

# ================= 完美部署配置 =================
# 这里不再直接写字符串，而是通过 st.secrets 读取
# 这保证了你的 API Key 在 GitHub 上也是加密隐藏的
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    st.error("请在 Streamlit 管理后台配置您的 DEEPSEEK_API_KEY")
    st.stop()

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com" # 如果用的是 SiliconFlow，请改回对应的 URL
)

# ... (后续代码保持不变) ...
# 注意：确保 model="deepseek-chat" 与你的服务商匹配
# ================= 核心 Prompt (灵魂) =================
SYSTEM_PROMPT = """
你是一位资深的高中英语阅卷老师。请分析学生的作文。

要求：
1. 分析维度：语法(Grammar)、词汇(Vocabulary)、逻辑(Logic)、结构(Structure)。
2. 评分标准：满分25分。
3. 输出格式：必须是严格的 JSON 格式，不要包含 markdown 标记（如 ```json）。

JSON 结构示例：
{
  "score": {
    "total": 22,
    "grammar": 8,
    "vocabulary": 7,
    "logic": 7,
    "structure": 6
  },
  "comment": "你的文章结构清晰，但在时态使用上有一些错误...",
  "suggestions": [
    {
      "original": "bad sentence",
      "improved": "good sentence",
      "reason": "explanation here"
    }
  ]
}
"""

# ================= 页面布局 =================
st.set_page_config(page_title="高中英语作文 AI 精批 (DeepSeek版)", layout="wide")

st.title("📝 高中英语作文 AI 提分神器")
st.caption("Powered by DeepSeek-V3 (国产之光 🚀)")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✍️ 输入作文")
    st.info("提示：DeepSeek 暂不支持直接读图，请使用手机提取文字后粘贴到下方。")
    
    # 这里只保留文本输入框
    user_text = st.text_area("在此粘贴你的英语作文...", height=400, placeholder="例如：Running is good for health...")
    
    start_btn = st.button("开始 AI 批改", type="primary")

# ================= 处理逻辑 =================
if start_btn:
    if not user_text:
        st.warning("请先输入作文内容！")
    else:
        with col2:
            with st.spinner("DeepSeek 老师正在极速阅卷中..."):
                try:
                    # 1. 调用 DeepSeek API
                    response = client.chat.completions.create(
                        model="deepseek-chat",  # DeepSeek V3 模型名称
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_text}
                        ],
                        #这一步是为了防止DeepSeek有时候不返回JSON
                        response_format={ "type": "json_object" }, 
                        temperature=1.3 # DeepSeek 建议稍微高一点的温度以获得更好效果
                    )
                    
                    # 2. 获取返回的文本
                    content = response.choices[0].message.content
                    
                    # 3. 解析 JSON
                    result = json.loads(content)
                    
                    # 4. 展示结果
                    st.success("批改完成！")
                    
                    # --- 展示分数 ---
                    s = result.get('score', {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🏆 总分", f"{s.get('total', 0)}/25")
                    c2.metric("语法", s.get('grammar', 0))
                    c3.metric("词汇", s.get('vocabulary', 0))
                    c4.metric("逻辑", s.get('logic', 0))
                    
                    # --- 雷达图 ---
                    try:
                        scores = [s.get('grammar',0), s.get('vocabulary',0), s.get('logic',0), s.get('structure',0), s.get('grammar',0)]
                        df = pd.DataFrame(dict(r=scores, theta=['语法','词汇','逻辑','结构','语法']))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass 

                    # --- 点评 ---
                    st.info(f"💡 **名师点评：** {result.get('comment', '无点评')}")
                    
                    # --- 建议 ---
                    st.subheader("✨ 提分建议")
                    for item in result.get('suggestions', []):
                        with st.expander(f"❌ {item.get('original', '原文')}"):
                            st.markdown(f"**✅ 建议:** `{item.get('improved', '')}`")
                            st.caption(f"原因: {item.get('reason', '')}")

                except Exception as e:
                    st.error(f"发生错误: {e}")
                    # 如果解析JSON失败，打印原始内容方便调试
                    # st.text(content)
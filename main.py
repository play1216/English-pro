import streamlit as st
from openai import OpenAI
import json
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
import easyocr
import numpy as np
from PIL import Image

# ================= 1. 安全配置区域 =================
load_dotenv() # 加载 .env 文件
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    st.error("⚠️ 未找到 API Key，请在项目根目录创建 .env 文件并填入 key。")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# ================= 2. 初始化 OCR 模型 (带缓存) =================
# @st.cache_resource 是 Streamlit 的神器，它能把模型存要在内存里
# 这样除了第一次启动慢一点，后面每次识别都是秒开
@st.cache_resource
def load_ocr_reader():
    # ['en'] 表示只识别英文，这会比识别中文快很多
    return easyocr.Reader(['en'], gpu=False) # 如果你电脑有显卡，可以把 False 改成 True

# ================= 3. 核心 Prompt =================
SYSTEM_PROMPT = """
你是一位资深的高中英语阅卷老师。请分析学生的作文。
要求：
1. 分析维度：语法、词汇、逻辑、结构。
2. 评分标准：满分25分。
3. 输出格式：严格的 JSON 格式。
JSON 结构示例：
{
  "score": { "total": 20, "grammar": 7, "vocabulary": 6, "logic": 7, "structure": 0 },
  "comment": "点评内容...",
  "suggestions": [ { "original": "...", "improved": "...", "reason": "..." } ]
}
"""

# ================= 4. 页面布局 =================
st.set_page_config(page_title="英语作文 AI 精批 (OCR版)", layout="wide")
st.title("📝 高中英语作文 AI 提分神器")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 上传或输入")
    
    # 图片上传组件
    uploaded_file = st.file_uploader("上传手写作文照片 (自动识别)", type=['jpg', 'png', 'jpeg'])
    
    # 文本框 (如果识别有误，用户可以手动修改)
    default_text = ""
    
    # --- 核心逻辑：如果有图片，先进行 OCR ---
    if uploaded_file:
        with st.spinner("👀 正在识别图片中的文字... (首次运行需下载模型，请稍候)"):
            try:
                # 1. 加载模型
                reader = load_ocr_reader()
                # 2. 处理图片格式
                image = Image.open(uploaded_file)
                image_np = np.array(image) # 转成 numpy 数组给 easyocr 用
                # 3. 开始识别
                result = reader.readtext(image_np, detail=0, paragraph=True)
                # 4. 拼接结果
                default_text = "\n".join(result)
                st.success("✅ 识别成功！请在下方核对文字：")
            except Exception as e:
                st.error(f"OCR 识别失败: {e}")

    # 这里的 value 就是识别出来的文字，用户可以手动修改错别字
    user_text = st.text_area("作文内容 (识别结果可修改)", value=default_text, height=300, placeholder="粘贴或等待图片识别结果...")
    
    start_btn = st.button("开始 AI 批改", type="primary")

# ================= 5. 批改逻辑 (DeepSeek) =================
if start_btn:
    if not user_text:
        st.warning("内容为空，无法批改！")
    else:
        with col2:
            with st.spinner("🤖 DeepSeek 老师正在阅卷..."):
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_text}
                        ],
                        response_format={ "type": "json_object" },
                        temperature=1.2
                    )
                    
                    # 解析与展示
                    content = response.choices[0].message.content
                    result = json.loads(content)
                    
                    st.balloons() # 放个气球庆祝一下
                    
                    # 分数展示
                    s = result.get('score', {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("总分", f"{s.get('total',0)}/25")
                    c2.metric("语法", s.get('grammar',0))
                    c3.metric("词汇", s.get('vocabulary',0))
                    c4.metric("逻辑", s.get('logic',0))
                    
                    # 雷达图
                    try:
                        scores = [s.get('grammar',0), s.get('vocabulary',0), s.get('logic',0), s.get('structure',0), s.get('grammar',0)]
                        df = pd.DataFrame(dict(r=scores, theta=['语法','词汇','逻辑','结构','语法']))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        st.plotly_chart(fig, use_container_width=True)
                    except: pass

                    st.info(f"💡 **点评：** {result.get('comment','')}")
                    
                    st.subheader("✨ 提分建议")
                    for item in result.get('suggestions', []):
                        with st.expander(f"❌ {item.get('original','')}"):
                            st.write(f"✅ **建议:** {item.get('improved','')}")
                            st.caption(item.get('reason',''))

                except Exception as e:
                    st.error(f"API 调用出错: {e}")
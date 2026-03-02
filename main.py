import streamlit as st
import pandas as pd
import plotly.express as px
from modules.grading import EssayGrader  # 导入我们刚写的类
from modules.multi_ocr_engine import PaddleOCREngine
from modules.subscription import SubscriptionManager

# ================= 1. 初始化引擎 =================
st.set_page_config(page_title="DeepSeek 英语精批 Pro", layout="wide", page_icon="📝")
grader = EssayGrader()
subscription = SubscriptionManager()

# OCR配置参数（在侧边栏设置后初始化）
ocr_lang = 'en'  # 默认英语
ocr_use_gpu = False  # 默认CPU
ocr_enable_mkldnn = False  # 默认禁用MKLDNN
ocr_preprocess = True  # 默认值

# ================= 2. 侧边栏配置 =================
with st.sidebar:
    st.title("⚙️ 设置面板")
    st.markdown("---")
    
    # 模式选择
    mode = st.radio("输入方式", ["✍️ 文本输入", "📸 拍照上传(OCR)"])
    
    st.markdown("---")
    essay_type = st.radio(
        "作文类型",
        ("applied", "continuation"),
        format_func=lambda x: "应用文 (15分)" if x == "applied" else "读后续写 (25分)"
    )
    st.info("💡 提示：OCR 首次运行需要加载模型，请耐心等待 10-20 秒。")
    
    # OCR 配置选项
    if mode == "📸 拍照上传(OCR)":
        st.markdown("---")
        st.subheader("🔧 OCR 设置")
        
        # 语言选择
        ocr_lang = st.selectbox(
            "识别语言",
            ['en', 'ch'],
            format_func=lambda x: "英语" if x == 'en' else "中文"
        )
        
        # 性能选项
        col_a, col_b = st.columns(2)
        with col_a:
            ocr_use_gpu = st.checkbox("使用 GPU", value=False, help="需要 CUDA 支持")
        with col_b:
            ocr_enable_mkldnn = st.checkbox("启用 MKLDNN", value=False, help="CPU 加速优化")
        
        # 预处理选项
        ocr_preprocess = st.checkbox("图像预处理", value=True, help="提高识别准确率")
        
        # 智能校正选项
        ocr_smart_correction = st.checkbox(
            "智能单词校正", 
            value=True, 
            help="自动校正拼写错误，识别被划线的单词，排除横线干扰"
        )

    st.markdown("---")
    st.subheader("💳 会员与用量")
    col_p, col_l = st.columns(2)
    with col_p:
        st.metric("价格", f"¥{subscription.get_price()}/月")
    with col_l:
        st.metric("月上限", f"{subscription.get_limit()}条")
    used = subscription.current_usage()
    remaining = subscription.remaining()
    df_usage = pd.DataFrame({"类型": ["已使用", "剩余"], "数量": [used, remaining]})
    fig_usage = px.bar(df_usage, x="类型", y="数量", text="数量", color="类型", height=250)
    fig_usage.update_traces(textposition="outside")
    fig_usage.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_usage, use_container_width=True)
    if subscription.is_subscribed():
        st.success("已开通会员")
    else:
        if st.button(f"开通会员（¥{subscription.get_price()}/月）", use_container_width=True):
            subscription.subscribe()
            st.success("开通成功")
            st.rerun()

# 初始化OCR引擎（在侧边栏配置之后）
ocr_engine = PaddleOCREngine(lang=ocr_lang, use_gpu=ocr_use_gpu, enable_mkldnn=ocr_enable_mkldnn)

# ================= 3. 主界面 =================
st.title("📝 高中英语作文 AI 精批系统")
st.markdown("#### *Powered by DeepSeek-V3 & PaddleOCR*")

col1, col2 = st.columns([1, 1])

# --- 左侧：输入区 ---
with col1:
    st.subheader("1. 作文提交")
    
    final_input_text = ""

    # A. 拍照模式
    if mode == "📸 拍照上传(OCR)":
        uploaded_img = st.file_uploader("上传作文图片", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_img:
            # 展示缩略图
            st.image(uploaded_img, caption="已上传图片", use_container_width=True)
            
            # 识别按钮
            if st.button("🔍 开始识别", type="primary"):
                with st.spinner("🤖 正在进行OCR识别..."):
                    # 转换图像
                    import numpy as np
                    import cv2
                    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    
                    if image is None:
                        st.error("❌ 无法解析图片，请确认文件完整。")
                    else:
                        # 使用PaddleOCR提取文字
                        result = ocr_engine.extract_text(image)
                        if result.text.strip():
                            extracted = result.text
                            st.success("✅ 识别完成！")
                            # 将识别结果放入 session_state
                            st.session_state['ocr_result'] = extracted
                            st.rerun()
                        else:
                            st.error("❌ 未能识别到有效文字，请尝试重新拍照或调整图片角度。")
                    
                    uploaded_img.seek(0)
    
    # B. 文本展示/编辑区 (无论是OCR还是手输，都在这里汇总)
    # 使用 session_state 保持 OCR 的结果
    default_text = st.session_state.get('ocr_result', "")
    
    user_text = st.text_area(
        "📝 作文内容 (已智能校对，下划线标注需检查的单词)", 
        value=default_text,
        height=400,
        placeholder="在此直接输入或等待智能OCR识别结果..."
    )
    
    submit_btn = st.button("🚀 开始 AI 老师批改", type="primary", use_container_width=True)

# --- 右侧：结果区 ---
with col2:
    st.subheader("2. 批改报告")
    
    if submit_btn:
        if len(user_text) < 10:
            st.warning("⚠️ 内容太短，无法批改。")
        else:
            if not subscription.is_subscribed():
                st.error(f"需要开通会员（¥{subscription.get_price()}/月，月上限{subscription.get_limit()}条）")
            elif subscription.remaining() <= 0:
                st.error("已达到本月500条使用上限")
            else:
                with st.spinner("👩‍🏫 阅卷组长正在评分..."):
                    result = grader.grade(user_text, essay_type)
                    
                    if "error" in result:
                        st.error(result['error'])
                    else:
                        subscription.increment(1)
                        s = result.get('score', {})
                        total = s.get('total', 0)
                        rank = s.get('rank', '未定档')
                        max_score = 15 if essay_type == 'applied' else 25
                        color = "#28a745" if (total/max_score) > 0.8 else "#ffc107"
                        st.markdown(f"""
                            <div style="padding:15px; border-radius:10px; background:#f0f2f6; text-align:center;">
                                <h3 style="margin:0; color:#555;">{rank}</h3>
                                <h1 style="font-size:4rem; margin:0; color:{color};">{total} <span style="font-size:1.5rem; color:#999;">/ {max_score}</span></h1>
                            </div>
                        """, unsafe_allow_html=True)
                        radar = s.get('radar', {})
                        if radar:
                            df = pd.DataFrame(dict(
                                r=list(radar.values()),
                                theta=['语法', '词汇', '逻辑', '结构']
                            ))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_traces(fill='toself')
                            st.plotly_chart(fig, use_container_width=True)
                        st.info(f"📌 **总评**：{result.get('comment')}")
                        st.markdown("### ✍️ 句型升级")
                        for advice in result.get('suggestions', []):
                            with st.expander(f"❌ {advice['original'][:30]}..."):
                                st.write(f"**✅ 改写**：{advice['improved']}")
                                st.caption(f"💡 原因：{advice['reason']}")

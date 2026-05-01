import streamlit as st
import pandas as pd
import plotly.express as px
from modules.afdian import AfdianClient
from modules.grading import EssayGrader  # 导入我们刚写的类
from modules.auth import AuthManager, render_auth_gate
from modules.image_utils import load_image_rgb
from modules.membership import render_membership_panel
from modules.multi_ocr_engine import PaddleOCREngine
from modules.subscription import SubscriptionManager
from modules.ui import apply_app_theme, render_app_header, render_panel_title, render_score_card

# ================= 1. 初始化引擎 =================
st.set_page_config(page_title="DeepSeek 英语精批 Pro", layout="wide", page_icon="📝")
apply_app_theme()
auth_manager = AuthManager()
if not auth_manager.is_logged_in():
    render_auth_gate(auth_manager)

current_user = auth_manager.get_current_user()
grader = EssayGrader()
membership_days = int(st.secrets.get("MEMBERSHIP_DAYS", 31))
subscription = SubscriptionManager(user_id=current_user, membership_days=membership_days)
afdian_client = AfdianClient()
st.session_state.setdefault("essay_prompt_text", "")
st.session_state.setdefault("essay_body_text", st.session_state.get("ocr_result", ""))

# OCR配置参数（在侧边栏设置后初始化）
ocr_lang = 'en'  # 默认英语
ocr_use_gpu = False  # 默认CPU
ocr_enable_mkldnn = False  # 默认禁用MKLDNN
ocr_preprocess = True  # 默认值

# ================= 2. 侧边栏配置 =================
with st.sidebar:
    st.title("⚙️ 设置面板")
    st.markdown("---")
    st.subheader("👤 当前用户")
    st.write(current_user)
    if st.button("退出登录", use_container_width=True):
        auth_manager.logout()
        st.rerun()
    
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
    render_membership_panel(subscription, afdian_client, panel_key="main")

# 初始化OCR引擎（在侧边栏配置之后）
ocr_engine = PaddleOCREngine(lang=ocr_lang, use_gpu=ocr_use_gpu, enable_mkldnn=ocr_enable_mkldnn)

# ================= 3. 主界面 =================
essay_type_label = "应用文 (15分)" if essay_type == "applied" else "读后续写 (25分)"
render_app_header(
    "高中英语作文 AI 精批系统",
    "支持题目定向批改、正文 OCR 识别和按用户计费的精批工作台。",
    current_user,
    essay_type_label,
)

col1, col2 = st.columns([1.02, 0.98], gap="large")

# --- 左侧：输入区 ---
with col1:
    render_panel_title("1. 作文提交", "先补充题目或材料，再录入学生作文，批改会更贴题。")
    task_input_label = "作文题目 / 写作任务"
    task_input_placeholder = "例如：假设你是李华，请给外教写一封邀请信，说明活动时间、地点和内容。"
    prompt_upload_label = "上传作文题目图片"
    if essay_type == "continuation":
        task_input_label = "原文材料 / 段首句 / 续写要求"
        task_input_placeholder = "请粘贴读后续写原文、两段首句和关键词，这样 AI 才能判断是否衔接自然、有没有跑题。"
        prompt_upload_label = "上传原文材料图片"

    with st.expander("📷 拍照导入题目 / 材料", expanded=False):
        uploaded_prompt_img = st.file_uploader(
            prompt_upload_label,
            type=["jpg", "png", "jpeg"],
            key="essay_prompt_uploader_main",
        )
        if uploaded_prompt_img:
            st.image(uploaded_prompt_img, caption="待识别题目图片", use_container_width=True)
            if st.button("识别题目图片", key="recognize_prompt_main", use_container_width=True):
                with st.spinner("🔎 正在识别作文题目..."):
                    prompt_image = load_image_rgb(uploaded_prompt_img)
                    prompt_result = ocr_engine.extract_text(prompt_image)
                    if prompt_result.text.strip():
                        extracted_prompt = prompt_result.text.strip()
                        st.session_state["essay_prompt_result"] = extracted_prompt
                        st.session_state["essay_prompt_text"] = extracted_prompt
                        st.success("题目识别完成，已回填到输入框。")
                        st.rerun()
                    else:
                        st.error("未识别到有效题目信息，请尝试更清晰的图片。")

    render_panel_title(task_input_label, "支持手动输入，也支持拍照导入。")
    essay_prompt = st.text_area(
        task_input_label,
        height=180,
        placeholder=task_input_placeholder,
        key="essay_prompt_text",
    )

    # A. 拍照模式
    if mode == "📸 拍照上传(OCR)":
        render_panel_title("作文正文图片导入", "适合直接拍学生答题纸或手写作文。")
        uploaded_img = st.file_uploader("上传作文图片", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_img:
            # 展示缩略图
            st.image(uploaded_img, caption="已上传图片", use_container_width=True)
            
            # 识别按钮
            if st.button("🔍 开始识别", type="primary"):
                with st.spinner("🤖 正在进行OCR识别..."):
                    image = load_image_rgb(uploaded_img)
                    
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
                            st.session_state["essay_body_text"] = extracted
                            st.rerun()
                        else:
                            st.error("❌ 未能识别到有效文字，请尝试重新拍照或调整图片角度。")
                    
                    uploaded_img.seek(0)
    
    # B. 文本展示/编辑区 (无论是OCR还是手输，都在这里汇总)
    render_panel_title("学生作文正文", "识别结果会自动回填，你也可以直接手动修改。")
    user_text = st.text_area(
        "📝 作文内容 (已智能校对，下划线标注需检查的单词)", 
        height=400,
        placeholder="在此直接输入或等待智能OCR识别结果...",
        key="essay_body_text",
    )
    
    submit_btn = st.button("🚀 开始 AI 老师批改", type="primary", use_container_width=True)

# --- 右侧：结果区 ---
with col2:
    render_panel_title("2. 批改报告", "系统会结合题目要求、语言表现和结构完成度给出综合判断。")
    
    if submit_btn:
        if len(user_text) < 10:
            st.warning("⚠️ 内容太短，无法批改。")
        else:
            if not subscription.can_grade():
                if subscription.is_subscribed():
                    st.error(f"已达到本月 {subscription.get_limit()} 篇会员上限")
                else:
                    st.error("本账号 3 次免费额度已用完，请先完成爱发电付费开通。")
            else:
                with st.spinner("👩‍🏫 阅卷组长正在评分..."):
                    result = grader.grade(user_text, essay_type, essay_prompt=essay_prompt)
                    
                    if "error" in result:
                        st.error(result['error'])
                    else:
                        subscription.increment(1)
                        s = result.get('score', {})
                        total = s.get('total', 0)
                        rank = s.get('rank', '未定档')
                        max_score = 15 if essay_type == 'applied' else 25
                        render_score_card(rank, total, max_score)
                        radar = s.get('radar', {})
                        if radar:
                            df = pd.DataFrame(dict(
                                r=list(radar.values()),
                                theta=['语法', '词汇', '逻辑', '结构']
                            ))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_traces(fill='toself')
                            st.plotly_chart(fig, use_container_width=True)
                        task_focus = result.get("task_focus", {})
                        if task_focus:
                            st.markdown("### 🎯 题目贴合度")
                            st.write(task_focus.get("summary", ""))
                            st.write(task_focus.get("task_completion", ""))
                            missed_points = [point for point in task_focus.get("missed_points", []) if point]
                            if missed_points:
                                st.warning("待补强要点：" + "；".join(missed_points))
                        st.info(f"📌 **总评**：{result.get('comment')}")
                        if not subscription.is_subscribed():
                            st.caption(f"本账号剩余免费批改次数：{subscription.free_remaining()} / {subscription.get_free_limit()}")
                        st.markdown("### ✍️ 句型升级")
                        for advice in result.get('suggestions', []):
                            with st.expander(f"❌ {advice['original'][:30]}..."):
                                st.write(f"**✅ 改写**：{advice['improved']}")
                                st.caption(f"💡 原因：{advice['reason']}")

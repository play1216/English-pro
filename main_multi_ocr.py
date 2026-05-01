import streamlit as st
import pandas as pd
import plotly.express as px
from modules.afdian import AfdianClient
from modules.auth import AuthManager, render_auth_gate
from modules.grading import EssayGrader
from modules.image_utils import load_image_bgr
from modules.membership import render_membership_panel
from modules.multi_ocr_engine import MultiOCREngine, GoogleVisionOCREngine, AzureOCREngine, ScnetOCREngine
from modules.ocr_config import OCRConfigManager
from modules.simple_logic_validator import SimpleLogicValidator
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

# 初始化OCR配置管理器
config_manager = OCRConfigManager()

# 初始化多OCR引擎
multi_ocr_engine = MultiOCREngine()

# 添加API引擎
enabled_apis = config_manager.get_enabled_apis()
for api_name, api_config in enabled_apis.items():
    if api_name == 'google_vision':
        multi_ocr_engine.add_api_engine(GoogleVisionOCREngine(api_config.api_key))
    elif api_name == 'azure_ocr':
        multi_ocr_engine.add_api_engine(AzureOCREngine(api_config.api_key, api_config.endpoint))
    elif api_name == 'scnet_ocr':
        multi_ocr_engine.add_api_engine(ScnetOCREngine(api_config.api_key, api_config.endpoint))

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
    mode = st.radio("输入方式", ["✍️ 文本输入", "📸 拍照上传(多OCR)"])
    
    st.markdown("---")
    essay_type = st.radio(
        "作文类型",
        ("applied", "continuation"),
        format_func=lambda x: "应用文 (15分)" if x == "applied" else "读后续写 (25分)"
    )
    
    # OCR配置页面链接
    if mode == "📸 拍照上传(多OCR)":
        if st.button("🔑 OCR API 配置", type="secondary"):
            st.switch_page("OCR配置")
        
        st.info("💡 提示：多OCR系统会自动选择最佳识别结果，提高准确率。")
        
        # OCR策略选择
        st.markdown("**🎯 OCR 策略**")
        ocr_strategy = st.selectbox(
            "识别策略",
            ["成本优先", "质量优先", "平衡模式"],
            help="选择OCR识别的优先策略"
        )
        
        # 预算设置
        st.markdown("**💰 预算控制**")
        daily_limit = st.number_input(
            "日预算限制 ($)",
            min_value=0.0,
            value=0.10,
            step=0.01,
            help="每日API调用最大预算"
        )
        
        monthly_limit = st.number_input(
            "月预算限制 ($)",
            min_value=0.0,
            value=3.00,
            step=0.10,
            help="每月API调用最大预算"
        )
        
        # 使用报告
        usage_report = config_manager.get_usage_report()
        budget_status = config_manager.check_budget_limits(daily_limit, monthly_limit)
        
        st.markdown("**📊 使用统计**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("今日消费", f"${usage_report['today_cost']:.4f}")
        with col2:
            st.metric("本月消费", f"${usage_report['month_cost']:.4f}")
        with col3:
            st.metric("总消费", f"${usage_report['total_cost']:.4f}")
        
        # 预算状态
        if not budget_status['can_use_api']:
            st.error("⚠️ 已达到预算限制，API调用将被暂停")
        else:
            st.success(f"✅ 预算充足：日剩余 ${budget_status['daily_remaining']:.4f}")
    
    st.markdown("---")
    render_membership_panel(subscription, afdian_client, panel_key="multi")

# ================= 3. 主界面 =================
essay_type_label = "应用文 (15分)" if essay_type == "applied" else "读后续写 (25分)"
render_app_header(
    "高中英语作文 AI 精批系统",
    "多引擎 OCR、题目定向批改和按用户计费的精批工作台。",
    current_user,
    essay_type_label,
)

col1, col2 = st.columns([1.02, 0.98], gap="large")

# --- 左侧：输入区 ---
with col1:
    render_panel_title("1. 作文提交", "支持先识别题目材料，再识别正文，最后统一批改。")
    task_input_label = "作文题目 / 写作任务"
    task_input_placeholder = "例如：假设你是李华，请写一封建议信，说明问题并给出两条建议。"
    prompt_upload_label = "上传作文题目图片"
    if essay_type == "continuation":
        task_input_label = "原文材料 / 段首句 / 续写要求"
        task_input_placeholder = "请粘贴读后续写原文、段首句和关键信息，系统会结合材料判断情节衔接与续写方向。"
        prompt_upload_label = "上传原文材料图片"

    with st.expander("📷 拍照导入题目 / 材料", expanded=False):
        uploaded_prompt_img = st.file_uploader(
            prompt_upload_label,
            type=["jpg", "png", "jpeg"],
            key="essay_prompt_uploader_multi",
        )
        if uploaded_prompt_img:
            st.image(uploaded_prompt_img, caption="待识别题目图片", use_container_width=True)
            if st.button("识别题目图片", key="recognize_prompt_multi", use_container_width=True):
                with st.spinner("🔎 正在识别作文题目..."):
                    prompt_image = load_image_bgr(uploaded_prompt_img)
                    prompt_result = multi_ocr_engine.extract_text_with_voting(
                        prompt_image,
                        max_engines=2,
                        budget_limit=0.01,
                    )
                    if prompt_result["text"].strip():
                        extracted_prompt = prompt_result["text"].strip()
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
    if mode == "📸 拍照上传(多OCR)":
        render_panel_title("作文正文图片导入", "可组合多种 OCR 引擎，提升识别稳定性。")
        uploaded_img = st.file_uploader("上传作文图片", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_img:
            # 展示缩略图
            st.image(uploaded_img, caption="已上传图片", use_container_width=True)
            
            # 高级选项
            with st.expander("🔧 高级OCR选项"):
                max_engines = st.slider(
                    "最大使用引擎数",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="同时使用的OCR引擎数量"
                )
                
                confidence_threshold = st.slider(
                    "置信度阈值",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="低于此值的结果将被标记为需要人工检查"
                )
                
                enable_logic_check = st.checkbox(
                    "启用逻辑校验",
                    value=True,
                    help="检查识别结果的逻辑连贯性"
                )
            
            # 识别按钮
            if st.button("🔍 开始多引擎识别", type="primary"):
                with st.spinner("🤖 正在进行多引擎OCR识别..."):
                    try:
                        logic_validator = SimpleLogicValidator()
                        image = load_image_bgr(uploaded_img)
                        
                        if image is None:
                            st.error("❌ 无法解析图片，请确认文件完整。")
                        else:
                            # 根据策略设置参数
                            if ocr_strategy == "成本优先":
                                budget_limit = daily_limit * 0.1  # 使用10%的日预算
                                max_engines_to_use = 2
                            elif ocr_strategy == "质量优先":
                                budget_limit = daily_limit * 0.5  # 使用50%的日预算
                                max_engines_to_use = max_engines
                            else:  # 平衡模式
                                budget_limit = daily_limit * 0.2  # 使用20%的日预算
                                max_engines_to_use = min(3, max_engines)
                            
                            # 执行多OCR识别
                            result = multi_ocr_engine.extract_text_with_voting(
                                image, 
                                max_engines=max_engines_to_use,
                                budget_limit=budget_limit
                            )
                            
                            if result['text'].strip():
                                # 逻辑校验
                                if enable_logic_check:
                                    validation = logic_validator.validate_essay_logic(result['text'])
                                    
                                    if not validation['is_logical']:
                                        st.warning("⚠️ 识别结果逻辑不够通顺，可能需要人工校对")
                                
                                # 标注低置信度单词
                                if result['confidence'] < confidence_threshold:
                                    problematic_words = logic_validator.identify_problematic_words(result['text'])
                                    final_text = logic_validator.suggest_corrections(result['text'], problematic_words)
                                    needs_review = True
                                else:
                                    final_text = result['text']
                                    needs_review = False
                                
                                # 显示结果
                                st.success("✅ 多引擎识别完成！")
                                
                                # 详细信息
                                st.markdown("**📊 识别详情**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("置信度", f"{result['confidence']:.1%}")
                                with col2:
                                    st.metric("使用引擎", ", ".join(result['engines_used']))
                                with col3:
                                    st.metric("成本", f"${result['total_cost']:.4f}")
                                
                                # 成本追踪
                                config_manager.track_usage("multi_ocr", result['total_cost'])
                                
                                # 如果需要人工检查，显示提示
                                if needs_review:
                                    st.info("🔍 部分单词标注了下划线，请检查确认")
                                
                                # 将识别结果放入 session_state
                                st.session_state['ocr_result'] = final_text
                                st.session_state["essay_body_text"] = final_text
                                st.session_state['ocr_details'] = result
                                st.rerun()
                            else:
                                st.error("❌ 未能识别到有效文字，请尝试重新拍照或调整图片角度。")
                    
                    except Exception as e:
                        st.error(f"❌ 识别过程发生错误: {str(e)}")
                    finally:
                        uploaded_img.seek(0)
    
    # B. 文本展示/编辑区
    render_panel_title("学生作文正文", "识别结果会自动回填，你也可以继续人工润色。")
    user_text = st.text_area(
        "📝 作文内容 (多引擎智能识别)", 
        height=400,
        placeholder="在此直接输入或等待多OCR识别结果...",
        key="essay_body_text",
    )
    
    # 显示识别详情（如果有）
    if 'ocr_details' in st.session_state and mode == "📸 拍照上传(多OCR)":
        with st.expander("📋 识别详情"):
            details = st.session_state['ocr_details']
            
            st.write("**🏆 最佳结果**")
            st.write(details['text'])
            
            if details.get('all_results'):
                st.write("**🔍 所有引擎结果**")
                for engine, text, conf in details['all_results']:
                    with st.expander(f"{engine} (置信度: {conf:.1%})"):
                        st.write(text)
    
    submit_btn = st.button("🚀 开始 AI 老师批改", type="primary", use_container_width=True)

# --- 右侧：结果区 ---
with col2:
    render_panel_title("2. 批改报告", "系统会结合题目、正文和衔接质量生成更细的评价。")
    
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

# ================= 4. OCR配置页面 =================
def show_ocr_config_page():
    st.title("🔑 OCR API 配置")
    st.markdown("---")
    
    # 返回主页按钮
    if st.button("🏠 返回主页", type="secondary"):
        st.switch_page("main")
    
    # 显示配置界面
    config_manager.render_config_ui()
    
    # 成本报告
    st.markdown("---")
    st.subheader("📊 成本分析")
    cost_report = multi_ocr_engine.get_cost_report()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("总请求次数", cost_report['request_count'])
        st.metric("总成本", f"${cost_report['total_cost']:.4f}")
        st.metric("平均成本/请求", f"${cost_report['average_cost_per_request']:.4f}")
    
    with col2:
        if cost_report['engine_usage']:
            st.write("**引擎使用统计**")
            for engine, usage in cost_report['engine_usage'].items():
                st.write(f"- {engine}: {usage['count']} 次, ${usage['cost']:.4f}")

# 页面路由
page = st.sidebar.selectbox("选择页面", ["主页", "OCR配置"])

if page == "OCR配置":
    show_ocr_config_page()
else:
    # 隐藏配置页面的侧边栏内容
    pass

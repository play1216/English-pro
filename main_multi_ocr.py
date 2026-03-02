import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import cv2
from modules.grading import EssayGrader
from modules.multi_ocr_engine import MultiOCREngine, GoogleVisionOCREngine, AzureOCREngine
from modules.ocr_config import OCRConfigManager
from modules.simple_logic_validator import SimpleLogicValidator

# ================= 1. 初始化引擎 =================
st.set_page_config(page_title="DeepSeek 英语精批 Pro", layout="wide", page_icon="📝")
grader = EssayGrader()

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

# ================= 2. 侧边栏配置 =================
with st.sidebar:
    st.title("⚙️ 设置面板")
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

# ================= 3. 主界面 =================
st.title("📝 高中英语作文 AI 精批系统 (多OCR版)")
st.markdown("#### *Powered by DeepSeek-V3 & Multi-OCR Engine*")

col1, col2 = st.columns([1, 1])

# --- 左侧：输入区 ---
with col1:
    st.subheader("1. 作文提交")
    
    final_input_text = ""

    # A. 拍照模式
    if mode == "📸 拍照上传(多OCR)":
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
                        # 转换图像
                        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, 1)
                        
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
                                    logic_validator = SimpleLogicValidator()
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
                                st.session_state['ocr_details'] = result
                                st.rerun()
                            else:
                                st.error("❌ 未能识别到有效文字，请尝试重新拍照或调整图片角度。")
                    
                    except Exception as e:
                        st.error(f"❌ 识别过程发生错误: {str(e)}")
                    finally:
                        uploaded_img.seek(0)
    
    # B. 文本展示/编辑区
    default_text = st.session_state.get('ocr_result', "")
    
    user_text = st.text_area(
        "📝 作文内容 (多引擎智能识别)", 
        value=default_text,
        height=400,
        placeholder="在此直接输入或等待多OCR识别结果..."
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
    st.subheader("2. 批改报告")
    
    if submit_btn:
        if len(user_text) < 10:
            st.warning("⚠️ 内容太短，无法批改。")
        else:
            with st.spinner("👩‍🏫 阅卷组长正在评分..."):
                result = grader.grade(user_text, essay_type)
                
                if "error" in result:
                    st.error(result['error'])
                else:
                    # 解析结果
                    s = result.get('score', {})
                    total = s.get('total', 0)
                    rank = s.get('rank', '未定档')
                    max_score = 15 if essay_type == 'applied' else 25
                    
                    # 1. 顶部大分
                    color = "#28a745" if (total/max_score) > 0.8 else "#ffc107"
                    st.markdown(f"""
                        <div style="padding:15px; border-radius:10px; background:#f0f2f6; text-align:center;">
                            <h3 style="margin:0; color:#555;">{rank}</h3>
                            <h1 style="font-size:4rem; margin:0; color:{color};">{total} <span style="font-size:1.5rem; color:#999;">/ {max_score}</span></h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 2. 维度雷达图
                    radar = s.get('radar', {})
                    if radar:
                        df = pd.DataFrame(dict(
                            r=list(radar.values()),
                            theta=['语法', '词汇', '逻辑', '结构']
                        ))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        fig.update_traces(fill='toself')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 3. 点评与建议
                    st.info(f"📌 **总评**：{result.get('comment')}")
                    
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

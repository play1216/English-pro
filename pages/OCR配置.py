import streamlit as st

from modules.auth import AuthManager, render_auth_gate
from modules.multi_ocr_engine import MultiOCREngine
from modules.ocr_config import OCRConfigManager
from modules.ui import apply_app_theme


st.set_page_config(page_title="OCR API 配置", layout="wide", page_icon="🔧")
apply_app_theme()

auth_manager = AuthManager()
if not auth_manager.is_logged_in():
    render_auth_gate(auth_manager)

st.title("🔧 OCR API 配置")
st.markdown("---")

if st.button("🏠 返回主页面", type="secondary"):
    st.switch_page("main_multi_ocr.py")

config_manager = OCRConfigManager()
config_manager.render_config_ui()

st.markdown("---")
st.subheader("📊 成本分析")
multi_ocr_engine = MultiOCREngine()
cost_report = multi_ocr_engine.get_cost_report()

col1, col2 = st.columns(2)
with col1:
    st.metric("总请求次数", cost_report["request_count"])
    st.metric("总成本", f"${cost_report['total_cost']:.4f}")
    st.metric("平均成本/请求", f"${cost_report['average_cost_per_request']:.4f}")

with col2:
    if cost_report["engine_usage"]:
        st.write("**引擎使用统计**")
        for engine, usage in cost_report["engine_usage"].items():
            st.write(f"- {engine}: {usage['count']} 次, ${usage['cost']:.4f}")
    else:
        st.caption("暂无调用记录。")

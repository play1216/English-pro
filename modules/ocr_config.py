import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict

import streamlit as st


@dataclass
class OCRAPIConfig:
    name: str
    api_key: str
    endpoint: str = ""
    cost_per_request: float = 0.0
    enabled: bool = False
    description: str = ""


class OCRConfigManager:
    def __init__(self):
        self.config_file = os.path.join(".streamlit", "ocr_config.local.json")
        self.legacy_config_file = "ocr_config.json"
        self.usage_file = "ocr_usage.json"
        self.config_source = "local"
        self._ensure_storage_dir()
        self.load_config()
        self.load_usage()

    def _ensure_storage_dir(self):
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

    def load_config(self):
        try:
            if "OCR_CONFIG" in st.secrets:
                config_data = st.secrets["OCR_CONFIG"]
                if isinstance(config_data, str):
                    config_data = json.loads(config_data)
                self.apis = {
                    name: OCRAPIConfig(**data)
                    for name, data in config_data.get("apis", {}).items()
                }
                self.config_source = "secrets"
                return

            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                self.apis = {
                    name: OCRAPIConfig(**data)
                    for name, data in config_data.get("apis", {}).items()
                }
                self.config_source = "local"
                return

            if os.path.exists(self.legacy_config_file):
                with open(self.legacy_config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                self.apis = {
                    name: OCRAPIConfig(**data)
                    for name, data in config_data.get("apis", {}).items()
                }
                self.config_source = "legacy"
                self.save_config()
                self.config_source = "local"
                return

            self.apis = self._get_default_configs()
            self.config_source = "local"
            self.save_config()
        except Exception as exc:
            logging.error("加载 OCR 配置失败: %s", exc)
            self.apis = self._get_default_configs()
            self.config_source = "local"

    def save_config(self):
        try:
            config_data = {
                "apis": {
                    name: {
                        "name": api.name,
                        "api_key": api.api_key,
                        "endpoint": api.endpoint,
                        "cost_per_request": api.cost_per_request,
                        "enabled": api.enabled,
                        "description": api.description,
                    }
                    for name, api in self.apis.items()
                }
            }

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            st.error(f"保存 OCR 配置失败: {exc}")

    def load_usage(self):
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, "r", encoding="utf-8") as f:
                    self.usage_data = json.load(f)
            else:
                self.usage_data = {
                    "daily_usage": {},
                    "monthly_usage": {},
                    "total_usage": 0.0,
                    "request_count": 0,
                }
                self.save_usage()
        except Exception as exc:
            st.error(f"加载使用统计失败: {exc}")
            self.usage_data = {
                "daily_usage": {},
                "monthly_usage": {},
                "total_usage": 0.0,
                "request_count": 0,
            }

    def save_usage(self):
        try:
            with open(self.usage_file, "w", encoding="utf-8") as f:
                json.dump(self.usage_data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            st.error(f"保存使用统计失败: {exc}")

    def _get_default_configs(self) -> Dict[str, OCRAPIConfig]:
        return {
            "google_vision": OCRAPIConfig(
                name="Google Vision",
                api_key="",
                cost_per_request=0.0015,
                description="Google Cloud Vision API，高精度 OCR，约 $1.5/1000 次",
            ),
            "azure_ocr": OCRAPIConfig(
                name="Azure Computer Vision",
                api_key="",
                endpoint="https://<region>.api.cognitive.microsoft.com",
                cost_per_request=0.001,
                description="Azure Computer Vision，稳定可靠，约 $1/1000 次",
            ),
            "baidu_ocr": OCRAPIConfig(
                name="百度 OCR",
                api_key="",
                endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic",
                cost_per_request=0.0001,
                description="百度智能云 OCR，中文优化，约 ¥0.5/1000 次",
            ),
            "scnet_ocr": OCRAPIConfig(
                name="Scnet OCR",
                api_key="",
                endpoint="https://api.scnet.cn/api/llm/v1/ocr/recognize",
                cost_per_request=0.0,
                description="Scnet OCR，自定义服务端点。",
            ),
        }

    def get_enabled_apis(self) -> Dict[str, OCRAPIConfig]:
        return {
            name: api
            for name, api in self.apis.items()
            if api.enabled and api.api_key
        }

    def update_api_config(self, api_name: str, persist=True, **kwargs):
        if api_name in self.apis:
            for key, value in kwargs.items():
                if hasattr(self.apis[api_name], key):
                    setattr(self.apis[api_name], key, value)
            if persist:
                self.save_config()

    def track_usage(self, api_name: str, cost: float):
        today = datetime.now().strftime("%Y-%m-%d")
        this_month = datetime.now().strftime("%Y-%m")

        if today not in self.usage_data["daily_usage"]:
            self.usage_data["daily_usage"][today] = 0.0
        self.usage_data["daily_usage"][today] += cost

        if this_month not in self.usage_data["monthly_usage"]:
            self.usage_data["monthly_usage"][this_month] = 0.0
        self.usage_data["monthly_usage"][this_month] += cost

        self.usage_data["total_usage"] += cost
        self.usage_data["request_count"] += 1
        self.save_usage()

    def get_usage_report(self) -> Dict:
        today = datetime.now().strftime("%Y-%m-%d")
        this_month = datetime.now().strftime("%Y-%m")
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.usage_data["daily_usage"] = {
            date: cost
            for date, cost in self.usage_data["daily_usage"].items()
            if date >= cutoff_date
        }

        return {
            "today_cost": self.usage_data["daily_usage"].get(today, 0.0),
            "month_cost": self.usage_data["monthly_usage"].get(this_month, 0.0),
            "total_cost": self.usage_data["total_usage"],
            "request_count": self.usage_data["request_count"],
            "average_cost_per_request": (
                self.usage_data["total_usage"] / self.usage_data["request_count"]
                if self.usage_data["request_count"] > 0
                else 0
            ),
        }

    def check_budget_limits(self, daily_limit: float, monthly_limit: float) -> Dict:
        usage_report = self.get_usage_report()
        daily_remaining = daily_limit - usage_report["today_cost"]
        monthly_remaining = monthly_limit - usage_report["month_cost"]

        return {
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
            "daily_used": usage_report["today_cost"],
            "monthly_used": usage_report["month_cost"],
            "daily_remaining": max(0, daily_remaining),
            "monthly_remaining": max(0, monthly_remaining),
            "within_daily_limit": usage_report["today_cost"] <= daily_limit,
            "within_monthly_limit": usage_report["month_cost"] <= monthly_limit,
            "can_use_api": daily_remaining > 0 and monthly_remaining > 0,
        }

    def render_config_ui(self):
        st.subheader("🔑 OCR API 配置")

        st.write("**💰 预算控制**")
        col1, col2 = st.columns(2)
        with col1:
            daily_limit = st.number_input(
                "日预算限制 ($)",
                min_value=0.0,
                value=0.10,
                step=0.01,
                help="每日 API 调用最大预算",
            )
        with col2:
            monthly_limit = st.number_input(
                "月预算限制 ($)",
                min_value=0.0,
                value=3.00,
                step=0.10,
                help="每月 API 调用最大预算",
            )

        usage_report = self.get_usage_report()
        budget_status = self.check_budget_limits(daily_limit, monthly_limit)

        st.write("**📊 使用统计**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("今日消费", f"${usage_report['today_cost']:.4f}")
        with col2:
            st.metric("本月消费", f"${usage_report['month_cost']:.4f}")
        with col3:
            st.metric("总消费", f"${usage_report['total_cost']:.4f}")

        if not budget_status["can_use_api"]:
            st.error("⚠️ 已达到预算限制，API 调用将被暂停")
        else:
            st.success(f"✅ 预算充足：日剩余 ${budget_status['daily_remaining']:.4f}")

        use_secrets = self.config_source == "secrets"
        if use_secrets:
            st.info("当前 OCR 配置来自 `.streamlit/secrets.toml`，为了避免密钥落盘，下面仅展示状态，不提供页面内改写。")
        elif self.config_source == "legacy":
            st.warning("检测到旧版 `ocr_config.json`，配置已迁移到 `.streamlit/ocr_config.local.json`。")
        else:
            st.caption("本地 OCR 密钥将保存到 `.streamlit/ocr_config.local.json`，不会再写入仓库根目录。")

        st.write("**🔧 API 配置**")
        for api_name, api_config in self.apis.items():
            with st.expander(f"{api_config.name} 配置"):
                if use_secrets:
                    st.write(f"启用状态：{'已启用' if api_config.enabled else '未启用'}")
                    st.write(f"密钥状态：{'已配置' if api_config.api_key else '未配置'}")
                    if api_config.endpoint:
                        st.write(f"端点：{api_config.endpoint}")
                else:
                    enabled = st.checkbox(
                        "启用此 API",
                        value=api_config.enabled,
                        key=f"{api_name}_enabled",
                    )
                    api_key = st.text_input(
                        "API 密钥",
                        value=api_config.api_key,
                        type="password",
                        key=f"{api_name}_api_key",
                        help="输入您的 API 密钥",
                    )

                    endpoint = api_config.endpoint
                    if api_name in {"azure_ocr", "baidu_ocr", "scnet_ocr"}:
                        endpoint = st.text_input(
                            "API 端点",
                            value=api_config.endpoint,
                            key=f"{api_name}_endpoint",
                        )

                    self.update_api_config(
                        api_name,
                        persist=False,
                        enabled=enabled,
                        api_key=api_key,
                        endpoint=endpoint,
                    )

                st.info(f"💡 {api_config.description}")
                st.write(f"💰 每次调用成本: ${api_config.cost_per_request:.4f}")

        if not use_secrets and st.button("💾 保存配置", type="primary"):
            self.save_config()
            st.success("配置已保存。")
            st.rerun()

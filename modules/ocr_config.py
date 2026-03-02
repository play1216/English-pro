import streamlit as st
import json
import os
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class OCRAPIConfig:
    """OCR API配置"""
    name: str
    api_key: str
    endpoint: str = ""
    cost_per_request: float = 0.0
    enabled: bool = False
    description: str = ""

class OCRConfigManager:
    """OCR配置管理器"""
    
    def __init__(self):
        self.config_file = "ocr_config.json"
        self.usage_file = "ocr_usage.json"
        self.load_config()
        self.load_usage()
    
    def load_config(self):
        """加载配置"""
        try:
            # 1. 首先尝试从 st.secrets 读取（针对 Streamlit Cloud 部署）
            if "OCR_CONFIG" in st.secrets:
                try:
                    config_data = st.secrets["OCR_CONFIG"]
                    # 如果 secrets 中是字符串（JSON），则解析
                    if isinstance(config_data, str):
                        config_data = json.loads(config_data)
                    self.apis = {name: OCRAPIConfig(**data) for name, data in config_data.get('apis', {}).items()}
                    return
                except Exception as e:
                    st.warning(f"从 Secrets 加载 OCR 配置失败，尝试加载本地文件: {e}")

            # 2. 然后尝试加载本地文件
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self.apis = {name: OCRAPIConfig(**data) for name, data in config_data.get('apis', {}).items()}
            else:
                self.apis = self._get_default_configs()
                self.save_config()
        except Exception as e:
            st.error(f"加载OCR配置失败: {e}")
            self.apis = self._get_default_configs()
    
    def save_config(self):
        """保存配置"""
        try:
            config_data = {
                'apis': {
                    name: {
                        'name': api.name,
                        'api_key': api.api_key,
                        'endpoint': api.endpoint,
                        'cost_per_request': api.cost_per_request,
                        'enabled': api.enabled,
                        'description': api.description
                    }
                    for name, api in self.apis.items()
                }
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"保存OCR配置失败: {e}")
    
    def load_usage(self):
        """加载使用统计"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r', encoding='utf-8') as f:
                    self.usage_data = json.load(f)
            else:
                self.usage_data = {
                    'daily_usage': {},
                    'monthly_usage': {},
                    'total_usage': 0.0,
                    'request_count': 0
                }
                self.save_usage()
        except Exception as e:
            st.error(f"加载使用统计失败: {e}")
            self.usage_data = {
                'daily_usage': {},
                'monthly_usage': {},
                'total_usage': 0.0,
                'request_count': 0
            }
    
    def save_usage(self):
        """保存使用统计"""
        try:
            with open(self.usage_file, 'w', encoding='utf-8') as f:
                json.dump(self.usage_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"保存使用统计失败: {e}")
    
    def _get_default_configs(self) -> Dict[str, OCRAPIConfig]:
        """获取默认配置"""
        return {
            'google_vision': OCRAPIConfig(
                name='Google Vision',
                api_key='',
                cost_per_request=0.0015,
                description='Google Cloud Vision API，高精度OCR，$1.5/1000次'
            ),
            'azure_ocr': OCRAPIConfig(
                name='Azure Computer Vision',
                api_key='',
                endpoint='https://<region>.api.cognitive.microsoft.com/vision/v3.2/ocr',
                cost_per_request=0.001,
                description='Azure Computer Vision，稳定可靠，$1/1000次'
            ),
            'baidu_ocr': OCRAPIConfig(
                name='百度OCR',
                api_key='',
                endpoint='https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic',
                cost_per_request=0.0001,
                description='百度智能云OCR，中文优化，¥0.5/1000次'
            )
        }
    
    def get_enabled_apis(self) -> Dict[str, OCRAPIConfig]:
        """获取启用的API"""
        return {name: api for name, api in self.apis.items() if api.enabled and api.api_key}
    
    def update_api_config(self, api_name: str, **kwargs):
        """更新API配置"""
        if api_name in self.apis:
            for key, value in kwargs.items():
                if hasattr(self.apis[api_name], key):
                    setattr(self.apis[api_name], key, value)
            self.save_config()
    
    def track_usage(self, api_name: str, cost: float):
        """追踪使用情况"""
        today = datetime.now().strftime('%Y-%m-%d')
        this_month = datetime.now().strftime('%Y-%m')
        
        # 更新日使用量
        if today not in self.usage_data['daily_usage']:
            self.usage_data['daily_usage'][today] = 0.0
        self.usage_data['daily_usage'][today] += cost
        
        # 更新月使用量
        if this_month not in self.usage_data['monthly_usage']:
            self.usage_data['monthly_usage'][this_month] = 0.0
        self.usage_data['monthly_usage'][this_month] += cost
        
        # 更新总使用量
        self.usage_data['total_usage'] += cost
        self.usage_data['request_count'] += 1
        
        self.save_usage()
    
    def get_usage_report(self) -> Dict:
        """获取使用报告"""
        today = datetime.now().strftime('%Y-%m-%d')
        this_month = datetime.now().strftime('%Y-%m')
        
        # 清理旧数据（保留30天）
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.usage_data['daily_usage'] = {
            date: cost for date, cost in self.usage_data['daily_usage'].items()
            if date >= cutoff_date
        }
        
        return {
            'today_cost': self.usage_data['daily_usage'].get(today, 0.0),
            'month_cost': self.usage_data['monthly_usage'].get(this_month, 0.0),
            'total_cost': self.usage_data['total_usage'],
            'request_count': self.usage_data['request_count'],
            'average_cost_per_request': (
                self.usage_data['total_usage'] / self.usage_data['request_count']
                if self.usage_data['request_count'] > 0 else 0
            )
        }
    
    def check_budget_limits(self, daily_limit: float, monthly_limit: float) -> Dict:
        """检查预算限制"""
        usage_report = self.get_usage_report()
        
        daily_remaining = daily_limit - usage_report['today_cost']
        monthly_remaining = monthly_limit - usage_report['month_cost']
        
        return {
            'daily_limit': daily_limit,
            'monthly_limit': monthly_limit,
            'daily_used': usage_report['today_cost'],
            'monthly_used': usage_report['month_cost'],
            'daily_remaining': max(0, daily_remaining),
            'monthly_remaining': max(0, monthly_remaining),
            'within_daily_limit': usage_report['today_cost'] <= daily_limit,
            'within_monthly_limit': usage_report['month_cost'] <= monthly_limit,
            'can_use_api': daily_remaining > 0 and monthly_remaining > 0
        }
    
    def render_config_ui(self):
        """渲染配置界面"""
        st.subheader("🔑 OCR API 配置")
        
        # 预算设置
        st.write("**💰 预算控制**")
        col1, col2 = st.columns(2)
        with col1:
            daily_limit = st.number_input(
                "日预算限制 ($)",
                min_value=0.0,
                value=0.10,
                step=0.01,
                help="每日API调用最大预算"
            )
        with col2:
            monthly_limit = st.number_input(
                "月预算限制 ($)",
                min_value=0.0,
                value=3.00,
                step=0.10,
                help="每月API调用最大预算"
            )
        
        # 使用报告
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
        
        # 预算状态
        if not budget_status['can_use_api']:
            st.error("⚠️ 已达到预算限制，API调用将被暂停")
        else:
            st.success(f"✅ 预算充足：日剩余 ${budget_status['daily_remaining']:.4f}")
        
        # API配置
        st.write("**🔧 API 配置**")
        for api_name, api_config in self.apis.items():
            with st.expander(f"{api_config.name} 配置"):
                enabled = st.checkbox(
                    "启用此API",
                    value=api_config.enabled,
                    key=f"{api_name}_enabled"
                )
                
                api_key = st.text_input(
                    "API密钥",
                    value=api_config.api_key,
                    type="password",
                    key=f"{api_name}_api_key",
                    help="输入您的API密钥"
                )
                
                if api_name in ['azure_ocr']:
                    endpoint = st.text_input(
                        "API端点",
                        value=api_config.endpoint,
                        key=f"{api_name}_endpoint",
                        help="Azure服务的端点URL"
                    )
                    self.update_api_config(api_name, enabled=enabled, api_key=api_key, endpoint=endpoint)
                else:
                    self.update_api_config(api_name, enabled=enabled, api_key=api_key)
                
                st.info(f"💡 {api_config.description}")
                st.write(f"💰 每次调用成本: ${api_config.cost_per_request:.4f}")
        
        # 保存配置按钮
        if st.button("💾 保存配置", type="primary"):
            self.save_config()
            st.success("配置已保存！")
            st.rerun()

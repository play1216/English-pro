import os
import json
import streamlit as st
from datetime import datetime

class SubscriptionManager:
    def __init__(self, data_path=None, price=5.9, monthly_limit=500):
        self.price = price
        self.monthly_limit = monthly_limit
        self.data_path = data_path or os.path.join(os.getcwd(), "grading_usage.json")
        self._data = {"subscribed": False, "usage": {}}
        self._load()

    def _month_key(self):
        return datetime.now().strftime("%Y-%m")

    def _load(self):
        """加载订阅数据"""
        try:
            # 1. 首先尝试从 st.secrets 读取（针对部署环境，可设置默认会员）
            if "SUBSCRIPTION_DATA" in st.secrets:
                try:
                    self._data = dict(st.secrets["SUBSCRIPTION_DATA"])
                    return
                except Exception:
                    pass

            # 2. 尝试加载本地文件
            if os.path.exists(self.data_path):
                with open(self.data_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            else:
                self._data = {"subscribed": False, "usage": {}}
                self._save()
        except Exception:
            self._data = {"subscribed": False, "usage": {}}
            self._save()

    def _save(self):
        try:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def is_subscribed(self):
        return bool(self._data.get("subscribed", False))

    def subscribe(self):
        self._data["subscribed"] = True
        self._save()

    def unsubscribe(self):
        self._data["subscribed"] = False
        self._save()

    def current_usage(self):
        key = self._month_key()
        return int(self._data.get("usage", {}).get(key, 0))

    def remaining(self):
        return max(self.monthly_limit - self.current_usage(), 0)

    def can_grade(self):
        return self.is_subscribed() and self.current_usage() < self.monthly_limit

    def increment(self, n=1):
        key = self._month_key()
        usage = self._data.setdefault("usage", {})
        usage[key] = int(usage.get(key, 0)) + int(n)
        self._save()

    def get_price(self):
        return self.price

    def get_limit(self):
        return self.monthly_limit

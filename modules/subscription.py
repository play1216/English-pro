import json
import os
from datetime import datetime, timedelta

import streamlit as st


class SubscriptionManager:
    def __init__(
        self,
        data_path=None,
        price=5.9,
        monthly_limit=500,
        user_id=None,
        free_limit=3,
        membership_days=31,
    ):
        self.price = price
        self.monthly_limit = monthly_limit
        self.user_id = user_id
        self.free_limit = free_limit
        self.membership_days = membership_days
        self.data_path = data_path or os.path.join(os.getcwd(), "grading_usage.json")
        self._data = {
            "subscribed": False,
            "usage": {},
            "users": {},
            "order_claims": {},
        }
        self._load()

    def _month_key(self):
        return datetime.now().strftime("%Y-%m")

    def _default_user_record(self):
        return {
            "subscribed": False,
            "usage": {},
            "free_used": 0,
            "claimed_orders": [],
            "last_payment": None,
            "activated_at": None,
            "expires_at": None,
        }

    def _parse_dt(self, value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    def _now(self):
        return datetime.now()

    def _is_record_subscribed(self, record):
        if not record.get("subscribed", False):
            return False
        expires_at = self._parse_dt(record.get("expires_at"))
        if not expires_at:
            # 兼容旧数据：没有 expires_at 时保持旧行为
            return True
        return expires_at > self._now()

    def _load(self):
        try:
            if "SUBSCRIPTION_DATA" in st.secrets:
                try:
                    self._data = dict(st.secrets["SUBSCRIPTION_DATA"])
                    self._ensure_data_shape()
                    return
                except Exception:
                    pass

            if os.path.exists(self.data_path):
                with open(self.data_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            else:
                self._save()

            self._ensure_data_shape()
        except Exception:
            self._data = {
                "subscribed": False,
                "usage": {},
                "users": {},
                "order_claims": {},
            }
            self._save()

    def _save(self):
        try:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _ensure_data_shape(self):
        default_data = {
            "subscribed": bool(self._data.get("subscribed", False)),
            "usage": dict(self._data.get("usage", {})),
            "users": dict(self._data.get("users", {})),
            "order_claims": dict(self._data.get("order_claims", {})),
        }
        self._data = default_data

        for user_id, record in list(self._data["users"].items()):
            normalized_record = self._default_user_record()
            normalized_record.update(record or {})
            normalized_record["usage"] = dict(normalized_record.get("usage", {}))
            normalized_record["claimed_orders"] = list(normalized_record.get("claimed_orders", []))
            self._data["users"][user_id] = normalized_record

    def _target_record(self):
        self._ensure_data_shape()
        if not self.user_id:
            return self._data

        users = self._data.setdefault("users", {})
        if self.user_id not in users:
            seed_record = self._default_user_record()
            if not users and (self._data.get("subscribed") or self._data.get("usage")):
                seed_record["subscribed"] = bool(self._data.get("subscribed", False))
                seed_record["usage"] = dict(self._data.get("usage", {}))
            users[self.user_id] = seed_record
            self._save()
        return users[self.user_id]

    def is_subscribed(self):
        return self._is_record_subscribed(self._target_record())

    def subscribe(self):
        record = self._target_record()
        record["subscribed"] = True
        now = self._now()
        record["activated_at"] = now.isoformat(timespec="seconds")
        record["expires_at"] = (now + timedelta(days=self.membership_days)).isoformat(timespec="seconds")
        self._save()

    def unsubscribe(self):
        self._target_record()["subscribed"] = False
        self._save()

    def current_usage(self):
        key = self._month_key()
        return int(self._target_record().get("usage", {}).get(key, 0))

    def remaining(self):
        return max(self.monthly_limit - self.current_usage(), 0)

    def free_used(self):
        return int(self._target_record().get("free_used", 0))

    def free_remaining(self):
        return max(self.free_limit - self.free_used(), 0)

    def has_free_quota(self):
        return self.free_remaining() > 0

    def can_grade(self):
        if self.is_subscribed():
            return self.current_usage() < self.monthly_limit
        return self.has_free_quota()

    def increment(self, n=1):
        record = self._target_record()
        if self.is_subscribed():
            key = self._month_key()
            usage = record.setdefault("usage", {})
            usage[key] = int(usage.get(key, 0)) + int(n)
        else:
            record["free_used"] = min(self.free_limit, self.free_used() + int(n))
        self._save()

    def activate_membership(self, order_no=None, payment_info=None):
        record = self._target_record()
        if order_no:
            owner = self._data.setdefault("order_claims", {}).get(order_no)
            if owner and owner != self.user_id:
                return False, "这个爱发电订单已经绑定到其他账号。"

            self._data["order_claims"][order_no] = self.user_id or "global"
            claimed_orders = record.setdefault("claimed_orders", [])
            if order_no not in claimed_orders:
                claimed_orders.append(order_no)

        now = self._now()
        current_expire = self._parse_dt(record.get("expires_at"))
        start_time = current_expire if current_expire and current_expire > now else now
        new_expire = start_time + timedelta(days=self.membership_days)

        record["subscribed"] = True
        record["activated_at"] = now.isoformat(timespec="seconds")
        record["expires_at"] = new_expire.isoformat(timespec="seconds")
        if payment_info:
            record["last_payment"] = payment_info
        self._save()
        return True, f"会员已开通，有效期至 {new_expire.strftime('%Y-%m-%d %H:%M')}。"

    def get_price(self):
        return self.price

    def get_limit(self):
        return self.monthly_limit

    def get_free_limit(self):
        return self.free_limit

    def get_last_payment(self):
        return self._target_record().get("last_payment")

    def get_membership_status(self):
        record = self._target_record()
        expires_at = self._parse_dt(record.get("expires_at"))
        active = self._is_record_subscribed(record)
        return {
            "is_active": active,
            "activated_at": record.get("activated_at"),
            "expires_at": record.get("expires_at"),
            "is_expiring_soon": bool(active and expires_at and (expires_at - self._now()) <= timedelta(days=3)),
            "days_left": (
                max((expires_at - self._now()).days, 0)
                if active and expires_at
                else 0
            ),
        }

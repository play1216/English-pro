import hashlib
import json
import time
from urllib import error, request

import streamlit as st


class AfdianClient:
    query_order_url = "https://afdian.net/api/open/query-order"

    def __init__(self, user_id=None, token=None, payment_url=None, plan_title=None):
        self.user_id = user_id or st.secrets.get("AFDIAN_USER_ID", "")
        self.token = token or st.secrets.get("AFDIAN_TOKEN", "")
        self.payment_url = payment_url or st.secrets.get("AFDIAN_PAYMENT_URL", "")
        self.plan_title = plan_title or st.secrets.get("AFDIAN_PLAN_TITLE", "")
        self.plan_keywords = self._parse_plan_keywords(st.secrets.get("AFDIAN_PLAN_KEYWORDS", ""))

    def is_configured(self):
        return bool(self.user_id and self.token)

    def _parse_plan_keywords(self, raw_keywords):
        if isinstance(raw_keywords, (list, tuple)):
            return [str(item).strip().lower() for item in raw_keywords if str(item).strip()]
        if not raw_keywords:
            return []
        return [item.strip().lower() for item in str(raw_keywords).split(",") if item.strip()]

    def _build_payload(self, params, ts=None):
        ts = ts or int(time.time())
        params_json = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
        raw = f"{self.token}params{params_json}ts{ts}user_id{self.user_id}"
        sign = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return {
            "user_id": self.user_id,
            "params": params_json,
            "ts": ts,
            "sign": sign,
        }

    def _post(self, payload):
        req = request.Request(
            self.query_order_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    def verify_order(self, order_no):
        order_no = (order_no or "").strip()
        if not order_no:
            return False, "请输入爱发电订单号。", None
        if not self.is_configured():
            return False, "爱发电参数未配置，请先在 secrets 中配置 AFDIAN_USER_ID 和 AFDIAN_TOKEN。", None

        payload = self._build_payload({"out_trade_no": order_no})
        try:
            response = self._post(payload)
        except error.URLError as exc:
            return False, f"连接爱发电失败：{exc.reason}", None
        except Exception as exc:
            return False, f"查询爱发电订单失败：{exc}", None

        if int(response.get("ec", -1)) != 200:
            return False, response.get("em", "爱发电返回了错误响应。"), None

        data = response.get("data") or {}
        orders = data.get("list") or []
        if not orders:
            return False, "没有查到这个订单，请确认订单号是否正确。", None

        order = None
        for item in orders:
            if str(item.get("out_trade_no", "")).strip() == order_no:
                order = item
                break
        if not order:
            return False, "没有找到匹配的订单号，请确认是否为当前账号下的订单。", None
        if int(order.get("status", 0)) != 2:
            return False, "订单尚未支付完成，请支付成功后再验证。", None

        plan_name = (order.get("show_item") or "").strip()
        normalized_plan_name = plan_name.lower()
        if self.plan_title and plan_name != self.plan_title:
            return False, f"订单项目不是当前会员套餐：{plan_name or '未识别'}。", None
        if self.plan_keywords:
            if not any(keyword in normalized_plan_name for keyword in self.plan_keywords):
                return False, f"订单项目未命中允许的套餐关键词：{plan_name or '未识别'}。", None

        payment_info = {
            "order_no": order.get("out_trade_no", order_no),
            "trade_no": order.get("trade_no", ""),
            "amount": order.get("show_amount", ""),
            "plan_name": order.get("show_item", ""),
            "paid_at": order.get("create_time", ""),
            "status": order.get("status", 0),
        }
        return True, "爱发电订单验证成功。", payment_info

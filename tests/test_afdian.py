import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.afdian import AfdianClient


class TestAfdianClient(unittest.TestCase):
    @patch("streamlit.secrets", {})
    def test_build_payload_generates_expected_sign(self):
        client = AfdianClient(user_id="user_x", token="token_y")
        payload = client._build_payload({"out_trade_no": "20260001"}, ts=1700000000)

        self.assertEqual(payload["user_id"], "user_x")
        self.assertEqual(payload["ts"], 1700000000)
        self.assertEqual(payload["params"], '{"out_trade_no":"20260001"}')
        self.assertEqual(payload["sign"], "040dbb8b110f48c7a621c06501f8ee20")

    @patch("streamlit.secrets", {})
    def test_verify_order_success(self):
        client = AfdianClient(
            user_id="user_x",
            token="token_y",
            plan_title="作文会员",
        )
        fake_response = {
            "ec": 200,
            "data": {
                "list": [
                    {
                        "out_trade_no": "20260001",
                        "status": 2,
                        "trade_no": "TRADE001",
                        "show_amount": "9.90",
                        "show_item": "作文会员",
                        "create_time": "2026-04-26 12:00:00",
                    }
                ]
            },
        }

        with patch.object(client, "_post", return_value=fake_response):
            success, message, payment_info = client.verify_order("20260001")

        self.assertTrue(success)
        self.assertIn("验证成功", message)
        self.assertEqual(payment_info["order_no"], "20260001")
        self.assertEqual(payment_info["status"], 2)

    @patch("streamlit.secrets", {})
    def test_verify_order_finds_matching_order_in_list(self):
        client = AfdianClient(user_id="user_x", token="token_y")
        fake_response = {
            "ec": 200,
            "data": {
                "list": [
                    {"out_trade_no": "20260000", "status": 2},
                    {"out_trade_no": "20260001", "status": 2, "show_item": "作文会员"},
                ]
            },
        }

        with patch.object(client, "_post", return_value=fake_response):
            success, _, payment_info = client.verify_order("20260001")

        self.assertTrue(success)
        self.assertEqual(payment_info["order_no"], "20260001")

    @patch("streamlit.secrets", {"AFDIAN_PLAN_KEYWORDS": "作文,会员"})
    def test_verify_order_checks_plan_keywords(self):
        client = AfdianClient(user_id="user_x", token="token_y")
        fake_response = {
            "ec": 200,
            "data": {
                "list": [
                    {"out_trade_no": "20260001", "status": 2, "show_item": "周边赞助"},
                ]
            },
        }
        with patch.object(client, "_post", return_value=fake_response):
            success, message, _ = client.verify_order("20260001")
        self.assertFalse(success)
        self.assertIn("套餐关键词", message)


if __name__ == "__main__":
    unittest.main()

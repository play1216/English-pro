import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.auth import AuthManager
from modules.subscription import SubscriptionManager


class TestAuthManager(unittest.TestCase):
    def test_register_and_authenticate_user(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "users.json")
            auth_manager = AuthManager(data_path=data_path)

            success, _ = auth_manager.register_user("TeacherA", "secret123")
            self.assertTrue(success)
            self.assertTrue(auth_manager.authenticate("TeacherA", "secret123"))
            self.assertFalse(auth_manager.authenticate("TeacherA", "wrongpass"))

    def test_register_duplicate_username(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "users.json")
            auth_manager = AuthManager(data_path=data_path)

            first_success, _ = auth_manager.register_user("student01", "secret123")
            second_success, message = auth_manager.register_user("Student01", "secret123")

            self.assertTrue(first_success)
            self.assertFalse(second_success)
            self.assertIn("用户名已存在", message)


class TestSubscriptionManager(unittest.TestCase):
    @patch("streamlit.secrets", {})
    def test_tracks_usage_per_user(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "grading_usage.json")
            alice = SubscriptionManager(data_path=data_path, user_id="alice")
            bob = SubscriptionManager(data_path=data_path, user_id="bob")

            alice.increment(2)
            bob.increment(1)
            alice.subscribe()
            alice.increment(2)

            self.assertTrue(alice.is_subscribed())
            self.assertFalse(bob.is_subscribed())
            self.assertEqual(alice.current_usage(), 2)
            self.assertEqual(bob.current_usage(), 1)
            self.assertEqual(alice.free_used(), 2)
            self.assertEqual(bob.free_remaining(), 2)

    @patch("streamlit.secrets", {})
    def test_activate_membership_claims_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "grading_usage.json")
            alice = SubscriptionManager(data_path=data_path, user_id="alice")
            bob = SubscriptionManager(data_path=data_path, user_id="bob")

            success, _ = alice.activate_membership("A123", {"amount": "9.90"})
            second_success, message = bob.activate_membership("A123", {"amount": "9.90"})

            self.assertTrue(success)
            self.assertFalse(second_success)
            self.assertIn("已经绑定", message)

    @patch("streamlit.secrets", {})
    def test_membership_has_expire_time_and_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "grading_usage.json")
            user = SubscriptionManager(data_path=data_path, user_id="alice", membership_days=31)
            success, _ = user.activate_membership("A100", {"amount": "9.90"})
            status = user.get_membership_status()

            self.assertTrue(success)
            self.assertTrue(status["is_active"])
            self.assertTrue(status["expires_at"])

    @patch("streamlit.secrets", {})
    def test_renew_membership_extends_expire_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "grading_usage.json")
            user = SubscriptionManager(data_path=data_path, user_id="alice", membership_days=31)
            user.activate_membership("A100", {"amount": "9.90"})
            first_expire = user.get_membership_status()["expires_at"]
            user.activate_membership("A101", {"amount": "9.90"})
            second_expire = user.get_membership_status()["expires_at"]

            self.assertTrue(second_expire > first_expire)

    @patch("streamlit.secrets", {})
    def test_legacy_data_is_still_readable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            month_key = datetime.now().strftime("%Y-%m")
            data_path = os.path.join(temp_dir, "grading_usage.json")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(f'{{"subscribed": true, "usage": {{"{month_key}": 3}}}}')

            subscription = SubscriptionManager(data_path=data_path)
            self.assertTrue(subscription.is_subscribed())
            self.assertEqual(subscription._data["usage"][month_key], 3)
            self.assertIn("users", subscription._data)

    @patch("streamlit.secrets", {})
    def test_legacy_single_user_data_migrates_to_first_login_user(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            month_key = datetime.now().strftime("%Y-%m")
            data_path = os.path.join(temp_dir, "grading_usage.json")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(f'{{"subscribed": true, "usage": {{"{month_key}": 3}}}}')

            subscription = SubscriptionManager(data_path=data_path, user_id="teacher_a")
            self.assertTrue(subscription.is_subscribed())
            self.assertEqual(subscription.current_usage(), 3)


if __name__ == "__main__":
    unittest.main()

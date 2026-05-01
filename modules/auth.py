import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime

import streamlit as st


class AuthManager:
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(os.getcwd(), "users.json")
        self._token_query_key = "auth_token"
        self._remember_days = int(st.secrets.get("AUTH_REMEMBER_DAYS", 30))
        secret_from_config = st.secrets.get("AUTH_TOKEN_SECRET", "")
        fallback_secret = f"local-dev-{os.path.abspath(self.data_path)}"
        self._token_secret = str(secret_from_config or fallback_secret)
        self._data = {"users": {}}
        self._load()
        self._restore_login_from_token()

    def _load(self):
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "users" in data:
                        self._data = data
                    else:
                        self._data = {"users": {}}
            else:
                self._save()
        except Exception:
            self._data = {"users": {}}
            self._save()

    def _save(self):
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def _normalize_username(self, username):
        return username.strip().lower()

    def _urlsafe_b64encode(self, value):
        return base64.urlsafe_b64encode(value.encode("utf-8")).decode("utf-8").rstrip("=")

    def _urlsafe_b64decode(self, value):
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode((value + padding).encode("utf-8")).decode("utf-8")

    def _token_sign(self, payload_b64):
        return hmac.new(
            self._token_secret.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _build_token(self, username):
        expires_at = int(time.time()) + self._remember_days * 24 * 60 * 60
        payload = f"{self._normalize_username(username)}|{expires_at}"
        payload_b64 = self._urlsafe_b64encode(payload)
        signature = self._token_sign(payload_b64)
        return f"{payload_b64}.{signature}"

    def _parse_token(self, token):
        if not token or "." not in token:
            return None

        payload_b64, signature = token.rsplit(".", 1)
        expected_signature = self._token_sign(payload_b64)
        if not hmac.compare_digest(signature, expected_signature):
            return None

        try:
            payload = self._urlsafe_b64decode(payload_b64)
            username, expires_at_raw = payload.split("|", 1)
            expires_at = int(expires_at_raw)
        except Exception:
            return None

        if expires_at < int(time.time()):
            return None

        normalized_username = self._normalize_username(username)
        if normalized_username not in self._data["users"]:
            return None

        return normalized_username

    def _get_query_param(self, key):
        try:
            if hasattr(st, "query_params"):
                return st.query_params.get(key)
            params = st.experimental_get_query_params()
            value = params.get(key)
            if isinstance(value, list):
                return value[0] if value else None
            return value
        except Exception:
            return None

    def _set_query_param(self, key, value):
        try:
            if hasattr(st, "query_params"):
                st.query_params[key] = value
            else:
                params = st.experimental_get_query_params()
                params[key] = value
                st.experimental_set_query_params(**params)
        except Exception:
            pass

    def _remove_query_param(self, key):
        try:
            if hasattr(st, "query_params"):
                if key in st.query_params:
                    del st.query_params[key]
            else:
                params = st.experimental_get_query_params()
                if key in params:
                    del params[key]
                    st.experimental_set_query_params(**params)
        except Exception:
            pass

    def _restore_login_from_token(self):
        if st.session_state.get("current_user"):
            return

        token = self._get_query_param(self._token_query_key)
        token_user = self._parse_token(token)
        if token_user:
            st.session_state["current_user"] = token_user
        elif token:
            self._remove_query_param(self._token_query_key)

    def _hash_password(self, password, salt=None):
        salt = salt or secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,
        ).hex()
        return salt, password_hash

    def _verify_password(self, password, salt, password_hash):
        _, computed_hash = self._hash_password(password, salt=salt)
        return hmac.compare_digest(computed_hash, password_hash)

    def register_user(self, username, password):
        normalized_username = self._normalize_username(username)
        if len(normalized_username) < 3:
            return False, "用户名至少需要 3 个字符。"
        if len(password) < 6:
            return False, "密码至少需要 6 个字符。"

        if normalized_username in self._data["users"]:
            return False, "用户名已存在，请直接登录。"

        salt, password_hash = self._hash_password(password)
        self._data["users"][normalized_username] = {
            "username": normalized_username,
            "password_hash": password_hash,
            "salt": salt,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save()
        return True, "注册成功，请登录。"

    def authenticate(self, username, password):
        normalized_username = self._normalize_username(username)
        user = self._data["users"].get(normalized_username)
        if not user:
            return False
        return self._verify_password(password, user["salt"], user["password_hash"])

    def login(self, username):
        normalized_username = self._normalize_username(username)
        st.session_state["current_user"] = normalized_username
        self._set_query_param(self._token_query_key, self._build_token(normalized_username))

    def logout(self):
        for key in [
            "current_user",
            "ocr_result",
            "ocr_details",
            "essay_prompt_result",
            "essay_prompt_text",
            "essay_body_text",
        ]:
            st.session_state.pop(key, None)
        self._remove_query_param(self._token_query_key)

    def get_current_user(self):
        return st.session_state.get("current_user")

    def is_logged_in(self):
        return bool(self.get_current_user())


def render_auth_gate(auth_manager):
    st.title("登录 AI 作文批改系统")
    st.caption("先登录，再进入批改页面。")

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            login_submitted = st.form_submit_button("登录", use_container_width=True)

        if login_submitted:
            if auth_manager.authenticate(username, password):
                auth_manager.login(username)
                st.success("登录成功，正在进入系统。")
                st.rerun()
            else:
                st.error("用户名或密码不正确。")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("新用户名")
            new_password = st.text_input("新密码", type="password")
            confirm_password = st.text_input("确认密码", type="password")
            register_submitted = st.form_submit_button("注册", use_container_width=True)

        if register_submitted:
            if new_password != confirm_password:
                st.error("两次输入的密码不一致。")
            else:
                success, message = auth_manager.register_user(new_username, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.stop()

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.ocr_config import OCRConfigManager


class TestOCRConfigManager(unittest.TestCase):
    @patch("streamlit.secrets", {})
    def test_legacy_config_migrates_to_local_private_file(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                with open("ocr_config.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "apis": {
                                "google_vision": {
                                    "name": "Google Vision",
                                    "api_key": "legacy-key",
                                    "endpoint": "",
                                    "cost_per_request": 0.0015,
                                    "enabled": True,
                                    "description": "legacy",
                                }
                            }
                        },
                        f,
                        ensure_ascii=False,
                    )

                manager = OCRConfigManager()
                self.assertEqual(manager.config_source, "local")
                self.assertTrue(os.path.exists(os.path.join(".streamlit", "ocr_config.local.json")))
            finally:
                os.chdir(original_cwd)

    @patch(
        "streamlit.secrets",
        {
            "OCR_CONFIG": {
                "apis": {
                    "google_vision": {
                        "name": "Google Vision",
                        "api_key": "secret-key",
                        "endpoint": "",
                        "cost_per_request": 0.0015,
                        "enabled": True,
                        "description": "from secrets",
                    }
                }
            }
        },
    )
    def test_secrets_config_does_not_create_local_key_file(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                manager = OCRConfigManager()
                self.assertEqual(manager.config_source, "secrets")
                self.assertFalse(os.path.exists(os.path.join(".streamlit", "ocr_config.local.json")))
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

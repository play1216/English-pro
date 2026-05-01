import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

# 将项目根目录添加到 sys.path 以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.grading import EssayGrader

class TestEssayGrader(unittest.TestCase):
    @patch('streamlit.secrets', {'DEEPSEEK_API_KEY': 'test_key'})
    @patch('openai.OpenAI')
    def setUp(self, mock_openai):
        self.mock_client = MagicMock()
        mock_openai.return_value = self.mock_client
        self.grader = EssayGrader()

    def test_get_system_prompt_applied(self):
        prompt = self.grader._get_system_prompt('applied', '假设你是李华，请给外教写一封邀请信。')
        self.assertIn('应用文写作', prompt)
        self.assertIn('15分', prompt)
        self.assertIn('邀请信', prompt)

    def test_get_system_prompt_continuation(self):
        prompt = self.grader._get_system_prompt('continuation', '阅读下面材料，根据所给段落开头续写两段。')
        self.assertIn('读后续写', prompt)
        self.assertIn('25分', prompt)
        self.assertIn('续写两段', prompt)

    def test_build_user_message(self):
        message = self.grader._build_user_message(
            "I went home happily.",
            "假设你是李华，请写一则通知。"
        )
        self.assertIn('作文题目/任务材料', message)
        self.assertIn('写一则通知', message)
        self.assertIn('I went home happily.', message)

    @patch('streamlit.secrets', {'DEEPSEEK_API_KEY': 'test_key'})
    def test_grade_success(self):
        # 模拟 API 返回的 JSON 结果
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "meta": {"type": "applied", "max_score": 15},
            "score": {"total": 12, "rank": "第四档", "radar": {"grammar": 8, "vocabulary": 8, "logic": 8, "structure": 8}},
            "task_focus": {"summary": "邀请信", "task_completion": "基本完成任务", "missed_points": []},
            "comment": "很好",
            "suggestions": []
        })
        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.grader.grade("Test essay content", "applied", "假设你是李华，请写一封邀请信。")
        
        self.assertEqual(result['score']['total'], 12)
        self.assertEqual(result['meta']['type'], 'applied')
        self.assertEqual(result['task_focus']['summary'], '邀请信')

    @patch('streamlit.secrets', {'DEEPSEEK_API_KEY': 'test_key'})
    def test_grade_api_error(self):
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        result = self.grader.grade("Test essay content", "applied", "假设你是李华，请写一封邀请信。")
        self.assertIn('error', result)
        self.assertIn('API 调用失败', result['error'])

if __name__ == '__main__':
    unittest.main()

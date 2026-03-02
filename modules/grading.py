import json
from openai import OpenAI
import streamlit as st

class EssayGrader:
    def __init__(self):
        """
        初始化 DeepSeek 客户端。
        优先从 Streamlit Secrets 读取 Key，保证安全。
        """
        self.api_key = None
        self.client = None
        
        # 安全读取 API Key
        if "DEEPSEEK_API_KEY" in st.secrets:
            self.api_key = st.secrets["DEEPSEEK_API_KEY"]
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com"
                )
            except Exception as e:
                print(f"Client Init Error: {e}")
        else:
            # 如果本地调试没有 secrets，可以在这里处理，或者交给 UI 层报错
            pass

    def _get_system_prompt(self, essay_type):
        """
        私有方法：生成符合高考阅卷标准的 System Prompt
        集成之前的 prompts.py 核心逻辑
        """
        if essay_type == 'applied':
            max_score = 15
            task_desc = "应用文写作（书信/通知/演讲稿等）"
            # 15分制的分档标准
            bands = """
            【第五档 (13-15分)】: 覆盖所有要点，应用了较多的语法结构和词汇，语言自然流畅，完全达到写作目的。
            【第四档 (10-12分)】: 覆盖所有要点，漏掉1-2个次重点，语法结构和词汇满足任务要求，有少量错误但不影响理解。
            【第三档 (7-9分)】: 漏掉一些要点，词汇语法结构单调，错误较多，影响理解。
            【第二档 (4-6分)】: 漏掉较多要点，语法错误多，很难理解。
            【第一档 (1-3分)】: 只有几个单词，无法传达信息。
            """
        else:
            max_score = 25
            task_desc = "读后续写 (Continuation Writing)"
            # 25分制的分档标准
            bands = """
            【第五档 (21-25分)】: 与原文衔接自然，内容逻辑丰富，使用了丰富的高级词汇和句式，几乎无语法错误。
            【第四档 (16-20分)】: 与原文衔接较好，故事逻辑通顺，有少量语法错误但不影响理解。
            【第三档 (11-15分)】: 基本能接续故事，但逻辑有跳跃，词汇单调，错误较多。
            【第二档 (6-10分)】: 故事逻辑混乱，大量语法错误。
            【第一档 (0-5分)】: 偏题或只写了很少内容。
            """

        # 返回核心 Prompt (逻辑锁 + 格式锁)
        return f"""
        你是一位严厉的中国高考英语阅卷组组长。
        当前任务：批改一篇{task_desc}。
        满分限制：**{max_score}分**。
        
        【评分流程 - 必须严格遵守】
        1. **先定档**：阅读文章，对照以下档次标准，决定文章属于哪个档次。
           {bands}
        2. **后打分**：在确定档次后，给出一个具体的**总分 (Total Score)**。
           - 严禁出现超过 {max_score} 分的情况！
           - 严禁使用百分制！
        3. **最后生成维度分**：根据总分，反推并在 JSON 中生成四个维度分（语法/词汇/逻辑/结构），这四个分数只是为了展示，不需要严格相加等于总分。

        【批改要求】
        1. **英文改错**：不要翻译原句！直接给出 Original (英文) -> Improved (英文) -> Reason (中文)。
        2. **中文点评**：语气要像高中老师，指出学生在“高级句型”、“连接词”、“书面语”方面的优缺点。

        【输出格式】
        必须返回严格的纯 JSON 格式：
        {{
            "meta": {{
                "type": "{essay_type}",
                "max_score": {max_score}
            }},
            "score": {{
                "total": (整数, 0-{max_score}之间),
                "rank": "(字符串, 例如: 第四档)",
                "radar": {{ 
                    "grammar": (0-10分), 
                    "vocabulary": (0-10分), 
                    "logic": (0-10分), 
                    "structure": (0-10分) 
                }}
            }},
            "comment": "(100字左右的中文总评)",
            "suggestions": [
                {{
                    "original": "(英文原句)",
                    "improved": "(优化后的英文高级句型)",
                    "reason": "(中文点评：例如'原句使用了run-on sentence，建议改为定语从句...')"
                }}
            ]
        }}
        """

    def grade(self, text, essay_type):
        """
        公开方法：执行批改逻辑
        """
        # 1. 检查客户端是否初始化成功
        if not self.client:
            return {"error": "DeepSeek API Key 未配置，请检查 .streamlit/secrets.toml"}

        # 2. 获取 Prompt
        system_prompt = self._get_system_prompt(essay_type)

        try:
            # 3. 调用 API
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                # 温度设为 0.1 或 0.2，让 AI 变得极其理性，严格遵守分数限制
                temperature=0.2 
            )
            
            # 4. 解析结果
            content = response.choices[0].message.content
            # 处理可能的 Markdown 代码块包裹 (虽然 json_object 模式通常不会有，但为了保险)
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            
            return json.loads(content)
            
        except json.JSONDecodeError:
            return {"error": "AI 返回的数据格式有误，请重试。"}
        except Exception as e:
            return {"error": f"API 调用失败: {str(e)}"}
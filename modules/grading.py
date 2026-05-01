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

    def _get_system_prompt(self, essay_type, essay_prompt=""):
        """
        私有方法：生成符合高考阅卷标准的 System Prompt
        集成之前的 prompts.py 核心逻辑
        """
        if essay_type == 'applied':
            max_score = 15
            task_desc = "应用文写作（书信/通知/演讲稿等）"
            task_focus = """
            【应用文专项要求】
            1. 必须先判断题目要求中的写作身份、对象、目的、要点和语气是否完成。
            2. 重点检查格式是否得体，信息点是否覆盖，语气是否符合真实交流场景。
            3. 评论时要明确指出是否跑题、漏点、语域不当、格式不规范。
            """
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
            task_focus = """
            【读后续写专项要求】
            1. 必须先判断续写是否紧扣所给材料、首句和人物关系。
            2. 重点检查情节推进、情感线、人物行为动机、前后呼应和结尾完成度。
            3. 评论时要明确指出是否与原文脱节、情节跳跃、人物失真或续写方向偏离。
            """
            # 25分制的分档标准
            bands = """
            【第五档 (21-25分)】: 与原文衔接自然，内容逻辑丰富，使用了丰富的高级词汇和句式，几乎无语法错误。
            【第四档 (16-20分)】: 与原文衔接较好，故事逻辑通顺，有少量语法错误但不影响理解。
            【第三档 (11-15分)】: 基本能接续故事，但逻辑有跳跃，词汇单调，错误较多。
            【第二档 (6-10分)】: 故事逻辑混乱，大量语法错误。
            【第一档 (0-5分)】: 偏题或只写了很少内容。
            """

        prompt_context = essay_prompt.strip() if essay_prompt.strip() else "用户没有提供作文题目或原文材料，请按通用标准批改，并在总评中提醒补充题目信息会更准确。"

        # 返回核心 Prompt (逻辑锁 + 格式锁)
        return f"""
        你是一位严厉的中国高考英语阅卷组组长。
        当前任务：批改一篇{task_desc}。
        满分限制：**{max_score}分**。

        【作文题目/任务材料】
        {prompt_context}
        
        【评分流程 - 必须严格遵守】
        1. **先定档**：阅读文章，对照以下档次标准，决定文章属于哪个档次。
           {bands}
        2. **后打分**：在确定档次后，给出一个具体的**总分 (Total Score)**。
           - 严禁出现超过 {max_score} 分的情况！
           - 严禁使用百分制！
        3. **最后生成维度分**：根据总分，反推并在 JSON 中生成四个维度分（语法/词汇/逻辑/结构），这四个分数只是为了展示，不需要严格相加等于总分。

        【批改要求】
        {task_focus}
        1. **英文改错**：不要翻译原句！直接给出 Original (英文) -> Improved (英文) -> Reason (中文)。
        2. **中文点评**：语气要像高中老师，指出学生在“切题性/任务完成度”、“高级句型”、“连接词”、“书面语”方面的优缺点。
        3. **题目关联**：如果用户提供了作文题目或材料，你必须把总评和建议明确关联到题目要求，不能只做泛泛点评。

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
            "task_focus": {{
                "summary": "(中文，一句话概括本题任务)",
                "task_completion": "(中文，评价是否切题、是否完成关键要求)",
                "missed_points": ["(中文，可为空)", "(中文，可为空)"]
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

    def _build_user_message(self, text, essay_prompt=""):
        prompt_context = essay_prompt.strip() if essay_prompt.strip() else "未提供"
        return f"""
作文题目/任务材料：
{prompt_context}

学生作文正文：
{text.strip()}
        """.strip()

    def grade(self, text, essay_type, essay_prompt=""):
        """
        公开方法：执行批改逻辑
        """
        # 1. 检查客户端是否初始化成功
        if not self.client:
            return {"error": "DeepSeek API Key 未配置，请检查 .streamlit/secrets.toml"}

        # 2. 获取 Prompt
        system_prompt = self._get_system_prompt(essay_type, essay_prompt)
        user_message = self._build_user_message(text, essay_prompt)

        try:
            # 3. 调用 API
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
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

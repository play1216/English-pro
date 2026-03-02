import streamlit as st
import numpy as np
import cv2
import logging
import os
from PIL import Image
import io
import re
from typing import List, Tuple, Dict
from difflib import get_close_matches
from collections import Counter
from modules.simple_logic_validator import SimpleLogicValidator

# ================= 1. 核心修复：强制锁定旧引擎 =================
# 在导入 paddle 之前，彻底关闭所有可能引起报错的新特性开关
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_executor_unittests'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'

try:
    import paddle
    # 强制禁用新执行引擎（针对 2.6.x 版本的双重保险）
    paddle.set_flags({'FLAGS_enable_pir_api': 0})
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    logging.info("PaddleOCR imported successfully")
except ImportError as e:
    PADDLE_AVAILABLE = False
    logging.error(f"PaddleOCR import error (ImportError): {e}")
except Exception as e:
    # 检查是否是已知的 torch/shm.dll 警告，这种情况下 PaddleOCR 仍然可用
    if "shm.dll" in str(e) or "torch" in str(e).lower():
        logging.warning(f"Known torch/shm warning detected, attempting to continue: {e}")
        try:
            # 尝试直接导入 PaddleOCR
            from paddleocr import PaddleOCR
            PADDLE_AVAILABLE = True
            logging.info("PaddleOCR imported successfully despite torch warning")
        except ImportError:
            PADDLE_AVAILABLE = False
            logging.error("PaddleOCR failed to import even after handling torch warning")
    else:
        PADDLE_AVAILABLE = False
        logging.error(f"PaddleOCR import error (Exception): {e}")

# 屏蔽 Paddle 内部无意义的警告
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
# 更严格地屏蔽 torch 相关警告
logging.getLogger("torch").setLevel(logging.ERROR)

class EnhancedOcrEngine:
    def __init__(self, lang='en', use_gpu=False, enable_mkldnn=False):
        self.is_ready = PADDLE_AVAILABLE
        self.lang = lang
        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        if not self.is_ready:
            st.warning("⚠️ 未检测到 paddleocr 库，请安装依赖。")
        
        # 初始化逻辑校验器
        self.logic_validator = SimpleLogicValidator()
        
        # 常用英语单词集合（用于语境分析）
        try:
            import english_words
            self.english_vocab = set(english_words.get_english_words_set(['web2']))
        except ImportError:
            # 如果没有 english_words 库，使用基础词汇
            self.english_vocab = {
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
                'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
                'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
                'one', 'all', 'would', 'there', 'their', 'what', 'so',
                'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
                'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
                'him', 'know', 'take', 'people', 'into', 'year', 'your',
                'good', 'some', 'could', 'them', 'see', 'other', 'than',
                'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think',
                'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
                'first', 'well', 'way', 'even', 'new', 'want', 'because',
                'any', 'these', 'give', 'day', 'most', 'us', 'is', 'water',
                'been', 'call', 'who', 'oil', 'sit', 'now', 'find', 'long',
                'down', 'day', 'did', 'get', 'has', 'him', 'his', 'how',
                'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
                'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too',
                'use'
            }
        
        # 作文常用词汇
        self.essay_vocab = {
            'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
            'although', 'because', 'since', 'while', 'whereas', 'despite',
            'important', 'significant', 'essential', 'crucial', 'vital',
            'believe', 'think', 'consider', 'suppose', 'maintain', 'argue',
            'should', 'must', 'ought', 'need', 'have to', 'will', 'would',
            'good', 'bad', 'better', 'worse', 'best', 'worst', 'great', 'terrible',
            'education', 'school', 'student', 'teacher', 'knowledge', 'learning',
            'society', 'culture', 'tradition', 'modern', 'development', 'progress',
            'technology', 'science', 'research', 'innovation', 'discovery',
            'environment', 'nature', 'pollution', 'climate', 'sustainable',
            'economy', 'business', 'market', 'trade', 'industry', 'commerce',
            'health', 'medicine', 'disease', 'treatment', 'prevention', 'fitness',
            'relationship', 'friendship', 'family', 'community', 'communication',
            'government', 'politics', 'democracy', 'freedom', 'justice', 'law'
        }
    
    def _remove_lines_and_noise(self, image):
        """去除横线和其他干扰"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 增强对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # 检测并去除水平线（作文纸横线）
            # 使用形态学操作检测水平线
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # 去除检测到的水平线
            without_horizontal = cv2.inpaint(enhanced, horizontal_lines, 3, cv2.INPAINT_TELEA)
            
            # 检测并去除垂直线
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(without_horizontal, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            cleaned = cv2.inpaint(without_horizontal, vertical_lines, 3, cv2.INPAINT_TELEA)
            
            return cleaned
        except Exception as e:
            logging.warning(f"去线处理失败，使用原图: {e}")
            return image
    
    def _detect_crossed_words(self, image):
        """检测被划线的单词区域"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 检测斜线（划线）
            # 使用边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 使用霍夫变换检测直线
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=20, maxLineGap=10)
            
            crossed_regions = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # 计算线条角度，筛选斜线（非水平线）
                    if x2 != x1:
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        if abs(angle) > 15 and abs(angle) < 165:  # 斜线
                            # 扩展线条区域以覆盖可能被划掉的文字
                            margin = 10
                            crossed_regions.append((
                                max(0, x1 - margin), max(0, y1 - margin),
                                min(gray.shape[1], x2 + margin), 
                                min(gray.shape[0], y2 + margin)
                            ))
            
            return crossed_regions
        except Exception as e:
            logging.warning(f"划线检测失败: {e}")
            return []
    
    def _enhance_crossed_text(self, image, crossed_regions):
        """增强被划线区域的文字可见性"""
        try:
            enhanced_img = image.copy()
            
            for x1, y1, x2, y2 in crossed_regions:
                # 提取被划线区域
                roi = enhanced_img[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # 使用自适应阈值增强文字
                    if len(roi.shape) == 3:
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    else:
                        roi_gray = roi
                    
                    # 应用局部对比度增强
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
                    enhanced_roi = clahe.apply(roi_gray)
                    
                    # 使用中值滤波去除划线噪声
                    filtered_roi = cv2.medianBlur(enhanced_roi, 3)
                    
                    # 放回原图
                    if len(enhanced_img.shape) == 3:
                        enhanced_img[y1:y2, x1:x2] = cv2.cvtColor(filtered_roi, cv2.COLOR_GRAY2BGR)
                    else:
                        enhanced_img[y1:y2, x1:x2] = filtered_roi
            
            return enhanced_img
        except Exception as e:
            logging.warning(f"被划线文字增强失败: {e}")
            return image

    def _smart_word_correction(self, text: str) -> str:
        """智能单词校正和语境分析"""
        try:
            words = text.split()
            corrected_words = []
            
            for i, word in enumerate(words):
                # 清理标点符号
                clean_word = re.sub(r'[^\w]', '', word.lower())
                original_word = word
                
                if not clean_word:
                    corrected_words.append(original_word)
                    continue
                
                # 检查是否是常见错误模式
                if len(clean_word) <= 2:
                    corrected_words.append(original_word)
                    continue
                
                # 获取上下文单词
                context_words = []
                if i > 0:
                    context_words.append(re.sub(r'[^\w]', '', words[i-1].lower()))
                if i < len(words) - 1:
                    context_words.append(re.sub(r'[^\w]', '', words[i+1].lower()))
                
                # 词汇检查和校正
                corrected_word = self._correct_single_word(clean_word, context_words)
                
                # 保持原始大小写和标点
                if word[0].isupper():
                    corrected_word = corrected_word.capitalize()
                
                # 恢复标点符号
                punctuation = re.findall(r'[^\w]', word)
                if punctuation:
                    corrected_word += punctuation[-1]
                
                corrected_words.append(corrected_word)
            
            return ' '.join(corrected_words)
        except Exception as e:
            logging.warning(f"智能校正失败: {e}")
            return text
    
    def _correct_single_word(self, word: str, context: List[str]) -> str:
        """单个单词校正"""
        # 如果单词已经在词典中，直接返回
        if word in self.english_vocab or word in self.essay_vocab:
            return word
        
        # 常见拼写错误映射
        common_corrections = {
            'teh': 'the', 'adn': 'and', 'taht': 'that', 'thier': 'their',
            'whihc': 'which', 'becuase': 'because', 'alot': 'a lot',
            'seperate': 'separate', 'recieve': 'receive', 'occured': 'occurred',
            'untill': 'until', 'wich': 'which', 'begining': 'beginning',
            'definately': 'definitely', 'neccessary': 'necessary', 'accomodate': 'accommodate',
            'writting': 'writing', 'intrest': 'interest', 'experiance': 'experience',
            'succesful': 'successful', 'acheive': 'achieve', 'knowlege': 'knowledge',
            'enviroment': 'environment', 'goverment': 'government', 'educaton': 'education'
        }
        
        # 检查常见错误
        if word in common_corrections:
            return common_corrections[word]
        
        # 基于上下文的校正
        context_suggestions = self._get_contextual_suggestions(word, context)
        if context_suggestions:
            return context_suggestions[0]
        
        # 使用编辑距离查找相似词
        candidates = get_close_matches(word, self.english_vocab, n=3, cutoff=0.7)
        if candidates:
            return candidates[0]
        
        # 如果没有找到合适的校正，返回原词
        return word
    
    def _get_contextual_suggestions(self, word: str, context: List[str]) -> List[str]:
        """基于上下文的单词建议"""
        suggestions = []
        
        # 基于前后词汇的语法规则
        if context:
            prev_word = context[0] if len(context) > 0 else ''
            next_word = context[1] if len(context) > 1 else ''
            
            # 常见搭配
            collocations = {
                ('very', 'good'): 'very good',
                ('very', 'well'): 'very well',
                ('make', 'decision'): 'make decision',
                ('take', 'care'): 'take care',
                ('look', 'forward'): 'look forward',
                ('due', 'to'): 'due to',
                ('according', 'to'): 'according to',
                ('in', 'order'): 'in order',
                ('as', 'well'): 'as well',
                ('such', 'as'): 'such as'
            }
            
            # 检查搭配
            for collocation in collocations:
                if prev_word in collocation and next_word in collocation:
                    suggestions.extend([collocation[2]])
        
        # 基于作文常用词汇
        essay_candidates = get_close_matches(word, self.essay_vocab, n=2, cutoff=0.6)
        suggestions.extend(essay_candidates)
        
        return suggestions
    
    def _analyze_text_structure(self, text: str) -> Dict:
        """分析文本结构，提供改进建议"""
        try:
            sentences = re.split(r'[.!?]+', text)
            words = text.split()
            
            analysis = {
                'word_count': len(words),
                'sentence_count': len([s for s in sentences if s.strip()]),
                'avg_words_per_sentence': len(words) / max(1, len([s for s in sentences if s.strip()])),
                'potential_errors': [],
                'suggestions': []
            }
            
            # 检测可能的OCR错误
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^\w]', '', word.lower())
                
                # 检测异常字符
                if re.search(r'[^\\w\\s]', word) and len(word) > 1:
                    analysis['potential_errors'].append(f"第{i+1}个词可能包含OCR错误: {word}")
                
                # 检测过短或过长的单词
                if len(clean_word) == 1 and clean_word not in ['a', 'i']:
                    analysis['suggestions'].append(f"单字母单词 '{word}' 可能需要检查")
            
            return analysis
        except Exception as e:
            logging.warning(f"文本结构分析失败: {e}")
            return {'word_count': 0, 'sentence_count': 0, 'potential_errors': [], 'suggestions': []}

    def _preprocess_image(self, image):
        """图像预处理以提高OCR准确率"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 降噪
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 二值化处理
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作去除噪点
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 去除横线和其他干扰
            cleaned = self._remove_lines_and_noise(cleaned)
            
            # 检测被划线的单词区域
            crossed_regions = self._detect_crossed_words(cleaned)
            
            # 增强被划线区域的文字可见性
            enhanced = self._enhance_crossed_text(cleaned, crossed_regions)
            
            return enhanced
        except Exception as e:
            logging.warning(f"图像预处理失败，使用原图: {e}")
            return image

    @staticmethod
    @st.cache_resource(show_spinner="AI 正在阅读图片文字...")
    def _load_model(lang='en', use_gpu=False, enable_mkldnn=False):
        if not PADDLE_AVAILABLE:
            return None
        
        try:
            # 配置 PaddleOCR - 针对英语作文和手写文字优化
            return PaddleOCR(
                use_angle_cls=True,      # 自动纠正文字方向
                lang=lang,               # 语言设置
                use_gpu=use_gpu,         # GPU设置
                enable_mkldnn=enable_mkldnn,  # OneDNN加速
                show_log=False,
                # 针对手写文字优化的参数
                det_db_thresh=0.2,       # 降低检测阈值以捕获更淡的文字
                det_db_box_thresh=0.4,   # 调整文本框阈值
                det_db_unclip_ratio=1.8, # 增加文本框扩展比例
                rec_batch_num=8,         # 增加批处理大小
                drop_score=0.3,          # 降低置信度阈值以捕获更多文字
                # 手写文字识别优化
                det_model_dir=None,      # 使用默认检测模型
                rec_model_dir=None,      # 使用默认识别模型
                cls_model_dir=None       # 使用默认分类模型
            )
        except Exception as e:
            st.error(f"OCR 引擎启动失败: {e}")
            logging.error(f"OCR model loading error: {e}")
            return None

    def extract_text_smart(self, uploaded_file, preprocess=True, max_attempts=3):
        """智能OCR提取，包含逻辑校验和自动重试"""
        if not self.is_ready:
            return "❌ 系统错误：OCR 组件未就绪。"

        ocr_model = self._load_model(self.lang, self.use_gpu, self.enable_mkldnn)
        if ocr_model is None:
            return "❌ 识别引擎加载失败。"

        try:
            # 验证文件类型
            if not uploaded_file.type.startswith('image/'):
                return "❌ 请上传有效的图片文件。"
            
            # 多次尝试识别，直到获得逻辑通顺的结果
            best_result = None
            best_score = 0
            
            for attempt in range(max_attempts):
                try:
                    # 重置文件指针
                    uploaded_file.seek(0)
                    
                    # 将上传的文件转为 OpenCV 图像格式
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)

                    if image is None:
                        continue
                    
                    # 图像预处理（每次尝试使用不同的参数）
                    if preprocess:
                        if attempt == 0:
                            # 第一次尝试：标准预处理
                            processed_image = self._preprocess_image(image)
                        elif attempt == 1:
                            # 第二次尝试：更激进的预处理
                            processed_image = self._preprocess_image_aggressive(image)
                        else:
                            # 第三次尝试：轻度预处理
                            processed_image = self._preprocess_image_light(image)
                    else:
                        processed_image = image

                    # 执行识别
                    result = ocr_model.ocr(processed_image)

                    if not result or not result[0]:
                        continue

                    # 过滤和拼接文字
                    text_lines = []
                    confidence_scores = []
                    
                    for line in result[0]:
                        if line and len(line) > 1:
                            text_content = line[1][0]
                            confidence = line[1][1] if len(line[1]) > 1 else 0.0
                            
                            # 根据尝试次数调整置信度阈值
                            threshold = 0.5 - (attempt * 0.1)
                            if confidence > threshold and len(text_content.strip()) > 0:
                                text_lines.append(text_content)
                                confidence_scores.append(confidence)
                    
                    if not text_lines:
                        continue
                    
                    # 拼接文字
                    extracted_text = "\n".join(text_lines)
                    
                    # 智能校正
                    corrected_text = self._smart_word_correction(extracted_text)
                    
                    # 逻辑校验
                    validation_result = self.logic_validator.validate_essay_logic(corrected_text)
                    
                    # 计算综合评分
                    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
                    combined_score = (avg_confidence + validation_result['confidence']) / 2
                    
                    # 如果逻辑通顺且评分更高，保存这个结果
                    if validation_result['is_logical'] and combined_score > best_score:
                        best_result = corrected_text
                        best_score = combined_score
                        
                        # 如果评分很高，直接返回
                        if combined_score > 0.85:
                            break
                    
                    # 如果逻辑不通顺但OCR置信度很高，也保存作为备选
                    elif avg_confidence > 0.8 and combined_score > best_score:
                        best_result = corrected_text
                        best_score = combined_score

                except Exception as e:
                    logging.warning(f"OCR尝试 {attempt + 1} 失败: {e}")
                    continue
            
            if best_result:
                # 对最佳结果进行最终处理
                # 识别有问题的单词并添加下划线
                problematic_words = self.logic_validator.identify_problematic_words(best_result)
                final_result = self.logic_validator.suggest_corrections(best_result, problematic_words)
                
                # 最终逻辑检查
                final_validation = self.logic_validator.validate_essay_logic(final_result)
                
                # 如果逻辑仍然有问题，添加简短提示
                if not final_validation['is_logical']:
                    final_result += f"\n\n⚠️ 文章可能存在逻辑问题，请检查"
                
                return final_result
            else:
                return "⚠️ 多次尝试后仍无法识别到有效文字，请尝试重新拍照或调整图片角度。"

        except Exception as e:
            logging.error(f"智能OCR提取失败: {e}")
            return f"❌ 识别发生错误: {str(e)}"
        finally:
            # 重置文件指针
            if uploaded_file:
                uploaded_file.seek(0)
    
    def _preprocess_image_aggressive(self, image):
        """更激进的图像预处理"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 强力降噪
            denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # 自适应阈值
            adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            
            # 更强的形态学操作
            kernel = np.ones((3,3), np.uint8)
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 去除横线和其他干扰
            cleaned = self._remove_lines_and_noise(cleaned)
            
            return cleaned
        except Exception as e:
            logging.warning(f"激进预处理失败，使用标准预处理: {e}")
            return self._preprocess_image(image)
    
    def _preprocess_image_light(self, image):
        """轻度图像预处理"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 轻度降噪
            denoised = cv2.GaussianBlur(gray, (3,3), 0)
            
            # 简单阈值
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binary
        except Exception as e:
            logging.warning(f"轻度预处理失败，使用原图: {e}")
            return image
        """提取图片文字"""
        if not self.is_ready:
            return "❌ 系统错误：OCR 组件未就绪。"

        ocr_model = self._load_model(self.lang, self.use_gpu, self.enable_mkldnn)
        if ocr_model is None:
            return "❌ 识别引擎加载失败。"

        try:
            # 验证文件类型
            if not uploaded_file.type.startswith('image/'):
                return "❌ 请上传有效的图片文件。"
            
            # 将上传的文件转为 OpenCV 图像格式
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            if image is None:
                return "❌ 无法解析图片，请确认文件完整。"
            
            # 图像预处理
            if preprocess:
                processed_image = self._preprocess_image(image)
            else:
                processed_image = image

            # 执行识别
            result = ocr_model.ocr(processed_image)

            if not result or not result[0]:
                return "⚠️ 未能识别到文字，请确保图片文字清晰且无反光。"

            # 过滤和拼接文字
            text_lines = []
            confidence_scores = []
            
            for line in result[0]:
                if line and len(line) > 1:
                    text_content = line[1][0]
                    confidence = line[1][1] if len(line[1]) > 1 else 0.0
                    
                    # 降低过滤阈值以捕获更多文字
                    if confidence > 0.3 and len(text_content.strip()) > 0:
                        text_lines.append(text_content)
                        confidence_scores.append(confidence)
            
            if not text_lines:
                return "⚠️ 未能识别到有效文字，请尝试重新拍照或调整图片角度。"
            
            # 计算平均置信度
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            # 拼接文字
            extracted_text = "\n".join(text_lines)
            
            # 智能校正和语境分析
            if enable_smart_correction:
                # 应用智能单词校正
                corrected_text = self._smart_word_correction(extracted_text)
                
                # 文本结构分析
                analysis = self._analyze_text_structure(corrected_text)
                
                # 构建详细结果
                result_text = corrected_text
                
                # 添加置信度信息
                result_text += f"\n\n📊 识别置信度: {avg_confidence:.1%}"
                
                # 添加文本分析信息
                if analysis['potential_errors']:
                    result_text += f"\n\n⚠️ 发现 {len(analysis['potential_errors'])} 个可能的识别错误"
                    for error in analysis['potential_errors'][:3]:  # 只显示前3个错误
                        result_text += f"\n• {error}"
                
                if analysis['suggestions']:
                    result_text += f"\n\n💡 建议:"
                    for suggestion in analysis['suggestions'][:2]:  # 只显示前2个建议
                        result_text += f"\n• {suggestion}"
                
                # 添加文本统计
                result_text += f"\n\n📝 文本统计: {analysis['word_count']} 词, {analysis['sentence_count']} 句"
                
                return result_text
            else:
                # 基础结果
                confidence_info = f"\n\n📊 识别置信度: {avg_confidence:.1%}"
                return extracted_text + confidence_info

        except Exception as e:
            logging.error(f"OCR extraction error: {e}")
            return f"❌ 识别发生错误: {str(e)}"
        finally:
            # 重置文件指针
            if uploaded_file:
                uploaded_file.seek(0)

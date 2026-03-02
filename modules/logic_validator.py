import re
import logging
from typing import List, Dict, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

# 下载必要的NLTK数据（首次使用）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class EssayLogicValidator:
    """作文逻辑校验器"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # 常见逻辑连接词
        self.logic_connectors = {
            'cause': ['because', 'since', 'as', 'due to', 'owing to', 'thanks to'],
            'effect': ['therefore', 'thus', 'hence', 'consequently', 'as a result'],
            'contrast': ['however', 'nevertheless', 'nonetheless', 'but', 'yet', 'still'],
            'addition': ['moreover', 'furthermore', 'additionally', 'besides', 'also'],
            'sequence': ['first', 'second', 'then', 'next', 'finally', 'lastly'],
            'example': ['for example', 'for instance', 'such as', 'like', 'namely']
        }
        
        # 作文常见主题词汇
        self.essay_topics = {
            'education': ['study', 'learn', 'education', 'school', 'student', 'teacher', 'knowledge'],
            'habits': ['habit', 'routine', 'practice', 'lifestyle', 'daily', 'regular'],
            'personal': ['my', 'i', 'me', 'myself', 'personal', 'individual'],
            'benefits': ['benefit', 'advantage', 'help', 'improve', 'enhance', 'better']
        }
        
        # 常见语法错误模式
        self.grammar_patterns = {
            'subject_verb': ['i am', 'he are', 'she are', 'it are', 'they is'],
            'article_errors': ['a hour', 'a university', 'an book', 'an house'],
            'preposition_errors': ['depend of', 'listen music', 'good on']
        }
    
    def validate_essay_logic(self, text: str) -> Dict:
        """验证作文逻辑性"""
        try:
            # 清理文本
            cleaned_text = self._clean_text(text)
            
            # 分句
            sentences = sent_tokenize(cleaned_text)
            
            if len(sentences) < 3:
                return {
                    'is_logical': False,
                    'confidence': 0.3,
                    'issues': ['文章过短，逻辑结构不完整'],
                    'suggestions': ['建议增加更多句子来完整表达观点']
                }
            
            # 各项检查
            coherence_score = self._check_coherence(sentences)
            grammar_score = self._check_grammar(cleaned_text)
            topic_consistency = self._check_topic_consistency(cleaned_text)
            structure_score = self._check_structure(sentences)
            
            # 综合评分
            overall_score = (coherence_score + grammar_score + topic_consistency + structure_score) / 4
            
            issues = []
            suggestions = []
            
            if coherence_score < 0.6:
                issues.append('句子间逻辑连接不够清晰')
                suggestions.append('建议使用更多逻辑连接词如however, therefore等')
            
            if grammar_score < 0.7:
                issues.append('存在语法错误')
                suggestions.append('检查主谓一致、冠词使用等语法问题')
            
            if topic_consistency < 0.6:
                issues.append('主题不够集中')
                suggestions.append('确保文章围绕中心主题展开')
            
            if structure_score < 0.6:
                issues.append('文章结构不够清晰')
                suggestions.append('建议使用更清晰的开头、主体、结尾结构')
            
            return {
                'is_logical': overall_score >= 0.6,
                'confidence': overall_score,
                'issues': issues,
                'suggestions': suggestions,
                'scores': {
                    'coherence': coherence_score,
                    'grammar': grammar_score,
                    'topic_consistency': topic_consistency,
                    'structure': structure_score
                }
            }
            
        except Exception as e:
            logging.error(f"逻辑验证失败: {e}")
            return {
                'is_logical': False,
                'confidence': 0.0,
                'issues': ['逻辑验证过程出错'],
                'suggestions': ['请检查文本格式']
            }
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符但保留标点
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def _check_coherence(self, sentences: List[str]) -> float:
        """检查句子连贯性"""
        connector_count = 0
        total_pairs = len(sentences) - 1
        
        if total_pairs <= 0:
            return 0.5
        
        for i in range(total_pairs):
            current = sentences[i].lower()
            next_sent = sentences[i+1].lower()
            
            # 检查逻辑连接词
            for category, connectors in self.logic_connectors.items():
                for connector in connectors:
                    if connector in current or connector in next_sent:
                        connector_count += 1
                        break
        
        return min(connector_count / total_pairs, 1.0)
    
    def _check_grammar(self, text: str) -> float:
        """检查语法正确性"""
        errors = 0
        words = word_tokenize(text.lower())
        
        # 检查常见语法错误模式
        for pattern in self.grammar_patterns['subject_verb']:
            if pattern in text.lower():
                errors += 1
        
        # 检查句子完整性（每个句子应该有主语和谓语）
        sentences = sent_tokenize(text)
        incomplete_sentences = 0
        
        for sent in sentences:
            words_in_sent = word_tokenize(sent)
            tagged = pos_tag(words_in_sent)
            
            # 检查是否有名词和动词
            has_noun = any(tag.startswith('NN') for word, tag in tagged)
            has_verb = any(tag.startswith('VB') for word, tag in tagged)
            
            if not (has_noun and has_verb):
                incomplete_sentences += 1
        
        if len(sentences) > 0:
            grammar_score = 1.0 - (incomplete_sentences / len(sentences))
            grammar_score = max(0, grammar_score - (errors * 0.1))
        else:
            grammar_score = 0.5
        
        return min(grammar_score, 1.0)
    
    def _check_topic_consistency(self, text: str) -> float:
        """检查主题一致性"""
        words = word_tokenize(text.lower())
        content_words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        if len(content_words) < 5:
            return 0.5
        
        # 统计主题词汇
        topic_scores = {}
        for topic, keywords in self.essay_topics.items():
            count = sum(1 for word in content_words if word in keywords)
            topic_scores[topic] = count / len(content_words)
        
        # 如果有明显的主题，给高分
        max_topic_score = max(topic_scores.values()) if topic_scores else 0
        
        # 检查词汇重复度（高重复度可能表示主题集中）
        word_freq = Counter(content_words)
        repetition_score = max(word_freq.values()) / len(content_words) if content_words else 0
        
        return (max_topic_score + repetition_score) / 2
    
    def _check_structure(self, sentences: List[str]) -> float:
        """检查文章结构"""
        if len(sentences) < 3:
            return 0.3
        
        # 简单的结构检查：开头、中间、结尾
        structure_score = 0.0
        
        # 检查开头句（通常引入主题）
        first_sent = sentences[0].lower()
        if any(word in first_sent for word in ['i', 'my', 'this', 'the', 'in', 'as']):
            structure_score += 0.3
        
        # 检查中间句的长度变化
        middle_sents = sentences[1:-1]
        if len(middle_sents) > 1:
            lengths = [len(sent.split()) for sent in middle_sents]
            length_variance = max(lengths) - min(lengths)
            if length_variance > 2:  # 有一定长度变化
                structure_score += 0.3
        
        # 检查结尾句（通常总结或建议）
        last_sent = sentences[-1].lower()
        if any(word in last_sent for word in ['therefore', 'thus', 'conclusion', 'suggest', 'recommend']):
            structure_score += 0.4
        
        return min(structure_score, 1.0)
    
    def identify_problematic_words(self, text: str) -> List[str]:
        """识别可能有问题的单词"""
        words = word_tokenize(text)
        problematic = []
        
        for word in words:
            # 检查是否包含数字（可能是OCR错误）
            if re.search(r'\d', word):
                problematic.append(word)
            
            # 检查异常字符组合
            if re.search(r'[^\w]', word) and len(word) > 1:
                problematic.append(word)
            
            # 检查过短的非常见词
            if len(word) == 1 and word.lower() not in ['a', 'i']:
                problematic.append(word)
        
        return problematic
    
    def suggest_corrections(self, text: str, problematic_words: List[str]) -> str:
        """为有问题的单词添加下划线标注"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            
            # 检查是否需要标注
            needs_underline = False
            for prob_word in problematic_words:
                if clean_word.lower() == prob_word.lower():
                    needs_underline = True
                    break
            
            if needs_underline:
                corrected_words.append(f"_{word}_")
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)

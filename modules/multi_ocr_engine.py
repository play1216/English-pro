import streamlit as st
import numpy as np
import cv2
import logging
import os
import json
import requests
import base64
from typing import List, Dict, Optional, Tuple
from PIL import Image
import io
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from modules.simple_logic_validator import SimpleLogicValidator

@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    confidence: float
    source: str
    cost: float = 0.0
    processing_time: float = 0.0

class BaseOCREngine(ABC):
    """OCR引擎基类"""
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """提取文字"""
        pass
    
    @abstractmethod
    def get_cost_per_request(self) -> float:
        """获取每次请求成本"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用"""
        pass

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR引擎"""
    
    def __init__(self, lang='en', use_gpu=False, enable_mkldnn=False):
        self.lang = lang
        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        self.ocr_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化PaddleOCR模型"""
        try:
            # 环境设置
            os.environ['FLAGS_enable_pir_api'] = '0'
            os.environ['FLAGS_enable_executor_unittests'] = '0'
            os.environ['FLAGS_use_mkldnn'] = '0'
            
            import paddle
            paddle.set_flags({'FLAGS_enable_pir_api': 0})
            from paddleocr import PaddleOCR
            
            self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                enable_mkldnn=self.enable_mkldnn,
                show_log=False,
                det_db_thresh=0.2,
                det_db_box_thresh=0.4,
                det_db_unclip_ratio=1.8,
                rec_batch_num=8,
                drop_score=0.3,
            )
            return True
        except Exception as e:
            logging.error(f"PaddleOCR初始化失败: {e}")
            return False
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """提取文字"""
        if not self.ocr_model:
            return OCRResult("", 0.0, "PaddleOCR")
        
        start_time = time.time()
        try:
            result = self.ocr_model.ocr(image)
            processing_time = time.time() - start_time
            
            if not result or not result[0]:
                return OCRResult("", 0.0, "PaddleOCR", 0.0, processing_time)
            
            # 过滤和拼接文字
            text_lines = []
            confidence_scores = []
            
            for line in result[0]:
                if line and len(line) > 1:
                    text_content = line[1][0]
                    confidence = line[1][1] if len(line[1]) > 1 else 0.0
                    
                    if confidence > 0.3 and len(text_content.strip()) > 0:
                        text_lines.append(text_content)
                        confidence_scores.append(confidence)
            
            if not text_lines:
                return OCRResult("", 0.0, "PaddleOCR", 0.0, processing_time)
            
            extracted_text = "\n".join(text_lines)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            return OCRResult(extracted_text, avg_confidence, "PaddleOCR", 0.0, processing_time)
            
        except Exception as e:
            logging.error(f"PaddleOCR识别失败: {e}")
            return OCRResult("", 0.0, "PaddleOCR", 0.0, time.time() - start_time)
    
    def get_cost_per_request(self) -> float:
        """PaddleOCR免费"""
        return 0.0
    
    def is_available(self) -> bool:
        """检查PaddleOCR是否可用"""
        return self.ocr_model is not None

class GoogleVisionOCREngine(BaseOCREngine):
    """Google Vision OCR引擎"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://vision.googleapis.com/v1/images:annotate"
        self.cost_per_request = 0.0015  # 约$1.5 per 1000 images
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """提取文字"""
        start_time = time.time()
        try:
            # 转换图像为base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构建请求
            request_data = {
                "requests": [{
                    "image": {
                        "content": image_base64
                    },
                    "features": [{
                        "type": "TEXT_DETECTION",
                        "maxResults": 1
                    }]
                }]
            }
            
            # 发送请求
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            response = requests.post(
                self.base_url,
                data=json.dumps(request_data),
                headers=headers,
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code != 200:
                logging.error(f"Google Vision API错误: {response.status_code} - {response.text}")
                return OCRResult("", 0.0, "Google Vision", self.cost_per_request, processing_time)
            
            result = response.json()
            
            # 解析结果
            if 'responses' not in result or not result['responses']:
                return OCRResult("", 0.0, "Google Vision", self.cost_per_request, processing_time)
            
            text_annotations = result['responses'][0].get('textAnnotations', [])
            if not text_annotations:
                return OCRResult("", 0.0, "Google Vision", self.cost_per_request, processing_time)
            
            # 获取完整文本和置信度
            full_text = text_annotations[0].get('description', '')
            
            # 计算平均置信度
            confidences = []
            for annotation in text_annotations[1:]:  # 跳过第一个（完整文本）
                if 'confidence' in annotation:
                    confidences.append(annotation['confidence'])
            
            avg_confidence = np.mean(confidences) if confidences else 0.8
            
            return OCRResult(full_text, avg_confidence, "Google Vision", self.cost_per_request, processing_time)
            
        except Exception as e:
            logging.error(f"Google Vision OCR失败: {e}")
            return OCRResult("", 0.0, "Google Vision", self.cost_per_request, time.time() - start_time)
    
    def get_cost_per_request(self) -> float:
        return self.cost_per_request
    
    def is_available(self) -> bool:
        """检查API密钥是否设置"""
        return bool(self.api_key and self.api_key.strip())

class AzureOCREngine(BaseOCREngine):
    """Azure Computer Vision OCR引擎"""
    
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/') + '/vision/v3.2/ocr'
        self.cost_per_request = 0.001  # 约$1 per 1000 images
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """提取文字"""
        start_time = time.time()
        try:
            # 转换图像为base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构建请求
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key,
                'Content-Type': 'application/octet-stream'
            }
            
            params = {
                'language': 'en',
                'detectOrientation': 'true'
            }
            
            response = requests.post(
                self.endpoint,
                data=buffer,
                headers=headers,
                params=params,
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code != 200:
                logging.error(f"Azure OCR API错误: {response.status_code} - {response.text}")
                return OCRResult("", 0.0, "Azure OCR", self.cost_per_request, processing_time)
            
            result = response.json()
            
            # 解析结果
            if 'regions' not in result:
                return OCRResult("", 0.0, "Azure OCR", self.cost_per_request, processing_time)
            
            # 提取文字
            lines = []
            confidences = []
            
            for region in result['regions']:
                for line in region['lines']:
                    line_text = ' '.join([word['text'] for word in line['words']])
                    lines.append(line_text)
                    
                    # Azure不直接提供置信度，使用默认值
                    confidences.append(0.8)
            
            full_text = '\n'.join(lines)
            avg_confidence = np.mean(confidences) if confidences else 0.8
            
            return OCRResult(full_text, avg_confidence, "Azure OCR", self.cost_per_request, processing_time)
            
        except Exception as e:
            logging.error(f"Azure OCR失败: {e}")
            return OCRResult("", 0.0, "Azure OCR", self.cost_per_request, time.time() - start_time)
    
    def get_cost_per_request(self) -> float:
        return self.cost_per_request
    
    def is_available(self) -> bool:
        """检查API密钥是否设置"""
        return bool(self.api_key and self.endpoint.strip())

class MultiOCREngine:
    """多OCR引擎管理器"""
    
    def __init__(self):
        self.engines: List[BaseOCREngine] = []
        self.logic_validator = SimpleLogicValidator()
        self.cost_tracker = {
            'total_cost': 0.0,
            'request_count': 0,
            'engine_usage': {}
        }
        
        # 初始化PaddleOCR（免费，始终可用）
        self.paddle_engine = PaddleOCREngine()
        if self.paddle_engine.is_available():
            self.engines.append(self.paddle_engine)
            logging.info("PaddleOCR引擎已加载")
    
    def add_api_engine(self, engine: BaseOCREngine):
        """添加API引擎"""
        if engine.is_available():
            self.engines.append(engine)
            engine_name = engine.__class__.__name__
            self.cost_tracker['engine_usage'][engine_name] = {
                'count': 0,
                'cost': 0.0
            }
            logging.info(f"{engine_name}引擎已添加")
        else:
            logging.warning(f"{engine.__class__.__name__}引擎不可用")
    
    def extract_text_with_voting(self, image: np.ndarray, max_engines: int = 3, budget_limit: float = 0.01) -> Dict:
        """使用多引擎投票机制提取文字"""
        results = []
        total_cost = 0.0
        
        # 按成本排序，优先使用便宜的引擎
        sorted_engines = sorted(self.engines, key=lambda x: x.get_cost_per_request())
        
        for i, engine in enumerate(sorted_engines[:max_engines]):
            if total_cost + engine.get_cost_per_request() > budget_limit:
                logging.info(f"达到预算限制，停止使用更多引擎")
                break
            
            try:
                result = engine.extract_text(image)
                if result.text.strip():  # 只保留有效结果
                    results.append(result)
                    total_cost += result.cost
                    
                    # 更新成本追踪
                    engine_name = engine.__class__.__name__
                    if engine_name in self.cost_tracker['engine_usage']:
                        self.cost_tracker['engine_usage'][engine_name]['count'] += 1
                        self.cost_tracker['engine_usage'][engine_name]['cost'] += result.cost
                    
                    logging.info(f"{engine_name}识别完成，置信度: {result.confidence:.2f}")
                
            except Exception as e:
                logging.error(f"{engine.__class__.__name__}识别失败: {e}")
        
        if not results:
            return {
                'text': '',
                'confidence': 0.0,
                'engines_used': [],
                'total_cost': 0.0,
                'processing_time': 0.0,
                'needs_review': True
            }
        
        # 投票机制选择最佳结果
        best_result = self._vote_best_result(results)
        
        # 更新总成本
        self.cost_tracker['total_cost'] += total_cost
        self.cost_tracker['request_count'] += 1
        
        return {
            'text': best_result.text,
            'confidence': best_result.confidence,
            'engines_used': [r.source for r in results],
            'total_cost': total_cost,
            'processing_time': sum(r.processing_time for r in results),
            'needs_review': best_result.confidence < 0.7,
            'all_results': [(r.source, r.text, r.confidence) for r in results]
        }
    
    def _vote_best_result(self, results: List[OCRResult]) -> OCRResult:
        """投票机制选择最佳结果"""
        if len(results) == 1:
            return results[0]
        
        # 计算每个结果的得分
        scored_results = []
        for result in results:
            # 基础分数：置信度
            base_score = result.confidence
            
            # 逻辑校验加分
            if result.text.strip():
                validation = self.logic_validator.validate_essay_logic(result.text)
                logic_bonus = validation['confidence'] * 0.3
            else:
                logic_bonus = 0
            
            # 文本长度加分（适中的长度更好）
            text_length = len(result.text.split())
            if 50 <= text_length <= 300:
                length_bonus = 0.1
            elif 20 <= text_length < 50 or 300 < text_length <= 500:
                length_bonus = 0.05
            else:
                length_bonus = 0
            
            # 成本惩罚（越便宜越好）
            cost_penalty = result.cost * 10
            
            total_score = base_score + logic_bonus + length_bonus - cost_penalty
            scored_results.append((total_score, result))
        
        # 选择得分最高的结果
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[0][1]
    
    def get_cost_report(self) -> Dict:
        """获取成本报告"""
        return {
            'total_cost': self.cost_tracker['total_cost'],
            'request_count': self.cost_tracker['request_count'],
            'average_cost_per_request': (
                self.cost_tracker['total_cost'] / self.cost_tracker['request_count']
                if self.cost_tracker['request_count'] > 0 else 0
            ),
            'engine_usage': self.cost_tracker['engine_usage'],
            'available_engines': [engine.__class__.__name__ for engine in self.engines]
        }
    
    def set_budget_limit(self, daily_limit: float, monthly_limit: float):
        """设置预算限制"""
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        # TODO: 实现预算追踪逻辑

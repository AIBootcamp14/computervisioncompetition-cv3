"""
Base Preprocessor Class

데이터 전처리를 위한 기본 클래스
Strategy Pattern과 Template Method Pattern 적용
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BasePreprocessor(ABC):
    """
    이미지 전처리를 위한 기본 클래스
    
    각 멤버는 이 클래스를 상속받아 자신만의 전처리 전략을 구현할 수 있음
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize_params: Optional[Dict[str, List[float]]] = None
    ):
        """
        Args:
            image_size: 목표 이미지 크기 (height, width)
            normalize_params: 정규화 파라미터 {'mean': [r,g,b], 'std': [r,g,b]}
        """
        self.image_size = image_size
        self.normalize_params = normalize_params or {
            'mean': [0.485, 0.456, 0.406],  # ImageNet 기본값
            'std': [0.229, 0.224, 0.225]
        }
        
        # 전처리 파이프라인 초기화
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
        self.test_transform = self._build_test_transform()
    
    @abstractmethod
    def _build_train_transform(self) -> A.Compose:
        """
        훈련용 데이터 증강 파이프라인 구성
        각 멤버가 자신만의 전략으로 구현
        
        Returns:
            A.Compose: Albumentations 변환 파이프라인
        """
        pass
    
    @abstractmethod
    def _build_val_transform(self) -> A.Compose:
        """
        검증용 전처리 파이프라인 구성
        
        Returns:
            A.Compose: Albumentations 변환 파이프라인
        """
        pass
    
    def _build_test_transform(self) -> A.Compose:
        """
        테스트용 전처리 파이프라인 구성 (기본 구현)
        
        Returns:
            A.Compose: Albumentations 변환 파이프라인
        """
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(**self.normalize_params),
            ToTensorV2()
        ])
    
    def get_transform(self, mode: str = 'train') -> A.Compose:
        """
        모드에 따른 변환 파이프라인 반환
        
        Args:
            mode: 'train', 'val', 'test' 중 하나
            
        Returns:
            A.Compose: 해당 모드의 변환 파이프라인
        """
        transform_map = {
            'train': self.train_transform,
            'val': self.val_transform,
            'test': self.test_transform
        }
        
        if mode not in transform_map:
            raise ValueError(f"Unknown mode: {mode}. Must be one of {list(transform_map.keys())}")
            
        return transform_map[mode]
    
    def preprocess_batch(
        self, 
        images: List[np.ndarray], 
        mode: str = 'train'
    ) -> List[np.ndarray]:
        """
        배치 단위로 이미지 전처리
        
        Args:
            images: 원본 이미지 리스트
            mode: 처리 모드
            
        Returns:
            List[np.ndarray]: 전처리된 이미지 리스트
        """
        transform = self.get_transform(mode)
        processed_images = []
        
        for image in images:
            # 이미지 유효성 검사
            if not self._validate_image(image):
                raise ValueError("Invalid image format")
                
            # 변환 적용
            transformed = transform(image=image)
            processed_images.append(transformed['image'])
            
        return processed_images
    
    def _validate_image(self, image: np.ndarray) -> bool:
        """
        이미지 유효성 검사
        
        Args:
            image: 검사할 이미지
            
        Returns:
            bool: 유효성 여부
        """
        if not isinstance(image, np.ndarray):
            return False
            
        if len(image.shape) != 3:
            return False
            
        if image.shape[2] != 3:  # RGB 채널
            return False
            
        if image.dtype != np.uint8:
            return False
            
        return True
    
    def get_preprocessing_stats(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        이미지 데이터의 통계 정보 계산
        
        Args:
            images: 분석할 이미지 리스트
            
        Returns:
            Dict[str, Any]: 통계 정보
        """
        if not images:
            return {}
            
        # 이미지 크기 분포
        sizes = [img.shape[:2] for img in images]
        heights, widths = zip(*sizes)
        
        # 픽셀 값 통계 (채널별)
        all_pixels = np.concatenate([img.reshape(-1, 3) for img in images])
        
        stats = {
            'count': len(images),
            'size_stats': {
                'height': {
                    'mean': np.mean(heights),
                    'std': np.std(heights),
                    'min': np.min(heights),
                    'max': np.max(heights)
                },
                'width': {
                    'mean': np.mean(widths),
                    'std': np.std(widths),
                    'min': np.min(widths),
                    'max': np.max(widths)
                }
            },
            'pixel_stats': {
                'mean': np.mean(all_pixels, axis=0).tolist(),
                'std': np.std(all_pixels, axis=0).tolist(),
                'min': np.min(all_pixels, axis=0).tolist(),
                'max': np.max(all_pixels, axis=0).tolist()
            }
        }
        
        return stats
    
    def suggest_normalization_params(self, images: List[np.ndarray]) -> Dict[str, List[float]]:
        """
        데이터셋 기반 정규화 파라미터 추천
        
        Args:
            images: 분석할 이미지 리스트
            
        Returns:
            Dict[str, List[float]]: 추천 정규화 파라미터
        """
        if not images:
            return self.normalize_params
            
        # 모든 픽셀값을 0-1 범위로 정규화한 후 통계 계산
        all_pixels = np.concatenate([
            (img.astype(np.float32) / 255.0).reshape(-1, 3) 
            for img in images
        ])
        
        mean = np.mean(all_pixels, axis=0).tolist()
        std = np.std(all_pixels, axis=0).tolist()
        
        return {'mean': mean, 'std': std}

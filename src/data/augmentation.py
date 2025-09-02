"""
Document Augmentation Module

문서 이미지 증강을 위한 클래스들
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class DocumentAugmentation:
    """
    문서 이미지 전용 데이터 증강 클래스
    
    문서의 특성을 고려한 증강 기법들을 제공
    """
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            image_size: 목표 이미지 크기
        """
        self.image_size = image_size
        self.augmentation_configs = self._define_augmentation_configs()
    
    def _define_augmentation_configs(self) -> Dict[str, Dict]:
        """증강 설정 정의"""
        return {
            'light': {
                'description': '가벼운 증강 (기본)',
                'transforms': [
                    A.Resize(height=self.image_size[0], width=self.image_size[1]),
                    A.HorizontalFlip(p=0.3),
                    A.Rotate(limit=5, p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                ]
            },
            'medium': {
                'description': '중간 강도 증강',
                'transforms': [
                    A.Resize(height=self.image_size[0], width=self.image_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10, p=0.4),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.OneOf([
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
                    ], p=0.3),
                ]
            },
            'heavy': {
                'description': '강한 증강 (데이터가 부족한 경우)',
                'transforms': [
                    A.Resize(height=self.image_size[0], width=self.image_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.OneOf([
                        A.GaussNoise(var_limit=(10.0, 80.0), p=0.5),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                        A.MultiplicativeNoise(multiplier=[0.9, 1.1], elementwise=True, p=0.5),
                    ], p=0.4),
                    A.OneOf([
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.5),
                    ], p=0.4),
                    A.OneOf([
                        A.CLAHE(clip_limit=2, p=0.5),
                        A.Sharpen(p=0.5),
                        A.Emboss(p=0.5),
                    ], p=0.3),
                ]
            },
            'document_specific': {
                'description': '문서 전용 증강',
                'transforms': [
                    A.Resize(height=self.image_size[0], width=self.image_size[1]),
                    # 문서 스캔 시뮬레이션
                    A.Perspective(scale=(0.05, 0.1), p=0.3),
                    # 조명 변화 (스캔/촬영 환경)
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                    # 블러 (스캔 품질)
                    A.OneOf([
                        A.MotionBlur(blur_limit=3, p=0.5),
                        A.GaussianBlur(blur_limit=3, p=0.5),
                    ], p=0.2),
                    # 노이즈 (스캔 아티팩트)
                    A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
                    # 회전 (문서 기울어짐)
                    A.Rotate(limit=3, p=0.3),
                ]
            }
        }
    
    def get_augmentation_pipeline(
        self, 
        config_name: str = 'medium',
        normalize_params: Optional[Dict[str, List[float]]] = None
    ) -> A.Compose:
        """
        증강 파이프라인 생성
        
        Args:
            config_name: 증강 설정 이름
            normalize_params: 정규화 파라미터
            
        Returns:
            A.Compose: 증강 파이프라인
        """
        if config_name not in self.augmentation_configs:
            raise ValueError(f"Unknown config: {config_name}")
        
        config = self.augmentation_configs[config_name]
        transforms = config['transforms'].copy()
        
        # 정규화 및 텐서 변환 추가
        if normalize_params:
            transforms.append(A.Normalize(**normalize_params))
        else:
            transforms.append(A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def get_test_time_augmentation(
        self,
        normalize_params: Optional[Dict[str, List[float]]] = None
    ) -> List[A.Compose]:
        """
        테스트 타임 증강 (TTA) 파이프라인들 생성
        
        Args:
            normalize_params: 정규화 파라미터
            
        Returns:
            List[A.Compose]: TTA 파이프라인 리스트
        """
        normalize = A.Normalize(**normalize_params) if normalize_params else A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        tta_pipelines = [
            # 원본
            A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                normalize,
                ToTensorV2()
            ]),
            # 수평 뒤집기
            A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=1.0),
                normalize,
                ToTensorV2()
            ]),
            # 작은 회전들
            A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Rotate(limit=5, p=1.0),
                normalize,
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Rotate(limit=-5, p=1.0),
                normalize,
                ToTensorV2()
            ]),
        ]
        
        return tta_pipelines
    
    def analyze_augmentation_impact(
        self,
        original_image: np.ndarray,
        config_name: str = 'medium',
        num_samples: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        증강 효과 분석을 위한 샘플 생성
        
        Args:
            original_image: 원본 이미지
            config_name: 증강 설정
            num_samples: 생성할 샘플 수
            
        Returns:
            Dict[str, np.ndarray]: 증강된 이미지들
        """
        pipeline = self.get_augmentation_pipeline(config_name, normalize_params=None)
        
        samples = {'original': original_image}
        
        for i in range(num_samples):
            # 정규화와 텐서 변환 제외하고 증강만 적용
            aug_pipeline = A.Compose(self.augmentation_configs[config_name]['transforms'])
            augmented = aug_pipeline(image=original_image)['image']
            samples[f'augmented_{i+1}'] = augmented
        
        return samples
    
    def get_available_configs(self) -> Dict[str, str]:
        """
        사용 가능한 증강 설정 목록 반환
        
        Returns:
            Dict[str, str]: 설정명과 설명
        """
        return {
            name: config['description'] 
            for name, config in self.augmentation_configs.items()
        }

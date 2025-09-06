
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple

class AdvancedAugmentationStrategy:
    """
    고급 증강 전략: 클래스별 맞춤 + Domain Adaptation
    EDA 분석 결과를 반영한 지능형 증강
    """
    
    def __init__(self):
        self.vehicle_classes = [2, 16]
        self.document_classes = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.minority_classes = [1, 13, 14]
        
    def get_class_specific_transforms(self, target_class: int, image_size: int = 512):
        """클래스별 맞춤 증강 반환"""
        
        if target_class in self.vehicle_classes:
            return self._get_vehicle_transforms(image_size)
        elif target_class in self.minority_classes:
            return self._get_minority_class_transforms(image_size)
        else:
            return self._get_document_transforms(image_size)
    
    def _get_vehicle_transforms(self, image_size: int):
        """차량 관련 클래스 전용 증강"""
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # 차량 특화 증강
            A.RandomBrightnessContrast(
                brightness_limit=0.4,  # 강화된 밝기 조정
                contrast_limit=0.4, 
                p=0.9
            ),
            
            # 색상 지터링 강화 (다양한 조명)
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3, 
                saturation=0.3,
                hue=0.1,
                p=0.7
            ),
            
            # 노이즈 증가 (실제 촬영 환경)
            A.OneOf([
                A.GaussNoise(var_limit=(20, 80)),  # 더 강한 노이즈
                A.MultiplicativeNoise(multiplier=[0.8, 1.2]),
            ], p=0.6),
            
            # 원근 변형 감소 (이미 다양한 각도)
            A.Perspective(scale=(0.02, 0.05), p=0.2),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_document_transforms(self, image_size: int):
        """문서 클래스 전용 증강"""
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # 문서 특화 증강
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=45, p=0.8, border_mode=cv2.BORDER_CONSTANT),  # 더 강한 회전
            
            # 원근 변형 강화 (스캔 왜곡)
            A.Perspective(scale=(0.1, 0.2), p=0.5),
            
            # 블러 추가 (Test 데이터 대응)
            A.OneOf([
                A.MotionBlur(blur_limit=5),  # 더 강한 블러
                A.GaussianBlur(blur_limit=5),
            ], p=0.4),
            
            # 대비 조정
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.4,  # 문서 품질 다양화
                p=0.8
            ),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_minority_class_transforms(self, image_size: int):
        """소수 클래스 전용 강화 증강"""
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # 기본 증강
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            
            # 소수 클래스 특화 증강
            A.OneOf([
                A.ElasticTransform(p=0.3),  # 탄성 변형
                A.GridDistortion(p=0.3),    # 격자 왜곡
                A.OpticalDistortion(p=0.3), # 광학 왜곡
            ], p=0.5),
            
            # 강화된 노이즈
            A.OneOf([
                A.GaussNoise(var_limit=(15, 60)),
                A.MultiplicativeNoise(multiplier=[0.85, 1.15]),
            ], p=0.6),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# Domain Adaptation 클래스
class DomainAdaptationAugmentation:
    """Train 데이터를 Test 데이터 특성에 맞게 적응시키는 증강"""
    
    def __init__(self, adaptation_epoch: int = 0):
        self.adaptation_epoch = adaptation_epoch
        
    def get_domain_adapted_transforms(self, image_size: int = 512):
        """에포크에 따른 점진적 Domain Adaptation"""
        
        transforms = [A.Resize(image_size, image_size)]
        
        # 점진적 적응 스케줄
        if self.adaptation_epoch >= 10:
            # 밝기 적응 (Test 데이터가 더 밝음)
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,  # Test 대응
                    contrast_limit=0.2,
                    p=0.8
                )
            )
        
        if self.adaptation_epoch >= 20:
            # 블러 적응 (Test 데이터가 더 흐림)
            transforms.append(
                A.OneOf([
                    A.MotionBlur(blur_limit=4),
                    A.GaussianBlur(blur_limit=4),
                ], p=0.5)
            )
        
        if self.adaptation_epoch >= 30:
            # 노이즈 적응 (Test 데이터가 노이즈 적음)
            transforms.append(
                A.GaussNoise(var_limit=(5, 25), p=0.3)  # 약한 노이즈
            )
        
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)

# 사용 예시
def get_advanced_transforms(target_class: int, epoch: int = 0, image_size: int = 512):
    """
    고급 증강 전략 통합 함수
    
    Args:
        target_class: 클래스 ID (0-16)
        epoch: 현재 에포크 (Domain Adaptation용)
        image_size: 입력 이미지 크기
    """
    
    # 클래스별 맞춤 증강
    class_aug = AdvancedAugmentationStrategy()
    base_transforms = class_aug.get_class_specific_transforms(target_class, image_size)
    
    # Domain Adaptation 추가
    if epoch > 10:  # 일정 에포크 후 적응 시작
        domain_aug = DomainAdaptationAugmentation(epoch)
        domain_transforms = domain_aug.get_domain_adapted_transforms(image_size)
        return domain_transforms
    
    return base_transforms

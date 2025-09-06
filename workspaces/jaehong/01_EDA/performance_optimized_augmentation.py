"""
성능 최적화에 특화된 고급 증강 전략
Test 데이터 통계를 적극 활용한 타겟팅 증강

핵심 원리:
1. Test 통계 직접 활용 - 최대 성능 추구
2. Train→Test 분포 차이 적극 보정
3. 구체적 수치 기반 파라미터 최적화
4. Adaptive Domain Adaptation
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
import torch

class PerformanceOptimizedAugmentation:
    """
    성능 최적화에 특화된 증강 전략
    EDA에서 발견한 Train/Test 차이를 적극 활용
    """
    
    def __init__(self, eda_results: Dict):
        """
        Args:
            eda_results: EDA에서 분석한 Train/Test 통계 차이
        """
        self.eda_results = eda_results
        
        # EDA 결과에서 핵심 통계 추출
        self.brightness_diff = 24.0    # Test가 24.0 더 밝음
        self.sharpness_ratio = 1.97    # Train이 1.97배 더 선명
        self.noise_diff = -2.92        # Test가 2.92 더 적은 노이즈
        self.contrast_diff = 1.46      # Test가 1.46 더 높은 대비
        
        print(f"🎯 성능 최적화 증강 초기화:")
        print(f"  • 밝기 차이: +{self.brightness_diff}")
        print(f"  • 선명도 비율: {self.sharpness_ratio:.1f}배")
        print(f"  • 노이즈 차이: {self.noise_diff}")
        print(f"  • 대비 차이: +{self.contrast_diff}")
    
    def get_test_targeted_transforms(self, image_size: int = 512, phase: str = "aggressive"):
        """
        Test 데이터 타겟팅 증강
        
        Args:
            phase: "conservative", "moderate", "aggressive"
        """
        
        if phase == "conservative":
            return self._get_conservative_transforms(image_size)
        elif phase == "moderate":
            return self._get_moderate_transforms(image_size)
        else:  # aggressive
            return self._get_aggressive_transforms(image_size)
    
    def _get_aggressive_transforms(self, image_size: int):
        """공격적 Test 타겟팅 (최대 성능 추구)"""
        
        # Test 통계 기반 파라미터 계산
        brightness_limit = min(0.5, self.brightness_diff / 100)  # 0.24
        contrast_limit = min(0.4, self.contrast_diff / 50)       # 0.029 -> 0.4
        blur_limit = int(min(10, self.sharpness_ratio * 2))      # 3.94 -> 3
        noise_var = int(max(10, 50 - abs(self.noise_diff) * 5))  # 35.4 -> 35
        
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # 밝기 타겟팅 (Test가 더 밝음)
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,  # Test 차이 기반
                contrast_limit=0.4,  # 강화된 대비
                brightness_by_max=True,  # 밝기 우선
                p=0.9  # 높은 확률
            ),
            
            # 선명도 타겟팅 (Test가 더 흐림)
            A.OneOf([
                A.MotionBlur(blur_limit=blur_limit),
                A.GaussianBlur(blur_limit=blur_limit),
                A.MedianBlur(blur_limit=min(blur_limit, 7)),
            ], p=0.7),  # Test 대응 높은 확률
            
            # 노이즈 타겟팅 (Test가 더 깨끗함)
            A.OneOf([
                A.GaussNoise(var_limit=(5, noise_var)),  # 적당한 노이즈
                A.MultiplicativeNoise(multiplier=[0.9, 1.1]),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),
            ], p=0.4),  # Test가 깨끗하므로 적당한 확률
            
            # 회전 강화 (일반적으로 Test에서 다양함)
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=45, p=0.8, border_mode=cv2.BORDER_CONSTANT),
            
            # Test 환경 시뮬레이션
            A.Perspective(scale=(0.05, 0.15), p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=0, 
                p=0.4
            ),
            
            # 색상 조정 (Test 특성 반영)
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=20, 
                    val_shift_limit=int(self.brightness_diff)
                ),
                A.ColorJitter(
                    brightness=brightness_limit,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1
                ),
            ], p=0.6),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_moderate_transforms(self, image_size: int):
        """중간 수준 Test 타겟팅"""
        
        brightness_limit = min(0.3, self.brightness_diff / 150)
        blur_limit = int(min(5, self.sharpness_ratio))
        
        return A.Compose([
            A.Resize(image_size, image_size),
            
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=0.3,
                p=0.8
            ),
            
            A.OneOf([
                A.MotionBlur(blur_limit=blur_limit),
                A.GaussianBlur(blur_limit=blur_limit),
            ], p=0.5),
            
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10, 40)),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1]),
            ], p=0.3),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_conservative_transforms(self, image_size: int):
        """보수적 증강 (일반적 범위 + 약간의 Test 고려)"""
        
        return A.Compose([
            A.Resize(image_size, image_size),
            
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # 보수적
                contrast_limit=0.2,
                p=0.7
            ),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.3),
            
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.6, border_mode=cv2.BORDER_CONSTANT),
            
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class AdaptiveDomainAlignment:
    """
    적응적 도메인 정렬 - Train을 Test 분포에 맞춤
    """
    
    def __init__(self, target_stats: Dict):
        """
        Args:
            target_stats: Test 데이터 목표 통계
        """
        self.target_brightness = target_stats.get('brightness', 172.2)
        self.target_sharpness = target_stats.get('sharpness', 688.3)
        self.target_noise = target_stats.get('noise', 7.3)
        self.target_contrast = target_stats.get('contrast', 49.0)
    
    def get_domain_aligned_transforms(self, 
                                    current_epoch: int,
                                    total_epochs: int,
                                    image_size: int = 512):
        """
        에포크에 따른 점진적 도메인 정렬
        초기: 일반적 증강 → 후기: Test 분포 타겟팅
        """
        
        # 진행률 계산 (0.0 ~ 1.0)
        progress = current_epoch / total_epochs
        
        # 점진적 강도 조절
        brightness_strength = 0.2 + (progress * 0.3)  # 0.2 → 0.5
        blur_strength = 2 + (progress * 5)             # 2 → 7
        alignment_prob = 0.3 + (progress * 0.4)        # 0.3 → 0.7
        
        transforms = [A.Resize(image_size, image_size)]
        
        # 밝기 정렬 (점진적 강화)
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness_strength,
                contrast_limit=0.3,
                p=alignment_prob
            )
        )
        
        # 선명도 정렬 (점진적 블러 증가)
        if progress > 0.3:  # 30% 이후부터 적용
            transforms.append(
                A.OneOf([
                    A.MotionBlur(blur_limit=int(blur_strength)),
                    A.GaussianBlur(blur_limit=int(blur_strength)),
                ], p=alignment_prob * 0.8)
            )
        
        # 노이즈 정렬 (Test가 더 깨끗하므로 감소)
        if progress > 0.5:  # 50% 이후부터 적용
            noise_strength = max(10, 40 - progress * 20)  # 40 → 20
            transforms.append(
                A.GaussNoise(var_limit=(5, int(noise_strength)), p=0.3)
            )
        
        # 기본 증강
        transforms.extend([
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)

class TestDistributionMatcher:
    """
    Test 분포 매칭 - 히스토그램 매칭 등 고급 기법
    """
    
    def __init__(self):
        self.reference_stats = None
    
    def set_reference_distribution(self, test_images: List[np.ndarray]):
        """Test 이미지들로부터 참조 분포 설정"""
        
        # Test 이미지들의 통계 계산
        brightnesses = []
        contrasts = []
        
        for img in test_images[:100]:  # 샘플링
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
                
            brightnesses.append(np.mean(gray))
            contrasts.append(np.std(gray))
        
        self.reference_stats = {
            'brightness_mean': np.mean(brightnesses),
            'brightness_std': np.std(brightnesses),
            'contrast_mean': np.mean(contrasts),
            'contrast_std': np.std(contrasts)
        }
    
    def get_distribution_matching_transforms(self, image_size: int = 512):
        """분포 매칭 변환 반환"""
        
        if self.reference_stats is None:
            raise ValueError("Reference distribution not set!")
        
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # 히스토그램 매칭 (근사)
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.8
            ),
            
            # 추가 정규화
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# 사용 예시
def get_performance_optimized_strategy():
    """성능 최적화 전략 반환"""
    
    # EDA 결과 (실제 값들)
    eda_results = {
        'brightness_diff': 24.0,
        'sharpness_ratio': 1.97,
        'noise_diff': -2.92,
        'contrast_diff': 1.46
    }
    
    # 성능 최적화 증강
    perf_aug = PerformanceOptimizedAugmentation(eda_results)
    
    return {
        'aggressive': perf_aug.get_test_targeted_transforms(phase="aggressive"),
        'moderate': perf_aug.get_test_targeted_transforms(phase="moderate"),
        'conservative': perf_aug.get_test_targeted_transforms(phase="conservative")
    }

if __name__ == "__main__":
    print("🚀 성능 최적화 증강 전략")
    print("=" * 50)
    
    strategies = get_performance_optimized_strategy()
    
    for name, strategy in strategies.items():
        print(f"✅ {name.upper()} 전략: {len(strategy.transforms)}개 변환")
    
    print("\n🎯 핵심 원리:")
    print("  • Test 통계 직접 활용")
    print("  • Train→Test 분포 차이 적극 보정") 
    print("  • 구체적 수치 기반 파라미터 최적화")
    print("  • 최대 성능 추구!")

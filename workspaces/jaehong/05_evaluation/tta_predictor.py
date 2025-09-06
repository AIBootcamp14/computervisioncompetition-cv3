"""
🔮 Advanced TTA Prediction System
시니어 그랜드마스터 수준의 TTA 예측 시스템

Features:
- 8가지 TTA 전략
- 가중 평균 결합
- 신뢰도 기반 필터링
- GPU 최적화
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
from tqdm import tqdm

# 04_training 모듈 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "04_training"))
from model import DocumentClassifier, ModelFactory


class TTADataset(Dataset):
    """TTA용 데이터셋"""
    
    def __init__(self, image_paths: List[str], transforms: List[A.Compose]):
        """
        Args:
            image_paths: 이미지 경로 리스트
            transforms: TTA 변환 리스트
        """
        self.image_paths = image_paths
        self.transforms = transforms
        self.n_tta = len(transforms)
    
    def __len__(self):
        return len(self.image_paths) * self.n_tta
    
    def __getitem__(self, idx):
        # 실제 이미지 인덱스와 TTA 인덱스 계산
        img_idx = idx // self.n_tta
        tta_idx = idx % self.n_tta
        
        # 이미지 로드
        image_path = self.image_paths[img_idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # TTA 변환 적용
        transformed = self.transforms[tta_idx](image=image)
        
        return {
            'image': transformed['image'],
            'img_idx': img_idx,
            'tta_idx': tta_idx,
            'image_path': str(image_path)
        }


class TTATransformFactory:
    """TTA 변환 팩토리"""
    
    @staticmethod
    def create_tta_transforms(image_size: int = 512) -> List[A.Compose]:
        """8가지 TTA 변환 생성"""
        
        # 기본 정규화
        normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transforms = []
        
        # 1. 원본
        transforms.append(A.Compose([
            A.Resize(image_size, image_size),
            normalize,
            ToTensorV2()
        ]))
        
        # 2. 수평 뒤집기
        transforms.append(A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            normalize,
            ToTensorV2()
        ]))
        
        # 3. 5도 회전
        transforms.append(A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(5, 5), p=1.0),
            normalize,
            ToTensorV2()
        ]))
        
        # 4. -5도 회전  
        transforms.append(A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(-5, -5), p=1.0),
            normalize,
            ToTensorV2()
        ]))
        
        # 5. 밝기 증가
        transforms.append(A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0),
            normalize,
            ToTensorV2()
        ]))
        
        # 6. 밝기 감소
        transforms.append(A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=0, p=1.0),
            normalize,
            ToTensorV2()
        ]))
        
        # 7. 대비 증가
        transforms.append(A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.1, 0.1), p=1.0),
            normalize,
            ToTensorV2()
        ]))
        
        # 8. 대비 감소
        transforms.append(A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.1, -0.1), p=1.0),
            normalize,
            ToTensorV2()
        ]))
        
        return transforms


class TTAPredictor:
    """고급 TTA 예측기"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        tta_weights: Optional[List[float]] = None
    ):
        """
        Args:
            model: 예측 모델
            device: 디바이스
            tta_weights: TTA별 가중치
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # TTA 가중치 (원본에 더 높은 가중치)
        if tta_weights is None:
            self.tta_weights = [0.3, 0.15, 0.1, 0.1, 0.1, 0.1, 0.075, 0.075]
        else:
            self.tta_weights = tta_weights
        
        # 가중치 정규화
        total_weight = sum(self.tta_weights)
        self.tta_weights = [w / total_weight for w in self.tta_weights]
        
        print(f"🔮 TTA 예측기 초기화:")
        print(f"   디바이스: {device}")
        print(f"   TTA 수: {len(self.tta_weights)}")
        print(f"   가중치: {[f'{w:.3f}' for w in self.tta_weights]}")
    
    def predict_with_tta(
        self,
        image_paths: List[str],
        image_size: int = 512,
        batch_size: int = 32,
        confidence_threshold: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        TTA를 사용한 예측
        
        Args:
            image_paths: 이미지 경로 리스트
            image_size: 이미지 크기
            batch_size: 배치 크기
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            predictions: 예측 클래스
            probabilities: 예측 확률
            stats: 통계 정보
        """
        print(f"🔮 TTA 예측 시작:")
        print(f"   이미지 수: {len(image_paths)}")
        print(f"   TTA 수: {len(self.tta_weights)}")
        print(f"   총 예측: {len(image_paths) * len(self.tta_weights)}")
        
        # TTA 변환 생성
        tta_transforms = TTATransformFactory.create_tta_transforms(image_size)
        
        # TTA 데이터셋 생성
        tta_dataset = TTADataset(image_paths, tta_transforms)
        tta_loader = DataLoader(
            tta_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 예측 수행
        all_predictions = []
        all_image_indices = []
        all_tta_indices = []
        
        with torch.no_grad():
            for batch in tqdm(tta_loader, desc="TTA 예측"):
                images = batch['image'].to(self.device, non_blocking=True)
                img_indices = batch['img_idx'].cpu().numpy()
                tta_indices = batch['tta_idx'].cpu().numpy()
                
                # 모델 예측
                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
                
                all_predictions.append(probs.cpu().numpy())
                all_image_indices.extend(img_indices)
                all_tta_indices.extend(tta_indices)
        
        # 예측 결합
        all_predictions = np.vstack(all_predictions)
        
        # 이미지별로 TTA 결과 집계
        final_probabilities = np.zeros((len(image_paths), all_predictions.shape[1]))
        confidence_scores = np.zeros(len(image_paths))
        
        for i in range(len(image_paths)):
            # 해당 이미지의 모든 TTA 예측 수집
            mask = np.array(all_image_indices) == i
            image_predictions = all_predictions[mask]
            image_tta_indices = np.array(all_tta_indices)[mask]
            
            # 가중 평균 계산
            weighted_probs = np.zeros_like(image_predictions[0])
            for pred, tta_idx in zip(image_predictions, image_tta_indices):
                weight = self.tta_weights[tta_idx]
                weighted_probs += pred * weight
            
            final_probabilities[i] = weighted_probs
            
            # 신뢰도 계산 (최대 확률)
            confidence_scores[i] = np.max(weighted_probs)
        
        # 최종 예측 클래스
        final_predictions = np.argmax(final_probabilities, axis=1)
        
        # 신뢰도 필터링
        if confidence_threshold > 0:
            low_confidence_mask = confidence_scores < confidence_threshold
            low_confidence_count = np.sum(low_confidence_mask)
            print(f"⚠️ 낮은 신뢰도 예측: {low_confidence_count}개 (임계값: {confidence_threshold})")
        
        # 통계 정보
        stats = {
            'total_images': len(image_paths),
            'total_tta_predictions': len(all_predictions),
            'mean_confidence': float(np.mean(confidence_scores)),
            'std_confidence': float(np.std(confidence_scores)),
            'min_confidence': float(np.min(confidence_scores)),
            'max_confidence': float(np.max(confidence_scores)),
            'low_confidence_count': int(np.sum(confidence_scores < confidence_threshold)) if confidence_threshold > 0 else 0,
            'tta_weights': self.tta_weights,
            'class_distribution': {
                int(cls): int(count) for cls, count in 
                zip(*np.unique(final_predictions, return_counts=True))
            }
        }
        
        print(f"✅ TTA 예측 완료:")
        print(f"   평균 신뢰도: {stats['mean_confidence']:.4f}")
        print(f"   신뢰도 범위: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")
        
        return final_predictions, final_probabilities, stats
    
    def predict_single_image(
        self,
        image_path: str,
        image_size: int = 512
    ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """단일 이미지 TTA 예측"""
        
        predictions, probabilities, stats = self.predict_with_tta(
            [image_path], image_size, batch_size=8
        )
        
        return predictions[0], probabilities[0], {
            'confidence': stats['mean_confidence'],
            'tta_weights': self.tta_weights
        }
    
    def analyze_tta_contribution(
        self,
        image_paths: List[str],
        sample_size: int = 100,
        image_size: int = 512
    ) -> Dict[str, Any]:
        """TTA 기여도 분석"""
        
        if len(image_paths) > sample_size:
            # 샘플링
            indices = np.random.choice(len(image_paths), sample_size, replace=False)
            sample_paths = [image_paths[i] for i in indices]
        else:
            sample_paths = image_paths
        
        print(f"🔍 TTA 기여도 분석 (샘플: {len(sample_paths)}개)")
        
        # 각 TTA별 개별 예측
        tta_transforms = TTATransformFactory.create_tta_transforms(image_size)
        tta_names = ["original", "h_flip", "rotate_5", "rotate_-5", 
                    "bright_up", "bright_down", "contrast_up", "contrast_down"]
        
        tta_results = {}
        
        for tta_idx, (transform, name) in enumerate(zip(tta_transforms, tta_names)):
            dataset = TTADataset(sample_paths, [transform])
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            
            predictions = []
            confidences = []
            
            with torch.no_grad():
                for batch in loader:
                    images = batch['image'].to(self.device)
                    logits = self.model(images)
                    probs = F.softmax(logits, dim=1)
                    
                    pred_classes = torch.argmax(probs, dim=1)
                    pred_confidences = torch.max(probs, dim=1)[0]
                    
                    predictions.extend(pred_classes.cpu().numpy())
                    confidences.extend(pred_confidences.cpu().numpy())
            
            tta_results[name] = {
                'predictions': predictions,
                'mean_confidence': float(np.mean(confidences)),
                'weight': self.tta_weights[tta_idx]
            }
        
        # 일치율 분석
        original_preds = tta_results['original']['predictions']
        agreement_rates = {}
        
        for name, result in tta_results.items():
            if name != 'original':
                agreement = np.mean(np.array(result['predictions']) == np.array(original_preds))
                agreement_rates[name] = float(agreement)
        
        analysis = {
            'tta_results': tta_results,
            'agreement_with_original': agreement_rates,
            'best_tta_by_confidence': max(tta_results.keys(), 
                                        key=lambda x: tta_results[x]['mean_confidence']),
            'sample_size': len(sample_paths)
        }
        
        print(f"✅ TTA 분석 완료:")
        print(f"   최고 신뢰도 TTA: {analysis['best_tta_by_confidence']}")
        print(f"   원본과 평균 일치율: {np.mean(list(agreement_rates.values())):.3f}")
        
        return analysis


# 사용 예시
if __name__ == "__main__":
    print("🔮 TTA 예측기 테스트는 main_evaluation.py에서 실행됩니다.")

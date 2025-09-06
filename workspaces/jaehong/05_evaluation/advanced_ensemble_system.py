"""
🎭 Advanced Ensemble System
클린 코드와 클린 아키텍처를 적용한 고급 앙상블 시스템

Features:
- 다중 모델 앙상블 (Voting, Averaging, Stacking)
- 고급 TTA (Test Time Augmentation) 전략
- 동적 가중치 계산
- 불확실성 추정
- 메모리 효율적인 구현
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import warnings
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import pickle

warnings.filterwarnings('ignore')


class BaseEnsemble(ABC):
    """
    앙상블의 추상 기반 클래스
    클린 아키텍처 원칙: 의존성 역전
    """
    
    def __init__(self, models: List[nn.Module], device: str = "cuda"):
        self.models = models
        self.device = device
        self.model_weights = None
        
        # 모델들을 평가 모드로 설정
        for model in self.models:
            model.eval()
            model.to(device)
    
    @abstractmethod
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """예측 추상 메서드"""
        pass
    
    @abstractmethod
    def predict_proba(self, data_loader) -> np.ndarray:
        """확률 예측 추상 메서드"""
        pass


class VotingEnsemble(BaseEnsemble):
    """
    투표 기반 앙상블
    각 모델의 예측을 투표로 결합
    """
    
    def __init__(
        self, 
        models: List[nn.Module], 
        weights: Optional[List[float]] = None,
        device: str = "cuda"
    ):
        super().__init__(models, device)
        self.weights = weights or [1.0] * len(models)
        
        # 가중치 정규화
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"🗳️ 투표 앙상블 초기화: {len(models)}개 모델")
        print(f"   가중치: {[f'{w:.3f}' for w in self.weights]}")
    
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """투표 기반 예측"""
        all_predictions = []
        all_probabilities = []
        
        print(f"🔮 투표 앙상블 예측 중...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['image'].to(self.device)
                batch_predictions = []
                batch_probabilities = []
                
                # 각 모델의 예측 수집
                for model_idx, model in enumerate(self.models):
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    
                    batch_predictions.append(outputs.cpu().numpy())
                    batch_probabilities.append(probs.cpu().numpy())
                
                # 가중 평균으로 결합
                weighted_probs = np.zeros_like(batch_probabilities[0])
                for i, (probs, weight) in enumerate(zip(batch_probabilities, self.weights)):
                    weighted_probs += probs * weight
                
                predictions = np.argmax(weighted_probs, axis=1)
                
                all_predictions.extend(predictions)
                all_probabilities.append(weighted_probs)
                
                if batch_idx % 50 == 0:
                    print(f"   진행률: {batch_idx+1}/{len(data_loader)} 배치")
        
        final_probabilities = np.vstack(all_probabilities)
        
        print(f"✅ 투표 앙상블 예측 완료!")
        return np.array(all_predictions), final_probabilities
    
    def predict_proba(self, data_loader) -> np.ndarray:
        """확률 예측"""
        _, probabilities = self.predict(data_loader)
        return probabilities


class StackingEnsemble(BaseEnsemble):
    """
    스태킹 앙상블
    메타 러너를 사용하여 모델들의 예측을 결합
    """
    
    def __init__(
        self, 
        models: List[nn.Module], 
        meta_learner=None,
        device: str = "cuda"
    ):
        super().__init__(models, device)
        self.meta_learner = meta_learner or LogisticRegression(
            random_state=42, 
            max_iter=1000,
            multi_class='ovr'
        )
        self.is_fitted = False
        
        print(f"🏗️ 스태킹 앙상블 초기화: {len(models)}개 모델")
        print(f"   메타 러너: {type(self.meta_learner).__name__}")
    
    def fit_meta_learner(self, val_loader, val_targets: np.ndarray):
        """
        메타 러너 훈련
        검증 데이터로 각 모델의 예측을 생성하고 메타 러너 훈련
        """
        print(f"🎓 메타 러너 훈련 중...")
        
        # 각 모델의 예측 수집
        meta_features = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                batch_meta_features = []
                
                for model in self.models:
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    batch_meta_features.append(probs.cpu().numpy())
                
                # 모델별 예측을 연결하여 메타 특성 생성
                batch_meta = np.concatenate(batch_meta_features, axis=1)
                meta_features.append(batch_meta)
        
        X_meta = np.vstack(meta_features)
        
        # 메타 러너 훈련
        self.meta_learner.fit(X_meta, val_targets)
        self.is_fitted = True
        
        # 성능 검증
        meta_predictions = self.meta_learner.predict(X_meta)
        meta_accuracy = accuracy_score(val_targets, meta_predictions)
        meta_f1 = f1_score(val_targets, meta_predictions, average='macro')
        
        print(f"✅ 메타 러너 훈련 완료!")
        print(f"   메타 러너 정확도: {meta_accuracy:.4f}")
        print(f"   메타 러너 F1: {meta_f1:.4f}")
    
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """스태킹 기반 예측"""
        if not self.is_fitted:
            raise ValueError("메타 러너가 훈련되지 않았습니다. fit_meta_learner()를 먼저 호출하세요.")
        
        print(f"🔮 스태킹 앙상블 예측 중...")
        
        meta_features = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['image'].to(self.device)
                batch_meta_features = []
                
                for model in self.models:
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    batch_meta_features.append(probs.cpu().numpy())
                
                batch_meta = np.concatenate(batch_meta_features, axis=1)
                meta_features.append(batch_meta)
                
                if batch_idx % 50 == 0:
                    print(f"   진행률: {batch_idx+1}/{len(data_loader)} 배치")
        
        X_meta = np.vstack(meta_features)
        
        # 메타 러너로 최종 예측
        predictions = self.meta_learner.predict(X_meta)
        probabilities = self.meta_learner.predict_proba(X_meta)
        
        print(f"✅ 스태킹 앙상블 예측 완료!")
        return predictions, probabilities
    
    def predict_proba(self, data_loader) -> np.ndarray:
        """확률 예측"""
        _, probabilities = self.predict(data_loader)
        return probabilities


class AdvancedTTAEnsemble:
    """
    고급 TTA (Test Time Augmentation) 앙상블
    다양한 증강과 모델을 결합
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        tta_transforms: List,
        tta_weights: Optional[List[float]] = None,
        device: str = "cuda"
    ):
        self.model = model.to(device).eval()
        self.tta_transforms = tta_transforms
        self.device = device
        
        # TTA 가중치 설정
        if tta_weights is None:
            self.tta_weights = [1.0] * len(tta_transforms)
        else:
            self.tta_weights = tta_weights
        
        # 가중치 정규화
        total_weight = sum(self.tta_weights)
        self.tta_weights = [w / total_weight for w in self.tta_weights]
        
        print(f"🔄 고급 TTA 앙상블 초기화:")
        print(f"   TTA 변환 수: {len(tta_transforms)}")
        print(f"   TTA 가중치: {[f'{w:.3f}' for w in self.tta_weights]}")
    
    def predict_with_tta(
        self, 
        image_paths: List[str], 
        image_size: int = 512,
        batch_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        TTA를 사용한 예측
        각 이미지에 대해 여러 증강을 적용하고 결과를 평균
        """
        print(f"🔮 고급 TTA 예측 중... ({len(image_paths)}개 이미지)")
        
        all_predictions = []
        all_probabilities = []
        confidence_scores = []
        
        # 이미지별 TTA 예측
        for img_idx, img_path in enumerate(image_paths):
            tta_probs = []
            
            # 이미지 로드
            import cv2
            image = cv2.imread(img_path)
            if image is None:
                # 기본 이미지 사용
                image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 128
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 각 TTA 변환 적용
            with torch.no_grad():
                for transform_idx, (transform, weight) in enumerate(zip(self.tta_transforms, self.tta_weights)):
                    # 변환 적용
                    augmented = transform(image=image)
                    tensor_image = augmented['image'].unsqueeze(0).to(self.device)
                    
                    # 예측
                    outputs = self.model(tensor_image)
                    probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                    
                    tta_probs.append(probs * weight)
            
            # TTA 결과 평균
            final_probs = np.sum(tta_probs, axis=0)
            prediction = np.argmax(final_probs)
            confidence = np.max(final_probs)
            
            all_predictions.append(prediction)
            all_probabilities.append(final_probs)
            confidence_scores.append(confidence)
            
            if (img_idx + 1) % 100 == 0:
                print(f"   진행률: {img_idx+1}/{len(image_paths)} 이미지")
        
        # 통계 계산
        stats = {
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores),
            'high_confidence_ratio': np.mean(np.array(confidence_scores) > 0.8)
        }
        
        print(f"✅ 고급 TTA 예측 완료!")
        print(f"   평균 신뢰도: {stats['mean_confidence']:.4f}")
        print(f"   높은 신뢰도 비율: {stats['high_confidence_ratio']:.2%}")
        
        return (
            np.array(all_predictions),
            np.array(all_probabilities),
            stats
        )


class EnsembleManager:
    """
    앙상블 관리자
    다양한 앙상블 전략을 통합 관리
    """
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.models = {}
        self.ensembles = {}
        
        print(f"🎭 앙상블 관리자 초기화")
        print(f"   워크스페이스: {workspace_root}")
    
    def load_model(
        self, 
        model_name: str, 
        model_path: str, 
        model_config: Dict[str, Any],
        device: str = "cuda"
    ) -> nn.Module:
        """
        모델 로드 및 등록
        """
        print(f"📥 모델 로드 중: {model_name}")
        
        # 모델 아키텍처에 따른 동적 임포트
        try:
            from ..03_modeling.improved_model_factory import ImprovedModelFactory
            
            model = ImprovedModelFactory.create_model(
                architecture=model_config.get('architecture', 'efficientnet_b3'),
                num_classes=model_config.get('num_classes', 17),
                pretrained=False,  # 체크포인트에서 로드
                **model_config
            )
            
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device).eval()
            self.models[model_name] = model
            
            print(f"✅ 모델 로드 완료: {model_name}")
            return model
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {model_name}, 오류: {e}")
            return None
    
    def create_voting_ensemble(
        self, 
        model_names: List[str], 
        weights: Optional[List[float]] = None,
        ensemble_name: str = "voting_ensemble"
    ) -> VotingEnsemble:
        """투표 앙상블 생성"""
        models = [self.models[name] for name in model_names if name in self.models]
        
        if len(models) == 0:
            raise ValueError("유효한 모델이 없습니다.")
        
        ensemble = VotingEnsemble(models, weights)
        self.ensembles[ensemble_name] = ensemble
        
        return ensemble
    
    def create_stacking_ensemble(
        self,
        model_names: List[str],
        meta_learner=None,
        ensemble_name: str = "stacking_ensemble"
    ) -> StackingEnsemble:
        """스태킹 앙상블 생성"""
        models = [self.models[name] for name in model_names if name in self.models]
        
        if len(models) == 0:
            raise ValueError("유효한 모델이 없습니다.")
        
        ensemble = StackingEnsemble(models, meta_learner)
        self.ensembles[ensemble_name] = ensemble
        
        return ensemble
    
    def create_tta_ensemble(
        self,
        model_name: str,
        tta_transforms: List,
        tta_weights: Optional[List[float]] = None,
        ensemble_name: str = "tta_ensemble"
    ) -> AdvancedTTAEnsemble:
        """TTA 앙상블 생성"""
        if model_name not in self.models:
            raise ValueError(f"모델을 찾을 수 없습니다: {model_name}")
        
        ensemble = AdvancedTTAEnsemble(
            self.models[model_name],
            tta_transforms,
            tta_weights
        )
        self.ensembles[ensemble_name] = ensemble
        
        return ensemble
    
    def evaluate_ensemble(
        self,
        ensemble_name: str,
        val_loader,
        val_targets: np.ndarray
    ) -> Dict[str, float]:
        """앙상블 성능 평가"""
        if ensemble_name not in self.ensembles:
            raise ValueError(f"앙상블을 찾을 수 없습니다: {ensemble_name}")
        
        print(f"📊 앙상블 평가 중: {ensemble_name}")
        
        ensemble = self.ensembles[ensemble_name]
        
        if isinstance(ensemble, AdvancedTTAEnsemble):
            # TTA 앙상블은 다른 방식으로 평가
            print("⚠️ TTA 앙상블은 별도 평가 방법을 사용하세요.")
            return {}
        
        predictions, probabilities = ensemble.predict(val_loader)
        
        # 메트릭 계산
        accuracy = accuracy_score(val_targets, predictions)
        f1_macro = f1_score(val_targets, predictions, average='macro')
        f1_weighted = f1_score(val_targets, predictions, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confidence_mean': np.mean(np.max(probabilities, axis=1)),
            'confidence_std': np.std(np.max(probabilities, axis=1))
        }
        
        print(f"✅ 앙상블 평가 완료:")
        print(f"   정확도: {accuracy:.4f}")
        print(f"   F1 (Macro): {f1_macro:.4f}")
        print(f"   F1 (Weighted): {f1_weighted:.4f}")
        
        return results
    
    def save_ensemble_config(self, ensemble_name: str, config_path: str):
        """앙상블 설정 저장"""
        if ensemble_name not in self.ensembles:
            raise ValueError(f"앙상블을 찾을 수 없습니다: {ensemble_name}")
        
        config = {
            'ensemble_name': ensemble_name,
            'ensemble_type': type(self.ensembles[ensemble_name]).__name__,
            'model_count': len(self.ensembles[ensemble_name].models) if hasattr(self.ensembles[ensemble_name], 'models') else 1,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        print(f"💾 앙상블 설정 저장: {config_path}")


# 사용 예시 및 테스트
if __name__ == "__main__":
    print("🎭 고급 앙상블 시스템 테스트")
    
    # 앙상블 관리자 생성
    manager = EnsembleManager("/home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong")
    
    # 예시 모델 설정들
    model_configs = {
        'efficientnet_b3': {
            'architecture': 'efficientnet_b3',
            'num_classes': 17,
            'dropout_rate': 0.4
        },
        'efficientnet_b4': {
            'architecture': 'efficientnet_b4', 
            'num_classes': 17,
            'dropout_rate': 0.4
        },
        'convnext_base': {
            'architecture': 'convnext_base',
            'num_classes': 17,
            'dropout_rate': 0.4
        }
    }
    
    print(f"✅ 고급 앙상블 시스템 초기화 완료!")
    print(f"   지원 앙상블 타입: Voting, Stacking, TTA")
    print(f"   모델 설정: {len(model_configs)}개 준비")

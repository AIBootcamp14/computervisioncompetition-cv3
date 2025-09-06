
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from collections import defaultdict, Counter

class PseudoLabelingManager:
    """
    Progressive Pseudo Labeling 관리자
    EDA 분석 결과를 반영한 지능형 pseudo labeling
    """
    
    def __init__(self, 
                 num_classes: int = 17,
                 confidence_thresholds: Dict = {'conservative': 0.95, 'moderate': 0.9, 'aggressive': 0.85},
                 class_weights: Optional[Dict] = None):
        
        self.num_classes = num_classes
        self.confidence_thresholds = confidence_thresholds
        self.class_weights = class_weights or {}
        
        # Progressive schedule
        self.progressive_schedule = {'phase_1': {'epochs': '1-10', 'threshold': 0.95, 'max_pseudo_ratio': 0.1, 'strategy': 'Only highest confidence'}, 'phase_2': {'epochs': '11-20', 'threshold': 0.9, 'max_pseudo_ratio': 0.2, 'strategy': 'Class-balanced selection'}, 'phase_3': {'epochs': '21-30', 'threshold': 0.85, 'max_pseudo_ratio': 0.3, 'strategy': 'Uncertainty-based selection'}}
        
        # Quality control
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.pseudo_history = defaultdict(list)
        
    def get_current_phase(self, epoch: int) -> Dict:
        """현재 에포크에 따른 phase 정보 반환"""
        if epoch <= 10:
            return self.progressive_schedule['phase_1']
        elif epoch <= 20:
            return self.progressive_schedule['phase_2']
        else:
            return self.progressive_schedule['phase_3']
    
    def select_pseudo_labels(self, 
                           predictions: torch.Tensor,
                           confidences: torch.Tensor,
                           features: torch.Tensor,
                           epoch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pseudo label 선택
        
        Args:
            predictions: 모델 예측 (N, num_classes)
            confidences: 예측 신뢰도 (N,)
            features: 특성 벡터 (N, feature_dim)
            epoch: 현재 에포크
            
        Returns:
            selected_indices: 선택된 샘플 인덱스
            pseudo_labels: Pseudo label
        """
        
        phase = self.get_current_phase(epoch)
        threshold = phase['threshold']
        max_ratio = phase['max_pseudo_ratio']
        
        # 1. 신뢰도 기반 필터링
        high_conf_mask = confidences > threshold
        
        if not high_conf_mask.any():
            return torch.tensor([]), torch.tensor([])
        
        # 2. 이상치 제거
        if len(features[high_conf_mask]) > 10:  # 최소 샘플 수 확보
            outlier_mask = self._detect_outliers(features[high_conf_mask])
            high_conf_mask[high_conf_mask.clone()] = ~outlier_mask
        
        # 3. 클래스 균형 고려 선택
        selected_indices = self._balanced_selection(
            predictions[high_conf_mask],
            confidences[high_conf_mask], 
            high_conf_mask.nonzero().squeeze(),
            max_ratio,
            phase['strategy']
        )
        
        if len(selected_indices) == 0:
            return torch.tensor([]), torch.tensor([])
        
        pseudo_labels = predictions[selected_indices].argmax(dim=1)
        
        # 4. 품질 기록
        self._record_pseudo_quality(selected_indices, pseudo_labels, confidences[selected_indices])
        
        return selected_indices, pseudo_labels
    
    def _detect_outliers(self, features: torch.Tensor) -> torch.Tensor:
        """Isolation Forest를 사용한 이상치 탐지"""
        features_np = features.detach().cpu().numpy()
        outlier_pred = self.isolation_forest.fit_predict(features_np)
        return torch.tensor(outlier_pred == -1)  # -1이 이상치
    
    def _balanced_selection(self, 
                          predictions: torch.Tensor,
                          confidences: torch.Tensor,
                          indices: torch.Tensor,
                          max_ratio: float,
                          strategy: str) -> torch.Tensor:
        """클래스 균형을 고려한 선택"""
        
        max_samples = int(len(predictions) * max_ratio)
        pred_classes = predictions.argmax(dim=1)
        
        if strategy == "Only highest confidence":
            # 단순히 신뢰도 높은 순으로
            _, top_indices = confidences.topk(min(max_samples, len(confidences)))
            return indices[top_indices]
            
        elif strategy == "Class-balanced selection":
            # 클래스별 균등 선택
            selected = []
            samples_per_class = max_samples // self.num_classes
            
            for class_id in range(self.num_classes):
                class_mask = pred_classes == class_id
                if not class_mask.any():
                    continue
                    
                class_confidences = confidences[class_mask]
                class_indices = indices[class_mask]
                
                # 해당 클래스에서 가장 신뢰도 높은 샘플들 선택
                num_select = min(samples_per_class, len(class_confidences))
                _, top_class_indices = class_confidences.topk(num_select)
                selected.extend(class_indices[top_class_indices].tolist())
            
            return torch.tensor(selected)
            
        elif strategy == "Uncertainty-based selection":
            # 불확실성 기반 선택 (entropy 사용)
            entropies = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
            
            # 신뢰도와 불확실성의 균형
            combined_score = confidences - 0.1 * entropies  # 불확실성 페널티
            _, selected_indices = combined_score.topk(min(max_samples, len(combined_score)))
            
            return indices[selected_indices]
        
        return torch.tensor([])
    
    def _record_pseudo_quality(self, indices: torch.Tensor, labels: torch.Tensor, confidences: torch.Tensor):
        """Pseudo label 품질 기록"""
        for idx, label, conf in zip(indices, labels, confidences):
            self.pseudo_history[int(idx)].append({
                'label': int(label),
                'confidence': float(conf),
                'timestamp': len(self.pseudo_history[int(idx)])
            })
    
    def get_pseudo_statistics(self) -> Dict:
        """Pseudo labeling 통계 반환"""
        if not self.pseudo_history:
            return {'total_pseudo_samples': 0}
        
        total_samples = len(self.pseudo_history)
        label_distribution = Counter()
        avg_confidence = 0
        
        for sample_history in self.pseudo_history.values():
            if sample_history:
                latest = sample_history[-1]
                label_distribution[latest['label']] += 1
                avg_confidence += latest['confidence']
        
        avg_confidence /= total_samples if total_samples > 0 else 1
        
        return {
            'total_pseudo_samples': total_samples,
            'label_distribution': dict(label_distribution),
            'average_confidence': avg_confidence,
            'class_balance_ratio': max(label_distribution.values()) / min(label_distribution.values()) if label_distribution else 0
        }

class EnsemblePseudoLabeling:
    """앙상블 기반 고품질 Pseudo Labeling"""
    
    def __init__(self, models: List[nn.Module], agreement_threshold: int = 3):
        self.models = models
        self.agreement_threshold = agreement_threshold
        
    def get_ensemble_pseudo_labels(self, 
                                 dataloader,
                                 device: torch.device) -> Tuple[List, List, List]:
        """
        앙상블 합의 기반 pseudo label 생성
        
        Returns:
            pseudo_data: 선택된 데이터
            pseudo_labels: 합의된 라벨  
            confidence_scores: 신뢰도 점수
        """
        
        all_predictions = []
        all_data = []
        
        # 각 모델의 예측 수집
        for model in self.models:
            model.eval()
            predictions = []
            data_batch = []
            
            with torch.no_grad():
                for batch_data, _ in dataloader:
                    batch_data = batch_data.to(device)
                    outputs = model(batch_data)
                    predictions.append(F.softmax(outputs, dim=1))
                    data_batch.append(batch_data)
            
            all_predictions.append(torch.cat(predictions, dim=0))
            if not all_data:  # 첫 번째 모델에서만 데이터 저장
                all_data = torch.cat(data_batch, dim=0)
        
        # 앙상블 합의 확인
        ensemble_preds = torch.stack(all_predictions)  # (num_models, num_samples, num_classes)
        pred_labels = ensemble_preds.argmax(dim=2)  # (num_models, num_samples)
        
        pseudo_data, pseudo_labels, confidence_scores = [], [], []
        
        for i in range(pred_labels.shape[1]):  # 각 샘플에 대해
            sample_preds = pred_labels[:, i]
            
            # 합의 확인 (과반수 이상 동일 예측)
            label_counts = torch.bincount(sample_preds, minlength=17)
            max_count = label_counts.max()
            
            if max_count >= self.agreement_threshold:
                agreed_label = label_counts.argmax()
                
                # 해당 라벨에 대한 평균 신뢰도
                confidence = ensemble_preds[:, i, agreed_label].mean()
                
                pseudo_data.append(all_data[i])
                pseudo_labels.append(agreed_label)
                confidence_scores.append(confidence)
        
        return pseudo_data, pseudo_labels, confidence_scores

# 사용 예시
def setup_pseudo_labeling(num_classes: int = 17, class_weights: Dict = None):
    """Pseudo Labeling 설정"""
    
    # EDA에서 계산된 클래스 가중치 사용
    if class_weights is None:
        class_weights = {}
    
    pseudo_manager = PseudoLabelingManager(
        num_classes=num_classes,
        class_weights=class_weights
    )
    
    return pseudo_manager

def progressive_training_with_pseudo_labels(model, 
                                          train_loader,
                                          pseudo_manager: PseudoLabelingManager,
                                          epoch: int):
    """Progressive Pseudo Labeling을 적용한 훈련"""
    
    # 현재 phase 확인
    phase = pseudo_manager.get_current_phase(epoch)
    print(f"Epoch {epoch}: {phase['strategy']} (threshold={phase['threshold']})")
    
    # Pseudo label 선택 및 훈련에 적용하는 로직은
    # 실제 훈련 루프에서 구현
    
    return phase

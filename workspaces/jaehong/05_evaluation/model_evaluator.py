"""
📊 Advanced Model Evaluation System
시니어 그랜드마스터 수준의 모델 평가 시스템

Features:
- 종합적인 성능 분석
- 클래스별 상세 분석
- 혼동 행렬 시각화
- 오분류 사례 분석
- 신뢰도 분석
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import json
from PIL import Image
import cv2
from tqdm import tqdm

# 04_training 모듈 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "04_training"))
from model import DocumentClassifier, ModelFactory


class ModelEvaluator:
    """고급 모델 평가기"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        class_weights: Optional[Dict[int, float]] = None,
        device: str = "cuda"
    ):
        """
        Args:
            model: 평가할 모델
            class_names: 클래스 이름 리스트
            class_weights: 클래스 가중치
            device: 디바이스
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.class_weights = class_weights or {}
        self.model.eval()
        
        print(f"📊 모델 평가기 초기화:")
        print(f"   클래스 수: {self.num_classes}")
        print(f"   디바이스: {device}")
        print(f"   클래스 가중치: {'적용' if class_weights else '없음'}")
    
    def evaluate_comprehensive(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_dir: Path,
        experiment_name: str = "evaluation"
    ) -> Dict[str, Any]:
        """종합적인 모델 평가"""
        
        print(f"🔍 종합 평가 시작...")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 예측 수행
        predictions, targets, probabilities, image_paths = self._predict_all(data_loader)
        
        # 기본 메트릭 계산
        basic_metrics = self._calculate_basic_metrics(targets, predictions, probabilities)
        
        # 클래스별 분석
        class_analysis = self._analyze_by_class(targets, predictions, probabilities)
        
        # 혼동 행렬 생성
        confusion_results = self._create_confusion_matrix(targets, predictions, save_dir)
        
        # 신뢰도 분석
        confidence_analysis = self._analyze_confidence(probabilities, targets, predictions, save_dir)
        
        # 오분류 분석
        misclassification_analysis = self._analyze_misclassifications(
            targets, predictions, probabilities, image_paths, save_dir
        )
        
        # ROC/PR 곡선 (다중 클래스)
        roc_pr_analysis = self._create_roc_pr_curves(targets, probabilities, save_dir)
        
        # 클래스 불균형 분석
        imbalance_analysis = self._analyze_class_imbalance(targets, predictions)
        
        # 종합 결과
        comprehensive_results = {
            'experiment_name': experiment_name,
            'basic_metrics': basic_metrics,
            'class_analysis': class_analysis,
            'confusion_matrix': confusion_results,
            'confidence_analysis': confidence_analysis,
            'misclassification_analysis': misclassification_analysis,
            'roc_pr_analysis': roc_pr_analysis,
            'imbalance_analysis': imbalance_analysis,
            'sample_count': len(targets)
        }
        
        # 결과 저장
        self._save_evaluation_results(comprehensive_results, save_dir)
        
        # 요약 리포트 생성
        self._generate_summary_report(comprehensive_results, save_dir)
        
        print(f"✅ 종합 평가 완료:")
        print(f"   정확도: {basic_metrics['accuracy']:.4f}")
        print(f"   Macro F1: {basic_metrics['macro_f1']:.4f}")
        print(f"   Weighted F1: {basic_metrics['weighted_f1']:.4f}")
        
        return comprehensive_results
    
    def _predict_all(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """모든 데이터에 대해 예측 수행"""
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_image_paths = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="예측 중"):
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch.get('target', torch.zeros(images.size(0), dtype=torch.long))
                image_paths = batch.get('image_id', [''] * images.size(0))
                
                # 모델 예측
                logits = self.model(images)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # 결과 수집
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_image_paths.extend(image_paths)
        
        return (
            np.array(all_predictions),
            np.array(all_targets),
            np.vstack(all_probabilities),
            all_image_paths
        )
    
    def _calculate_basic_metrics(
        self, 
        targets: np.ndarray, 
        predictions: np.ndarray, 
        probabilities: np.ndarray
    ) -> Dict[str, float]:
        """기본 메트릭 계산"""
        
        return {
            'accuracy': float(accuracy_score(targets, predictions)),
            'macro_f1': float(f1_score(targets, predictions, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(targets, predictions, average='weighted', zero_division=0)),
            'macro_precision': float(precision_score(targets, predictions, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(targets, predictions, average='macro', zero_division=0)),
            'weighted_precision': float(precision_score(targets, predictions, average='weighted', zero_division=0)),
            'weighted_recall': float(recall_score(targets, predictions, average='weighted', zero_division=0))
        }
    
    def _analyze_by_class(
        self,
        targets: np.ndarray,
        predictions: np.ndarray, 
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """클래스별 상세 분석"""
        
        # 클래스별 메트릭
        per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
        per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)
        
        # 클래스별 지원 샘플 수
        unique_targets, target_counts = np.unique(targets, return_counts=True)
        support_dict = dict(zip(unique_targets, target_counts))
        
        # 클래스별 평균 신뢰도
        class_confidences = {}
        for class_idx in range(self.num_classes):
            class_mask = targets == class_idx
            if np.sum(class_mask) > 0:
                class_probs = probabilities[class_mask, class_idx]
                class_confidences[class_idx] = float(np.mean(class_probs))
            else:
                class_confidences[class_idx] = 0.0
        
        # 결과 구성
        class_analysis = {}
        for i in range(self.num_classes):
            class_analysis[f'class_{i}'] = {
                'name': self.class_names[i] if i < len(self.class_names) else f'class_{i}',
                'f1_score': float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
                'precision': float(per_class_precision[i]) if i < len(per_class_precision) else 0.0,
                'recall': float(per_class_recall[i]) if i < len(per_class_recall) else 0.0,
                'support': int(support_dict.get(i, 0)),
                'avg_confidence': class_confidences.get(i, 0.0),
                'class_weight': self.class_weights.get(i, 1.0),
                'is_minority': self.class_weights.get(i, 1.0) > 1.5
            }
        
        return class_analysis
    
    def _create_confusion_matrix(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        save_dir: Path
    ) -> Dict[str, Any]:
        """혼동 행렬 생성 및 시각화"""
        
        # 혼동 행렬 계산
        cm = confusion_matrix(targets, predictions)
        
        # 정규화된 혼동 행렬
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 시각화
        plt.figure(figsize=(15, 12))
        
        # 절대값 혼동 행렬
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names[:cm.shape[1]],
                   yticklabels=self.class_names[:cm.shape[0]])
        plt.title('Confusion Matrix (Absolute)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 정규화된 혼동 행렬
        plt.subplot(2, 2, 2)
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.class_names[:cm.shape[1]],
                   yticklabels=self.class_names[:cm.shape[0]])
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 클래스별 정확도
        plt.subplot(2, 2, 3)
        class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        plt.bar(range(len(class_accuracy)), class_accuracy)
        plt.title('Per-Class Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(class_accuracy)), 
                  [f'C{i}' for i in range(len(class_accuracy))], rotation=45)
        
        # 클래스 분포
        plt.subplot(2, 2, 4)
        class_distribution = np.sum(cm, axis=1)
        plt.bar(range(len(class_distribution)), class_distribution)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(range(len(class_distribution)), 
                  [f'C{i}' for i in range(len(class_distribution))], rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'confusion_matrix': cm.tolist(),
            'normalized_confusion_matrix': cm_normalized.tolist(),
            'per_class_accuracy': class_accuracy.tolist(),
            'class_distribution': class_distribution.tolist()
        }
    
    def _analyze_confidence(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        save_dir: Path
    ) -> Dict[str, Any]:
        """신뢰도 분석"""
        
        # 최대 확률 (신뢰도)
        max_probs = np.max(probabilities, axis=1)
        
        # 정답/오답별 신뢰도
        correct_mask = predictions == targets
        correct_confidences = max_probs[correct_mask]
        incorrect_confidences = max_probs[~correct_mask]
        
        # 신뢰도 구간별 정확도
        confidence_bins = np.arange(0, 1.1, 0.1)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(correct_mask[bin_mask])
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(np.sum(bin_mask))
            else:
                bin_accuracies.append(0.0)
                bin_counts.append(0)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        
        # 신뢰도 분포
        plt.subplot(2, 2, 1)
        plt.hist([correct_confidences, incorrect_confidences], 
                bins=20, alpha=0.7, label=['Correct', 'Incorrect'])
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        
        # 신뢰도 vs 정확도
        plt.subplot(2, 2, 2)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        plt.plot(bin_centers, bin_accuracies, 'o-')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        
        # 신뢰도 구간별 샘플 수
        plt.subplot(2, 2, 3)
        plt.bar(bin_centers, bin_counts)
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Samples per Confidence Bin')
        
        # 클래스별 평균 신뢰도
        plt.subplot(2, 2, 4)
        class_confidences = []
        for class_idx in range(self.num_classes):
            class_mask = targets == class_idx
            if np.sum(class_mask) > 0:
                class_conf = np.mean(max_probs[class_mask])
                class_confidences.append(class_conf)
            else:
                class_confidences.append(0.0)
        
        plt.bar(range(self.num_classes), class_confidences)
        plt.xlabel('Class')
        plt.ylabel('Average Confidence')
        plt.title('Per-Class Average Confidence')
        plt.xticks(range(self.num_classes), [f'C{i}' for i in range(self.num_classes)], rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'overall_confidence': {
                'mean': float(np.mean(max_probs)),
                'std': float(np.std(max_probs)),
                'min': float(np.min(max_probs)),
                'max': float(np.max(max_probs))
            },
            'correct_predictions': {
                'mean_confidence': float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
                'count': int(len(correct_confidences))
            },
            'incorrect_predictions': {
                'mean_confidence': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
                'count': int(len(incorrect_confidences))
            },
            'calibration': {
                'bin_accuracies': bin_accuracies,
                'bin_counts': bin_counts,
                'bin_centers': bin_centers.tolist()
            },
            'per_class_confidence': class_confidences
        }
    
    def _analyze_misclassifications(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        image_paths: List[str],
        save_dir: Path,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """오분류 분석"""
        
        # 오분류 찾기
        misclassified_mask = predictions != targets
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            return {'misclassification_count': 0}
        
        # 신뢰도 기준으로 정렬 (높은 신뢰도로 틀린 것들)
        misclassified_confidences = np.max(probabilities[misclassified_indices], axis=1)
        sorted_indices = np.argsort(misclassified_confidences)[::-1]
        
        # 상위 K개 오분류 사례
        top_misclassified = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = misclassified_indices[sorted_indices[i]]
            top_misclassified.append({
                'index': int(idx),
                'image_path': image_paths[idx] if idx < len(image_paths) else '',
                'true_class': int(targets[idx]),
                'predicted_class': int(predictions[idx]),
                'confidence': float(misclassified_confidences[sorted_indices[i]]),
                'true_class_prob': float(probabilities[idx, targets[idx]]),
                'predicted_class_prob': float(probabilities[idx, predictions[idx]])
            })
        
        # 오분류 패턴 분석
        misclass_patterns = {}
        for true_class in range(self.num_classes):
            true_mask = targets == true_class
            misclass_in_true = misclassified_mask & true_mask
            
            if np.sum(misclass_in_true) > 0:
                pred_classes = predictions[misclass_in_true]
                unique_preds, counts = np.unique(pred_classes, return_counts=True)
                
                patterns = []
                for pred_class, count in zip(unique_preds, counts):
                    patterns.append({
                        'predicted_as': int(pred_class),
                        'count': int(count),
                        'percentage': float(count / np.sum(true_mask) * 100)
                    })
                
                misclass_patterns[f'true_class_{true_class}'] = sorted(
                    patterns, key=lambda x: x['count'], reverse=True
                )
        
        return {
            'misclassification_count': int(np.sum(misclassified_mask)),
            'misclassification_rate': float(np.sum(misclassified_mask) / len(targets)),
            'top_misclassified': top_misclassified,
            'misclassification_patterns': misclass_patterns
        }
    
    def _create_roc_pr_curves(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        save_dir: Path
    ) -> Dict[str, Any]:
        """ROC 및 PR 곡선 생성 (다중 클래스)"""
        
        # 이진화
        targets_binarized = label_binarize(targets, classes=range(self.num_classes))
        
        # 클래스별 ROC/PR 계산
        roc_auc_scores = {}
        pr_auc_scores = {}
        
        plt.figure(figsize=(15, 10))
        
        # ROC 곡선
        plt.subplot(2, 3, 1)
        for i in range(min(self.num_classes, 10)):  # 최대 10개 클래스만 표시
            if np.sum(targets == i) > 0:  # 해당 클래스가 존재하는 경우만
                fpr, tpr, _ = roc_curve(targets_binarized[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                roc_auc_scores[f'class_{i}'] = float(roc_auc)
                
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # PR 곡선
        plt.subplot(2, 3, 2)
        for i in range(min(self.num_classes, 10)):
            if np.sum(targets == i) > 0:
                precision, recall, _ = precision_recall_curve(targets_binarized[:, i], probabilities[:, i])
                pr_auc = auc(recall, precision)
                pr_auc_scores[f'class_{i}'] = float(pr_auc)
                
                plt.plot(recall, precision, label=f'Class {i} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 클래스별 AUC 점수 바 차트
        plt.subplot(2, 3, 3)
        if roc_auc_scores:
            classes = list(roc_auc_scores.keys())
            scores = list(roc_auc_scores.values())
            plt.bar(range(len(classes)), scores)
            plt.xlabel('Class')
            plt.ylabel('ROC AUC')
            plt.title('ROC AUC by Class')
            plt.xticks(range(len(classes)), [c.replace('class_', 'C') for c in classes], rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'roc_auc_scores': roc_auc_scores,
            'pr_auc_scores': pr_auc_scores,
            'macro_roc_auc': float(np.mean(list(roc_auc_scores.values()))) if roc_auc_scores else 0.0,
            'macro_pr_auc': float(np.mean(list(pr_auc_scores.values()))) if pr_auc_scores else 0.0
        }
    
    def _analyze_class_imbalance(
        self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """클래스 불균형 분석"""
        
        # 실제 클래스 분포
        unique_targets, target_counts = np.unique(targets, return_counts=True)
        true_distribution = dict(zip(unique_targets.astype(int), target_counts.astype(int)))
        
        # 예측 클래스 분포
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        pred_distribution = dict(zip(unique_preds.astype(int), pred_counts.astype(int)))
        
        # 불균형 비율 계산
        max_count = max(target_counts)
        imbalance_ratios = {}
        for class_idx, count in true_distribution.items():
            imbalance_ratios[class_idx] = float(max_count / count)
        
        # 소수 클래스 성능 분석
        minority_threshold = 1.5
        minority_classes = [cls for cls, ratio in imbalance_ratios.items() if ratio > minority_threshold]
        
        minority_performance = {}
        for cls in minority_classes:
            class_mask = targets == cls
            if np.sum(class_mask) > 0:
                class_predictions = predictions[class_mask]
                class_accuracy = np.mean(class_predictions == cls)
                minority_performance[cls] = {
                    'accuracy': float(class_accuracy),
                    'support': int(np.sum(class_mask)),
                    'imbalance_ratio': imbalance_ratios[cls],
                    'class_weight': self.class_weights.get(cls, 1.0)
                }
        
        return {
            'true_distribution': true_distribution,
            'predicted_distribution': pred_distribution,
            'imbalance_ratios': imbalance_ratios,
            'minority_classes': minority_classes,
            'minority_performance': minority_performance,
            'most_imbalanced_class': max(imbalance_ratios.keys(), key=lambda x: imbalance_ratios[x]),
            'least_imbalanced_class': min(imbalance_ratios.keys(), key=lambda x: imbalance_ratios[x])
        }
    
    def _save_evaluation_results(self, results: Dict[str, Any], save_dir: Path):
        """평가 결과 저장"""
        
        # JSON 저장 (numpy 타입 변환)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_converted = convert_numpy_types(results)
        results_file = save_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_converted, f, indent=2, default=str)
        
        print(f"💾 평가 결과 저장: {results_file}")
    
    def _generate_summary_report(self, results: Dict[str, Any], save_dir: Path):
        """요약 리포트 생성"""
        
        report_lines = [
            "# 📊 모델 평가 요약 리포트",
            f"**실험명**: {results['experiment_name']}",
            f"**평가 샘플 수**: {results['sample_count']}",
            "",
            "## 🎯 전체 성능",
            f"- **정확도**: {results['basic_metrics']['accuracy']:.4f}",
            f"- **Macro F1**: {results['basic_metrics']['macro_f1']:.4f}",
            f"- **Weighted F1**: {results['basic_metrics']['weighted_f1']:.4f}",
            f"- **Macro Precision**: {results['basic_metrics']['macro_precision']:.4f}",
            f"- **Macro Recall**: {results['basic_metrics']['macro_recall']:.4f}",
            "",
            "## 🔍 신뢰도 분석",
            f"- **평균 신뢰도**: {results['confidence_analysis']['overall_confidence']['mean']:.4f}",
            f"- **정답 예측 신뢰도**: {results['confidence_analysis']['correct_predictions']['mean_confidence']:.4f}",
            f"- **오답 예측 신뢰도**: {results['confidence_analysis']['incorrect_predictions']['mean_confidence']:.4f}",
            "",
            "## ❌ 오분류 분석",
            f"- **오분류 수**: {results['misclassification_analysis']['misclassification_count']}",
            f"- **오분류 비율**: {results['misclassification_analysis']['misclassification_rate']:.4f}",
            "",
            "## ⚖️ 클래스 불균형 영향",
            f"- **소수 클래스 수**: {len(results['imbalance_analysis']['minority_classes'])}",
            f"- **가장 불균형한 클래스**: {results['imbalance_analysis']['most_imbalanced_class']}",
            "",
            "## 🏆 클래스별 최고/최저 성능",
        ]
        
        # 클래스별 F1 점수 정렬
        class_f1_scores = [(cls, info['f1_score']) for cls, info in results['class_analysis'].items()]
        class_f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        report_lines.extend([
            f"- **최고 성능**: {class_f1_scores[0][0]} (F1: {class_f1_scores[0][1]:.4f})",
            f"- **최저 성능**: {class_f1_scores[-1][0]} (F1: {class_f1_scores[-1][1]:.4f})",
        ])
        
        # 마크다운 파일 저장
        report_file = save_dir / 'evaluation_summary.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📋 요약 리포트 생성: {report_file}")


# 사용 예시
if __name__ == "__main__":
    print("📊 모델 평가기 테스트는 main_evaluation.py에서 실행됩니다.")

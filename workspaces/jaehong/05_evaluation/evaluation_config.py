"""
🎯 Evaluation Configuration Manager
시니어 그랜드마스터 수준의 평가 설정 관리

Features:
- 01~04 단계 완전 연계
- 다양한 평가 전략
- TTA 최적화 설정
- 앙상블 구성 자동화
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import torch

# 04_training 모듈 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "04_training"))
from config import ConfigManager


@dataclass
class TTAConfig:
    """TTA 설정"""
    enabled: bool = True
    n_tta: int = 8
    strategies: List[str] = None
    weights: List[float] = None
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = [
                "original",
                "horizontal_flip", 
                "rotate_5",
                "rotate_minus_5",
                "brightness_up",
                "brightness_down",
                "contrast_up",
                "contrast_down"
            ]
        
        if self.weights is None:
            # 원본에 더 높은 가중치
            self.weights = [0.3, 0.15, 0.1, 0.1, 0.1, 0.1, 0.075, 0.075]


@dataclass  
class EnsembleConfig:
    """앙상블 설정"""
    enabled: bool = True
    strategy: str = "weighted_soft_voting"
    models: List[str] = None
    weights: List[float] = None
    confidence_threshold: float = 0.9
    
    def __post_init__(self):
        if self.models is None:
            self.models = ["best_model"]
        if self.weights is None:
            self.weights = [1.0]


@dataclass
class EvaluationConfig:
    """평가 설정"""
    # 기본 설정
    experiment_name: str = "evaluation_v1"
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4
    
    # TTA 설정
    tta: TTAConfig = None
    
    # 앙상블 설정
    ensemble: EnsembleConfig = None
    
    # 분석 설정
    generate_confusion_matrix: bool = True
    generate_classification_report: bool = True
    analyze_misclassifications: bool = True
    visualize_predictions: bool = True
    save_prediction_samples: int = 50
    
    # 출력 설정
    create_submission: bool = True
    save_probabilities: bool = True
    
    def __post_init__(self):
        if self.tta is None:
            self.tta = TTAConfig()
        if self.ensemble is None:
            self.ensemble = EnsembleConfig()


class EvaluationConfigManager:
    """평가 설정 관리자 - 01~04 단계 완전 연계"""
    
    def __init__(self, workspace_root: str, training_experiment_name: str):
        """
        Args:
            workspace_root: 워크스페이스 루트 경로
            training_experiment_name: 04_training 실험 이름
        """
        self.workspace_root = Path(workspace_root)
        self.training_experiment_name = training_experiment_name
        
        # 경로 설정
        self.training_dir = self.workspace_root / "04_training"
        self.evaluation_dir = self.workspace_root / "05_evaluation"
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # 04_training 설정 관리자 생성
        self.training_config_manager = ConfigManager(str(self.workspace_root))
        
        # 04_training 결과 로드
        self._load_training_results()
        
        print(f"🎯 Evaluation 설정 관리자 초기화:")
        print(f"   Training 실험: {training_experiment_name}")
        print(f"   모델 경로: {self.model_path}")
        print(f"   최고 F1: {self.training_results.get('best_f1', 'N/A')}")
    
    def _load_training_results(self):
        """04_training 결과 로드"""
        # 훈련 결과 디렉토리
        training_output_dir = self.training_dir / "outputs" / self.training_experiment_name
        
        # 모델 파일 경로
        self.model_path = training_output_dir / "best_model.pth"
        
        # 훈련 설정 로드
        config_path = training_output_dir / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.training_config = json.load(f)
        else:
            print(f"⚠️ 훈련 설정 파일을 찾을 수 없습니다: {config_path}")
            self.training_config = {}
        
        # 훈련 결과 로드 (체크포인트에서)
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            self.training_results = {
                'best_f1': checkpoint.get('best_score', 0.0),
                'epoch': checkpoint.get('epoch', 0),
                'model_config': checkpoint.get('config', {}).get('model', {}),
                'training_config': checkpoint.get('config', {}).get('training', {})
            }
        else:
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {self.model_path}")
            self.training_results = {'best_f1': 0.0}
    
    def create_evaluation_config(
        self,
        experiment_name: str = None,
        enable_tta: bool = True,
        n_tta: int = 8,
        enable_ensemble: bool = False,
        **overrides
    ) -> EvaluationConfig:
        """평가 설정 생성"""
        
        if experiment_name is None:
            experiment_name = f"eval_{self.training_experiment_name}"
        
        # GPU 메모리 기반 배치 크기 조정
        batch_size = 32
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            if gpu_memory < 8:
                batch_size = 16
            elif gpu_memory >= 16:
                batch_size = 64
        
        # TTA 설정
        tta_config = TTAConfig(
            enabled=enable_tta,
            n_tta=n_tta if enable_tta else 1
        )
        
        # 앙상블 설정
        ensemble_config = EnsembleConfig(
            enabled=enable_ensemble
        )
        
        # 기본 설정 생성
        config = EvaluationConfig(
            experiment_name=experiment_name,
            batch_size=batch_size,
            tta=tta_config,
            ensemble=ensemble_config
        )
        
        # 오버라이드 적용
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def get_model_info(self) -> Dict[str, Any]:
        """훈련된 모델 정보 반환"""
        return {
            'model_path': str(self.model_path),
            'architecture': self.training_results.get('model_config', {}).get('architecture', 'efficientnetv2_s'),
            'num_classes': self.training_results.get('model_config', {}).get('num_classes', 17),
            'image_size': self.training_results.get('model_config', {}).get('image_size', 512),
            'best_f1': self.training_results.get('best_f1', 0.0),
            'training_epochs': self.training_results.get('epoch', 0),
            'class_weights': self.training_config_manager.class_weights,
            'minority_classes': self.training_config_manager.get_minority_classes()
        }
    
    def get_data_info(self) -> Dict[str, Any]:
        """데이터 정보 반환"""
        return {
            'data_root': str(self.training_config_manager.data_root),
            'num_classes': 17,
            'class_names': [f"class_{i}" for i in range(17)],
            'class_weights': self.training_config_manager.class_weights,
            'minority_classes': self.training_config_manager.get_minority_classes()
        }
    
    def save_evaluation_config(self, config: EvaluationConfig, output_dir: Path):
        """평가 설정 저장"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정을 딕셔너리로 변환
        config_dict = {
            "evaluation": asdict(config),
            "model_info": self.get_model_info(),
            "data_info": self.get_data_info(),
            "training_experiment": self.training_experiment_name,
            "integration": {
                "eda_results": str(self.training_config_manager.eda_path),
                "preprocessing_config": str(self.training_config_manager.preprocessing_path),
                "modeling_results": str(self.training_config_manager.modeling_path),
                "training_output": str(self.training_dir / "outputs" / self.training_experiment_name)
            }
        }
        
        # JSON 저장
        config_file = output_dir / "evaluation_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"💾 평가 설정 저장: {config_file}")
        return config_file
    
    def create_tta_strategies(self, n_tta: int = 8) -> List[Dict[str, Any]]:
        """TTA 전략 생성"""
        strategies = [
            {"name": "original", "transform": "none", "weight": 0.3},
            {"name": "horizontal_flip", "transform": "hflip", "weight": 0.15},
            {"name": "rotate_5", "transform": "rotate", "params": {"degrees": 5}, "weight": 0.1},
            {"name": "rotate_minus_5", "transform": "rotate", "params": {"degrees": -5}, "weight": 0.1},
            {"name": "brightness_up", "transform": "brightness", "params": {"factor": 1.1}, "weight": 0.1},
            {"name": "brightness_down", "transform": "brightness", "params": {"factor": 0.9}, "weight": 0.1},
            {"name": "contrast_up", "transform": "contrast", "params": {"factor": 1.1}, "weight": 0.075},
            {"name": "contrast_down", "transform": "contrast", "params": {"factor": 0.9}, "weight": 0.075},
        ]
        
        return strategies[:n_tta]
    
    def optimize_tta_config(self, validation_results: Dict[str, float]) -> TTAConfig:
        """검증 결과를 바탕으로 TTA 설정 최적화"""
        # 성능이 좋으면 더 많은 TTA, 나쁘면 적게
        best_f1 = validation_results.get('f1', 0.0)
        
        if best_f1 > 0.85:
            n_tta = 8
        elif best_f1 > 0.75:
            n_tta = 6
        else:
            n_tta = 4
        
        return TTAConfig(
            enabled=True,
            n_tta=n_tta,
            strategies=self.create_tta_strategies(n_tta)
        )


# 사용 예시
if __name__ == "__main__":
    # 설정 관리자 생성
    workspace_root = "/home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong"
    training_experiment = "kaggle_real_training_v1"
    
    eval_config_manager = EvaluationConfigManager(workspace_root, training_experiment)
    
    # 평가 설정 생성
    eval_config = eval_config_manager.create_evaluation_config(
        experiment_name="grandmaster_evaluation_v1",
        enable_tta=True,
        n_tta=8,
        enable_ensemble=False
    )
    
    print(f"\n🎯 평가 설정 생성 완료:")
    print(f"   실험명: {eval_config.experiment_name}")
    print(f"   TTA: {eval_config.tta.enabled} ({eval_config.tta.n_tta}개)")
    print(f"   앙상블: {eval_config.ensemble.enabled}")
    print(f"   배치 크기: {eval_config.batch_size}")
    
    # 모델 정보 출력
    model_info = eval_config_manager.get_model_info()
    print(f"   모델 F1: {model_info['best_f1']:.4f}")
    print(f"   소수 클래스: {model_info['minority_classes']}")

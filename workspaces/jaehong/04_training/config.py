"""
🎯 Training Configuration Manager
시니어 캐글러 수준의 설정 관리 시스템

Features:
- 01~03 단계 결과 자동 로드
- 다양한 실험 설정 템플릿
- 환경별 최적화 설정
- Reproducible 실험 보장
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import torch


@dataclass
class ModelConfig:
    """모델 설정"""
    architecture: str = "efficientnetv2_s"  # 균형: 적절한 모델 용량
    num_classes: int = 17
    image_size: int = 512  # 02_preprocessing에서 최적화된 크기
    dropout_rate: float = 0.3  # 균형: 0.5 → 0.3 (언더피팅 해결)
    pretrained: bool = True
    
    # Loss function
    loss_type: str = "focal"  # 클래스 불균형 대응
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    label_smoothing: float = 0.1  # 균형: 0.15 → 0.1 (언더피팅 해결)


@dataclass 
class TrainingConfig:
    """훈련 설정"""
    # Basic training - 균형잡힌 설정
    epochs: int = 25  # 적절한 훈련 길이
    batch_size: int = 32
    learning_rate: float = 8e-5  # 5e-5 → 8e-5: 적절한 학습속도
    weight_decay: float = 1.2e-2  # 2e-2 → 1.2e-2: 적절한 정규화
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine_warm_restarts"
    warmup_epochs: int = 5
    
    # Regularization - 균형잡힌 정규화
    gradient_clip_norm: float = 1.0
    mixup_alpha: float = 0.3  # 0.4 → 0.3: 적절한 정규화
    cutmix_alpha: float = 1.0  # 1.5 → 1.0: 적절한 정규화
    
    # Early stopping - 균형잡힌 설정
    early_stopping_patience: int = 7  # 5 → 7: 적절한 인내심
    monitor_metric: str = "val_f1"
    min_delta: float = 0.002  # 0.005 → 0.002: 적절한 개선 기준
    
    # Technical
    use_amp: bool = True  # Mixed Precision
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    """실험 설정"""
    experiment_name: str = "kaggle_training"
    use_wandb: bool = True
    wandb_project: str = "document-classification-cv3"
    
    # Cross validation
    n_folds: int = 5
    fold_strategy: str = "stratified"
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True


class ConfigManager:
    """설정 관리자 - 01~03 단계 결과 자동 연계"""
    
    def __init__(self, workspace_root: str):
        """
        Args:
            workspace_root: 워크스페이스 루트 경로
        """
        self.workspace_root = Path(workspace_root)
        self.data_root = self.workspace_root.parent.parent / "data"
        
        # 01~03 단계 경로
        self.eda_path = self.workspace_root / "01_EDA" / "eda_results"
        self.preprocessing_path = self.workspace_root / "02_preprocessing" 
        self.modeling_path = self.workspace_root / "03_modeling"
        
        # 01~03 단계 결과 로드
        self._load_previous_results()
        
    def _load_previous_results(self):
        """01~03 단계 결과 로드"""
        print("📊 이전 단계 결과 로드 중...")
        
        # 01_EDA 결과
        self.class_weights = self._load_class_weights()
        self.competition_strategy = self._load_competition_strategy()
        
        # 02_preprocessing 결과  
        self.preprocessing_config = self._load_preprocessing_config()
        
        # 03_modeling 결과
        self.modeling_results = self._load_modeling_results()
        
        print(f"✅ 이전 단계 결과 로드 완료")
        print(f"   소수 클래스: {self.get_minority_classes()}")
        print(f"   최적 이미지 크기: {self.preprocessing_config.get('image_size', 512)}")
        print(f"   검증된 모델: {self.modeling_results.get('model_specification', {}).get('backbone', 'efficientnetv2_s')}")
    
    def _load_class_weights(self) -> Dict[str, float]:
        """클래스 가중치 로드"""
        try:
            with open(self.eda_path / "class_weights.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ 클래스 가중치 파일을 찾을 수 없습니다. 균등 가중치 사용.")
            return {str(i): 1.0 for i in range(17)}
    
    def _load_competition_strategy(self) -> Dict[str, Any]:
        """경쟁 전략 로드"""
        try:
            with open(self.eda_path / "competition_strategy.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ 경쟁 전략 파일을 찾을 수 없습니다.")
            return {}
    
    def _load_preprocessing_config(self) -> Dict[str, Any]:
        """전처리 설정 로드"""
        try:
            with open(self.preprocessing_path / "preprocessing_config.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ 전처리 설정 파일을 찾을 수 없습니다.")
            return {"image_size": 512, "device": "cuda"}
    
    def _load_modeling_results(self) -> Dict[str, Any]:
        """모델링 결과 로드"""
        try:
            with open(self.modeling_path / "final_modeling_results.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ 모델링 결과 파일을 찾을 수 없습니다.")
            return {"model_specification": {"backbone": "efficientnetv2_s"}}
    
    def get_minority_classes(self) -> list:
        """소수 클래스 반환"""
        return [int(k) for k, v in self.class_weights.items() if float(v) > 1.5]
    
    def get_class_weights_tensor(self, device: str = "cuda") -> torch.Tensor:
        """클래스 가중치 텐서 반환"""
        weights = [float(self.class_weights[str(i)]) for i in range(17)]
        return torch.tensor(weights, dtype=torch.float32, device=device)
    
    def create_model_config(self, **overrides) -> ModelConfig:
        """모델 설정 생성"""
        config = ModelConfig(
            architecture=self.modeling_results.get("model_specification", {}).get("backbone", "efficientnetv2_s"),
            image_size=self.preprocessing_config.get("image_size", 512),
            num_classes=17
        )
        
        # 오버라이드 적용
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def create_training_config(self, mode: str = "default", **overrides) -> TrainingConfig:
        """훈련 설정 생성
        
        Args:
            mode: 훈련 모드 ("default", "fast", "high_quality", "debug")
        """
        # 01_EDA 전략 기반 기본 설정
        base_config = TrainingConfig()
        
        # 모드별 설정 조정
        if mode == "fast":
            base_config.epochs = 20
            base_config.batch_size = 64
            base_config.early_stopping_patience = 3
        elif mode == "high_quality":
            base_config.epochs = 100
            base_config.learning_rate = 5e-5
            base_config.early_stopping_patience = 15
        elif mode == "debug":
            base_config.epochs = 2
            base_config.batch_size = 8
            
        # GPU 메모리에 따른 배치 크기 조정
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            if gpu_memory < 8:  # 8GB 미만
                base_config.batch_size = min(base_config.batch_size, 16)
            elif gpu_memory >= 16:  # 16GB 이상
                base_config.batch_size = min(base_config.batch_size * 2, 64)
        
        # 오버라이드 적용
        for key, value in overrides.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
                
        return base_config
    
    def create_experiment_config(self, **overrides) -> ExperimentConfig:
        """실험 설정 생성"""
        config = ExperimentConfig()
        
        # 오버라이드 적용
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        return config
    
    def get_augmentation_strategy(self) -> Dict[str, Any]:
        """01_EDA에서 도출된 증강 전략 반환"""
        return self.competition_strategy.get("augmentation", [])
    
    def get_training_strategy(self) -> Dict[str, Any]:
        """01_EDA에서 도출된 훈련 전략 반환"""
        return self.competition_strategy.get("training", [])
    
    def save_config(self, model_config: ModelConfig, training_config: TrainingConfig, 
                   experiment_config: ExperimentConfig, output_dir: Path):
        """설정 저장"""
        config_dict = {
            "model": asdict(model_config),
            "training": asdict(training_config), 
            "experiment": asdict(experiment_config),
            "previous_results": {
                "class_weights": self.class_weights,
                "minority_classes": self.get_minority_classes(),
                "preprocessing_config": self.preprocessing_config,
                "modeling_backbone": self.modeling_results.get("model_specification", {}).get("backbone")
            }
        }
        
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"💾 설정 저장 완료: {output_dir / 'training_config.json'}")


# 사용 예시
if __name__ == "__main__":
    # 설정 관리자 생성
    workspace_root = "/home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong"
    config_manager = ConfigManager(workspace_root)
    
    # 다양한 실험 설정 생성
    print("\n🎯 실험 설정 템플릿:")
    
    # 1. 기본 실험
    model_config = config_manager.create_model_config()
    training_config = config_manager.create_training_config("default")
    experiment_config = config_manager.create_experiment_config(experiment_name="baseline_experiment")
    
    print(f"✅ 기본 실험 설정:")
    print(f"   모델: {model_config.architecture}")
    print(f"   이미지 크기: {model_config.image_size}")
    print(f"   배치 크기: {training_config.batch_size}")
    print(f"   에포크: {training_config.epochs}")
    print(f"   소수 클래스: {config_manager.get_minority_classes()}")

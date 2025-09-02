"""
Experiment Tracker with wandb

wandb를 활용한 실험 추적 및 관리 클래스
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class ExperimentTracker:
    """
    wandb를 활용한 실험 추적 클래스
    
    각 멤버가 독립적으로 실험을 관리할 수 있도록 설계
    """
    
    def __init__(
        self,
        project_name: str = "doc-classification",
        member_name: str = "member1",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        offline_mode: bool = False
    ):
        """
        Args:
            project_name: wandb 프로젝트 이름
            member_name: 멤버 이름 (태그로 사용)
            experiment_name: 실험 이름
            config: 실험 설정
            offline_mode: 오프라인 모드 여부
        """
        self.project_name = project_name
        self.member_name = member_name
        self.experiment_name = experiment_name or f"{member_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.offline_mode = offline_mode
        
        # wandb 초기화
        self._init_wandb()
        
        # 실험 메타데이터 저장
        self.experiment_metadata = {
            'member': member_name,
            'start_time': datetime.now().isoformat(),
            'experiment_id': self.run.id if self.run else None,
            'metrics_history': [],
            'model_checkpoints': []
        }
    
    def _init_wandb(self) -> None:
        """wandb 초기화"""
        try:
            if self.offline_mode:
                os.environ["WANDB_MODE"] = "offline"
            
            self.run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                tags=[self.member_name, "document-classification"],
                reinit=True
            )
            
            print(f"✅ wandb 초기화 완료: {self.experiment_name}")
            
        except Exception as e:
            print(f"⚠️ wandb 초기화 실패: {e}")
            print("오프라인 모드로 계속 진행합니다.")
            self.run = None
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        실험 설정 로깅
        
        Args:
            config: 실험 설정 딕셔너리
        """
        self.config.update(config)
        
        if self.run:
            wandb.config.update(config)
        
        # 로컬에도 저장
        self._save_config_locally(config)
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        메트릭 로깅
        
        Args:
            metrics: 메트릭 딕셔너리
            step: 스텝 번호
            prefix: 메트릭 이름 접두사 (train_, val_ 등)
        """
        # 접두사 적용
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # wandb 로깅
        if self.run:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        
        # 로컬 히스토리 저장
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        }
        self.experiment_metadata['metrics_history'].append(log_entry)
        
        # 콘솔 출력
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in metrics.items()])
        print(f"📊 Step {step}: {metrics_str}")
    
    def log_model_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: Path,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        모델 체크포인트 로깅
        
        Args:
            model: 저장할 모델
            checkpoint_path: 체크포인트 파일 경로
            metrics: 해당 체크포인트의 성능 메트릭
            is_best: 최고 성능 모델 여부
        """
        # 체크포인트 정보 기록
        checkpoint_info = {
            'path': str(checkpoint_path),
            'metrics': metrics,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat()
        }
        self.experiment_metadata['model_checkpoints'].append(checkpoint_info)
        
        # wandb에 아티팩트로 업로드
        if self.run:
            artifact_name = f"model_checkpoint_{self.member_name}"
            if is_best:
                artifact_name += "_best"
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                metadata=checkpoint_info
            )
            artifact.add_file(str(checkpoint_path))
            self.run.log_artifact(artifact)
        
        print(f"💾 모델 체크포인트 저장: {checkpoint_path}")
        if is_best:
            print("🏆 새로운 최고 성능 모델!")
    
    def log_images(
        self,
        images: Dict[str, Union[np.ndarray, plt.Figure]],
        step: Optional[int] = None
    ) -> None:
        """
        이미지 로깅
        
        Args:
            images: 이미지 딕셔너리 (이름: 이미지)
            step: 스텝 번호
        """
        if not self.run:
            return
        
        wandb_images = {}
        for name, image in images.items():
            if isinstance(image, plt.Figure):
                wandb_images[name] = wandb.Image(image)
            elif isinstance(image, np.ndarray):
                wandb_images[name] = wandb.Image(image)
            else:
                print(f"⚠️ 지원하지 않는 이미지 형식: {type(image)}")
        
        if step is not None:
            wandb.log(wandb_images, step=step)
        else:
            wandb.log(wandb_images)
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        step: Optional[int] = None
    ) -> None:
        """
        Confusion Matrix 로깅
        
        Args:
            y_true: 실제 라벨
            y_pred: 예측 라벨
            class_names: 클래스 이름 목록
            step: 스텝 번호
        """
        if not self.run:
            return
        
        # wandb confusion matrix 생성
        cm = wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=class_names
        )
        
        log_data = {"confusion_matrix": cm}
        if step is not None:
            wandb.log(log_data, step=step)
        else:
            wandb.log(log_data)
    
    def log_learning_curve(
        self,
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = "accuracy"
    ) -> None:
        """
        학습 곡선 로깅
        
        Args:
            train_scores: 훈련 점수 리스트
            val_scores: 검증 점수 리스트
            metric_name: 메트릭 이름
        """
        if not self.run:
            return
        
        # 학습 곡선 플롯 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_scores) + 1)
        
        ax.plot(epochs, train_scores, 'o-', label=f'Train {metric_name}', color='blue')
        ax.plot(epochs, val_scores, 'o-', label=f'Val {metric_name}', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'Learning Curve - {metric_name.capitalize()}')
        ax.legend()
        ax.grid(True)
        
        # wandb에 로깅
        wandb.log({f"learning_curve_{metric_name}": wandb.Image(fig)})
        plt.close(fig)
    
    def save_experiment_summary(self, save_path: Optional[Path] = None) -> Path:
        """
        실험 요약 저장
        
        Args:
            save_path: 저장 경로
            
        Returns:
            Path: 저장된 파일 경로
        """
        if save_path is None:
            save_path = Path(f"experiment_summary_{self.member_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # 실험 종료 시간 추가
        self.experiment_metadata['end_time'] = datetime.now().isoformat()
        
        # 최고 성능 메트릭 찾기
        if self.experiment_metadata['metrics_history']:
            best_metrics = self._find_best_metrics()
            self.experiment_metadata['best_metrics'] = best_metrics
        
        # JSON으로 저장
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"📋 실험 요약 저장: {save_path}")
        return save_path
    
    def _save_config_locally(self, config: Dict[str, Any]) -> None:
        """설정을 로컬에 저장"""
        config_path = Path(f"config_{self.member_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def _find_best_metrics(self) -> Dict[str, Any]:
        """최고 성능 메트릭 찾기"""
        best_metrics = {}
        
        # 모든 메트릭 수집
        all_metrics = {}
        for entry in self.experiment_metadata['metrics_history']:
            for metric_name, value in entry['metrics'].items():
                if isinstance(value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # 각 메트릭의 최고값 찾기
        for metric_name, values in all_metrics.items():
            if 'loss' in metric_name.lower():
                # loss는 최소값이 최고
                best_metrics[f"best_{metric_name}"] = min(values)
            else:
                # 다른 메트릭들은 최대값이 최고
                best_metrics[f"best_{metric_name}"] = max(values)
        
        return best_metrics
    
    def finish_experiment(self) -> None:
        """실험 종료"""
        if self.run:
            # 최종 요약 저장
            summary_path = self.save_experiment_summary()
            
            # wandb 아티팩트로 요약 업로드
            artifact = wandb.Artifact(
                name=f"experiment_summary_{self.member_name}",
                type="summary"
            )
            artifact.add_file(str(summary_path))
            self.run.log_artifact(artifact)
            
            # wandb 실행 종료
            wandb.finish()
            print(f"🏁 실험 종료: {self.experiment_name}")
        else:
            # 오프라인 모드에서도 요약 저장
            self.save_experiment_summary()
            print(f"🏁 실험 종료 (오프라인): {self.experiment_name}")
    
    def __enter__(self):
        """Context manager 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.finish_experiment()

"""
🏆 캐글 1등 달성을 위한 완전한 훈련 실행 스크립트
Kaggle Competition Winner - Complete Training Pipeline

🎯 목표: 캐글 대회 1등 달성
🚀 전략: EDA + 전처리 + 고급 모델링의 완전한 통합

실행 방법:
python kaggle_winner_training.py --strategy diverse_ensemble --target_score 0.95

Clean Code & Clean Architecture:
- Command Pattern: 실행 명령 캡슐화
- Template Method: 훈련 파이프라인 템플릿
- Observer Pattern: 진행 상황 모니터링
- Strategy Pattern: 다양한 훈련 전략
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 프로젝트 모듈들
sys.path.append('../01_EDA')
sys.path.append('../02_preprocessing')

try:
    from grandmaster_processor import (
        create_grandmaster_processor, 
        load_competition_data,
        GrandmasterConfig as ProcessorConfig,
        ProcessingStrategy
    )
    from grandmaster_modeling_strategy import (
        GrandmasterTrainer,
        GrandmasterModelConfig,
        ModelingStrategy,
        create_grandmaster_modeling_system
    )
    from improved_model_factory import ImprovedModelFactory
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 의존성 로드 실패: {e}")
    print("   기본 구현을 사용합니다.")
    DEPENDENCIES_AVAILABLE = False

warnings.filterwarnings('ignore')

class KaggleWinnerPipeline:
    """
    🏆 캐글 우승 파이프라인
    모든 컴포넌트를 통합한 완전한 훈련 시스템
    """
    
    def __init__(self, 
                 data_path: str = "/home/james/doc-classification/computervisioncompetition-cv3/data",
                 strategy: str = "diverse_ensemble",
                 target_score: float = 0.95,
                 experiment_name: str = None,
                 fast_mode: bool = False):
        """
        Args:
            data_path: 데이터 경로
            strategy: 모델링 전략
            target_score: 목표 점수
            experiment_name: 실험 이름
            fast_mode: 빠른 실행 모드 (최소 설정으로 최대 속도)
        """
        
        self.data_path = Path(data_path)
        self.strategy = strategy
        self.target_score = target_score
        self.experiment_name = experiment_name or f"kaggle_winner_{datetime.now().strftime('%m%d_%H%M')}"
        self.fast_mode = fast_mode
        
        # 결과 저장 경로
        self.results_dir = Path(f"./results/{self.experiment_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트들
        self.processor = None
        self.trainer = None
        self.competition_data = None
        
        print(f"🏆 캐글 우승 파이프라인 초기화")
        print(f"   실험명: {self.experiment_name}")
        print(f"   전략: {strategy}")
        print(f"   목표 점수: {target_score}")
        print(f"   빠른 모드: {'ON' if fast_mode else 'OFF'}")
        print(f"   결과 저장: {self.results_dir}")
    
    def setup_components(self):
        """모든 컴포넌트 설정"""
        
        print("\n=== 🔧 컴포넌트 설정 ===")
        
        # 빠른 모드 설정
        if self.fast_mode:
            print("⚡ 빠른 모드 활성화 - 최소 설정으로 최대 속도")
            image_size = 224  # 작은 이미지 크기
            strategy = "basic"  # 기본 전략
        else:
            image_size = 640
            strategy = "eda_optimized"
        
        # 1. 전처리 프로세서 생성
        if DEPENDENCIES_AVAILABLE:
            print("📊 전처리 프로세서 생성 중...")
            self.processor = create_grandmaster_processor(
                strategy=strategy,
                image_size=image_size,
                experiment_name=self.experiment_name
            )
            print("✅ 전처리 프로세서 생성 완료")
        else:
            print("⚠️ 전처리 프로세서를 사용할 수 없습니다.")
            return False
        
        # 2. 모델링 시스템 생성
        print("🧠 모델링 시스템 생성 중...")
        
        # 빠른 모드용 설정
        if self.fast_mode:
            model_config = GrandmasterModelConfig(
                strategy=ModelingStrategy.SINGLE_BEST,
                target_score=0.85,  # 낮은 목표 점수
                image_size=image_size,
                batch_size=16,  # 작은 배치 크기
                ensemble_size=1,  # 단일 모델
                ensemble_architectures=["efficientnet_b0"],  # 가장 빠른 모델
                max_epochs=5,  # 적은 에포크
                early_stopping_patience=2,
                use_pseudo_labeling=False,
                use_knowledge_distillation=False,
                use_mixup_cutmix=False,
                use_test_time_augmentation=False
            )
            self.trainer = GrandmasterTrainer(model_config)
        else:
            self.trainer, _ = create_grandmaster_modeling_system(
                strategy=self.strategy,
                target_score=self.target_score,
                experiment_name=self.experiment_name
            )
        
        # 3. 모델들 생성
        print("🏗️ 앙상블 모델들 생성 중...")
        models = self.trainer.create_ensemble_models()
        self.trainer.setup_training_components()
        
        print(f"✅ 컴포넌트 설정 완료 (모델 수: {len(models)}개)")
        return True
    
    def load_and_prepare_data(self, fold_idx: int = 0):
        """데이터 로드 및 준비"""
        
        print(f"\n=== 📊 데이터 준비 (Fold {fold_idx}) ===")
        
        if not DEPENDENCIES_AVAILABLE:
            print("⚠️ 데이터 로더를 사용할 수 없습니다.")
            return False
        
        try:
            # 대회 데이터 로드
            self.competition_data = load_competition_data(self.processor, fold_idx)
            
            datasets = self.competition_data['datasets']
            dataloaders = self.competition_data['dataloaders']
            
            print(f"✅ 데이터 로드 완료:")
            print(f"   훈련: {len(datasets['train'])}개")
            print(f"   검증: {len(datasets['valid'])}개")
            print(f"   테스트: {len(datasets['test'])}개")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def run_training(self):
        """실제 훈련 실행"""
        
        print(f"\n=== 🚀 훈련 시작 ===")
        
        if not self.competition_data:
            print("❌ 데이터가 준비되지 않았습니다.")
            return None
        
        dataloaders = self.competition_data['dataloaders']
        
        try:
            # 앙상블 훈련 실행
            training_results = self.trainer.train_ensemble(
                train_loader=dataloaders['train'],
                valid_loader=dataloaders['valid'],
                test_loader=dataloaders.get('test', None)
            )
            
            print(f"\n✅ 훈련 완료!")
            print(f"   최고 검증 점수: {training_results['best_score']:.4f}")
            print(f"   훈련 에포크: {training_results['final_epoch']}")
            
            # 결과 저장
            self._save_training_results(training_results)
            
            return training_results
            
        except Exception as e:
            print(f"❌ 훈련 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_predictions(self):
        """최종 예측 생성 (TTA 포함)"""
        
        print(f"\n=== 🔮 최종 예측 생성 ===")
        
        if not self.competition_data:
            print("❌ 데이터가 준비되지 않았습니다.")
            return None
        
        test_loader = self.competition_data['dataloaders'].get('test')
        if not test_loader:
            print("❌ 테스트 데이터 로더가 없습니다.")
            return None
        
        try:
            # TTA를 사용한 예측
            predictions = self.trainer.predict_with_tta(test_loader)
            
            print(f"✅ 예측 완료: {len(predictions)}개 샘플")
            
            # 제출 파일 생성
            submission_path = self._create_submission_file(predictions)
            
            print(f"📄 제출 파일 생성: {submission_path}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_pipeline(self, num_folds: int = 5):
        """완전한 파이프라인 실행 (K-Fold)"""
        
        print(f"\n🏆 캐글 우승 파이프라인 시작!")
        print(f"   K-Fold: {num_folds}")
        print("=" * 60)
        
        # 컴포넌트 설정
        if not self.setup_components():
            print("❌ 컴포넌트 설정 실패")
            return False
        
        all_results = {}
        fold_scores = []
        
        # K-Fold 훈련
        for fold in range(num_folds):
            print(f"\n🔄 Fold {fold+1}/{num_folds} 시작")
            print("-" * 40)
            
            # 데이터 준비
            if not self.load_and_prepare_data(fold_idx=fold):
                print(f"❌ Fold {fold+1} 데이터 준비 실패")
                continue
            
            # 훈련 실행
            fold_results = self.run_training()
            if fold_results:
                fold_score = fold_results['best_score']
                fold_scores.append(fold_score)
                all_results[f'fold_{fold}'] = fold_results
                
                print(f"✅ Fold {fold+1} 완료: {fold_score:.4f}")
                
                # 목표 점수 달성 시 조기 종료 옵션
                if fold_score >= self.target_score:
                    print(f"🎉 목표 점수 달성! Fold {fold+1}에서 {fold_score:.4f}")
            else:
                print(f"❌ Fold {fold+1} 훈련 실패")
        
        # 최종 결과
        if fold_scores:
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            print(f"\n🏆 K-Fold 훈련 완료!")
            print(f"   평균 점수: {avg_score:.4f} ± {std_score:.4f}")
            print(f"   최고 점수: {max(fold_scores):.4f}")
            print(f"   목표 달성: {'✅' if avg_score >= self.target_score else '❌'}")
            
            # 최종 예측 (최고 성능 fold 사용)
            best_fold = fold_scores.index(max(fold_scores))
            print(f"\n🔮 최고 성능 Fold {best_fold+1}로 최종 예측 생성")
            
            # 최고 fold로 다시 설정
            self.load_and_prepare_data(fold_idx=best_fold)
            predictions = self.generate_predictions()
            
            # 최종 결과 저장
            final_results = {
                'avg_score': float(avg_score),
                'std_score': float(std_score),
                'max_score': float(max(fold_scores)),
                'fold_scores': [float(s) for s in fold_scores],
                'best_fold': int(best_fold),
                'target_achieved': avg_score >= self.target_score,
                'all_fold_results': all_results
            }
            
            self._save_final_results(final_results)
            
            return True
        else:
            print("❌ 모든 fold 훈련 실패")
            return False
    
    def _save_training_results(self, results: Dict[str, Any]):
        """훈련 결과 저장"""
        
        results_path = self.results_dir / "training_results.json"
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {
            'best_score': float(results['best_score']),
            'final_epoch': int(results['final_epoch']),
            'experiment_name': self.experiment_name,
            'strategy': self.strategy,
            'target_score': self.target_score,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 훈련 결과 저장: {results_path}")
    
    def _make_json_serializable(self, obj):
        """
        JSON 직렬화 가능하도록 변환
        
        Args:
            obj: 직렬화할 객체
            
        Returns:
            JSON 직렬화 가능한 객체
        """
        import numpy as np
        
        # 딕셔너리 처리
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        
        # 리스트/튜플 처리
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        
        # Numpy 정수 타입들
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, 
                             np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        # Numpy 실수 타입들
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        
        # Numpy 불린 타입 (핵심 수정 부분)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        
        # Numpy 배열
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # 일반 Python 타입들은 그대로 반환
        else:
            return obj
    
    def _save_final_results(self, results: Dict[str, Any]):
        """최종 결과 저장"""
        
        results_path = self.results_dir / "final_results.json"
        
        # JSON 직렬화 가능하도록 변환
        serializable_results = self._make_json_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 최종 결과 저장: {results_path}")
    
    def _create_submission_file(self, predictions: np.ndarray) -> Path:
        """제출 파일 생성"""
        
        # 샘플 제출 파일 로드
        sample_submission_path = self.data_path / "sample_submission.csv"
        
        if sample_submission_path.exists():
            submission_df = pd.read_csv(sample_submission_path)
            submission_df['target'] = predictions
        else:
            # 기본 제출 파일 생성
            submission_df = pd.DataFrame({
                'ID': [f'test_{i}.jpg' for i in range(len(predictions))],
                'target': predictions
            })
        
        # 제출 파일 저장
        submission_path = self.results_dir / f"submission_{self.experiment_name}.csv"
        submission_df.to_csv(submission_path, index=False)
        
        return submission_path


def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(description="캐글 1등 달성 훈련 파이프라인")
    
    parser.add_argument("--data_path", type=str, 
                       default="/home/james/doc-classification/computervisioncompetition-cv3/data",
                       help="데이터 경로")
    parser.add_argument("--strategy", type=str, default="diverse_ensemble",
                       choices=["single_best", "diverse_ensemble", "stacking_ensemble"],
                       help="모델링 전략")
    parser.add_argument("--target_score", type=float, default=0.95,
                       help="목표 점수")
    parser.add_argument("--num_folds", type=int, default=5,
                       help="K-Fold 수")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="실험 이름")
    parser.add_argument("--single_fold", type=int, default=None,
                       help="단일 fold만 실행 (디버깅용)")
    
    args = parser.parse_args()
    
    print("🏆 캐글 1등 달성 훈련 파이프라인")
    print("=" * 50)
    print(f"데이터 경로: {args.data_path}")
    print(f"전략: {args.strategy}")
    print(f"목표 점수: {args.target_score}")
    print(f"K-Fold: {args.num_folds}")
    print("=" * 50)
    
    # 파이프라인 생성
    pipeline = KaggleWinnerPipeline(
        data_path=args.data_path,
        strategy=args.strategy,
        target_score=args.target_score,
        experiment_name=args.experiment_name
    )
    
    try:
        if args.single_fold is not None:
            # 단일 fold 실행 (디버깅용)
            print(f"🔍 단일 Fold {args.single_fold} 실행 (디버깅 모드)")
            
            if pipeline.setup_components():
                if pipeline.load_and_prepare_data(fold_idx=args.single_fold):
                    results = pipeline.run_training()
                    if results:
                        pipeline.generate_predictions()
                        print("✅ 단일 fold 실행 완료")
                    else:
                        print("❌ 훈련 실패")
                else:
                    print("❌ 데이터 준비 실패")
            else:
                print("❌ 컴포넌트 설정 실패")
        else:
            # 완전한 파이프라인 실행
            success = pipeline.run_complete_pipeline(num_folds=args.num_folds)
            
            if success:
                print(f"\n🎉 캐글 우승 파이프라인 완료!")
                print(f"   결과 확인: {pipeline.results_dir}")
            else:
                print(f"\n❌ 파이프라인 실행 실패")
                return 1
                
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 중단됨")
        return 1
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
🏆 그랜드마스터 훈련 실행기
03_modeling의 최고 성능 시스템과 완전 연동

Features:
- 7개 최고 성능 모델 앙상블
- RTX 4090 Laptop 최적화
- EDA 기반 Domain Adaptation
- Progressive Pseudo Labeling
- Test-Time Augmentation

Usage:
    python grandmaster_train.py --mode quick_test
    python grandmaster_train.py --mode full_training
    python grandmaster_train.py --mode competition_ready
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import torch

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "01_EDA"))
sys.path.append(str(project_root / "02_preprocessing"))  
sys.path.append(str(project_root / "03_modeling"))

try:
    from kaggle_winner_training import KaggleWinnerPipeline
    from grandmaster_modeling_strategy import (
        GrandmasterTrainer, GrandmasterModelConfig, 
        ModelingStrategy, create_grandmaster_modeling_system
    )
    GRANDMASTER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 그랜드마스터 시스템 로드 실패: {e}")
    GRANDMASTER_AVAILABLE = False

warnings.filterwarnings('ignore')


class GrandmasterExecutor:
    """
    🏆 그랜드마스터 훈련 실행기
    03_modeling 시스템의 편리한 실행 인터페이스
    """
    
    def __init__(self, 
                 data_path: str = None,
                 experiment_name: str = None):
        
        # 기본 경로 설정
        if data_path is None:
            data_path = str(project_root.parent / "data")
        
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.results_dir = Path("./training_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"🏆 그랜드마스터 훈련 실행기 초기화")
        print(f"   데이터 경로: {data_path}")
        print(f"   실험명: {experiment_name}")
        print(f"   결과 저장: {self.results_dir}")
    
    def quick_test(self):
        """빠른 테스트 - 1개 모델, 1 fold"""
        
        print("\n🔍 빠른 테스트 모드")
        print("=" * 40)
        
        if not GRANDMASTER_AVAILABLE:
            print("❌ 그랜드마스터 시스템을 사용할 수 없습니다.")
            return False
        
        try:
            # 단순화된 설정으로 파이프라인 생성
            pipeline = KaggleWinnerPipeline(
                data_path=self.data_path,
                strategy="single_best",  # 단일 모델
                target_score=0.90,      # 낮은 목표
                experiment_name=self.experiment_name or "quick_test"
            )
            
            # 컴포넌트 설정
            if pipeline.setup_components():
                # 단일 fold 테스트
                if pipeline.load_and_prepare_data(fold_idx=0):
                    results = pipeline.run_training()
                    if results:
                        print(f"✅ 테스트 완료! 점수: {results['best_score']:.4f}")
                        return True
            
            return False
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def full_training(self):
        """전체 훈련 - 앙상블, 5-fold"""
        
        print("\n🚀 전체 훈련 모드")
        print("=" * 40)
        
        if not GRANDMASTER_AVAILABLE:
            print("❌ 그랜드마스터 시스템을 사용할 수 없습니다.")
            return False
        
        try:
            pipeline = KaggleWinnerPipeline(
                data_path=self.data_path,
                strategy="diverse_ensemble",  # 7개 모델 앙상블
                target_score=0.95,          # 높은 목표
                experiment_name=self.experiment_name or "full_training"
            )
            
            # 전체 파이프라인 실행
            success = pipeline.run_complete_pipeline(num_folds=5)
            
            if success:
                print(f"🎉 전체 훈련 완료!")
                print(f"   결과 확인: {pipeline.results_dir}")
                return True
            else:
                print("❌ 훈련 실패")
                return False
                
        except Exception as e:
            print(f"❌ 훈련 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def competition_ready(self):
        """대회 준비 - 최고 설정"""
        
        print("\n🏆 대회 준비 모드 (캐글 1등 도전!)")
        print("=" * 50)
        
        if not GRANDMASTER_AVAILABLE:
            print("❌ 그랜드마스터 시스템을 사용할 수 없습니다.")
            return False
        
        # GPU 정보 출력
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        try:
            pipeline = KaggleWinnerPipeline(
                data_path=self.data_path,
                strategy="diverse_ensemble",
                target_score=0.95,  # 캐글 1등 목표
                experiment_name=self.experiment_name or "competition_ready"
            )
            
            # 최고 설정으로 파이프라인 실행
            success = pipeline.run_complete_pipeline(num_folds=5)
            
            if success:
                print(f"\n🏆 캐글 1등 준비 완료!")
                print(f"   제출 파일: {pipeline.results_dir}/submission_*.csv")
                print(f"   모든 결과: {pipeline.results_dir}")
                
                # 추가 정보
                print(f"\n📊 최종 체크리스트:")
                print(f"   ✅ 7개 최고 성능 모델 앙상블")
                print(f"   ✅ EDA 기반 Domain Adaptation") 
                print(f"   ✅ Progressive Pseudo Labeling")
                print(f"   ✅ Test-Time Augmentation")
                print(f"   ✅ 5-Fold Cross Validation")
                print(f"   ✅ RTX 4090 최적화")
                
                return True
            else:
                print("❌ 대회 준비 실패")
                return False
                
        except Exception as e:
            print(f"❌ 대회 준비 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def debug_mode(self):
        """디버그 모드 - 시스템 점검"""
        
        print("\n🔧 디버그 모드")
        print("=" * 30)
        
        # 시스템 점검
        print("📋 시스템 점검:")
        print(f"   Python: {sys.version}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 모듈 점검
        print(f"\n📦 모듈 점검:")
        print(f"   그랜드마스터 시스템: {'✅' if GRANDMASTER_AVAILABLE else '❌'}")
        
        # 데이터 점검
        data_path = Path(self.data_path)
        print(f"\n📁 데이터 점검:")
        print(f"   데이터 경로: {data_path}")
        print(f"   경로 존재: {'✅' if data_path.exists() else '❌'}")
        
        if data_path.exists():
            train_csv = data_path / "train.csv"
            test_csv = data_path / "sample_submission.csv"
            train_dir = data_path / "train"
            test_dir = data_path / "test"
            
            print(f"   train.csv: {'✅' if train_csv.exists() else '❌'}")
            print(f"   sample_submission.csv: {'✅' if test_csv.exists() else '❌'}")
            print(f"   train/ 폴더: {'✅' if train_dir.exists() else '❌'}")
            print(f"   test/ 폴더: {'✅' if test_dir.exists() else '❌'}")
            
            if train_dir.exists():
                train_files = len(list(train_dir.glob("*.jpg")))
                print(f"   훈련 이미지: {train_files}개")
            
            if test_dir.exists():
                test_files = len(list(test_dir.glob("*.jpg")))
                print(f"   테스트 이미지: {test_files}개")
        
        # 권장사항
        print(f"\n💡 권장사항:")
        if not GRANDMASTER_AVAILABLE:
            print("   - 03_modeling 폴더의 의존성을 확인하세요")
        if not torch.cuda.is_available():
            print("   - GPU 사용을 위해 CUDA를 설치하세요")
        if not data_path.exists():
            print("   - 올바른 데이터 경로를 지정하세요")
        
        return True


def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(description="🏆 그랜드마스터 훈련 실행기")
    
    parser.add_argument("--mode", type=str, default="quick_test",
                       choices=["quick_test", "full_training", "competition_ready", "debug"],
                       help="실행 모드")
    parser.add_argument("--data_path", type=str, default=None,
                       help="데이터 경로")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="실험 이름")
    
    args = parser.parse_args()
    
    print("🏆 그랜드마스터 훈련 실행기")
    print("=" * 50)
    print(f"모드: {args.mode}")
    print(f"데이터 경로: {args.data_path or '기본값 사용'}")
    print(f"실험명: {args.experiment_name or '자동 생성'}")
    print("=" * 50)
    
    # 실행기 생성
    executor = GrandmasterExecutor(
        data_path=args.data_path,
        experiment_name=args.experiment_name
    )
    
    # 모드별 실행
    try:
        if args.mode == "quick_test":
            success = executor.quick_test()
        elif args.mode == "full_training":
            success = executor.full_training()
        elif args.mode == "competition_ready":
            success = executor.competition_ready()
        elif args.mode == "debug":
            success = executor.debug_mode()
        else:
            print(f"❌ 알 수 없는 모드: {args.mode}")
            success = False
        
        if success:
            print(f"\n🎉 '{args.mode}' 모드 실행 완료!")
            return 0
        else:
            print(f"\n❌ '{args.mode}' 모드 실행 실패!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 중단됨")
        return 1
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
🚀 캐글 대회 실행 스크립트 (간단 버전)
Quick Start Script for Kaggle Competition

사용법:
python run_competition.py
"""

import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from kaggle_winner_training import KaggleWinnerPipeline
    
    def quick_start():
        """빠른 시작 - 기본 설정으로 실행"""
        
        print("🏆 캐글 대회 빠른 시작!")
        print("=" * 40)
        
        # 빠른 모드로 파이프라인 생성
        pipeline = KaggleWinnerPipeline(
            strategy="single_best",
            target_score=0.85,
            experiment_name="quick_start",
            fast_mode=True  # 빠른 모드 활성화
        )
        
        # 단일 fold로 테스트 실행
        print("🔍 테스트 실행 (Fold 0)")
        
        if pipeline.setup_components():
            if pipeline.load_and_prepare_data(fold_idx=0):
                results = pipeline.run_training()
                if results:
                    predictions = pipeline.generate_predictions()
                    if predictions is not None:
                        print("✅ 테스트 실행 완료!")
                        print(f"   검증 점수: {results['best_score']:.4f}")
                        print(f"   결과 저장: {pipeline.results_dir}")
                        return True
        
        print("❌ 실행 실패")
        return False
    
    def full_training():
        """전체 훈련 - K-Fold 실행"""
        
        print("🏆 전체 훈련 시작!")
        print("=" * 40)
        
        pipeline = KaggleWinnerPipeline(
            strategy="diverse_ensemble",
            target_score=0.95,
            experiment_name="full_training"
        )
        
        return pipeline.run_complete_pipeline(num_folds=5)
    
    if __name__ == "__main__":
        print("캐글 대회 실행 옵션:")
        print("1. 빠른 테스트 (1 fold)")
        print("2. 전체 훈련 (5 folds)")
        
        choice = input("선택하세요 (1 또는 2): ").strip()
        
        if choice == "1":
            success = quick_start()
        elif choice == "2":
            success = full_training()
        else:
            print("잘못된 선택입니다. 빠른 테스트를 실행합니다.")
            success = quick_start()
        
        if success:
            print("\n🎉 실행 완료!")
        else:
            print("\n❌ 실행 실패!")
            
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    print("필요한 의존성이 설치되어 있는지 확인하세요.")

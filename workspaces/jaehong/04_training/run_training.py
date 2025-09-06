#!/usr/bin/env python3
"""
⚡ 간단한 훈련 실행 스크립트
사용자 친화적인 인터페이스

Usage:
    python run_training.py
"""

import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from grandmaster_train import GrandmasterExecutor
    
    def interactive_training():
        """대화형 훈련 실행"""
        
        print("🏆 캐글 문서 분류 대회 훈련 시스템")
        print("=" * 50)
        
        # 데이터 경로 확인
        default_data_path = str(Path(__file__).parent.parent.parent / "data")
        print(f"기본 데이터 경로: {default_data_path}")
        
        data_path = input("데이터 경로를 입력하세요 (엔터: 기본값 사용): ").strip()
        if not data_path:
            data_path = default_data_path
        
        # 실행 모드 선택
        print("\n실행 모드를 선택하세요:")
        print("1. 🔍 빠른 테스트 (1개 모델, 빠름)")
        print("2. 🚀 전체 훈련 (7개 모델 앙상블, 5-fold)")
        print("3. 🏆 대회 준비 (최고 설정, 캐글 1등 도전)")
        print("4. 🔧 디버그 (시스템 점검)")
        
        choice = input("\n선택하세요 (1-4): ").strip()
        
        mode_map = {
            "1": "quick_test",
            "2": "full_training", 
            "3": "competition_ready",
            "4": "debug"
        }
        
        mode = mode_map.get(choice, "quick_test")
        
        # 실험명 입력
        experiment_name = input("\n실험명을 입력하세요 (엔터: 자동 생성): ").strip()
        if not experiment_name:
            experiment_name = None
        
        print(f"\n🚀 실행 시작!")
        print(f"   모드: {mode}")
        print(f"   데이터: {data_path}")
        print(f"   실험명: {experiment_name or '자동 생성'}")
        print("-" * 50)
        
        # 실행기 생성 및 실행
        executor = GrandmasterExecutor(
            data_path=data_path,
            experiment_name=experiment_name
        )
        
        if mode == "quick_test":
            success = executor.quick_test()
        elif mode == "full_training":
            success = executor.full_training()
        elif mode == "competition_ready":
            success = executor.competition_ready()
        elif mode == "debug":
            success = executor.debug_mode()
        
        if success:
            print(f"\n🎉 실행 완료!")
            
            if mode in ["full_training", "competition_ready"]:
                print(f"\n📄 다음 단계:")
                print(f"   1. training_results/ 폴더에서 결과 확인")
                print(f"   2. submission_*.csv 파일을 캐글에 제출")
                print(f"   3. 리더보드 점수 확인")
        else:
            print(f"\n❌ 실행 실패!")
        
        return success
    
    if __name__ == "__main__":
        try:
            success = interactive_training()
            exit(0 if success else 1)
        except KeyboardInterrupt:
            print(f"\n⏹️ 사용자에 의해 중단됨")
            exit(1)
        except Exception as e:
            print(f"\n💥 오류: {e}")
            exit(1)
            
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    print("grandmaster_train.py 파일이 있는지 확인하세요.")
    exit(1)

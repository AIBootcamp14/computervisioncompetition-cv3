#!/usr/bin/env python3
"""
🚀 통합 평가 실행기
사용자 친화적인 평가 시스템 실행 스크립트

Features:
- 간단한 메뉴 인터페이스
- 다양한 평가 모드 지원
- 자동 설정 및 실행
- 결과 요약 제공
"""

import os
import sys
import argparse
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def print_banner():
    """배너 출력"""
    print("🎯" + "="*50)
    print("🏆 캐글 그랜드마스터 평가 시스템")
    print("🎯" + "="*50)
    print()

def print_menu():
    """메뉴 출력"""
    print("📋 평가 모드 선택:")
    print("1. 🚀 빠른 평가 (단일 모델)")
    print("2. 🎭 앙상블 평가 (다중 모델)")
    print("3. 🔍 상세 분석 (성능 분석)")
    print("4. 📊 TTA 예측 (Test-Time Augmentation)")
    print("5. 🎯 전체 평가 (모든 기능)")
    print("6. ❌ 종료")
    print()

def get_data_path():
    """데이터 경로 입력 받기"""
    default_path = "/home/james/doc-classification/computervisioncompetition-cv3/data"
    
    print(f"📁 데이터 경로를 입력하세요 (기본값: {default_path}):")
    data_path = input().strip()
    
    if not data_path:
        data_path = default_path
    
    if not Path(data_path).exists():
        print(f"❌ 경로가 존재하지 않습니다: {data_path}")
        return None
    
    return data_path

def run_quick_evaluation(data_path):
    """빠른 평가 실행"""
    print("\n🚀 빠른 평가 모드")
    print("="*30)
    
    try:
        from main_evaluation import main as eval_main
        
        # 빠른 평가 설정
        sys.argv = [
            'main_evaluation.py',
            '--mode', 'quick',
            '--data_path', data_path,
            '--output_dir', 'evaluation_results/quick_eval'
        ]
        
        eval_main()
        print("✅ 빠른 평가 완료!")
        
    except Exception as e:
        print(f"❌ 빠른 평가 실패: {e}")

def run_ensemble_evaluation(data_path):
    """앙상블 평가 실행"""
    print("\n🎭 앙상블 평가 모드")
    print("="*30)
    
    try:
        from main_evaluation import main as eval_main
        
        # 앙상블 평가 설정
        sys.argv = [
            'main_evaluation.py',
            '--mode', 'ensemble',
            '--data_path', data_path,
            '--output_dir', 'evaluation_results/ensemble_eval'
        ]
        
        eval_main()
        print("✅ 앙상블 평가 완료!")
        
    except Exception as e:
        print(f"❌ 앙상블 평가 실패: {e}")

def run_detailed_analysis(data_path):
    """상세 분석 실행"""
    print("\n🔍 상세 분석 모드")
    print("="*30)
    
    try:
        from performance_analyzer import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer(data_path)
        analyzer.run_full_analysis()
        print("✅ 상세 분석 완료!")
        
    except Exception as e:
        print(f"❌ 상세 분석 실패: {e}")

def run_tta_prediction(data_path):
    """TTA 예측 실행"""
    print("\n📊 TTA 예측 모드")
    print("="*30)
    
    try:
        from tta_predictor import TTAPredictor
        
        predictor = TTAPredictor(data_path)
        predictor.run_tta_evaluation()
        print("✅ TTA 예측 완료!")
        
    except Exception as e:
        print(f"❌ TTA 예측 실패: {e}")

def run_full_evaluation(data_path):
    """전체 평가 실행"""
    print("\n🎯 전체 평가 모드")
    print("="*30)
    
    try:
        from main_evaluation import main as eval_main
        
        # 전체 평가 설정
        sys.argv = [
            'main_evaluation.py',
            '--mode', 'full',
            '--data_path', data_path,
            '--output_dir', 'evaluation_results/full_eval'
        ]
        
        eval_main()
        print("✅ 전체 평가 완료!")
        
    except Exception as e:
        print(f"❌ 전체 평가 실패: {e}")

def main():
    """메인 함수"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("선택하세요 (1-6): ").strip()
            
            if choice == '1':
                data_path = get_data_path()
                if data_path:
                    run_quick_evaluation(data_path)
                    
            elif choice == '2':
                data_path = get_data_path()
                if data_path:
                    run_ensemble_evaluation(data_path)
                    
            elif choice == '3':
                data_path = get_data_path()
                if data_path:
                    run_detailed_analysis(data_path)
                    
            elif choice == '4':
                data_path = get_data_path()
                if data_path:
                    run_tta_prediction(data_path)
                    
            elif choice == '5':
                data_path = get_data_path()
                if data_path:
                    run_full_evaluation(data_path)
                    
            elif choice == '6':
                print("👋 평가 시스템을 종료합니다.")
                break
                
            else:
                print("❌ 잘못된 선택입니다. 1-6 중에서 선택하세요.")
                
        except KeyboardInterrupt:
            print("\n👋 평가 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()

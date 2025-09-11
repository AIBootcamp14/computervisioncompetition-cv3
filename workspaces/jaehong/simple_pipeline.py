#!/usr/bin/env python3
"""
🚀 간단한 파이프라인 실행기 (Simple Pipeline Runner)
사용자 친화적인 간단한 인터페이스

사용법:
    python simple_pipeline.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime


def print_banner():
    """배너 출력"""
    print("🏆" + "="*50)
    print("🚀 간단 파이프라인")
    print("🏆" + "="*50)
    print()


def print_menu():
    """메뉴 출력"""
    print("📋 실행 모드 선택:")
    print("1. 🔍 빠른 테스트 (Quick Test)")
    print("2. 🚀 전체 훈련 (Full Training)")
    print("3. 🏆 대회 준비 (Competition Ready)")
    print("4. 🔧 개별 단계 실행")
    print("5. ❌ 종료")
    print()


def run_command(command, description):
    """명령어 실행"""
    print(f"🔄 {description} 실행 중...")
    print(f"   명령어: {command}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        duration = time.time() - start_time
        print(f"✅ {description} 완료 ({duration:.1f}초)")
        
        if result.stdout:
            print("📋 출력:")
            print(result.stdout)
            
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ {description} 실패 ({duration:.1f}초)")
        print(f"   오류: {e}")
        
        if e.stdout:
            print("📋 표준 출력:")
            print(e.stdout)
            
        if e.stderr:
            print("❌ 오류 출력:")
            print(e.stderr)
            
        return False


def run_quick_test():
    """빠른 테스트 실행"""
    print("🔍 빠른 테스트 모드")
    print("-" * 30)
    
    # 현재 디렉토리로 이동
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    # 파이프라인 실행
    command = f"python run_pipeline.py --mode quick_test --verbose"
    return run_command(command, "빠른 테스트 파이프라인")


def run_full_training():
    """전체 훈련 실행"""
    print("🚀 전체 훈련 모드")
    print("-" * 30)
    
    # 현재 디렉토리로 이동
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    # 파이프라인 실행
    command = f"python run_pipeline.py --mode full_training --verbose"
    return run_command(command, "전체 훈련 파이프라인")


def run_competition_ready():
    """대회 준비 모드 실행"""
    print("🏆 대회 준비 모드")
    print("-" * 30)
    
    # 현재 디렉토리로 이동
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    # 파이프라인 실행
    command = f"python run_pipeline.py --mode competition_ready --verbose"
    return run_command(command, "대회 준비 파이프라인")


def run_individual_steps():
    """개별 단계 실행"""
    print("🔧 개별 단계 실행")
    print("-" * 30)
    
    steps = {
        "1": ("01_EDA", "competition_eda.py", "탐색적 데이터 분석"),
        "2": ("02_preprocessing", "grandmaster_processor.py", "데이터 전처리"),
        "3": ("03_modeling", "run_competition.py", "모델링 전략"),
        "4": ("04_training", "run_training.py", "모델 훈련"),
        "5": ("05_evaluation", "run_evaluation.py", "모델 평가"),
        "6": ("06_submission", "run_submission.py", "제출 파일 생성")
    }
    
    print("📋 실행할 단계 선택:")
    for key, (folder, script, description) in steps.items():
        print(f"{key}. {description} ({folder}/{script})")
    print("0. 돌아가기")
    
    choice = input("\n선택하세요: ").strip()
    
    if choice == "0":
        return True
    
    if choice in steps:
        folder, script, description = steps[choice]
        
        # 해당 폴더로 이동하여 스크립트 실행
        current_dir = Path(__file__).parent
        target_dir = current_dir / folder
        
        if target_dir.exists():
            os.chdir(target_dir)
            command = f"python {script}"
            return run_command(command, description)
        else:
            print(f"❌ 폴더가 존재하지 않습니다: {target_dir}")
            return False
    else:
        print("❌ 잘못된 선택입니다.")
        return False


def check_requirements():
    """필수 요구사항 확인"""
    print("🔍 시스템 요구사항 확인 중...")
    
    # Python 버전 확인
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 필수 폴더 확인
    current_dir = Path(__file__).parent
    required_folders = ["01_EDA", "02_preprocessing", "03_modeling", "04_training", "05_evaluation", "06_submission"]
    
    missing_folders = []
    for folder in required_folders:
        if not (current_dir / folder).exists():
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"❌ 누락된 폴더: {', '.join(missing_folders)}")
        return False
    
    print("✅ 모든 필수 폴더 존재")
    
    # 데이터 경로 확인
    data_path = Path("/home/james/doc-classification/computervisioncompetition-cv3/data")
    if not data_path.exists():
        print(f"⚠️ 데이터 경로가 존재하지 않습니다: {data_path}")
        print("   다른 경로를 사용하거나 데이터를 준비해주세요.")
    
    return True


def main():
    """메인 함수"""
    print_banner()
    
    # 요구사항 확인
    if not check_requirements():
        print("\n❌ 시스템 요구사항을 충족하지 않습니다.")
        return
    
    print("\n✅ 시스템 요구사항 확인 완료!")
    print()
    
    while True:
        print_menu()
        choice = input("선택하세요 (1-5): ").strip()
        
        if choice == "1":
            success = run_quick_test()
        elif choice == "2":
            success = run_full_training()
        elif choice == "3":
            success = run_competition_ready()
        elif choice == "4":
            success = run_individual_steps()
        elif choice == "5":
            print("👋 프로그램을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 1-5 중에서 선택해주세요.")
            continue
        
        if choice in ["1", "2", "3"]:
            if success:
                print("\n🎉 파이프라인 실행 완료!")
                print(f"📁 결과 확인: {Path(__file__).parent}")
            else:
                print("\n❌ 파이프라인 실행 실패!")
                print("   로그를 확인하고 문제를 해결해주세요.")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    main()

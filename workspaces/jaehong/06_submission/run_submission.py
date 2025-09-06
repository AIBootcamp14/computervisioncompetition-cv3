#!/usr/bin/env python3
"""
🚀 통합 제출 실행기
사용자 친화적인 제출 시스템 실행 스크립트

Features:
- 간단한 메뉴 인터페이스
- 다양한 제출 모드 지원
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
    print("🚀" + "="*50)
    print("🏆 캐글 그랜드마스터 제출 시스템")
    print("🚀" + "="*50)
    print()

def print_menu():
    """메뉴 출력"""
    print("📋 제출 모드 선택:")
    print("1. 🎯 최종 제출 (Final Submission)")
    print("2. 🔧 개선된 제출 (Fixed Submission)")
    print("3. 📊 제출 결과 분석")
    print("4. 🗂️ 제출 파일 관리")
    print("5. ❌ 종료")
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

def run_final_submission(data_path):
    """최종 제출 실행"""
    print("\n🎯 최종 제출 모드")
    print("="*30)
    
    try:
        from final_submission import main as final_main
        
        # 최종 제출 설정
        sys.argv = [
            'final_submission.py',
            '--data_path', data_path,
            '--output_dir', 'final_submissions'
        ]
        
        final_main()
        print("✅ 최종 제출 완료!")
        
    except Exception as e:
        print(f"❌ 최종 제출 실패: {e}")

def run_fixed_submission(data_path):
    """개선된 제출 실행"""
    print("\n🔧 개선된 제출 모드")
    print("="*30)
    
    try:
        from fixed_submission_generator import main as fixed_main
        
        # 개선된 제출 설정
        sys.argv = [
            'fixed_submission_generator.py',
            '--data_path', data_path,
            '--output_dir', 'fixed_submissions'
        ]
        
        fixed_main()
        print("✅ 개선된 제출 완료!")
        
    except Exception as e:
        print(f"❌ 개선된 제출 실패: {e}")

def analyze_submissions():
    """제출 결과 분석"""
    print("\n📊 제출 결과 분석")
    print("="*30)
    
    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        # 제출 파일들 찾기
        submission_files = list(Path('.').glob('**/*submission*.csv'))
        
        if not submission_files:
            print("❌ 제출 파일을 찾을 수 없습니다.")
            return
        
        print(f"📁 발견된 제출 파일: {len(submission_files)}개")
        
        for file in submission_files:
            print(f"\n📄 {file.name}:")
            try:
                df = pd.read_csv(file)
                print(f"   - 샘플 수: {len(df)}")
                print(f"   - 클래스 수: {df['class'].nunique()}")
                print(f"   - 클래스 분포:")
                
                class_counts = df['class'].value_counts().sort_index()
                for cls, count in class_counts.head(5).items():
                    print(f"     클래스 {cls}: {count}개")
                
                if len(class_counts) > 5:
                    print(f"     ... 외 {len(class_counts) - 5}개 클래스")
                    
            except Exception as e:
                print(f"   ❌ 분석 실패: {e}")
        
        print("\n✅ 제출 결과 분석 완료!")
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")

def manage_submissions():
    """제출 파일 관리"""
    print("\n🗂️ 제출 파일 관리")
    print("="*30)
    
    try:
        from pathlib import Path
        
        # 제출 폴더들 확인
        submission_dirs = ['final_submissions', 'fixed_submissions']
        
        for dir_name in submission_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                files = list(dir_path.glob('*'))
                print(f"\n📁 {dir_name}/ ({len(files)}개 파일):")
                
                # 파일 크기별로 정렬
                files_with_size = [(f, f.stat().st_size) for f in files if f.is_file()]
                files_with_size.sort(key=lambda x: x[1], reverse=True)
                
                for file, size in files_with_size[:5]:  # 상위 5개만 표시
                    size_mb = size / (1024 * 1024)
                    print(f"   📄 {file.name} ({size_mb:.2f}MB)")
                
                if len(files_with_size) > 5:
                    print(f"   ... 외 {len(files_with_size) - 5}개 파일")
            else:
                print(f"\n📁 {dir_name}/ (폴더 없음)")
        
        print("\n✅ 제출 파일 관리 완료!")
        
    except Exception as e:
        print(f"❌ 관리 실패: {e}")

def main():
    """메인 함수"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("선택하세요 (1-5): ").strip()
            
            if choice == '1':
                data_path = get_data_path()
                if data_path:
                    run_final_submission(data_path)
                    
            elif choice == '2':
                data_path = get_data_path()
                if data_path:
                    run_fixed_submission(data_path)
                    
            elif choice == '3':
                analyze_submissions()
                
            elif choice == '4':
                manage_submissions()
                
            elif choice == '5':
                print("👋 제출 시스템을 종료합니다.")
                break
                
            else:
                print("❌ 잘못된 선택입니다. 1-5 중에서 선택하세요.")
                
        except KeyboardInterrupt:
            print("\n👋 제출 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()

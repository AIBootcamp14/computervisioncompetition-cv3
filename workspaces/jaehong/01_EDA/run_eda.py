#!/usr/bin/env python3
"""
EDA 실행 스크립트

사용법:
    python run_eda.py
    또는
    python run_eda.py --data-root /path/to/data --sample-size 200
"""

import argparse
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from competition_eda import CompetitionEDA


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='문서 분류 대회 EDA 실행')
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='/home/james/doc-classification/computervisioncompetition-cv3/data',
        help='데이터 루트 디렉토리 (기본값: ../../../data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./eda_results',
        help='결과 저장 디렉토리 (기본값: ./eda_results)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=200,
        help='분석할 샘플 크기 (기본값: 200)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='빠른 분석 모드 (샘플 크기 축소)'
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # 데이터 경로 확인
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_root}")
        print("💡 --data-root 옵션으로 올바른 경로를 지정해주세요.")
        sys.exit(1)
    
    # 필수 파일들 확인
    required_files = ['train.csv', 'meta.csv', 'sample_submission.csv']
    required_dirs = ['train', 'test']
    
    for file_name in required_files:
        if not (data_root / file_name).exists():
            print(f"❌ 필수 파일이 없습니다: {data_root / file_name}")
            sys.exit(1)
    
    for dir_name in required_dirs:
        if not (data_root / dir_name).exists():
            print(f"❌ 필수 디렉토리가 없습니다: {data_root / dir_name}")
            sys.exit(1)
    
    print(f"✅ 데이터 디렉토리 확인 완료: {data_root}")
    
    # 빠른 모드 설정
    if args.quick:
        print("🚀 빠른 분석 모드 활성화")
        # 여기서 샘플 크기를 더 작게 설정할 수 있음
    
    try:
        # EDA 실행
        print("🏆 Competition EDA 시작...")
        eda = CompetitionEDA(data_root)
        eda.run_complete_analysis()
        
        print(f"\n🎉 EDA 완료!")
        print(f"📁 결과는 {eda.output_dir}에 저장되었습니다.")
        
        # 생성된 파일들 나열
        result_files = list(eda.output_dir.glob('*'))
        if result_files:
            print(f"\n📄 생성된 파일들:")
            for file_path in sorted(result_files):
                print(f"  • {file_path.name}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 중단했습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

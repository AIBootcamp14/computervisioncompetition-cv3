"""
🏆 Final Submission System
시니어 그랜드마스터 수준의 최종 제출 시스템

Features:
- 05_evaluation 결과 활용
- TTA 기반 최종 예측
- 다중 모델 앙상블
- 완벽한 검증 및 분석
"""

import argparse
import warnings
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime

# 경고 무시
warnings.filterwarnings('ignore')

# 05_evaluation 모듈 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "05_evaluation"))
from tta_predictor import TTAPredictor
from evaluation_config import EvaluationConfigManager

# 04_training 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent.parent / "04_training"))
from model import DocumentClassifier
from utils import set_seed, get_gpu_info, optimize_gpu_settings

# 로컬 모듈
from submission_generator import SubmissionGenerator


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="🏆 Final Submission Generation")
    
    # 필수 인수
    parser.add_argument("--training_experiment", type=str, required=True,
                       help="04_training 실험 이름")
    parser.add_argument("--submission_name", type=str, default="final_submission",
                       help="제출 파일 이름")
    
    # TTA 설정
    parser.add_argument("--enable_tta", action="store_true", default=True,
                       help="TTA 활성화")
    parser.add_argument("--n_tta", type=int, default=8,
                       help="TTA 수")
    parser.add_argument("--tta_weights", type=str, default=None,
                       help="TTA 가중치 (콤마로 구분)")
    
    # 앙상블 설정
    parser.add_argument("--enable_ensemble", action="store_true",
                       help="앙상블 활성화")
    parser.add_argument("--ensemble_files", type=str, nargs='+',
                       help="앙상블할 확률 파일들")
    parser.add_argument("--ensemble_weights", type=str, default=None,
                       help="앙상블 가중치 (콤마로 구분)")
    
    # 출력 설정
    parser.add_argument("--output_dir", type=str, default="final_submissions",
                       help="출력 디렉토리")
    parser.add_argument("--create_analysis", action="store_true", default=True,
                       help="분석 리포트 생성")
    
    # 기타
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="배치 크기")
    
    return parser.parse_args()


def load_trained_model(model_path: str, model_config: dict, device: str) -> torch.nn.Module:
    """훈련된 모델 로드"""
    
    print(f"🧠 훈련된 모델 로드 중...")
    
    # 모델 생성
    model = DocumentClassifier(
        architecture=model_config.get('architecture', 'efficientnetv2_s'),
        num_classes=model_config.get('num_classes', 17),
        dropout_rate=model_config.get('dropout_rate', 0.3)
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 모델 가중치 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ 모델 로드 완료:")
    print(f"   아키텍처: {model_config.get('architecture', 'efficientnetv2_s')}")
    print(f"   최고 F1: {checkpoint.get('best_score', 'N/A')}")
    
    return model


def create_single_model_submission(
    args,
    eval_config_manager: EvaluationConfigManager,
    device: str
) -> dict:
    """단일 모델 제출 파일 생성"""
    
    print(f"🎯 단일 모델 제출 파일 생성...")
    
    # 모델 정보 가져오기
    model_info = eval_config_manager.get_model_info()
    data_info = eval_config_manager.get_data_info()
    
    # 모델 로드
    model = load_trained_model(
        model_info['model_path'],
        {
            'architecture': model_info['architecture'],
            'num_classes': model_info['num_classes'],
            'dropout_rate': 0.3
        },
        device
    )
    
    # 테스트 이미지 경로 수집
    sample_df = pd.read_csv(data_info['data_root'] + "/sample_submission.csv")
    test_image_paths = [
        str(Path(data_info['data_root']) / "test" / img_id) 
        for img_id in sample_df['ID']
    ]
    
    if args.enable_tta:
        # TTA 예측
        print(f"🔮 TTA 예측 수행...")
        
        # TTA 가중치 파싱
        tta_weights = None
        if args.tta_weights:
            tta_weights = [float(w.strip()) for w in args.tta_weights.split(',')]
        
        tta_predictor = TTAPredictor(
            model=model,
            device=device,
            tta_weights=tta_weights
        )
        
        predictions, probabilities, tta_stats = tta_predictor.predict_with_tta(
            image_paths=test_image_paths,
            image_size=model_info['image_size'],
            batch_size=args.batch_size
        )
        
        confidence_scores = np.max(probabilities, axis=1)
        
        print(f"✅ TTA 예측 완료:")
        print(f"   평균 신뢰도: {tta_stats['mean_confidence']:.4f}")
        
    else:
        # 단일 예측 (TTA 없음)
        print(f"🎯 단일 예측 수행...")
        
        # 여기서는 간단히 TTA 1개만 사용
        tta_predictor = TTAPredictor(model=model, device=device)
        predictions, probabilities, _ = tta_predictor.predict_with_tta(
            image_paths=test_image_paths,
            image_size=model_info['image_size'],
            batch_size=args.batch_size
        )
        confidence_scores = np.max(probabilities, axis=1)
    
    # 제출 파일 생성
    submission_generator = SubmissionGenerator(
        sample_submission_path=str(Path(data_info['data_root']) / "sample_submission.csv"),
        class_names=data_info['class_names'],
        class_weights=data_info['class_weights']
    )
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'confidence_scores': confidence_scores,
        'submission_generator': submission_generator,
        'model_info': model_info
    }


def create_ensemble_submission(
    args,
    submission_generator: SubmissionGenerator
) -> dict:
    """앙상블 제출 파일 생성"""
    
    print(f"🎭 앙상블 제출 파일 생성...")
    
    # 앙상블 가중치 파싱
    weights = None
    if args.ensemble_weights:
        weights = [float(w.strip()) for w in args.ensemble_weights.split(',')]
    
    return submission_generator.create_ensemble_submission(
        prediction_files=args.ensemble_files,
        weights=weights,
        experiment_name=f"{args.submission_name}_ensemble",
        output_dir=Path(args.output_dir)
    )


def main():
    """메인 함수"""
    print("🏆 Final Submission Generation")
    print("=" * 60)
    
    # 인수 파싱
    args = parse_args()
    
    # 시스템 최적화
    optimize_gpu_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    # 재현성 보장
    set_seed(args.seed)
    
    # 워크스페이스 경로
    workspace_root = Path(__file__).parent.parent
    
    # 평가 설정 관리자 생성
    eval_config_manager = EvaluationConfigManager(
        str(workspace_root), 
        args.training_experiment
    )
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🏆 최종 제출 설정:")
    print(f"   훈련 실험: {args.training_experiment}")
    print(f"   제출 이름: {args.submission_name}")
    print(f"   TTA: {'✅' if args.enable_tta else '❌'} ({args.n_tta if args.enable_tta else 0}개)")
    print(f"   앙상블: {'✅' if args.enable_ensemble else '❌'}")
    print(f"   출력 디렉토리: {output_dir}")
    
    if args.enable_ensemble and args.ensemble_files:
        # 앙상블 제출
        print(f"   앙상블 파일: {len(args.ensemble_files)}개")
        
        # 임시로 submission_generator 생성
        data_info = eval_config_manager.get_data_info()
        submission_generator = SubmissionGenerator(
            sample_submission_path=str(Path(data_info['data_root']) / "sample_submission.csv"),
            class_names=data_info['class_names'],
            class_weights=data_info['class_weights']
        )
        
        submission_info = create_ensemble_submission(args, submission_generator)
        
    else:
        # 단일 모델 제출
        result = create_single_model_submission(args, eval_config_manager, device)
        
        # 제출 파일 생성
        submission_info = result['submission_generator'].create_submission(
            predictions=result['predictions'],
            probabilities=result['probabilities'],
            confidence_scores=result['confidence_scores'],
            experiment_name=args.submission_name,
            output_dir=output_dir,
            save_probabilities=True,
            create_analysis=args.create_analysis
        )
    
    # 최종 요약
    print(f"\n🏁 최종 제출 파일 생성 완료!")
    print(f"   제출 파일: {submission_info['submission_file']}")
    print(f"   예측 수: {submission_info['prediction_count']}")
    print(f"   사용된 클래스: {submission_info['unique_predictions']}개")
    
    if 'confidence_stats' in submission_info:
        conf_stats = submission_info['confidence_stats']
        print(f"   평균 신뢰도: {conf_stats['mean_confidence']:.4f}")
        print(f"   높은 신뢰도 예측: {conf_stats['high_confidence_count']}개")
    
    # Kaggle 제출 안내
    print(f"\n🚀 Kaggle 제출 준비 완료!")
    print(f"   1. 파일 다운로드: {submission_info['submission_file']}")
    print(f"   2. Kaggle 업로드")
    print(f"   3. 성능 확인")
    
    # 성능 예상 (훈련 결과 기반)
    if not args.enable_ensemble:
        model_f1 = result['model_info']['best_f1']
        if model_f1 > 0.85:
            print(f"   🎯 예상 LB 점수: 매우 높음 (0.80+)")
        elif model_f1 > 0.75:
            print(f"   🎯 예상 LB 점수: 높음 (0.70+)")
        else:
            print(f"   🎯 예상 LB 점수: 보통 (0.60+)")
    
    return submission_info


if __name__ == "__main__":
    try:
        submission_info = main()
        print("\n✅ 최종 제출 시스템 실행 성공!")
        
    except Exception as e:
        print(f"\n❌ 최종 제출 시스템 실행 실패: {e}")
        import traceback
        traceback.print_exc()

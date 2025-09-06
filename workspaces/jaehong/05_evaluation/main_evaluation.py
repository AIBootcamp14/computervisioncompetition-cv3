"""
🎯 Main Evaluation System
시니어 그랜드마스터 수준의 통합 평가 시스템

Features:
- 01~04 단계 완전 연계
- TTA 기반 고급 예측
- 종합적인 모델 분석
- 자동 제출 파일 생성
"""

import argparse
import warnings
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# 경고 무시
warnings.filterwarnings('ignore')

# 로컬 모듈 임포트
from evaluation_config import EvaluationConfigManager
from tta_predictor import TTAPredictor, TTATransformFactory
from model_evaluator import ModelEvaluator

# 06_submission 모듈 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "06_submission"))
from submission_generator import SubmissionGenerator

# 04_training 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent.parent / "04_training"))
from config import ConfigManager
from model import ModelFactory, DocumentClassifier
from data_loader import DataLoaderFactory
from utils import set_seed, get_gpu_info, optimize_gpu_settings, log_system_info


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="🎯 Document Classification Evaluation")
    
    # 기본 설정
    parser.add_argument("--training_experiment", type=str, required=True,
                       help="04_training 실험 이름")
    parser.add_argument("--experiment_name", type=str, default="grandmaster_eval",
                       help="평가 실험 이름")
    
    # TTA 설정
    parser.add_argument("--enable_tta", action="store_true", default=True,
                       help="TTA 활성화")
    parser.add_argument("--n_tta", type=int, default=8,
                       help="TTA 수")
    parser.add_argument("--tta_weights", type=str, default=None,
                       help="TTA 가중치 (콤마로 구분)")
    
    # 평가 설정
    parser.add_argument("--batch_size", type=int, default=None,
                       help="배치 크기 (None이면 자동 설정)")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                       help="신뢰도 임계값")
    
    # 출력 설정
    parser.add_argument("--create_submission", action="store_true", default=True,
                       help="제출 파일 생성")
    parser.add_argument("--save_probabilities", action="store_true", default=True,
                       help="예측 확률 저장")
    parser.add_argument("--detailed_analysis", action="store_true", default=True,
                       help="상세 분석 수행")
    
    # 검증 설정
    parser.add_argument("--validate_on_train", action="store_true",
                       help="훈련 데이터로 검증 수행")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                       help="검증 데이터 비율")
    
    # 기타
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="출력 디렉토리")
    
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
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 가중치 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ 모델 로드 완료:")
    print(f"   아키텍처: {model_config.get('architecture', 'efficientnetv2_s')}")
    print(f"   파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def create_test_data_loader(data_factory: DataLoaderFactory, batch_size: int, image_size: int) -> DataLoader:
    """테스트 데이터 로더 생성"""
    
    print(f"🗃️ 테스트 데이터 로더 생성 중...")
    
    test_loader = data_factory.create_test_loader(
        batch_size=batch_size,
        image_size=image_size,
        num_workers=4,
        tta=False
    )
    
    print(f"✅ 테스트 데이터 로더 생성 완료:")
    print(f"   배치 수: {len(test_loader)}")
    print(f"   총 샘플 수: {len(test_loader.dataset)}")
    
    return test_loader


def create_validation_data_loader(
    data_factory: DataLoaderFactory, 
    batch_size: int, 
    image_size: int, 
    val_ratio: float
) -> DataLoader:
    """검증 데이터 로더 생성"""
    
    print(f"🔍 검증 데이터 로더 생성 중...")
    
    _, val_loader = data_factory.create_train_val_loaders(
        batch_size=batch_size,
        val_ratio=val_ratio,
        image_size=image_size,
        augmentation_level="none",  # 검증 시에는 증강 없음
        use_weighted_sampler=False
    )
    
    print(f"✅ 검증 데이터 로더 생성 완료:")
    print(f"   배치 수: {len(val_loader)}")
    print(f"   검증 샘플 수: {len(val_loader.dataset)}")
    
    return val_loader


def main():
    """메인 함수"""
    print("🎯 Document Classification Evaluation")
    print("=" * 60)
    
    # 인수 파싱
    args = parse_args()
    
    # 시스템 정보 출력
    system_info = log_system_info()
    optimize_gpu_settings()
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ 사용 디바이스: {device}")
    
    # 재현성 보장
    set_seed(args.seed)
    
    # 워크스페이스 경로
    workspace_root = Path(__file__).parent.parent
    
    # 평가 설정 관리자 생성
    print(f"\n📊 평가 설정 관리자 초기화...")
    eval_config_manager = EvaluationConfigManager(
        str(workspace_root), 
        args.training_experiment
    )
    
    # TTA 가중치 파싱
    tta_weights = None
    if args.tta_weights:
        tta_weights = [float(w.strip()) for w in args.tta_weights.split(',')]
        if len(tta_weights) != args.n_tta:
            raise ValueError(f"TTA 가중치 수({len(tta_weights)})가 TTA 수({args.n_tta})와 다릅니다.")
    
    # 배치 크기 설정 (None이면 기본값 사용)
    batch_size = args.batch_size if args.batch_size is not None else 32
    
    # 평가 설정 생성
    eval_config = eval_config_manager.create_evaluation_config(
        experiment_name=args.experiment_name,
        enable_tta=args.enable_tta,
        n_tta=args.n_tta,
        batch_size=batch_size
    )
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 저장
    config_file = eval_config_manager.save_evaluation_config(eval_config, output_dir)
    
    print(f"\n🎯 평가 설정:")
    print(f"   실험명: {eval_config.experiment_name}")
    print(f"   훈련 실험: {args.training_experiment}")
    print(f"   TTA: {'✅' if eval_config.tta.enabled else '❌'} ({eval_config.tta.n_tta}개)")
    print(f"   배치 크기: {eval_config.batch_size}")
    print(f"   출력 디렉토리: {output_dir}")
    
    # 모델 정보 및 데이터 정보 가져오기
    model_info = eval_config_manager.get_model_info()
    data_info = eval_config_manager.get_data_info()
    
    print(f"   모델 F1: {model_info['best_f1']:.4f}")
    print(f"   소수 클래스: {model_info['minority_classes']}")
    
    # 훈련된 모델 로드
    model = load_trained_model(
        model_info['model_path'],
        {
            'architecture': model_info['architecture'],
            'num_classes': model_info['num_classes'],
            'dropout_rate': 0.3
        },
        device
    )
    
    # 04_training 설정 관리자 생성 (데이터 로더용)
    training_config_manager = ConfigManager(str(workspace_root))
    data_factory = DataLoaderFactory(training_config_manager)
    
    # 검증 수행 (선택적)
    if args.validate_on_train:
        print(f"\n🔍 훈련 데이터 검증 수행...")
        
        val_loader = create_validation_data_loader(
            data_factory, eval_config.batch_size, model_info['image_size'], args.val_ratio
        )
        
        # 모델 평가기 생성
        evaluator = ModelEvaluator(
            model=model,
            class_names=data_info['class_names'],
            class_weights=data_info['class_weights'],
            device=device
        )
        
        # 종합 평가 수행
        validation_results = evaluator.evaluate_comprehensive(
            data_loader=val_loader,
            save_dir=output_dir / "validation_analysis",
            experiment_name=f"{args.experiment_name}_validation"
        )
        
        print(f"✅ 검증 완료:")
        print(f"   정확도: {validation_results['basic_metrics']['accuracy']:.4f}")
        print(f"   Macro F1: {validation_results['basic_metrics']['macro_f1']:.4f}")
    
    # 테스트 데이터 예측
    print(f"\n🔮 테스트 데이터 예측 시작...")
    
    # 테스트 데이터 로더 생성
    test_loader = create_test_data_loader(
        data_factory, eval_config.batch_size, model_info['image_size']
    )
    
    # 이미지 경로 수집
    sample_df = pd.read_csv(data_info['data_root'] + "/sample_submission.csv")
    test_image_paths = [
        str(Path(data_info['data_root']) / "test" / img_id) 
        for img_id in sample_df['ID']
    ]
    
    if eval_config.tta.enabled:
        # TTA 예측
        print(f"🔮 TTA 예측 수행...")
        
        tta_predictor = TTAPredictor(
            model=model,
            device=device,
            tta_weights=tta_weights
        )
        
        predictions, probabilities, tta_stats = tta_predictor.predict_with_tta(
            image_paths=test_image_paths,
            image_size=model_info['image_size'],
            batch_size=eval_config.batch_size,
            confidence_threshold=args.confidence_threshold
        )
        
        confidence_scores = np.max(probabilities, axis=1)
        
        print(f"✅ TTA 예측 완료:")
        print(f"   평균 신뢰도: {tta_stats['mean_confidence']:.4f}")
        print(f"   예측 분포: {len(tta_stats['class_distribution'])}개 클래스")
        
        # TTA 기여도 분석 (선택적)
        if args.detailed_analysis:
            print(f"🔍 TTA 기여도 분석...")
            tta_analysis = tta_predictor.analyze_tta_contribution(
                test_image_paths[:500],  # 샘플만 분석
                sample_size=100,
                image_size=model_info['image_size']
            )
            
            # TTA 분석 결과 저장
            import json
            with open(output_dir / "tta_analysis.json", 'w') as f:
                json.dump(tta_analysis, f, indent=2, default=str)
    
    else:
        # 단일 예측
        print(f"🎯 단일 예측 수행...")
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device, non_blocking=True)
                
                logits = model(images)
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.append(probs.cpu().numpy())
        
        predictions = np.array(all_predictions)
        probabilities = np.vstack(all_probabilities)
        confidence_scores = np.max(probabilities, axis=1)
        
        print(f"✅ 단일 예측 완료:")
        print(f"   평균 신뢰도: {np.mean(confidence_scores):.4f}")
    
    # 제출 파일 생성
    if args.create_submission:
        print(f"\n📤 제출 파일 생성...")
        
        submission_generator = SubmissionGenerator(
            sample_submission_path=str(Path(data_info['data_root']) / "sample_submission.csv"),
            class_names=data_info['class_names'],
            class_weights=data_info['class_weights']
        )
        
        submission_info = submission_generator.create_submission(
            predictions=predictions,
            probabilities=probabilities if args.save_probabilities else None,
            confidence_scores=confidence_scores,
            experiment_name=args.experiment_name,
            output_dir=output_dir / "submissions",
            save_probabilities=args.save_probabilities,
            create_analysis=args.detailed_analysis
        )
        
        print(f"✅ 제출 파일 생성 완료:")
        print(f"   파일: {submission_info['submission_file']}")
        print(f"   클래스 분포: {len(submission_info['class_distribution'])}개 클래스")
        
        if 'confidence_stats' in submission_info:
            conf_stats = submission_info['confidence_stats']
            print(f"   신뢰도: {conf_stats['mean_confidence']:.4f} ± {conf_stats['std_confidence']:.4f}")
    
    # 최종 요약
    print(f"\n🏁 평가 완료!")
    print(f"   실험명: {args.experiment_name}")
    print(f"   예측 수: {len(predictions)}")
    print(f"   평균 신뢰도: {np.mean(confidence_scores):.4f}")
    print(f"   출력 디렉토리: {output_dir}")
    
    # 성능 예상 출력
    if args.validate_on_train and 'validation_results' in locals():
        val_f1 = validation_results['basic_metrics']['macro_f1']
        print(f"   검증 F1: {val_f1:.4f}")
        
        # 간단한 성능 예측
        if val_f1 > 0.85:
            print(f"   🎯 예상 LB 점수: 매우 높음 (0.80+)")
        elif val_f1 > 0.75:
            print(f"   🎯 예상 LB 점수: 높음 (0.70+)")
        elif val_f1 > 0.65:
            print(f"   🎯 예상 LB 점수: 보통 (0.60+)")
        else:
            print(f"   🎯 예상 LB 점수: 개선 필요")
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'confidence_scores': confidence_scores,
        'submission_info': submission_info if args.create_submission else None,
        'validation_results': validation_results if args.validate_on_train else None
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ 평가 시스템 실행 성공!")
        
    except Exception as e:
        print(f"\n❌ 평가 시스템 실행 실패: {e}")
        import traceback
        traceback.print_exc()

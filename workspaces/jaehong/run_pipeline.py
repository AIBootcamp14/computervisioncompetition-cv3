#!/usr/bin/env python3
"""
통합 파이프라인 실행기
Complete Pipeline Runner for Kaggle Document Classification Competition

Clean Code & Clean Architecture 적용:
- Single Responsibility: 각 단계별 명확한 책임 분리
- Open/Closed: 새로운 단계 추가 시 기존 코드 수정 없이 확장 가능
- Dependency Inversion: 추상화된 인터페이스에 의존
- Interface Segregation: 단계별 작은 인터페이스들로 분리

전체 파이프라인:
01_EDA → 02_preprocessing → 03_modeling → 04_training → 05_evaluation → 06_submission
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 각 단계별 모듈 import
try:
    # 01_EDA
    sys.path.append(str(current_dir / "01_EDA"))
    from competition_eda import CompetitionEDA
    
    # 02_preprocessing  
    sys.path.append(str(current_dir / "02_preprocessing"))
    from grandmaster_processor import GrandmasterProcessor
    
    # 03_modeling
    sys.path.append(str(current_dir / "03_modeling"))
    from kaggle_winner_training import KaggleWinnerPipeline
    
    # 04_training
    sys.path.append(str(current_dir / "04_training"))
    from grandmaster_train import GrandmasterExecutor
    
    # 05_evaluation
    sys.path.append(str(current_dir / "05_evaluation"))
    from main_evaluation import MainEvaluator
    
    # 06_submission
    sys.path.append(str(current_dir / "06_submission"))
    from final_submission import FinalSubmissionGenerator
    
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("각 폴더의 의존성을 확인해주세요.")
    sys.exit(1)


class PipelineMode(Enum):
    """파이프라인 실행 모드"""
    QUICK_TEST = "quick_test"          # 빠른 테스트 (1개 모델, 1-fold)
    FULL_TRAINING = "full_training"    # 전체 훈련 (7개 모델, 5-fold)
    COMPETITION_READY = "competition_ready"  # 대회 준비 (최고 설정)
    DEBUG = "debug"                    # 디버그 모드


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    mode: PipelineMode = PipelineMode.QUICK_TEST
    data_path: str = "/home/james/doc-classification/computervisioncompetition-cv3/data"
    output_dir: str = "/home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong"
    experiment_name: str = field(default_factory=lambda: f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    skip_steps: List[str] = field(default_factory=list)
    verbose: bool = True
    save_intermediate: bool = True


class PipelineStep(ABC):
    """파이프라인 단계 추상 클래스"""
    
    def __init__(self, name: str, config: PipelineConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"Pipeline.{name}")
        
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """단계 실행"""
        pass
        
    @abstractmethod
    def can_skip(self, context: Dict[str, Any]) -> bool:
        """단계 건너뛰기 가능 여부"""
        pass


class EDAStep(PipelineStep):
    """01_EDA 단계"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("EDA", config)
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """EDA 실행"""
        self.logger.info("🔍 탐색적 데이터 분석 시작")
        
        try:
            # EDA 실행
            eda = CompetitionEDA(
                data_path=self.config.data_path,
                output_dir=Path(self.config.output_dir) / "01_EDA" / "eda_results"
            )
            
            # 전체 분석 실행
            results = eda.run_complete_analysis()
            
            context["eda_results"] = results
            context["eda_completed"] = True
            
            self.logger.info("✅ EDA 완료")
            return context
            
        except Exception as e:
            self.logger.error(f"❌ EDA 실패: {e}")
            raise
            
    def can_skip(self, context: Dict[str, Any]) -> bool:
        """EDA 건너뛰기 가능 여부"""
        eda_results_path = Path(self.config.output_dir) / "01_EDA" / "eda_results"
        return eda_results_path.exists() and len(list(eda_results_path.glob("*.json"))) > 0


class PreprocessingStep(PipelineStep):
    """02_preprocessing 단계"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("Preprocessing", config)
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """전처리 실행"""
        self.logger.info("🔧 데이터 전처리 시작")
        
        try:
            # 전처리 실행
            processor = GrandmasterProcessor(
                data_path=self.config.data_path,
                output_dir=Path(self.config.output_dir) / "02_preprocessing",
                experiment_name=self.config.experiment_name
            )
            
            # EDA 결과 반영
            if "eda_results" in context:
                processor.apply_eda_strategies(context["eda_results"])
            
            # 전처리 실행
            results = processor.run_complete_preprocessing()
            
            context["preprocessing_results"] = results
            context["preprocessing_completed"] = True
            
            self.logger.info("✅ 전처리 완료")
            return context
            
        except Exception as e:
            self.logger.error(f"❌ 전처리 실패: {e}")
            raise
            
    def can_skip(self, context: Dict[str, Any]) -> bool:
        """전처리 건너뛰기 가능 여부"""
        preprocessed_path = Path(self.config.output_dir) / "02_preprocessing"
        return preprocessed_path.exists() and len(list(preprocessed_path.glob("*.pkl"))) > 0


class ModelingStep(PipelineStep):
    """03_modeling 단계"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("Modeling", config)
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """모델링 실행"""
        self.logger.info("🧠 모델링 전략 실행 시작")
        
        try:
            # 모델링 전략 설정
            strategy = "single_best" if self.config.mode == PipelineMode.QUICK_TEST else "ensemble"
            target_score = 0.85 if self.config.mode == PipelineMode.QUICK_TEST else 0.95
            
            # 파이프라인 생성
            pipeline = KaggleWinnerPipeline(
                strategy=strategy,
                target_score=target_score,
                experiment_name=self.config.experiment_name,
                fast_mode=(self.config.mode == PipelineMode.QUICK_TEST)
            )
            
            # 설정 및 실행
            if pipeline.setup_components():
                context["modeling_pipeline"] = pipeline
                context["modeling_completed"] = True
                self.logger.info("✅ 모델링 전략 설정 완료")
            else:
                raise Exception("모델링 파이프라인 설정 실패")
                
            return context
            
        except Exception as e:
            self.logger.error(f"❌ 모델링 실패: {e}")
            raise
            
    def can_skip(self, context: Dict[str, Any]) -> bool:
        """모델링 건너뛰기 가능 여부"""
        return False  # 항상 실행


class TrainingStep(PipelineStep):
    """04_training 단계"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("Training", config)
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """훈련 실행"""
        self.logger.info("🏋️ 모델 훈련 시작")
        
        try:
            # 훈련 실행기 생성
            executor = GrandmasterExecutor(
                data_path=self.config.data_path,
                output_dir=Path(self.config.output_dir) / "04_training",
                experiment_name=self.config.experiment_name
            )
            
            # 모드별 설정
            if self.config.mode == PipelineMode.QUICK_TEST:
                results = executor.run_quick_test()
            elif self.config.mode == PipelineMode.FULL_TRAINING:
                results = executor.run_full_training()
            elif self.config.mode == PipelineMode.COMPETITION_READY:
                results = executor.run_competition_ready()
            else:  # DEBUG
                results = executor.run_debug()
            
            context["training_results"] = results
            context["training_completed"] = True
            
            self.logger.info("✅ 훈련 완료")
            return context
            
        except Exception as e:
            self.logger.error(f"❌ 훈련 실패: {e}")
            raise
            
    def can_skip(self, context: Dict[str, Any]) -> bool:
        """훈련 건너뛰기 가능 여부"""
        saved_models_path = Path(self.config.output_dir) / "04_training" / "saved_models"
        return saved_models_path.exists() and len(list(saved_models_path.glob("**/*.pt"))) > 0


class EvaluationStep(PipelineStep):
    """05_evaluation 단계"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("Evaluation", config)
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """평가 실행"""
        self.logger.info("📈 모델 평가 시작")
        
        try:
            # 평가기 생성
            evaluator = MainEvaluator(
                data_path=self.config.data_path,
                output_dir=Path(self.config.output_dir) / "05_evaluation",
                experiment_name=self.config.experiment_name
            )
            
            # 평가 실행
            results = evaluator.run_complete_evaluation()
            
            context["evaluation_results"] = results
            context["evaluation_completed"] = True
            
            self.logger.info("✅ 평가 완료")
            return context
            
        except Exception as e:
            self.logger.error(f"❌ 평가 실패: {e}")
            raise
            
    def can_skip(self, context: Dict[str, Any]) -> bool:
        """평가 건너뛰기 가능 여부"""
        evaluation_path = Path(self.config.output_dir) / "05_evaluation"
        return evaluation_path.exists() and len(list(evaluation_path.glob("*.json"))) > 0


class SubmissionStep(PipelineStep):
    """06_submission 단계"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("Submission", config)
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """제출 파일 생성"""
        self.logger.info("📤 최종 제출 파일 생성 시작")
        
        try:
            # 제출 생성기 생성
            generator = FinalSubmissionGenerator(
                data_path=self.config.data_path,
                output_dir=Path(self.config.output_dir) / "06_submission",
                experiment_name=self.config.experiment_name
            )
            
            # 제출 파일 생성
            results = generator.generate_final_submission()
            
            context["submission_results"] = results
            context["submission_completed"] = True
            
            self.logger.info("✅ 제출 파일 생성 완료")
            return context
            
        except Exception as e:
            self.logger.error(f"❌ 제출 파일 생성 실패: {e}")
            raise
            
    def can_skip(self, context: Dict[str, Any]) -> bool:
        """제출 건너뛰기 가능 여부"""
        submission_path = Path(self.config.output_dir) / "06_submission" / "final_submissions"
        return submission_path.exists() and len(list(submission_path.glob("*.csv"))) > 0


class GrandmasterPipeline:
    """통합 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.steps = self._create_steps()
        self.context = {}
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("GrandmasterPipeline")
        logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        log_file = Path(self.config.output_dir) / f"pipeline_{self.config.experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        return logger
        
    def _create_steps(self) -> List[PipelineStep]:
        """파이프라인 단계 생성"""
        return [
            EDAStep(self.config),
            PreprocessingStep(self.config),
            ModelingStep(self.config),
            TrainingStep(self.config),
            EvaluationStep(self.config),
            SubmissionStep(self.config)
        ]
        
    def run(self) -> bool:
        """전체 파이프라인 실행"""
        self.logger.info("🚀 파이프라인 시작")
        self.logger.info(f"📋 모드: {self.config.mode.value}")
        self.logger.info(f"📁 데이터 경로: {self.config.data_path}")
        self.logger.info(f"📂 출력 경로: {self.config.output_dir}")
        
        start_time = time.time()
        
        try:
            for i, step in enumerate(self.steps, 1):
                step_name = step.name
                
                # 건너뛰기 확인
                if step_name.lower() in self.config.skip_steps:
                    self.logger.info(f"⏭️ {step_name} 단계 건너뛰기 (사용자 설정)")
                    continue
                    
                if step.can_skip(self.context):
                    self.logger.info(f"⏭️ {step_name} 단계 건너뛰기 (이미 완료됨)")
                    continue
                
                # 단계 실행
                self.logger.info(f"🔄 [{i}/{len(self.steps)}] {step_name} 단계 실행")
                step_start = time.time()
                
                self.context = step.execute(self.context)
                
                step_duration = time.time() - step_start
                self.logger.info(f"✅ {step_name} 단계 완료 ({step_duration:.1f}초)")
                
                # 중간 결과 저장
                if self.config.save_intermediate:
                    self._save_intermediate_results(i, step_name)
            
            # 전체 완료
            total_duration = time.time() - start_time
            self.logger.info(f"🎉 전체 파이프라인 완료! ({total_duration:.1f}초)")
            
            # 최종 결과 요약
            self._print_final_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 실패: {e}")
            return False
            
    def _save_intermediate_results(self, step_num: int, step_name: str):
        """중간 결과 저장"""
        try:
            results_file = Path(self.config.output_dir) / f"intermediate_results_step_{step_num}_{step_name}.json"
            
            # 민감한 정보 제외하고 저장
            safe_context = {}
            for key, value in self.context.items():
                if not key.endswith("_pipeline") and not key.endswith("_executor"):
                    safe_context[key] = str(value) if not isinstance(value, (dict, list)) else value
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(safe_context, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"중간 결과 저장 실패: {e}")
            
    def _print_final_summary(self):
        """최종 결과 요약 출력"""
        print("\n" + "="*60)
        print("🏆 파이프라인 완료!")
        print("="*60)
        
        # 완료된 단계들
        completed_steps = [key.replace("_completed", "").title() 
                          for key, value in self.context.items() 
                          if key.endswith("_completed") and value]
        
        print(f"✅ 완료된 단계: {', '.join(completed_steps)}")
        
        # 주요 결과들
        if "training_results" in self.context:
            results = self.context["training_results"]
            if isinstance(results, dict) and "best_score" in results:
                print(f"🎯 최고 점수: {results['best_score']:.4f}")
        
        if "submission_results" in self.context:
            print(f"📤 제출 파일: {self.config.output_dir}/06_submission/final_submissions/")
        
        print(f"📁 전체 결과: {self.config.output_dir}")
        print("="*60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="🏆 통합 파이프라인")
    
    parser.add_argument(
        "--mode", 
        choices=["quick_test", "full_training", "competition_ready", "debug"],
        default="quick_test",
        help="실행 모드 선택"
    )
    
    parser.add_argument(
        "--data-path",
        default="/home/james/doc-classification/computervisioncompetition-cv3/data",
        help="데이터 경로"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong",
        help="출력 디렉토리"
    )
    
    parser.add_argument(
        "--experiment-name",
        help="실험 이름 (기본값: 자동 생성)"
    )
    
    parser.add_argument(
        "--skip-steps",
        nargs="+",
        help="건너뛸 단계들 (eda, preprocessing, modeling, training, evaluation, submission)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()
    
    # 설정 생성
    config = PipelineConfig(
        mode=PipelineMode(args.mode),
        data_path=args.data_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        skip_steps=args.skip_steps or [],
        verbose=args.verbose
    )
    
    # 출력 디렉토리 생성
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 파이프라인 실행
    pipeline = GrandmasterPipeline(config)
    success = pipeline.run()
    
    if success:
        print("\n🎉 파이프라인 실행 성공!")
        sys.exit(0)
    else:
        print("\n❌ 파이프라인 실행 실패!")
        sys.exit(1)


if __name__ == "__main__":
    main()

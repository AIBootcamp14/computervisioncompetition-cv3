# 🎯 05_evaluation - 평가 시스템

캐글 그랜드마스터 수준의 통합 평가 시스템입니다.

## 📁 파일 구조

### 핵심 파일들
- `main_evaluation.py` - 메인 평가 시스템
- `evaluation_config.py` - 평가 설정 관리
- `model_evaluator.py` - 모델 평가기
- `performance_analyzer.py` - 성능 분석기
- `advanced_ensemble_system.py` - 고급 앙상블 시스템
- `tta_predictor.py` - TTA 예측기

### 새로 추가된 파일들
- `run_evaluation.py` - 통합 실행 스크립트 (사용자 친화적)
- `quick_evaluator.py` - 간단한 평가 스크립트

## 🚀 사용법

### 1. 통합 실행 스크립트 사용 (권장)

```bash
cd /home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong/05_evaluation
python run_evaluation.py
```

메뉴에서 원하는 평가 모드를 선택하세요:
- 🚀 빠른 평가 (단일 모델)
- 🎭 앙상블 평가 (다중 모델)
- 🔍 상세 분석 (성능 분석)
- 📊 TTA 예측 (Test-Time Augmentation)
- 🎯 전체 평가 (모든 기능)

### 2. 간단한 평가

```bash
python quick_evaluator.py --model_path /path/to/model.pth --data_path /path/to/data
```

### 3. 직접 실행

```bash
python main_evaluation.py --mode quick --data_path /path/to/data
```

## 🎯 주요 기능

### 📊 평가 모드
- **빠른 평가**: 단일 모델의 기본 성능 확인
- **앙상블 평가**: 다중 모델 앙상블 성능
- **상세 분석**: 종합적인 성능 분석 및 시각화
- **TTA 예측**: Test-Time Augmentation을 활용한 고급 예측
- **전체 평가**: 모든 기능을 포함한 완전한 평가

### 🔧 고급 기능
- **다양한 앙상블 전략**: Voting, Averaging, Stacking
- **TTA 최적화**: 다양한 증강 기법 적용
- **성능 분석**: Confusion Matrix, ROC Curve, Feature Importance
- **자동 제출 파일 생성**: 캐글 제출 형식으로 결과 저장
- **메모리 효율적 구현**: 대용량 데이터 처리 최적화

## 📈 결과 출력

평가 결과는 다음 형태로 저장됩니다:
```
evaluation_results/
├── quick_eval/
│   ├── results.json
│   ├── confusion_matrix.png
│   └── performance_report.txt
├── ensemble_eval/
│   ├── ensemble_results.json
│   ├── model_comparison.png
│   └── submission.csv
└── full_eval/
    ├── comprehensive_analysis.json
    ├── visualizations/
    └── final_submission.csv
```

## ⚙️ 설정

`evaluation_config.py`에서 평가 설정을 관리할 수 있습니다:

```python
# 주요 설정
EVALUATION_MODES = ['quick', 'ensemble', 'detailed', 'tta', 'full']
ENSEMBLE_STRATEGIES = ['voting', 'averaging', 'stacking']
TTA_ROUNDS = 8
OUTPUT_FORMATS = ['json', 'csv', 'png']
```

## 🎨 시각화

성능 분석 시 다음 시각화가 생성됩니다:
- Confusion Matrix
- ROC Curves
- Precision-Recall Curves
- Feature Importance
- Model Comparison Charts
- Training History Plots

## 🔗 연계 시스템

이 평가 시스템은 다음 단계들과 완전히 연계됩니다:
- `01_EDA/` - 탐색적 데이터 분석 결과 활용
- `02_preprocessing/` - 전처리된 데이터 사용
- `03_modeling/` - 훈련된 모델 로드
- `04_training/` - 훈련 결과 활용

## 💡 팁

1. **빠른 테스트**: 먼저 `quick_evaluator.py`로 기본 성능을 확인하세요
2. **메모리 관리**: 대용량 데이터는 배치 크기를 조정하세요
3. **앙상블 최적화**: 여러 모델의 성능을 비교한 후 앙상블을 구성하세요
4. **TTA 활용**: 중요한 대회에서는 TTA를 반드시 사용하세요

## 🐛 문제 해결

### 일반적인 오류
- **CUDA 메모리 부족**: 배치 크기를 줄이거나 모델을 CPU로 이동
- **모델 로딩 실패**: 모델 파일 경로와 형식을 확인
- **데이터 경로 오류**: 데이터 디렉토리 구조를 확인

### 로그 확인
```bash
tail -f evaluation.log
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 데이터 경로가 올바른지
2. 모델 파일이 존재하는지
3. 필요한 라이브러리가 설치되어 있는지
4. GPU 메모리가 충분한지

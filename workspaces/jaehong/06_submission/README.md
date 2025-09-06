# 🚀 06_submission - 제출 시스템

캐글 그랜드마스터 수준의 통합 제출 시스템입니다.

## 📁 파일 구조

### 핵심 파일들
- `final_submission.py` - 최종 제출 시스템
- `fixed_submission_generator.py` - 개선된 제출 생성기

### 새로 추가된 파일들
- `run_submission.py` - 통합 실행 스크립트 (사용자 친화적)

### 제출 결과 폴더들
- `final_submissions/` - 최종 제출 결과들
- `fixed_submissions/` - 개선된 제출 결과들

## 🚀 사용법

### 1. 통합 실행 스크립트 사용 (권장)

```bash
cd /home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong/06_submission
python run_submission.py
```

메뉴에서 원하는 제출 모드를 선택하세요:
- 🎯 최종 제출 (Final Submission)
- 🔧 개선된 제출 (Fixed Submission)
- 📊 제출 결과 분석
- 🗂️ 제출 파일 관리

### 2. 직접 실행

```bash
# 최종 제출
python final_submission.py --data_path /path/to/data

# 개선된 제출
python fixed_submission_generator.py --data_path /path/to/data
```

## 🎯 주요 기능

### 📤 제출 모드
- **최종 제출**: 05_evaluation 결과를 활용한 최종 제출
- **개선된 제출**: 클래스 편향 보정 및 TTA 최적화된 제출
- **결과 분석**: 제출 파일들의 통계 및 분포 분석
- **파일 관리**: 제출 파일들의 크기 및 구조 관리

### 🔧 고급 기능
- **클래스 편향 보정**: 불균형한 클래스 분포 자동 보정
- **TTA 가중치 최적화**: Test-Time Augmentation 가중치 자동 조정
- **신뢰도 기반 후처리**: 낮은 신뢰도 예측 자동 보정
- **균형잡힌 예측 분포**: 모든 클래스에 대한 균등한 예측 분포 생성

## 📈 제출 결과

제출 결과는 다음 형태로 저장됩니다:
```
final_submissions/
├── final_submission_YYYYMMDD_HHMMSS.csv
├── final_submission_YYYYMMDD_HHMMSS_confidence.npy
├── final_submission_YYYYMMDD_HHMMSS_probabilities.npy
└── final_submission_YYYYMMDD_HHMMSS_analysis.json

fixed_submissions/
├── fixed_submission_YYYYMMDD_HHMMSS.csv
├── fixed_submission_YYYYMMDD_HHMMSS_confidence.npy
├── fixed_submission_YYYYMMDD_HHMMSS_probabilities.npy
└── fixed_submission_YYYYMMDD_HHMMSS_analysis.json
```

## ⚙️ 설정

주요 설정 옵션들:

```python
# 최종 제출 설정
FINAL_SUBMISSION_CONFIG = {
    'use_tta': True,
    'tta_rounds': 8,
    'ensemble_strategy': 'weighted',
    'confidence_threshold': 0.8
}

# 개선된 제출 설정
FIXED_SUBMISSION_CONFIG = {
    'class_bias_correction': True,
    'tta_weight_optimization': True,
    'confidence_postprocessing': True,
    'balanced_distribution': True
}
```

## 🎨 분석 기능

제출 결과 분석 시 다음 정보가 제공됩니다:
- 클래스별 예측 분포
- 신뢰도 통계 (평균, 표준편차, 최소/최대값)
- 고신뢰도/저신뢰도 비율
- 예측 일관성 분석
- 파일 크기 및 구조 정보

## 🔗 연계 시스템

이 제출 시스템은 다음 단계들과 완전히 연계됩니다:
- `01_EDA/` - 탐색적 데이터 분석 결과 활용
- `02_preprocessing/` - 전처리된 데이터 사용
- `03_modeling/` - 훈련된 모델 로드
- `04_training/` - 훈련 결과 활용
- `05_evaluation/` - 평가 결과 활용

## 💡 팁

1. **최종 제출 전**: 먼저 개선된 제출로 테스트해보세요
2. **클래스 균형**: 클래스 분포가 균등한지 확인하세요
3. **신뢰도 확인**: 낮은 신뢰도 예측들을 검토하세요
4. **파일 크기**: 제출 파일 크기가 적절한지 확인하세요

## 🐛 문제 해결

### 일반적인 오류
- **모델 로딩 실패**: 모델 파일 경로와 형식을 확인
- **데이터 경로 오류**: 데이터 디렉토리 구조를 확인
- **메모리 부족**: 배치 크기를 줄이거나 모델을 CPU로 이동
- **클래스 불균형**: 개선된 제출 모드를 사용

### 로그 확인
```bash
tail -f submission.log
```

## 📊 성능 비교

| 제출 방식 | 정확도 | 클래스 균형 | 신뢰도 | 추천도 |
|-----------|--------|-------------|--------|--------|
| 기본 제출 | 보통 | 불균형 | 보통 | ⭐⭐ |
| 최종 제출 | 좋음 | 보통 | 좋음 | ⭐⭐⭐ |
| 개선된 제출 | 우수 | 균형 | 우수 | ⭐⭐⭐⭐⭐ |

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 데이터 경로가 올바른지
2. 모델 파일이 존재하는지
3. 필요한 라이브러리가 설치되어 있는지
4. GPU 메모리가 충분한지

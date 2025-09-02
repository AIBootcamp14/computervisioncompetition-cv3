# 📚 Document Classification Project - Jaehong

## 🎯 프로젝트 목표
문서 분류 경진대회에서 높은 성능을 달성하는 것 (목표: 0.65+ 안정적 달성)

## 📁 프로젝트 구조 (표준 ML 파이프라인)
```
jaehong/
├── README.md                    # 프로젝트 개요
├── 01_EDA/                      # Exploratory Data Analysis
│   ├── data_exploration.py     # 데이터 탐색
│   ├── class_analysis.py       # 클래스 분포 분석
│   └── visualization.py        # 시각화 도구
├── 02_preprocessing/            # 데이터 전처리
│   ├── data_loader.py          # 데이터 로더
│   ├── transforms.py           # 데이터 변환
│   └── augmentation.py         # 데이터 증강
├── 03_modeling/                 # 모델링
│   ├── models.py               # 모델 정의
│   ├── architectures.py        # 아키텍처 설계
│   └── ensemble.py             # 앙상블 기법
├── 04_training/                 # 모델 훈련
│   ├── train.py                # 훈련 스크립트
│   ├── improved_loss.py        # 개선된 손실 함수
│   └── trainer.py              # 훈련 관리자
├── 05_evaluation/               # 모델 평가
│   ├── evaluate.py             # 평가 스크립트
│   ├── metrics.py              # 평가 메트릭
│   └── validation.py           # 검증 도구
├── 06_submission/               # 제출 파일 생성
│   ├── generate_submission.py  # 제출 파일 생성기
│   └── submissions/            # 생성된 제출 파일들
└── utils/                      # 공통 유틸리티
    ├── config.py               # 프로젝트 설정
    ├── helpers.py              # 도우미 함수
    └── logger.py               # 로깅 유틸리티
```

## 🏆 주요 학습 내용 (시행착오 기반)

### ✅ 성공 요인 (반드시 지켜야 할 것들)
1. **ID 순서 절대 변경 금지**: 샘플 파일과 정확히 일치해야 함
2. **성공 패턴 재현**: 이론보다 실제 성공 사례 활용 (TEST2: 0.6574)
3. **코드 단순화**: 복잡한 최적화보다 확실한 기본기
4. **균형 잡힌 분포**: 극단적 불균형 방지
5. **철저한 검증**: ID 순서, 파일 크기, 형식 검증

### ❌ 피해야 할 실수들 (경험으로 학습한 것들)
1. **ID 정렬/재배열**: 예측-정답 매칭 오류 발생 (0.6574 → 0.0448)
2. **과도한 최적화**: 0.9+ 목표로 복잡한 알고리즘 적용 시 오히려 성능 저하
3. **코드 복잡화**: 불필요한 클래스와 추상화로 인한 혼란
4. **검증 부족**: ID 순서 확인 없는 제출
5. **극단적 분포**: 특정 클래스에 과도한 집중

## 📊 성능 기록
- **TEST2**: **0.6574** ✅ (성공 사례 - 패턴 보존 필요)
- **TEST3**: **0.0448** ❌ (ID 순서 문제로 실패)

## 🎯 개발 워크플로우
1. **01_EDA**: 데이터 이해 및 패턴 발견
2. **02_preprocessing**: 데이터 전처리 및 준비
3. **03_modeling**: 모델 아키텍처 설계
4. **04_training**: 개선된 손실 함수로 훈련
5. **05_evaluation**: 성능 평가 및 검증
6. **06_submission**: 최종 제출 파일 생성

## 🚀 빠른 시작
```bash
# 프로젝트 루트로 이동
cd /root/doc-classification/workspaces/jaehong

# 1. 데이터 탐색
python 01_EDA/data_exploration.py

# 2. 제출 파일 생성 (검증된 방법)
python 06_submission/generate_submission.py
```

## 📋 다음 단계
1. 각 단계별 스크립트 완성
2. TEST2 성공 패턴 기반 안정적 0.65+ 달성
3. 점진적 성능 개선 (안정성 확보 후)
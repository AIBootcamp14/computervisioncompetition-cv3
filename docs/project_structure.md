# 📁 프로젝트 구조 가이드

## 전체 구조 개요

```
doc-classification/
├── 📄 README.md                    # 프로젝트 메인 가이드
├── 📦 requirements.txt             # 의존성 패키지
├── 🚫 .gitignore                   # Git 제외 파일
│
├── 📊 data/                        # 공통 데이터
│   ├── train.csv                   # 학습 라벨 데이터
│   ├── meta.csv                    # 클래스 메타데이터
│   ├── train/                      # 학습 이미지
│   ├── test/                       # 테스트 이미지
│   └── submissions/                # 제출 파일 모음
│
├── 🧑‍💻 개별 멤버 작업공간
│   ├── jaehong/                    # 재홍의 작업공간
│   ├── dongjun/                    # 동준의 작업공간
│   ├── junyoung/                   # 준영의 작업공간
│   └── yeeun/                      # 예은의 작업공간
│   ├── 01_EDA/                     # 탐색적 데이터 분석
│   ├── 02_preprocessing/           # 데이터 전처리
│   ├── 03_modeling/                # 모델 설계
│   ├── 04_training/                # 모델 학습
│   ├── 05_validation/              # 모델 검증
│   ├── 06_submission/              # 최종 제출
│   ├── configs/                    # 개인 설정 파일
│   ├── logs/                       # 실험 로그
│   ├── models/                     # 학습된 모델
│   ├── notebooks/                  # Jupyter 노트북
│   └── README.md                   # 개인 작업 가이드
│
├── 🔧 src/                         # 공통 소스 코드
│   ├── data/                       # 데이터 처리 모듈
│   ├── models/                     # 모델 클래스
│   ├── training/                   # 학습 로직
│   ├── evaluation/                 # 평가 모듈
│   └── utils/                      # 유틸리티 (wandb 포함)
│
├── 🤝 shared/                      # 팀 공유 자료
│   ├── best_models/                # 최고 성능 모델
│   ├── ensemble_results/           # 앙상블 결과
│   ├── logs/                       # 통합 실험 로그
│   └── data_insights/              # 데이터 분석 결과
│
├── 📝 templates/                   # 템플릿 파일들
│   ├── config_template.yaml        # 설정 템플릿
│   ├── notebook_template.ipynb     # 노트북 템플릿
│   └── experiment_template.py      # 실험 스크립트 템플릿
│
└── 📚 docs/                        # 문서화
    ├── project_structure.md        # 이 파일
    ├── data_description.md         # 데이터 설명
    └── api_reference.md            # API 레퍼런스
```

## 폴더별 상세 설명

### 🧑‍💻 memberX/ - 개별 작업공간
각 멤버는 독립적인 ML 파이프라인을 가집니다:

- **01_EDA/**: 데이터 탐색, 시각화, 패턴 분석
- **02_preprocessing/**: 전처리 파이프라인, 데이터 증강
- **03_modeling/**: 모델 아키텍처, 하이퍼파라미터 설계
- **04_training/**: 학습 실행, wandb 추적
- **05_validation/**: 교차검증, 성능 평가
- **06_submission/**: 최종 예측, 제출 파일 생성

### 🔧 src/ - 공통 코드베이스
재사용 가능한 클래스와 함수들:

- **data/**: BaseDataset, DataLoader, Augmentation
- **models/**: BaseModel, ModelFactory, 특화 모델들
- **training/**: Trainer, LossFunction, Optimizer 설정
- **evaluation/**: 메트릭 계산, 모델 비교, 시각화
- **utils/**: ExperimentTracker (wandb), 유틸리티 함수

### 🤝 shared/ - 팀 협업
팀 전체가 공유하는 자료:

- **best_models/**: 각 멤버의 최고 성능 모델
- **ensemble_results/**: 앙상블 실험 결과
- **logs/**: 통합 실험 로그 및 비교
- **data_insights/**: 데이터 분석 공유 결과

## 작업 흐름

1. **Setup**: `memberX/` 폴더에서 작업 시작
2. **EDA**: `01_EDA/`에서 데이터 탐색
3. **Preprocessing**: `02_preprocessing/`에서 파이프라인 구축
4. **Modeling**: `03_modeling/`에서 모델 설계
5. **Training**: `04_training/`에서 학습 실행
6. **Validation**: `05_validation/`에서 성능 검증
7. **Submission**: `06_submission/`에서 최종 제출
8. **Share**: 최고 모델을 `shared/best_models/`에 공유

## 명명 규칙

### 파일명
- 스크립트: `snake_case.py`
- 노트북: `01_descriptive_name.ipynb`
- 설정: `config_name.yaml`
- 모델: `model_version_score.pth`

### 실험명 (wandb)
- 형식: `memberX_modeltype_experiment`
- 예시: `member1_resnet50_baseline`

## Git 브랜치 전략
```bash
main                    # 안정 버전
├── member1-pipeline    # 멤버1 전체 파이프라인
├── member2-pipeline    # 멤버2 전체 파이프라인
├── member3-pipeline    # 멤버3 전체 파이프라인
└── member4-pipeline    # 멤버4 전체 파이프라인
```

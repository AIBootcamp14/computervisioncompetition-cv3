# 🧑‍💻 동준(Dongjun) - 개인 작업공간

## 👋 환영합니다!
이 폴더는 동준님의 독립적인 ML 파이프라인 작업공간입니다.

## 📂 폴더 구조
```
dongjun/
├── 01_EDA/                     # 📊 탐색적 데이터 분석
├── 02_preprocessing/           # 🔧 데이터 전처리
├── 03_modeling/                # 🧠 모델 설계
├── 04_training/                # 🏋️ 모델 학습
├── 05_validation/              # ✅ 모델 검증
├── 06_submission/              # 📤 최종 제출
├── configs/                    # ⚙️ 설정 파일
├── logs/                       # 📝 실험 로그
├── models/                     # 💾 학습된 모델
├── notebooks/                  # 📓 Jupyter 노트북
└── README.md                   # 📋 이 파일
```

## 🚀 시작하기

### 1. 작업 환경 설정
```bash
# 프로젝트 루트에서
cd member2

# 가상환경 활성화 (이미 설정되어 있다면 생략)
source ../venv/bin/activate

# Jupyter Lab 실행
jupyter lab
```

### 2. wandb 설정
```bash
# wandb 로그인 (최초 1회)
wandb login

# 실험 태그에 member2 포함하기
```

## 📋 작업 체크리스트

### 📊 01_EDA (탐색적 데이터 분석)
- [ ] 데이터 로딩 및 기본 정보 확인
- [ ] 클래스 분포 분석
- [ ] 이미지 크기/품질 분석
- [ ] 시각화 및 인사이트 도출
- [ ] 전처리 전략 수립

**산출물**: `eda_analysis.ipynb`, `data_insights.md`

### 🔧 02_preprocessing (데이터 전처리)
- [ ] 이미지 전처리 파이프라인 구축
- [ ] 데이터 증강 전략 설계
- [ ] 정규화 파라미터 결정
- [ ] 전처리 함수 테스트

**산출물**: `preprocessing_pipeline.py`, `augmentation_config.yaml`

### 🧠 03_modeling (모델 설계)
- [ ] 모델 아키텍처 선택/설계
- [ ] 하이퍼파라미터 정의
- [ ] 손실 함수 및 옵티마이저 선택
- [ ] 모델 구현 및 테스트

**산출물**: `model_architecture.py`, `model_config.yaml`

### 🏋️ 04_training (모델 학습)
- [ ] 학습 파이프라인 구축
- [ ] wandb 실험 추적 설정
- [ ] 모델 학습 실행
- [ ] 체크포인트 저장

**산출물**: `training_script.py`, wandb 로그, 모델 체크포인트

### ✅ 05_validation (모델 검증)
- [ ] 교차 검증 실행
- [ ] 성능 메트릭 계산
- [ ] 모델 비교 및 분석
- [ ] 최종 모델 선택

**산출물**: `validation_results.json`, `model_comparison.ipynb`

### 📤 06_submission (최종 제출)
- [ ] 테스트 데이터 예측
- [ ] 제출 파일 생성
- [ ] 결과 분석 및 정리
- [ ] 최종 보고서 작성

**산출물**: `submission.csv`, `final_report.md`

## 🎯 개인 목표
- **성능 목표**: F1 Score > 0.85
- **실험 횟수**: 최소 10회 이상
- **모델 다양성**: 2개 이상 아키텍처 실험

## 📊 진행 상황

### 현재 최고 성능
- **F1 Score**: -
- **Accuracy**: -
- **모델**: -
- **실험명**: -

### 실험 히스토리
| 날짜 | 실험명 | 모델 | F1 Score | 비고 |
|------|--------|------|----------|------|
| - | - | - | - | - |

## 🔗 유용한 링크
- [프로젝트 README](../README.md)
- [데이터 설명](../docs/data_description.md)
- [공통 코드 가이드](../src/README.md)
- [wandb 프로젝트](https://wandb.ai/your-team/doc-classification)

## 💡 팁 & 주의사항

### 🎯 성공 전략
1. **작은 실험부터**: 간단한 베이스라인 먼저 구축
2. **체계적 기록**: 모든 실험을 wandb에 기록
3. **점진적 개선**: 한 번에 하나씩 변경사항 적용
4. **코드 재사용**: `../src/` 공통 모듈 적극 활용

### ⚠️ 주의사항
- 실험명에 반드시 `member2_` 접두사 포함
- 최고 성능 모델은 `../shared/best_models/`에 공유
- 개인정보가 포함된 샘플 이미지 외부 공유 금지
- 정기적으로 코드 백업 및 버전 관리

## 🤝 팀 협업
- 주간 진행상황 공유
- 인사이트 및 노하우 공유
- 최종 앙상블을 위한 모델 기여

---
**행운을 빕니다! 🍀**

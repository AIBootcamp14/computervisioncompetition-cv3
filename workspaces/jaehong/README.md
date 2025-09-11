# 🏆 문서 분류 시스템

준의 완전한 머신러닝 파이프라인입니다.

## 📁 폴더 구조

### 핵심 파이프라인 (01~06)
- **`01_EDA/`** - 탐색적 데이터 분석
- **`02_preprocessing/`** - 데이터 전처리 및 증강
- **`03_modeling/`** - 모델 아키텍처 및 전략
- **`04_training/`** - 모델 훈련 시스템
- **`05_evaluation/`** - 모델 평가 및 분석
- **`06_submission/`** - 최종 제출 시스템

## 🚀 빠른 시작

### 🎯 방법 1: 통합 파이프라인 실행 (추천)

#### 간단한 실행 (대화형)
```bash
python simple_pipeline.py
```

#### 고급 실행 (명령줄)
```bash
# 빠른 테스트 (10-30분)
python run_pipeline.py --mode quick_test

# 전체 훈련 (2-4시간)
python run_pipeline.py --mode full_training

# 대회 준비 (4-8시간)
python run_pipeline.py --mode competition_ready
```

### 🔧 방법 2: 개별 단계 실행
```bash
# 1단계: EDA 분석
cd 01_EDA && python competition_eda.py

# 2단계: 데이터 전처리
cd 02_preprocessing && python grandmaster_processor.py

# 3단계: 모델 훈련
cd 04_training && python run_training.py

# 4단계: 모델 평가
cd 05_evaluation && python run_evaluation.py

# 5단계: 최종 제출
cd 06_submission && python run_submission.py
```

### 📋 실행 모드 선택 가이드
- **빠른 테스트**: 코드 검증, 빠른 결과 확인 (1개 모델, 1-fold)
- **전체 훈련**: 실제 모델 훈련, 성능 평가 (7개 모델, 5-fold)
- **대회 준비**: 최고 성능 달성, 최종 제출 (최적화된 모든 설정)

## 🎯 주요 기능

### 📊 01_EDA - 탐색적 데이터 분석
- Train/Test 데이터 분포 분석
- 클래스 불균형 분석
- 이미지 품질 메트릭 계산
- 도메인 적응 전략 제안

### 🔧 02_preprocessing - 데이터 전처리
- EDA 기반 도메인 적응
- 클래스별 맞춤 증강
- Multi-Modal 데이터 처리
- Progressive Pseudo Labeling

### 🧠 03_modeling - 모델링 전략
- Multi-Architecture Ensemble
- Attention Mechanism
- Advanced Loss Functions
- Knowledge Distillation

### 🏋️ 04_training - 훈련 시스템
- Mixed Precision Training
- Gradient Accumulation
- Early Stopping
- 실시간 모니터링

### 📈 05_evaluation - 평가 시스템
- 다양한 평가 모드
- 성능 분석 및 시각화
- TTA 예측
- 앙상블 평가

### 📤 06_submission - 제출 시스템
- 최종 제출 생성
- 클래스 편향 보정
- 신뢰도 기반 후처리
- 제출 파일 관리

## ⚙️ 시스템 요구사항

### 하드웨어
- **GPU**: RTX 4090 Laptop (16GB VRAM) 권장
- **RAM**: 32GB 이상
- **Storage**: 50GB 이상 여유 공간

### 소프트웨어
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- 기타 필수 라이브러리들

## 📊 성능 지표

### 목표 성능
- **정확도**: 95% 이상
- **F1 점수**: 0.95 이상
- **클래스 균형**: 모든 클래스 균등 분포

### 최적화 설정
- **이미지 크기**: 64px (빠른 훈련) / 224px (고성능)
- **배치 크기**: 8-32 (GPU 메모리에 따라 조정)
- **에포크**: 1-40 (모드에 따라 조정)
- **앙상블**: 1-7개 모델

## 🔗 연계 시스템

각 단계는 이전 단계의 결과를 활용하여 완전히 연계됩니다:

```
01_EDA → 02_preprocessing → 03_modeling → 04_training → 05_evaluation → 06_submission
```

## 💡 사용 팁

### 빠른 테스트
```bash
# 빠른 테스트 모드 (1 에포크, 단일 모델)
cd 04_training && python grandmaster_train.py --mode quick_test
```

### 전체 훈련
```bash
# 전체 훈련 모드 (5 folds, 앙상블)
cd 04_training && python grandmaster_train.py --mode full_training
```

### 성능 최적화
- GPU 메모리 부족 시: 배치 크기 감소, 이미지 크기 축소
- 훈련 속도 향상: 에포크 수 감소, 모델 복잡도 축소
- 성능 향상: 앙상블 모델 수 증가, TTA 활용

## 🐛 문제 해결

### 일반적인 문제
1. **CUDA 메모리 부족**: 배치 크기나 이미지 크기 조정
2. **모델 로딩 실패**: 모델 파일 경로 확인
3. **데이터 경로 오류**: 데이터 디렉토리 구조 확인
4. **의존성 오류**: 필요한 라이브러리 설치 확인

### 로그 확인
```bash
# 훈련 로그 확인
tail -f training.log

# 평가 로그 확인
tail -f evaluation.log
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 각 폴더의 README.md 파일
2. 로그 파일의 오류 메시지
3. 시스템 요구사항 충족 여부
4. 데이터 경로 및 형식 확인

## 🏆 성과

이 시스템을 사용하여 달성할 수 있는 성과:
- **95% 이상 정확도** 달성
- **완전 자동화된 파이프라인** 구축
- **재사용 가능한 모듈화된 시스템** 개발

---


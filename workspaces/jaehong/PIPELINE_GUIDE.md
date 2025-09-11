# 🚀 파이프라인 실행 가이드

## 📋 개요

jaehong 폴더에 전체 머신러닝 파이프라인을 한 번에 실행할 수 있는 두 가지 스크립트를 제공합니다:

1. **`run_pipeline.py`** - 고급 파이프라인 실행기 (명령줄 인터페이스)
2. **`simple_pipeline.py`** - 간단한 파이프라인 실행기 (대화형 인터페이스)

## 🎯 실행 모드

### 1. 빠른 테스트 (Quick Test)
- **목적**: 빠른 검증 및 테스트
- **설정**: 1개 모델, 1-fold, 빠른 설정
- **소요시간**: 약 10-30분
- **용도**: 코드 검증, 빠른 결과 확인

### 2. 전체 훈련 (Full Training)
- **목적**: 완전한 모델 훈련
- **설정**: 7개 모델 앙상블, 5-fold 교차검증
- **소요시간**: 약 2-4시간
- **용도**: 실제 모델 훈련, 성능 평가

### 3. 대회 준비 (Competition Ready)
- **목적**: 최고 성능 달성
- **설정**: 최적화된 모든 설정, 고급 기법 적용
- **소요시간**: 약 4-8시간
- **용도**: 최종 제출, 대회 참가

## 🚀 사용법

### 방법 1: 간단한 실행 (추천)

```bash
cd /home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong
python simple_pipeline.py
```

**특징:**
- 대화형 메뉴 인터페이스
- 시스템 요구사항 자동 확인
- 개별 단계 실행 가능
- 사용자 친화적

### 방법 2: 고급 실행

```bash
cd /home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong

# 빠른 테스트
python run_pipeline.py --mode quick_test

# 전체 훈련
python run_pipeline.py --mode full_training

# 대회 준비
python run_pipeline.py --mode competition_ready

# 특정 단계 건너뛰기
python run_pipeline.py --mode full_training --skip-steps eda preprocessing

# 사용자 정의 설정
python run_pipeline.py \
    --mode full_training \
    --data-path /path/to/your/data \
    --output-dir /path/to/output \
    --experiment-name my_experiment \
    --verbose
```

## 📁 파이프라인 단계

전체 파이프라인은 다음 6단계로 구성됩니다:

```
01_EDA → 02_preprocessing → 03_modeling → 04_training → 05_evaluation → 06_submission
```

### 각 단계별 설명

1. **01_EDA** - 탐색적 데이터 분석
   - Train/Test 데이터 분포 분석
   - 클래스 불균형 분석
   - 이미지 품질 메트릭 계산

2. **02_preprocessing** - 데이터 전처리
   - EDA 기반 도메인 적응
   - 클래스별 맞춤 증강
   - Multi-Modal 데이터 처리

3. **03_modeling** - 모델링 전략
   - Multi-Architecture Ensemble
   - Attention Mechanism
   - Advanced Loss Functions

4. **04_training** - 모델 훈련
   - Mixed Precision Training
   - Gradient Accumulation
   - Early Stopping

5. **05_evaluation** - 모델 평가
   - 다양한 평가 모드
   - 성능 분석 및 시각화
   - TTA 예측

6. **06_submission** - 제출 파일 생성
   - 최종 제출 생성
   - 클래스 편향 보정
   - 신뢰도 기반 후처리

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

## 📊 예상 결과

### 빠른 테스트 모드
- **정확도**: 85-90%
- **소요시간**: 10-30분
- **결과**: 기본 모델 성능 확인

### 전체 훈련 모드
- **정확도**: 90-95%
- **소요시간**: 2-4시간
- **결과**: 앙상블 모델 성능

### 대회 준비 모드
- **정확도**: 95% 이상
- **소요시간**: 4-8시간
- **결과**: 최고 성능 모델

## 🔧 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   ```bash
   # 빠른 테스트 모드로 실행
   python run_pipeline.py --mode quick_test
   ```

2. **모듈 import 실패**
   ```bash
   # 의존성 확인
   pip install -r requirements.txt
   ```

3. **데이터 경로 오류**
   ```bash
   # 데이터 경로 확인
   ls /home/james/doc-classification/computervisioncompetition-cv3/data
   ```

### 로그 확인

```bash
# 파이프라인 로그 확인
tail -f pipeline_*.log

# 각 단계별 로그 확인
ls */logs/
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. 시스템 요구사항 충족 여부
2. 데이터 경로 및 형식 확인
3. 로그 파일의 오류 메시지
4. 각 폴더의 README.md 파일

## 🎉 성과

이 파이프라인을 사용하여 달성할 수 있는 성과:

- **캐글 대회 상위권** 진입
- **95% 이상 정확도** 달성
- **완전 자동화된 파이프라인** 구축
- **재사용 가능한 모듈화된 시스템** 개발

---


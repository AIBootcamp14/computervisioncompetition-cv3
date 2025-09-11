# 🏗️ 문서 분류 시스템 구조 다이어그램 설명

## 📋 전체 시스템 개요

문서 분류 시스템은 **6단계 파이프라인**으로 구성되어 있으며, 각 단계는 독립적으로 실행 가능하면서도 전체적으로 연계되어 작동합니다.

```
📊 데이터 입력 → 🔍 EDA → 🔧 전처리 → 🧠 모델링 → 🏋️ 훈련 → 📈 평가 → 📤 제출
```

---

## 🏆 1. 파이프라인 아키텍처 (pipeline_architecture.png)

### 📋 구조 설명

```
┌─────────────────────────────────────────────────────────────────┐
│                    🏆 Document Classification System            │
│                        Pipeline Architecture                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  01_EDA     │───▶│02_Preprocess│───▶│03_Modeling  │───▶│04_Training  │
│             │    │             │    │             │    │             │
│CompetitionEDA│    │Grandmaster  │    │KaggleWinner │    │Grandmaster  │
│ImageAnalyzer │    │Processor    │    │Pipeline     │    │Executor     │
│             │    │Grandmaster  │    │ImprovedDoc  │    │ConfigManager│
│             │    │Dataset      │    │Classifier   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                    │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│06_Submission│◀───│05_Evaluation│◀───│             │              │
│             │    │             │    │             │              │
│FinalSubmiss │    │MainEvaluator│    │             │              │
│Generator    │    │BaseEnsemble │    │             │              │
│Submission   │    │VotingEnsembl│    │             │              │
│Manager      │    │StackingEnsem│    │             │              │
└─────────────┘    └─────────────┘    └─────────────┘              │
                                                                    ▼
                                                           ┌─────────────┐
                                                           │   Results   │
                                                           │   Output    │
                                                           └─────────────┘
```

### 🔄 각 단계별 역할

1. **01_EDA (탐색적 데이터 분석)**
   - `CompetitionEDA`: 전체 데이터 분석 및 전략 수립
   - `ImageAnalyzer`: 이미지 품질 및 특성 분석

2. **02_Preprocessing (데이터 전처리)**
   - `GrandmasterProcessor`: EDA 결과 기반 전처리 실행
   - `GrandmasterDataset`: 전처리된 데이터셋 관리

3. **03_Modeling (모델링 전략)**
   - `KaggleWinnerPipeline`: 모델링 전략 수립
   - `ImprovedDocumentClassifier`: 개선된 분류 모델

4. **04_Training (모델 훈련)**
   - `GrandmasterExecutor`: 훈련 실행 및 관리
   - `ConfigManager`: 설정 관리 및 최적화

5. **05_Evaluation (모델 평가)**
   - `MainEvaluator`: 성능 평가 및 분석
   - `BaseEnsemble`: 앙상블 모델 관리

6. **06_Submission (제출 파일 생성)**
   - `FinalSubmissionGenerator`: 최종 제출 파일 생성
   - `SubmissionManager`: 제출 파일 관리

---

## 🧠 2. 모델 아키텍처 계층 (model_architecture.png)

### 📋 구조 설명

```
┌─────────────────────────────────────────────────────────────────┐
│                    🧠 Model Architecture Hierarchy              │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │ BaseDocumentClassifier   │
                    │      <<abstract>>       │
                    │                         │
                    │ + forward(x: Tensor)    │
                    │ + get_features(x: Tensor)│
                    └─────────────────────────┘
                              ▲
                              │ 상속
                    ┌─────────┼─────────┐
                    │         │         │
        ┌───────────▼───┐ ┌───▼───┐ ┌───▼───────────┐
        │ImprovedDoc    │ │Efficient│ │ConvNeXt       │
        │Classifier     │ │NetV2   │ │Classifier     │
        │               │ │        │ │               │
        │+ backbone      │ │+ model │ │+ model        │
        │+ attention     │ │+ class │ │+ classifier   │
        │+ pooling       │ │+ dropout│ │+ dropout     │
        │+ classifier    │ │        │ │               │
        └───────────────┘ └────────┘ └───────────────┘
                │
                │ 구성
        ┌───────┼───────┐
        │       │       │
┌───────▼───┐ ┌─▼─────┐ │
│Attention  │ │GeM    │ │
│Module     │ │Pooling│ │
│           │ │       │ │
│+ channel  │ │+ p    │ │
│+ spatial  │ │+ eps  │ │
└───────────┘ └───────┘ │
```

### 🎯 핵심 설계 원칙

1. **추상화 (Abstraction)**
   - `BaseDocumentClassifier`: 모든 분류 모델의 공통 인터페이스 정의
   - `forward()`, `get_features()` 메서드로 일관된 인터페이스 제공

2. **다형성 (Polymorphism)**
   - `ImprovedDocumentClassifier`: 개선된 모델 구현
   - `EfficientNetV2Classifier`: EfficientNet 기반 구현
   - `ConvNeXtClassifier`: ConvNeXt 기반 구현

3. **모듈화 (Modularity)**
   - `AttentionModule`: 채널/공간 어텐션 메커니즘
   - `GeMPooling`: Generalized Mean Pooling

---

## 🎭 3. 앙상블 시스템 (ensemble_system.png)

### 📋 구조 설명

```
┌─────────────────────────────────────────────────────────────────┐
│                    🎭 Ensemble System Architecture              │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │ BaseEnsemble            │
                    │      <<abstract>>       │
                    │                         │
                    │ + models: List[nn.Module]│
                    │ + device: str           │
                    │ + predict(data_loader)   │
                    │ + predict_proba(data_loader)│
                    └─────────────────────────┘
                              ▲
                              │ 상속
                    ┌─────────┼─────────┐
                    │         │         │
        ┌───────────▼───┐ ┌───▼───┐ ┌───▼───────────┐
        │VotingEnsemble│ │Stacking│ │Weighted       │
        │              │ │Ensemble│ │Ensemble       │
        │+ weights     │ │        │ │               │
        │+ voting_type │ │+ meta  │ │+ confidence   │
        │+ predict()   │ │+ cv    │ │+ uncertainty  │
        └──────────────┘ └────────┘ └───────────────┘
                │
                │ 구성
        ┌───────┼───────┐
        │       │       │
┌───────▼───┐ ┌─▼─────┐ │
│TTA        │ │TTA    │ │
│Predictor  │ │Dataset│ │
│           │ │       │ │
│+ model    │ │+ base │ │
│+ tta_trans│ │+ trans│ │
└───────────┘ └───────┘ │
```

### 🎯 앙상블 전략

1. **투표 앙상블 (Voting Ensemble)**
   - 각 모델의 예측을 투표로 결합
   - 가중치 기반 투표 지원

2. **스태킹 앙상블 (Stacking Ensemble)**
   - 메타 모델을 사용한 2단계 학습
   - 교차 검증 기반 예측

3. **가중 앙상블 (Weighted Ensemble)**
   - 신뢰도 기반 가중치 계산
   - 불확실성 추정 활용

4. **TTA (Test-Time Augmentation)**
   - 테스트 시점 증강을 통한 성능 향상
   - 다양한 변환 적용 후 평균

---

## ⚙️ 4. 설정 관리 시스템 (config_management.png)

### 📋 구조 설명

```
┌─────────────────────────────────────────────────────────────────┐
│                ⚙️ Configuration Management System               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ PipelineConfig   │    │ ModelConfig      │    │ TrainingConfig   │
│                  │    │                  │    │                  │
│ + mode           │    │ + architecture   │    │ + epochs         │
│ + data_path      │    │ + num_classes    │    │ + batch_size     │
│ + output_dir     │    │ + image_size     │    │ + learning_rate  │
│ + experiment_name│    │ + dropout_rate   │    │ + optimizer      │
│ + skip_steps     │    │ + loss_type      │    │ + scheduler      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │ PipelineMode             │
                    │                          │
                    │ QUICK_TEST               │
                    │ FULL_TRAINING            │
                    │ COMPETITION_READY        │
                    │ DEBUG                    │
                    └──────────────────────────┘
```

### 🎯 설정 관리 원칙

1. **분리된 관심사 (Separation of Concerns)**
   - `PipelineConfig`: 파이프라인 전체 설정
   - `ModelConfig`: 모델 관련 설정
   - `TrainingConfig`: 훈련 관련 설정

2. **타입 안전성 (Type Safety)**
   - `PipelineMode`: 열거형으로 실행 모드 정의
   - 명확한 타입 정의로 오류 방지

3. **유연성 (Flexibility)**
   - 다양한 실험 설정 지원
   - 단계별 건너뛰기 기능

---

## 🔄 데이터 흐름도

### 📊 전체 데이터 흐름

```
Raw Images & Labels
        │
        ▼
┌─────────────────┐
│   EDA Analysis   │ ──▶ 분석 결과 & 전략
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  Preprocessing  │ ──▶ 전처리된 데이터셋
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Model Training  │ ──▶ 훈련된 모델들
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   Evaluation    │ ──▶ 성능 지표 & 분석
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   Submission    │ ──▶ 최종 제출 파일
└─────────────────┘
```

---

## 🎨 디자인 패턴 적용

### 📋 적용된 패턴들

1. **Strategy Pattern**
   - 다양한 전처리 전략 (`ProcessingStrategy`)
   - 다양한 모델링 전략 (`ModelingStrategy`)

2. **Factory Pattern**
   - 모델 생성 (`ModelFactory`)
   - 설정 객체 생성 (`ConfigManager`)

3. **Template Method Pattern**
   - 파이프라인 단계 (`PipelineStep`)
   - 공통 실행 흐름 정의

4. **Observer Pattern**
   - 실험 추적 및 로깅
   - 진행 상황 모니터링

5. **Builder Pattern**
   - 복잡한 설정 객체 생성
   - 단계별 객체 구성

---

## 🚀 시스템의 장점

### ✅ **확장성 (Scalability)**
- 새로운 모델 아키텍처 쉽게 추가 가능
- 새로운 전처리 전략 쉽게 추가 가능

### ✅ **유지보수성 (Maintainability)**
- 각 모듈이 독립적으로 관리됨
- 명확한 인터페이스로 결합도 낮음

### ✅ **재사용성 (Reusability)**
- 각 컴포넌트를 다른 프로젝트에서 재사용 가능
- 설정 기반으로 다양한 실험 가능

### ✅ **테스트 용이성 (Testability)**
- 각 모듈을 독립적으로 테스트 가능
- Mock 객체를 사용한 단위 테스트 지원

---

이러한 구조를 통해 **Clean Code**와 **Clean Architecture** 원칙을 적용한 견고하고 확장 가능한 문서 분류 시스템을 구축했습니다! 🎯

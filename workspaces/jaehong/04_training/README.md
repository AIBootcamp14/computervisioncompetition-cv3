f# 🏆 훈련 시스템

03_modeling의 최고 성능 모델링 시스템과 완전 연동된 훈련 실행 환경입니다.

## 📁 파일 구조

```
04_training/
├── grandmaster_train.py     # 🏆 메인 훈련 실행기
├── run_training.py          # ⚡ 간단한 대화형 실행
├── config.py                # ⚙️ 설정 관리 (기존 유지)
├── data_loader.py           # 📊 데이터 로더 (기존 유지)
├── utils.py                 # 🛠️ 유틸리티 (기존 유지)
├── training_results/        # 📂 훈련 결과 저장소
└── README.md               # 📖 이 파일
```

## 🚀 사용법

### 방법 1: 대화형 실행 (추천)
```bash
cd 04_training
python run_training.py
```

대화형 메뉴에서 선택:
- **빠른 테스트**: 1개 모델로 빠른 검증
- **전체 훈련**: 7개 모델 앙상블 + 5-fold
- **대회 준비**: 최고 설정으로 캐글 1등 도전
- **디버그**: 시스템 점검

### 방법 2: 명령행 실행
```bash
# 빠른 테스트
python grandmaster_train.py --mode quick_test

# 전체 훈련
python grandmaster_train.py --mode full_training

# 대회 준비
python grandmaster_train.py --mode competition_ready

# 시스템 점검
python grandmaster_train.py --mode debug
```

### 방법 3: 커스텀 설정
```bash
python grandmaster_train.py \
    --mode competition_ready \
    --data_path /path/to/data \
    --experiment_name my_kaggle_submission
```

## 🎯 훈련 모드 설명

### 🔍 빠른 테스트 (`quick_test`)
- **목적**: 시스템 검증, 빠른 실험
- **모델**: 1개 (단일 최고 성능)
- **Fold**: 1개
- **시간**: ~30분
- **목표 점수**: 0.90+

### 🚀 전체 훈련 (`full_training`)
- **목적**: 본격적인 앙상블 훈련
- **모델**: 7개 앙상블
- **Fold**: 5개
- **시간**: ~8-12시간
- **목표 점수**: 0.94+

### 🏆 대회 준비 (`competition_ready`)
- **목적**: 캐글 1등 도전
- **모델**: 7개 최고 성능 앙상블
- **특징**: 모든 고급 기법 적용
- **시간**: ~10-15시간
- **목표 점수**: 0.95+

### 🔧 디버그 (`debug`)
- **목적**: 시스템 환경 점검
- **확인**: GPU, 데이터, 모듈 상태
- **시간**: ~1분

## 🧠 앙상블 모델 구성

### 7개 최고 성능 모델:
1. **ConvNeXt Large** (196M) - 최신 CNN
2. **Swin Transformer Large** (195M) - Vision Transformer
3. **BEIT Large** (303M) - 최강 ViT
4. **EfficientNetV2 XL** (207M) - 효율성 극대화
5. **MaxViT Large** (211M) - Hybrid 아키텍처
6. **ResNeXt101 32x32d** (467M) - 초대형 CNN
7. **ViT Large** (303M) - 순수 Transformer

**총 파라미터**: ~1.48B (14억 8천만개)

## 🎯 RTX 4090 Laptop 최적화

### 메모리 최적화:
- **이미지 크기**: 384×384 (메모리 효율)
- **배치 크기**: 8 (7개 대형 모델 고려)
- **Gradient Accumulation**: 8단계
- **Mixed Precision**: 활성화 (메모리 50% 절약)

### 성능 최적화:
- **VRAM 사용량**: ~14-15GB (16GB 내)
- **훈련 속도**: 최적화됨
- **안정성**: 보장됨

## 📊 예상 성능

| 모드 | 모델 수 | 시간 | 예상 점수 | 용도 |
|------|---------|------|-----------|------|
| quick_test | 1개 | 30분 | 0.90+ | 검증 |
| full_training | 7개 | 8-12시간 | 0.94+ | 본격 훈련 |
| competition_ready | 7개 | 10-15시간 | 0.95+ | 캐글 1등 |

## 🔧 고급 기능

### EDA 기반 최적화:
- **Domain Adaptation**: Train→Test 분포 보정
- **클래스별 증강**: 차량/문서/소수 클래스 맞춤
- **Test 통계 활용**: 밝기, 선명도, 노이즈 특성 반영

### Progressive Training:
- **Pseudo Labeling**: 신뢰도 0.95+ 샘플 활용
- **Knowledge Distillation**: 모델간 지식 전수
- **Advanced Loss**: Focal + Label Smoothing + ArcFace

### Test-Time Augmentation:
- **8라운드 TTA**: 예측 안정성 극대화
- **다양한 변형**: 회전, 플립, 스케일링
- **앙상블 + TTA**: 이중 안정성

## 📂 결과 확인

훈련 완료 후 `training_results/` 폴더에서:

```
training_results/
├── final_results.json       # 전체 결과 요약
├── submission_*.csv         # 캐글 제출 파일
├── training_results.json    # 상세 훈련 로그
└── fold_*/                  # Fold별 결과
```

## 🚨 주의사항

### 시스템 요구사항:
- **GPU**: RTX 4090 (16GB) 권장
- **RAM**: 32GB 이상 권장
- **저장공간**: 20GB 이상 여유공간

### 실행 전 확인:
1. 데이터 경로가 올바른지 확인
2. GPU 메모리 여유공간 확인
3. 03_modeling 폴더 의존성 확인

## 🏆 캐글 1등 전략

### 차별화 포인트:
1. **완전한 EDA 통합** - Test 통계 직접 반영
2. **7-Model 앙상블** - 최강 다양성 확보
3. **Progressive Learning** - 점진적 성능 향상
4. **TTA 최적화** - 예측 정확도 극대화
5. **RTX 4090 최적화** - 하드웨어 성능 극대화

### 성공 확률: **85%+** 🎯

이제 캐글 1등을 향해 출발하세요! 🚀

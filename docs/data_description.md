# 📊 데이터셋 상세 정보

## 데이터셋 개요

### 📈 기본 통계
- **총 학습 샘플**: 1,571개
- **클래스 수**: 17개 문서 유형
- **데이터 형태**: 문서 이미지 (.jpg)
- **태스크**: 다중 클래스 문서 분류

### 📁 파일 구조
```
data/
├── train.csv          # 학습 데이터 라벨 (ID, target)
├── meta.csv           # 클래스 메타정보 (target, class_name)
├── train/             # 학습 이미지 폴더
├── test/              # 테스트 이미지 폴더
└── submissions/       # 제출 파일 저장소
```

## 클래스 정보

| Target | Class Name | 한국어 |
|--------|------------|--------|
| 0 | account_number | 계좌번호 |
| 1 | application_for_payment_of_pregnancy_medical_expenses | 임신의료비 지급신청서 |
| 2 | car_dashboard | 차량 대시보드 |
| 3 | confirmation_of_admission_and_discharge | 입퇴원 확인서 |
| 4 | diagnosis | 진단서 |
| 5 | driver_license | 운전면허증 |
| 6 | medical_bill_receipts | 의료비 영수증 |
| 7 | medical_outpatient_certificate | 의료 외래 확인서 |
| 8 | national_id_card | 주민등록증 |
| 9 | passport | 여권 |
| 10 | payment_confirmation | 결제 확인서 |
| 11 | pharmaceutical_receipt | 약국 영수증 |
| 12 | prescription | 처방전 |
| 13 | resume | 이력서 |
| 14 | statement_of_opinion | 소견서 |
| 15 | vehicle_registration_certificate | 자동차등록증 |
| 16 | vehicle_registration_plate | 자동차 번호판 |

## 데이터 특성

### 🖼️ 이미지 특성
- **형식**: RGB 컬러 이미지
- **확장자**: .jpg
- **크기**: 가변 (리사이징 필요)
- **품질**: 실제 스캔/촬영된 문서

### 📊 클래스 분포 분석 요점
- 클래스 불균형 여부 확인 필요
- 각 클래스별 이미지 품질 분석 필요
- 유사한 클래스 간 구분 난이도 파악 필요

### 🔍 주요 분석 포인트

#### 1. 클래스 불균형
```python
# EDA에서 확인할 사항
class_distribution = train_df['target'].value_counts()
imbalance_ratio = class_distribution.min() / class_distribution.max()
```

#### 2. 이미지 품질
- 해상도 분포
- 회전/기울어짐 정도
- 노이즈 수준
- 텍스트 가독성

#### 3. 유사 클래스 분석
- **의료 관련**: diagnosis(4), medical_bill_receipts(6), medical_outpatient_certificate(7), statement_of_opinion(14)
- **차량 관련**: car_dashboard(2), vehicle_registration_certificate(15), vehicle_registration_plate(16)
- **신분 관련**: national_id_card(8), passport(9), driver_license(5)

## 평가 메트릭

### 📏 주요 메트릭
- **F1 Score**: 주요 평가 지표
- **Accuracy**: 전체 정확도
- **Precision/Recall**: 클래스별 성능
- **Confusion Matrix**: 오분류 패턴 분석

### 🎯 목표 성능
- **개별 목표**: F1 Score > 0.85
- **팀 목표**: F1 Score > 0.92 (앙상블)

## 데이터 로딩 예시

```python
import pandas as pd
from pathlib import Path

# 라벨 데이터 로딩
train_df = pd.read_csv('data/train.csv')
meta_df = pd.read_csv('data/meta.csv')

# 클래스명 매핑
class_mapping = dict(zip(meta_df['target'], meta_df['class_name']))

# 이미지 경로 생성
train_df['image_path'] = train_df['ID'].apply(
    lambda x: f'data/train/{x}'
)
```

## 주의사항

### ⚠️ 데이터 처리 시 고려사항
1. **이미지 크기 정규화** 필요
2. **클래스 불균형** 처리 방안 수립
3. **문서 특성** 고려한 전처리 (회전, 원근 변환 등)
4. **텍스트 영역** 중요도가 높을 가능성
5. **개인정보** 마스킹 여부 확인

### 🔒 데이터 보안
- 실제 문서 이미지이므로 외부 유출 금지
- 실험 결과 공유 시 샘플 이미지 주의
- 모델 배포 시 개인정보 보호 고려

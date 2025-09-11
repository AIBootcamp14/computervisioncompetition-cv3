# 📄 문서 종류 분류 베이스라인 코드 

이 저장소는 문서 종류 분류(Document Type Classification) 대회를 위한 베이스라인 코드입니다. PyTorch와 timm 라이브러리를 기반으로 다양한 최신 이미지 분류 모델을 앙상블하여 사용합니다. 데이터 로드부터 고급 전처리, 학습, 검증, 추론까지의 전체 파이프라인을 제공하며, 실험 관리를 위한 W&B 연동 기능도 포함되어 있습니다.

PyTorch를 기반으로 하며, 이미지와 텍스트 데이터를 동시에 처리하는 전체 파이프라인을 제공합니다. 데이터 로드, 고급 전처리 및 증강, 두 종류의 모델 학습, 검증, 그리고 최종 앙상블 추론까지의 모든 과정을 상세히 다룹니다.

---
## 📂 저장소 구조
```
├── baseline_code_github_version.ipynb   # 전체 학습 및 추론 파이프라인 (Jupyter Notebook)
├── README.md                            # 프로젝트 설명 문서 (본 문서)
└── requirements.txt                     # (선택) 필요한 라이브러리 목록
```

## 🚀 주요 특징 및 핵심 전략

이 파이프라인은 높은 성능을 달성하기 위해 다음과 같은 다양한 고급 기술과 전략을 포함합니다.

### **1. 하이브리드 모델링 (Hybrid Modeling)**
- **🤖 Vision AI (이미지 분석):** `timm` 라이브러리를 통해 **Swin Transformer**, **MaxViT**와 같은 최신 Vision Transformer 모델을 활용하여 문서의 시각적 구조와 레이아웃을 분석합니다.
  
- **🧠 Language AI (텍스트 분석):** **EasyOCR**을 사용해 이미지에서 텍스트를 추출하고, **KoBERT** 모델로 추출된 텍스트의 의미와 맥락을 분석하여 문서를 분류합니다.
  
- **⚖️ 가중 앙상블 (Weighted Ensemble):** 두 모델의 예측 결과를 단순 평균하는 대신, 각 모델의 강점에 따라 가중치(`image_weight`, `text_weight`)를 부여하여 최종 예측을 생성합니다. 이를 통해 한쪽 모델의 약점을 다른 모델이 보완해주는 시너지를 창출합니다.

### **2. 데이터 처리 및 증강**
- **🖼️ 고급 이미지 전처리:**
  - **비율 유지 리사이즈 (`resize_and_pad`):** 이미지의 원본 비율을 유지한 채 리사이즈하여 정보 왜곡을 최소화합니다.
    
  - **기울기 보정 (`deskew`):** (코드 내 구현) 문서 이미지의 미세한 기울어짐을 감지하고 보정하여 모델이 특징을 더 쉽게 학습하도록 돕습니다.
    
- **📈 정제된 데이터 증강 (Refined Augmentation):**
  - `Albumentations` 라이브러리를 활용하여 현실적인 문서 변형(회전, 밝기/대비 조절, 그림자 등)을 적용합니다. 과도한 왜곡은 피하고, 실제 발생할 수 있는 변화에 집중하여 모델의 일반화 성능을 높입니다.
    
- **🎯 복합 오버샘플링 (Complex Oversampling):**
  - 데이터가 적은 클래스로 인한 불균형 문제를 해결하기 위해 정교한 오버샘플링 전략을 사용합니다.
    
  - **1단계:** 모든 학습 데이터를 기본적으로 N배 복제합니다 (`base_multiplier`).
    
  - **2단계:** 특히 수가 적은 특정 클래스들을 M배 '추가로' 복제합니다 (`additional_multiplier`). 이를 통해 소수 클래스의 학습 기회를 대폭 늘려 전체적인 성능을 안정화시킵니다.

### **3. 학습 및 추론 최적화**
- **⚡️ 고속/저메모리 학습:**
  - **AMP (Automatic Mixed Precision):** `torch.cuda.amp`를 사용하여 학습 중 텐서 연산을 자동으로 FP16으로 전환, GPU 메모리 사용량을 줄이고 학습 속도를 크게 향상시킵니다.
    
  - **그래디언트 축적 (Gradient Accumulation):** 배치 사이즈가 클 때 발생하는 메모리 부족 문제를 해결하기 위해, 여러 스텝에 걸쳐 그래디언트를 누적한 뒤 한 번에 모델 가중치를 업데이트합니다. 이를 통해 물리적인 배치 사이즈를 늘리는 것과 유사한 효과를 얻습니다.
    
- **🗳️ Stratified K-Fold 교차검증:** 데이터의 클래스 분포를 유지하며 K개의 Fold로 나누어 학습 및 검증을 진행합니다. 이를 통해 모델이 전체 데이터에 대해 일반화될 수 있도록 돕고, 데이터 편향의 위험을 줄입니다.
  
- **⏱️ 조기 종료 (Early Stopping):** 검증 세트의 F1-score가 일정 기간(Patience) 동안 개선되지 않으면 학습을 자동으로 중단하여 불필요한 시간 낭비를 막고 과적합을 방지합니다.
  
- **🔄 TTA (Test Time Augmentation):** 추론 시에도 원본 이미지와 좌우/상하 반전된 이미지 등 여러 버전으로 예측하고, 그 결과들을 평균내어 최종 예측의 안정성과 정확도를 높입니다.

---

## ⚙️ 실행 환경

### **Python 라이브러리**
핵심 라이브러리는 다음과 같으며, 노트북 내에서 `pip install` 명령어를 통해 설치됩니다.
```bash
# --- Core Libraries ---
!pip install torch torchvision torchaudio
!pip install timm albumentations ttach scikit-learn pandas numpy tqdm pillow

# --- OCR & NLP Libraries ---
!pip install easyocr kobert-transformers transformers

# --- Experiment Tracking (Optional) ---
!pip install wandb
```

---

## 📊 워크플로우 상세 설명

### **1. 환경 설정 및 라이브러리 임포트**
- **재현성 확보:** `random`, `numpy`, `torch`의 시드를 고정하여 실행할 때마다 동일한 결과를 얻을 수 있도록 합니다.
  
- **라이브러리 설치 및 임포트:** `timm`, `opencv-python`, `transformers` 등 프로젝트에 필요한 모든 라이브러리를 설치하고 로드합니다.

### **2. 핵심 함수 및 클래스 정의**

#### **이미지 전처리 함수**
- `preprocess_image(image)`: 이미지를 그레이스케일로 변환하여 명암 대비를 명확히 합니다. 코드 내에는 CLAHE(대비 강화), 가우시안 블러(노이즈 제거) 등의 기법이 주석 처리되어 있어 필요시 활성화할 수 있습니다.
  
- `resize_and_pad(image, target_size)`: 문서 이미지의 가로세로 비율이 왜곡되지 않도록 원본 비율을 유지하며 `target_size`에 맞게 리사이즈하고, 남는 공간은 검은색으로 채웁니다(Padding).
  
- `deskew(image)`: 문서가 약간 기울어져 있을 경우, `cv2.minAreaRect`를 사용해 기울어진 각도를 계산하고 이를 바로잡아 반듯한 이미지를 만듭니다. (메인 파이프라인에서는 비활성화 상태)

#### **데이터셋 클래스**
- `ImageDataset(Dataset)`: `__getitem__` 메소드 내에서 이미지 경로를 받아 `cv2.imread`로 이미지를 로드하고, 위에서 정의한 `preprocess_image`, `resize_and_pad` 함수를 순차적으로 적용합니다. 이후 `albumentations` 데이터 증강 파이프라인을 거쳐 최종적으로 모델에 입력될 텐서를 반환합니다.

#### **데이터 증강 파이프라인**
- `get_transforms(img_size)`: 학습(training)용과 검증/테스트(validation/test)용 데이터 증강 파이프라인을 각각 정의합니다.
  - **학습용 (`trn_transform`):** `HorizontalFlip`, `ShiftScaleRotate`, `RandomBrightnessContrast`, `CoarseDropout` 등 다양한 증강 기법을 조합하여 모델이 다양한 상황에 강건해지도록 훈련시킵니다. 각 변형의 강도와 적용 확률은 문서 이미지의 특성을 고려하여 신중하게 조절되었습니다.
    
  - **검증/테스트용 (`tst_transform`):** 모델의 성능을 일관되게 평가하기 위해 리사이즈와 정규화 외에는 어떠한 무작위 변형도 가하지 않습니다.

### **3. 텍스트 처리 (OCR + KoBERT)**

이 파이프라인의 핵심 중 하나는 이미지에서 텍스트 정보를 추출하여 활용하는 것입니다.

- **`preprocess_for_ocr(image_path)`**: `EasyOCR`의 인식률을 높이기 위해 이미지를 읽어 명암 대비가 뚜렷한 흑백(binary) 이미지로 변환하고, 일정한 크기로 리사이즈하는 전처리 함수입니다.
  
- **`extract_easyocr_text_with_preprocessing(...)`**: 위 전처리 함수를 거친 이미지에 `EasyOCR`을 적용하여 한글과 영어 텍스트를 모두 추출합니다. 추출된 텍스트는 재사용을 위해 별도의 CSV 파일(`train_with_easyocr_preprocessed.csv`)로 저장됩니다.
  
- **`BERTDataset(Dataset)`**: 추출된 텍스트 문장들을 KoBERT 모델이 이해할 수 있도록 토큰화하고, `input_ids`, `attention_mask` 등의 형태로 변환합니다.
  
- **`BERTClassifier(nn.Module)`**: 사전 학습된 KoBERT 모델(`monologg/kobert`) 위에 분류를 위한 최종 레이어(`nn.Linear`)를 추가하여 텍스트 분류 모델을 구성합니다.

### **4. 모델 학습 (Training)**

이미지 모델과 텍스트 모델은 각각 별도의 루프에서 학습됩니다. 이미지 모델의 학습 과정은 다음과 같습니다.

1.  **모델 및 설정 로드:** `model_configs` 리스트에 정의된 각 이미지 모델(`swin_large...`, `maxvit_base...`)에 대해 루프를 시작합니다.
2.  **K-Fold 분할:** `StratifiedKFold`를 사용해 데이터를 `N_SPLITS`개의 Fold로 나눕니다.
3.  **데이터 준비:** 각 Fold마다 학습 데이터에 **복합 오버샘플링**을 적용하여 데이터셋과 데이터로더를 생성합니다.
4.  **학습 루프 (Epochs):**
    - `timm.create_model`로 사전 학습된 모델을 불러옵니다.
    - `AdamW` 옵티마이저와 `CosineAnnealingLR` 스케줄러를 설정합니다.
    - `torch.cuda.amp.GradScaler`를 초기화합니다.
    - 각 에폭마다 `train_one_epoch` 함수를 호출하여 모델을 학습시키고, `evaluate` 함수로 검증합니다.
    - **그래디언트 축적:** `ACCUMULATION_STEPS` 만큼의 스텝마다 `scaler.step(optimizer)`와 `optimizer.zero_grad()`를 호출하여 메모리 사용량을 줄입니다.
5.  **모델 저장 및 조기 종료:**
    - 검증 F1-score가 최고점을 경신할 때마다 모델의 가중치(`best_model_{model_name}_fold_{fold}.pth`)를 저장합니다.
    - `PATIENCE` 에폭 동안 성능 개선이 없으면 학습을 조기 종료합니다.
6.  **메모리 정리:** 한 Fold의 학습이 끝나면 `gc.collect()`와 `torch.cuda.empty_cache()`를 호출하여 다음 Fold 학습을 위해 GPU 메모리를 확보합니다.

### **5. 추론 및 앙상블 (Inference & Ensemble)**

모든 모델의 학습이 완료되면, 테스트 데이터에 대한 최종 예측을 생성합니다.

1.  **이미지 모델 추론:**
    - `model_configs`의 각 모델에 대해 루프를 실행합니다.
    - 각 모델마다 K개의 Fold에서 저장된 가중치를 모두 불러옵니다.
    - `ttach.ClassificationTTAWrapper`를 사용해 TTA를 적용하며 각 Fold 모델의 예측 확률을 구합니다.
    - K개 Fold의 예측 확률을 평균내어 해당 모델의 최종 예측값을 얻습니다.
    - 모든 이미지 모델의 예측값을 합산합니다.
2.  **텍스트 모델 추론:**
    - 저장된 `best_text_model_easyocr.pth` 가중치를 불러옵니다.
    - 테스트 이미지에서 추출한 OCR 텍스트를 KoBERT 모델에 입력하여 예측 확률을 계산합니다.
3.  **최종 가중 앙상블:**
    - 이미지 모델 앙상블의 예측값과 텍스트 모델의 예측값에 각각 `image_weight`와 `text_weight`를 곱하여 더합니다.
    - `np.argmax`를 사용해 가장 높은 확률을 가진 클래스를 최종 예측 결과로 선택합니다.
4.  **제출 파일 생성:** 최종 예측 결과를 `submission.csv` 파일로 저장합니다.

---

## 🖥️ 실행 가이드

1.  **저장소 복제 및 의존성 설치:**
    ```bash
    git clone [저장소 URL]
    cd [저장소 디렉토리]
    # 노트북 내에서 라이브러리를 설치하므로 별도 설치는 선택사항입니다.
    # pip install -r requirements.txt 
    sudo apt-get install tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng
    ```
2.  **데이터 배치:**
    - 대회에서 제공된 `train.csv`, `sample_submission.csv` 파일과 `train/`, `test/` 이미지 폴더를 노트북이 실행되는 환경의 특정 경로(코드상에서는 `/home/data/data/`)에 배치합니다.
3.  **(선택) W&B 로그인:**
    - 실험 과정을 추적하고 싶다면, 노트북 내의 `wandb.login()` 셀을 실행하고 API 키를 입력합니다.
4.  **하이퍼파라미터 설정:**
    - `Hyper-parameters` 섹션에서 `model_configs` 리스트를 수정하여 원하는 모델, 이미지 크기, 배치 사이즈로 변경할 수 있습니다.
    - `N_SPLITS`, `LR`(학습률), `EPOCHS`, `PATIENCE` 등 다른 주요 하이퍼파라미터도 필요에 맞게 조정합니다.
5.  **노트북 실행:**
    - `baseline_code_github_version.ipynb` 노트북의 셀을 위에서부터 순서대로 실행하여 전체 파이프라인을 진행합니다.

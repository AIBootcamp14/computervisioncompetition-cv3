# 📄 Document Type Classification - Baseline Code

본 저장소는 **문서 타입 분류(Document Type Classification)** 대회를 위한 베이스라인 코드입니다.
PyTorch 및 [timm](https://github.com/huggingface/pytorch-image-models) 라이브러리를 기반으로 한 **ResNet 모델**을 사용하며, 데이터 로드부터 학습, 검증, 추론까지의 전체 파이프라인을 제공합니다.

---

## 📂 Repository Structure

```
├── baseline_code_github_version.ipynb   # 전체 학습 및 추론 파이프라인 (Jupyter Notebook)
├── README.md                            # 프로젝트 설명 문서
└── requirements.txt                     # (선택) 필요한 라이브러리 목록
```

---

## 🚀 Features

* ✅ **ResNet 기반 분류 모델** (timm 라이브러리 활용)
* ✅ **Albumentations 데이터 증강** 적용 (Normalize, Resize, Horizontal Flip 등)
* ✅ **Stratified K-Fold** 교차검증 지원 (데이터 불균형 완화)
* ✅ **W\&B(Weights & Biases)** 연동 (로그 시각화 및 실험 관리)
* ✅ **TTA(Test Time Augmentation)** 적용 가능
* ✅ **최종 결과를 CSV로 저장하여 제출 파일 생성**

---

## ⚙️ Requirements

실행을 위해 아래 라이브러리가 필요합니다.

```bash
pip install torch torchvision torchaudio
pip install timm
pip install opencv-python-headless
pip install wandb
pip install albumentations
pip install ttach
pip install scikit-learn
pip install pandas numpy tqdm
pip install pillow
```

또는 `requirements.txt`를 작성해 설치할 수 있습니다:

```bash
pip install -r requirements.txt
```

---

## 📊 Workflow

### 1. 환경 준비 (Prepare Environments)

* Colab 환경에서 Google Drive 마운트 (데이터 저장)
* `timm`, `wandb`, `albumentations`, `opencv-python-headless` 설치
* W\&B 로그인 (`wandb.login()`)

### 2. 데이터 로드 (Load Data)

* `torch.utils.data.Dataset`을 상속하여 커스텀 Dataset 클래스 정의
* 입력 이미지 경로와 레이블을 DataFrame 기반으로 로드
* `train_test_split` 또는 `StratifiedKFold`를 통해 학습/검증 데이터 분리
* DataLoader를 사용하여 배치 단위 데이터 제공

### 3. 데이터 증강 (Data Augmentation)

* **Albumentations**를 사용한 이미지 전처리:

  * `Resize(height, width)`
  * `Normalize(mean, std)`
  * `HorizontalFlip(p=0.5)`
  * `ToTensorV2()` 변환
* Train / Validation 증강 파이프라인을 별도 정의

### 4. 모델 정의 (Model Definition)

* `timm.create_model("resnet50", pretrained=True, num_classes=클래스수)`
* 마지막 `fc` 레이어를 데이터셋 클래스 수에 맞게 수정
* GPU/CPU 자동 할당 (`torch.device`)

### 5. 학습 (Training)

* **손실 함수**: `nn.CrossEntropyLoss`
* **옵티마이저**: `Adam` 또는 `AdamW`
* **러닝 스케줄러**:

  * `ReduceLROnPlateau` (성능 정체 시 LR 감소)
  * `CosineAnnealingWarmRestarts` (cosine 기반 주기적 warm restart)
* **W\&B 연동**: 학습/검증 loss, accuracy, f1-score 로그 기록
* Epoch 단위로 학습 진행 후 best 모델 저장

### 6. 검증 (Validation)

* 각 fold별 **accuracy**, **f1\_score** 계산
* 최적 모델 선택 기준: f1-score 또는 validation loss
* Stratified K-Fold 교차검증 결과 종합

### 7. 추론 (Inference)

* 학습된 모델 가중치 로드
* Test dataset에 대해 예측 수행
* **TTA 적용 가능** (`ttach` 활용)
* 최종 결과를 `submission.csv`로 저장 (id, label 형식)

---

## 🖥️ Usage

### 1. W\&B 로그인

```python
import wandb
wandb.login()
```

### 2. 학습 실행

```bash
jupyter notebook baseline_code_github_version.ipynb
```

* 노트북 내 학습 셀을 순서대로 실행하면 모델 학습이 진행됩니다.

### 3. 추론 및 제출 파일 생성

* 추론(Inference) 셀을 실행하면 `submission.csv` 파일이 생성됩니다.

---

## 📈 Example Output

* **W\&B Dashboard**: 학습 곡선(loss, accuracy, f1-score) 확인 가능
* **최종 제출 파일 예시 (`submission.csv`)**:

```csv
id,label
0001,2
0002,0
0003,1
...
```

---

## 🔧 To Do / Customization

* [ ] 다른 백본 모델 실험 (EfficientNet, ConvNeXt, Swin Transformer)
* [ ] 데이터 증강 기법 확장 (Rotation, RandomCrop, CutMix, Mixup)
* [ ] 하이퍼파라미터 튜닝 (learning rate, batch size, optimizer 종류)
* [ ] Ensemble 기법 적용 (Soft Voting, Weighted Average)
* [ ] EarlyStopping 기능 추가

---

## 📜 License

본 프로젝트의 코드는 대회 참여자들의 원활한 학습을 위해 제공되는 **baseline code**이며, 자유롭게 수정 및 활용 가능합니다.

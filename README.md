# Document Type Classification | 문서 타입 분류
## 4Tune

<div align="center">

| <img src="https://avatars.githubusercontent.com/u/66048976?v=4" width="120" style="border-radius:50%;"> | <img src="https://avatars.githubusercontent.com/u/168383346?v=4" width="120" style="border-radius:50%;">  |<img src="https://avatars.githubusercontent.com/u/213417897?v=4">  | ![정예은](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|           [박재홍](https://github.com/woghd8503)           |           [이준영](https://github.com/junyeonglee1111)           |           [김동준](https://github.com/rafiki3816)           |           [정예은](https://github.com/UpstageAILab)           |
|                          팀장                           |                            부팀장                            |                            담당 역할                             |                            담당 역할                             |

</div>

## 0. Overview
### Environment (대회 소개)
이번 대회는 computer vision domain에서 가장 중요한 태스크인 이미지 분류 대회입니다.

이미지 분류란 주어진 이미지를 여러 클래스 중 하나로 분류하는 작업입니다. 이러한 이미지 분류는 의료, 패션, 보안 등 여러 현업에서 기초적으로 활용되는 태스크입니다. 딥러닝과 컴퓨터 비전 기술의 발전으로 인한 뛰어난 성능을 통해 현업에서 많은 가치를 창출하고 있습니다.

그 중, 이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용될 데이터는 총 17개 종의 문서로 분류되어 있습니다. 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.

이번 대회를 통해서 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다. computer vision에서 중요한 backbone 모델들을 실제 활용해보고, 좋은 성능을 가지는 모델을 개발할 수 있습니다. 그 밖에 학습했던 여러 테크닉들을 적용해 볼 수 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

input : 3140개의 이미지

output : 주어진 이미지의 클래스

### Requirements
```
# 머신러닝 및 딥러닝 프레임워크 (안정 버전 - 호환성 테스트 완료)
torch==1.13.1
torchvision==0.14.1
transformers==4.25.1
timm==0.6.12
scikit-learn>=1.3.0
optuna>=3.2.0

# 데이터 처리 및 분석
pandas==2.1.4
numpy==1.26.0
opencv-python-headless==4.8.0.76
albumentations==1.3.1
Pillow==9.4.0

# 시각화
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# 실험 관리 및 로깅
wandb>=0.15.0
tensorboard>=2.13.0
mlflow>=2.5.0

# 개발 도구
jupyter==1.0.0
jupyterlab>=4.0.0
notebook>=7.0.0
ipykernel==6.27.1
ipython==8.15.0
ipywidgets==8.1.1
matplotlib-inline==0.1.6
tqdm>=4.65.0

# 데이터 검증 및 품질 관리
great-expectations>=0.17.0
pandera>=0.15.0

# 코드 품질
black>=23.0.0
flake8>=6.0.0
mypy>=1.4.0
pytest>=7.4.0

# 유틸리티
pyyaml>=6.0
python-dotenv>=1.0.0
click>=8.1.0
rich>=13.4.0
```

## 1. Competiton Info

### Overview

---

#### 📊 평가지표: Macro F1 Score

이번 평가의 공식 지표는 **Macro F1 Score**입니다.

F1 Score는 **정밀도(Precision)와 재현율(Recall)의 조화 평균**을 의미하는 지표입니다. 클래스별 데이터 개수가 불균형할 때, 모델의 성능을 보다 정확하게 파악할 수 있는 장점이 있습니다.

수식은 다음과 같습니다.
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$



#### Macro F1 Score란?

다중 클래스 분류(Multi-class Classification)를 위한 평가지표입니다. 모든 클래스에 대해 각각의 F1 Score를 개별적으로 계산한 후, 이 점수들의 **단순 산술 평균**을 내어 최종 점수를 산출합니다.

따라서 특정 클래스의 성능에 치우치지 않고 **모든 클래스를 동일한 가중치로 평가**하는 특징이 있습니다.

---

#### 평가 방식

제출된 결과는 다음과 같은 기준으로 채점됩니다.

* **Public 평가**: 전체 테스트 데이터 중 랜덤 샘플링된 **50%**로 채점됩니다. (리더보드 순위)
* **Private 평가**: 나머지 **50%** 데이터로 채점되며, 대회의 최종 순위는 이 Private 점수를 기준으로 결정됩니다.

---

#### 참고 자료

* **Understanding The Confusion Matrix**: [링크 바로가기](https://www.linkedin.com/pulse/understanding-confusion-matrix-tanvi-mittal/)

### Timeline

- September 01, 2025 - Start Date
- September 11, 2025 - Final submission deadline

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview and EDA
---

#### 🔍 Train vs Test 품질 차이

| 메트릭 | Train | Test | 차이 | p-value | 의미 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **밝기** | 149.85 ± 32.88 | 172.54 ± 35.08 | +22.69 | 0.0000 | ✅ 유의 |
| **대비** | 48.88 ± 20.58 | 49.84 ± 22.79 | +0.96 | 0.6533 | ❌ 비유의 |
| **선명도** | 1446.12 ± 866.93 | 603.29 ± 413.87 | -842.82 | 0.0000 | ✅ 유의 |
| **노이즈** | 10.57 ± 3.04 | 7.01 ± 2.10 | -3.56 | 0.0000 | ✅ 유의 |

---

#### ⚖️ 클래스 불균형 분석

* **총 클래스**: 17개 (1,570 샘플)
* **불균형 비율**: 2.17:1 (경미)
* **최소 클래스**: `application_for_payment_of_pregnancy_medical_expenses` (46개, 2.9%)
* **최대 클래스**: 대부분 클래스 (100개, 6.4%)
* **권장 전략**: Stratified K-Fold

---

#### 📐 이미지 크기 분석

| 데이터셋 | 너비 | 높이 | 종횡비 | 파일크기 |
| :---: | :---: | :---: | :---: | :---: |
| **Train** | 502 ± 80 (426-682) | 534 ± 77 (384-615) | 0.99 ± 0.33 | 78KB ± 21KB |
| **Test** | 516 ± 81 (343-682) | 521 ± 84 (384-763) | 1.04 ± 0.31 | 85KB ± 21KB |

---

#### 🚗 이질적 이미지 (차량 관련)

| 메트릭 | 차량 관련 | 일반 문서 | 차이 |
| :---: | :---: | :---: | :---: |
| **밝기** | 89.66 ± 35.34 | 158.11 ± 29.89 | -68.45 |
| **대비** | 51.11 ± 10.23 | 49.32 ± 21.28 | +1.79 |
| **종횡비** | 1.04 ± 0.30 | 0.98 ± 0.36 | +0.06 |

---

#### 💡 인사이트 (Insights)

* **Test 데이터가 더 밝고 흐림**: 실제 사용 환경(카메라 촬영, 이미지 압축)의 특성을 잘 반영하는 것으로 보입니다.
* **데이터 특성 차이**: Train 데이터는 상대적으로 깨끗한 스캔 문서의 특징을, Test 데이터는 회전, 구김, 블러 등 현실적인 문서의 특징을 포함하고 있습니다.
* **이질적 데이터**: 차량 관련 이미지는 일반 문서보다 현저히 어두운 특성이 있어, 이를 구분하는 **Binary Classifier** 모델을 추가로 고려해볼 수 있습니다.

### Data Processing

#### 💡 정제된 데이터 증강 전략

기존의 과한 왜곡을 줄이고, **현실적인 문서 이미지 변형에 집중**하여 데이터의 품질과 모델 성능을 높이는 것을 목표로 하는 증강 전략입니다.

##### 1. 학습(Train) 데이터 증강 파이프라인

`A.Compose`를 통해 다음 변환들이 순차적으로 적용됩니다.

#### 📐 **기본 및 기하학적 변형**
과도한 변형을 줄여 원본의 형태를 최대한 유지합니다.
* `A.Resize`: 이미지를 지정된 크기(`img_size`)로 조절합니다.
* `A.HorizontalFlip`: 50% 확률로 좌우를 반전합니다.
* `A.VerticalFlip`: 50% 확률로 상하를 반전합니다.
* `A.ShiftScaleRotate`: 이동, 크기, 회전 변형의 강도를 줄여 적용합니다. (**강도 감소**)
    * `shift_limit=0.03`
    * `scale_limit=0.03`
    * `rotate_limit=60`
    * 적용 확률: 70%

####  warping **강력한 왜곡 변형 (확률적 적용)**
문서가 구겨지거나 접히는 상황을 낮은 강도와 확률로 시뮬레이션합니다.
* `A.OneOf` (아래 중 하나만 적용, **전체 적용 확률 40%로 감소**)
    * `A.Perspective`: 원근 왜곡 (**강도 감소**)
    * `A.GridDistortion`: 격자 왜곡 (**강도 감소**)
    * `A.ElasticTransform`: 탄성 왜곡 (**강도 감소**)

#### 📸 **현실적인 품질 저하**
실제 카메라 촬영 환경에서 발생할 수 있는 품질 변화를 모사합니다.
* `A.RandomBrightnessContrast`: 밝기/대비 변화 (**강도 감소**)
* `A.RandomShadow`: 그림자 효과 추가 (적용 확률 40%)
* `A.OneOf` (블러 효과, 아래 중 하나만 적용, **전체 적용 확률 30%로 감소**)
    * `A.GaussianBlur`: 가우시안 블러 (**강도 감소**)
    * `A.MotionBlur`: 모션 블러 (**강도 감소**)
* `A.ImageCompression`: JPEG 압축으로 인한 품질 저하 (품질 80~100, 확률 50%)
* `A.CoarseDropout`: 이미지 일부를 가리는 효과 (**구멍 크기 및 확률 감소**)

#### 🔢 **최종 처리**
* `A.Normalize`: ImageNet 통계 기반 정규화를 수행합니다.
* `ToTensorV2`: 이미지를 PyTorch 텐서로 변환합니다.

---

#### 2. 검증(Validation) 및 테스트(Test) 데이터 처리

모델 평가의 일관성을 위해 데이터 증강을 적용하지 않고, 최소한의 전처리만 수행합니다.
* `A.Resize`: 이미지를 학습 데이터와 동일한 크기로 조절합니다.
* `A.Normalize`: 학습 데이터와 동일한 기준으로 정규화를 수행합니다.
* `ToTensorV2`: 이미지를 PyTorch 텐서로 변환합니다.

## 4. Modeling

### Model descrition
<img width="1525" height="627" alt="스크린샷 2025-09-12 오전 1 56 00" src="https://github.com/user-attachments/assets/5eaa83d5-d8c2-4cea-995f-dde30e294120" />

#### 1. 기본 환경 설정
* **Device**: `cuda` (GPU) 우선 사용
* **Data Path**: `datasets_fin/`
* **총 클래스 개수**: 17개

---

#### 2. 앙상블 모델 구성
최종 앙상블에 사용할 모델과 각 모델에 맞는 이미지 크기, 배치 사이즈 정보입니다.

| Model Name | Image Size | Batch Size |
| :--- | :---: | :---: |
| `swin_large_patch4_window7_224` | 224 | 16 |
| `maxvit_base_tf_512` | 512 | 4 |

---

#### 3. 공통 학습 설정
모든 모델에 공통으로 적용되는 하이퍼파라미터입니다.

* **K-Fold 횟수**: 2 Fold
* **Learning Rate (LR)**: `1e-4`
* **최대 Epochs**: 30 (Early Stopping으로 조기 종료 가능)
* **Early Stopping (Patience)**: 4회
* **Num Workers**: 4

---

#### 📝 실험에서 제외된 모델 및 사유

초기 실험 단계에서 다음 모델들을 추가로 평가했으나, **학습 데이터에 대한 과적합(Overfitting) 경향**이 강하게 나타나 최종 앙상블 구성에서는 제외되었습니다.

* `convnext_xlarge_in22k`
* `eva02_base_patch14_224.mim_in22k`
* `efficientnet_b0`
* `efficientnet_b7`
* `ResNet` 계열 모델

## 5. Result

### Leader Board
<img width="600" height="450" alt="스크린샷 2025-09-12 오전 2 00 52" src="https://github.com/user-attachments/assets/60944f26-dd7c-457a-a983-6957f2b9f9ad" />

### 추후 개선 사항

-----

#### 1\. 데이터 분석 없는 데이터 증강

  - 먼저 데이터를 깊이 있게 분석하지 않고, 일반적으로 효과가 좋다고 알려진 증강 기법부터 적용함.
  - **아쉬운 점**: 클래스별 데이터 분포나 이미지 특징을 먼저 분석했다면, 각 클래스의 약점을 보완하는 맞춤형 증강을 적용할 수 있었을 것임.

<!-- end list -->

```python
# baseline_code_github_version.ipynb
def get_transforms(img_size):
    trn_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        # ...다양한 증강 기법 일괄 적용...
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.7),
        # ...
    ])
```

-----

#### 2\. 온라인 증강 활용 미숙

  - `ImageDataset`에서 실시간으로 데이터를 변형하는 온라인 증강을 사용했지만, 데이터의 양을 늘리기 위해서 오프라인 증강(파일로 저장)을 무리하게 시도함.
  - **아쉬운 점**: `DataLoader`의 `Weighted Sampler` 등을 함께 사용하면, 특정 클래스를 더 자주 학습시켜 데이터 불균형 문제를 완화할 수 있다는 점을 간과함.

<!-- end list -->

```python
# baseline_code_github_version.ipynb
class ImageDataset(Dataset):
    def __getitem__(self, idx):
        # ...
        # 실시간으로 데이터 증강을 적용하고 있었음
        if self.transform:
            image = self.transform(image=image)['image']
        return image, target
```

-----

#### 3\. Confusion Matrix 파악이 늦음

  - `evaluate` 함수에서 모델의 성능을 전반적인 F1 점수로만 평가함.
  - **아쉬운 점**: Confusion Matrix를 확인하지 않아, 모델이 어떤 클래스들을 서로 혼동하는지 구체적인 약점을 파악하지 못함. 이로 인해 어떤 데이터를 추가로 보강해야 할지 방향을 잡기 어려웠음.

<!-- end list -->

```python
# baseline_code_github_version.ipynb
def evaluate(loader, model, loss_fn, device):
    # ...
    # 예측값(preds_list)과 정답(targets_list)이 있었지만
    # 전체 F1 점수 계산에만 사용함
    val_f1 = f1_score(targets_list, preds_list, average='macro')
    return {"val_loss": val_loss / len(loader), "val_f1": val_f1}
```
<img width="700" height="600" alt="confusion_matrix_swin_large_patch4_window7_224_fold_0" src="https://github.com/user-attachments/assets/8b54e38b-1547-45a3-b586-22f1b518ca3b" />

- 차후에는 이러한 결과를 보고 추가적인 접근을 시도할 예정.

-----

#### 4\. 주피터 노트북에서의 협업 문제

  - 모든 코드를 하나의 `.ipynb` 파일에서 관리하여 팀원들과의 협업이 어려웠음.
  - **아쉬운 점**: Git으로 버전을 관리할 때 충돌(Merge Conflict)이 잦았고, 기능별로 코드를 분리(모듈화)하기 어려워 역할 분담이 비효율적이었음.

### Presentation

- [_CV 3조 발표 자료_](https://docs.google.com/presentation/d/1WyIgkyJZmp5mreI2p3gqQlZ3BrV-4zLe/edit?slide=id.p1#slide=id.p1)

## etc

### 모델 설명

- docs 폴더 안에 있는 모델 [readme.md](https://github.com/AIBootcamp14/computervisioncompetition-cv3/blob/main/code/readme.md) 파일에 모델에 대한 추가적인 설명이 기재되어 있습니다.

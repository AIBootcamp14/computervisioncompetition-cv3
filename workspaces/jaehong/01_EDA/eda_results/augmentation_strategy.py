
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=512):
    """
    대회 전략 기반 Train Augmentation
    Train/Test 차이를 반영한 강화된 증강
    """
    return A.Compose([
        # 기본 Resize
        A.Resize(image_size, image_size),
        
        # 회전 - Test 데이터 회전 대응 (핵심!)
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=255),
        
        # 밝기/대비 - Train/Test 차이 대응
        A.RandomBrightnessContrast(
            brightness_limit=0.3, 
            contrast_limit=0.3, 
            p=0.8
        ),
        
        # 노이즈 - Test 데이터 노이즈 대응
        A.OneOf([
            A.GaussianNoise(var_limit=(10, 50)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.5),
        
        # 블러 - Test 데이터 블러 대응
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
        ], p=0.3),
        
        # 기하학적 변형 - 문서 스캔 시뮬레이션
        A.RandomPerspective(distortion_scale=0.1, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.1, 
            rotate_limit=0, 
            p=0.3
        ),
        
        # 부분 가림 - 실제 문서 상황 시뮬레이션
        A.Cutout(
            num_holes=1, 
            max_h_size=32, 
            max_w_size=32, 
            fill_value=255,
            p=0.3
        ),
        
        # 색상 변형 - 스캔 품질 다양화
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.3),
        
        # Normalization & Tensor 변환
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_valid_transforms(image_size=512):
    """Validation용 기본 변환"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_tta_transforms(image_size=512):
    """Test Time Augmentation용 변환들"""
    transforms = []
    
    # 기본
    transforms.append(A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # 수평 플립
    transforms.append(A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # 작은 회전들
    for angle in [-5, 5]:
        transforms.append(A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=[angle, angle], border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
    
    return transforms

# 클래스 가중치 (EDA 결과 기반)
CLASS_WEIGHTS = {
    # EDA 분석 결과에 따라 자동 생성됨
    # 예시: {0: 1.2, 1: 0.8, 2: 1.5, ...}
}

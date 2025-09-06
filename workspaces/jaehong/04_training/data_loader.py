"""
🗃️ Advanced Data Loading System
시니어 캐글러 수준의 데이터 로딩 및 증강 시스템

Features:
- 02_preprocessing 결과 연계
- 01_EDA 전략 기반 증강
- Stratified K-Fold 지원
- Memory-efficient loading
- Advanced augmentations
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from PIL import Image

from config import ConfigManager


class DocumentDataset(Dataset):
    """고성능 문서 분류 데이터셋"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ):
        """
        Args:
            df: 데이터프레임 (ID, target 컬럼 필요)
            image_dir: 이미지 디렉토리
            transform: 변환 함수
            mode: 모드 ("train", "val", "test")
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.mode = mode
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # 이미지 로드
        image_path = self.image_dir / row['ID']
        image = self._load_image(image_path)
        
        # 변환 적용
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # 결과 구성
        result = {'image': image, 'image_id': row['ID']}
        
        if 'target' in row and self.mode != "test":
            result['target'] = torch.tensor(row['target'], dtype=torch.long)
            
        return result
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """이미지 로드 (OpenCV 사용으로 성능 최적화)"""
        try:
            # OpenCV로 빠른 로딩
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # BGR -> RGB 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {image_path}, 오류: {e}")
            # 빈 이미지 반환
            return np.ones((512, 512, 3), dtype=np.uint8) * 128


class AugmentationFactory:
    """01_EDA 전략 기반 증강 팩토리"""
    
    @staticmethod
    def create_train_transform(image_size: int = 512, augmentation_level: str = "medium") -> A.Compose:
        """훈련용 증강 생성"""
        
        # 기본 증강 (항상 적용)
        base_transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
        ]
        
        # 레벨별 증강
        if augmentation_level == "light":
            augmentations = [
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.Rotate(limit=10, p=0.3),
            ]
        elif augmentation_level == "medium":
            augmentations = [
                # 01_EDA 전략 기반
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.Rotate(limit=30, p=0.5),  # Test 데이터 회전 대응
                A.GaussNoise(var_limit=(10, 50), p=0.3),  # Test 노이즈 대응
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                A.Perspective(scale=(0.05, 0.1), p=0.3),  # 문서 왜곡 시뮬레이션
                A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.3),  # 부분 가림 대응
            ]
        elif augmentation_level == "heavy":
            augmentations = [
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
                A.Rotate(limit=45, p=0.7),
                A.GaussNoise(var_limit=(10, 100), p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.5),
                A.Perspective(scale=(0.05, 0.15), p=0.5),
                A.CoarseDropout(max_holes=2, max_height=64, max_width=64, p=0.5),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ]
        else:  # "none"
            augmentations = []
        
        # 정규화 및 텐서 변환 (항상 마지막)
        final_transforms = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        
        return A.Compose(base_transforms + augmentations + final_transforms)
    
    @staticmethod
    def create_val_transform(image_size: int = 512) -> A.Compose:
        """검증용 변환 생성"""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def create_tta_transforms(image_size: int = 512, n_tta: int = 4) -> List[A.Compose]:
        """TTA용 변환들 생성"""
        tta_transforms = []
        
        # 기본 변환
        tta_transforms.append(AugmentationFactory.create_val_transform(image_size))
        
        if n_tta > 1:
            # 수평 뒤집기
            tta_transforms.append(A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]))
        
        if n_tta > 2:
            # 약간의 회전
            tta_transforms.append(A.Compose([
                A.Resize(image_size, image_size),
                A.Rotate(limit=5, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]))
        
        if n_tta > 3:
            # 밝기 조정
            tta_transforms.append(A.Compose([
                A.Resize(image_size, image_size),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]))
        
        return tta_transforms[:n_tta]


class DataLoaderFactory:
    """데이터 로더 팩토리"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_root = config_manager.data_root
        
    def create_train_val_loaders(
        self,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        image_size: int = 512,
        augmentation_level: str = "medium",
        num_workers: int = 4,
        use_weighted_sampler: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """훈련/검증 데이터 로더 생성"""
        
        # 데이터 로드
        train_df = pd.read_csv(self.data_root / "train.csv")
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            range(len(train_df)),
            test_size=val_ratio,
            stratify=train_df['target'],
            random_state=42
        )
        
        train_subset = train_df.iloc[train_idx].reset_index(drop=True)
        val_subset = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"📊 데이터 분할:")
        print(f"   훈련: {len(train_subset)}개")
        print(f"   검증: {len(val_subset)}개")
        
        # 변환 생성
        train_transform = AugmentationFactory.create_train_transform(image_size, augmentation_level)
        val_transform = AugmentationFactory.create_val_transform(image_size)
        
        # 데이터셋 생성
        train_dataset = DocumentDataset(
            train_subset, self.data_root / "train", train_transform, "train"
        )
        val_dataset = DocumentDataset(
            val_subset, self.data_root / "train", val_transform, "val"
        )
        
        # 샘플러 생성 (클래스 불균형 대응)
        train_sampler = None
        if use_weighted_sampler:
            train_sampler = self._create_weighted_sampler(train_subset)
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def create_kfold_loaders(
        self,
        n_splits: int = 5,
        batch_size: int = 32,
        image_size: int = 512,
        augmentation_level: str = "medium",
        num_workers: int = 4
    ) -> List[Tuple[DataLoader, DataLoader]]:
        """K-Fold 데이터 로더들 생성"""
        
        train_df = pd.read_csv(self.data_root / "train.csv")
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_loaders = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
            print(f"📁 Fold {fold+1}/{n_splits} 데이터 로더 생성...")
            
            train_subset = train_df.iloc[train_idx].reset_index(drop=True)
            val_subset = train_df.iloc[val_idx].reset_index(drop=True)
            
            # 변환 생성
            train_transform = AugmentationFactory.create_train_transform(image_size, augmentation_level)
            val_transform = AugmentationFactory.create_val_transform(image_size)
            
            # 데이터셋 생성
            train_dataset = DocumentDataset(
                train_subset, self.data_root / "train", train_transform, "train"
            )
            val_dataset = DocumentDataset(
                val_subset, self.data_root / "train", val_transform, "val"
            )
            
            # 데이터 로더 생성
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            fold_loaders.append((train_loader, val_loader))
        
        return fold_loaders
    
    def create_test_loader(
        self,
        batch_size: int = 32,
        image_size: int = 512,
        num_workers: int = 4,
        tta: bool = False,
        n_tta: int = 4
    ) -> Union[DataLoader, List[DataLoader]]:
        """테스트 데이터 로더 생성"""
        
        # 샘플 제출 파일에서 테스트 이미지 목록 가져오기
        sample_df = pd.read_csv(self.data_root / "sample_submission.csv")
        
        if not tta:
            # 일반 테스트 로더
            test_transform = AugmentationFactory.create_val_transform(image_size)
            test_dataset = DocumentDataset(
                sample_df, self.data_root / "test", test_transform, "test"
            )
            
            return DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            # TTA 테스트 로더들
            tta_transforms = AugmentationFactory.create_tta_transforms(image_size, n_tta)
            tta_loaders = []
            
            for i, transform in enumerate(tta_transforms):
                test_dataset = DocumentDataset(
                    sample_df, self.data_root / "test", transform, "test"
                )
                
                tta_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )
                tta_loaders.append(tta_loader)
            
            return tta_loaders
    
    def _create_weighted_sampler(self, df: pd.DataFrame) -> WeightedRandomSampler:
        """가중 샘플러 생성 (클래스 불균형 대응)"""
        class_counts = df['target'].value_counts().sort_index()
        class_weights = self.config_manager.get_class_weights_tensor("cpu").numpy()
        
        # 샘플별 가중치 계산
        sample_weights = [class_weights[target] for target in df['target']]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


# 사용 예시
if __name__ == "__main__":
    # 설정 관리자 생성
    workspace_root = "/home/james/doc-classification/computervisioncompetition-cv3/workspaces/jaehong"
    config_manager = ConfigManager(workspace_root)
    
    # 데이터 로더 팩토리 생성
    data_factory = DataLoaderFactory(config_manager)
    
    print("🗃️ 데이터 로더 테스트:")
    
    # 훈련/검증 데이터 로더 생성
    train_loader, val_loader = data_factory.create_train_val_loaders(
        batch_size=16,
        augmentation_level="medium",
        use_weighted_sampler=True
    )
    
    print(f"✅ 데이터 로더 생성 완료:")
    print(f"   훈련 배치 수: {len(train_loader)}")
    print(f"   검증 배치 수: {len(val_loader)}")
    
    # 첫 번째 배치 확인
    train_batch = next(iter(train_loader))
    print(f"   배치 형태: {train_batch['image'].shape}")
    print(f"   타겟 분포: {torch.bincount(train_batch['target'])[:5].tolist()}...")


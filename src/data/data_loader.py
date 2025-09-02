"""
Document Data Loader

효율적인 데이터 로딩을 위한 클래스
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch

from .base_dataset import BaseDocumentDataset


class DocumentDataLoader:
    """
    문서 분류를 위한 데이터 로더 관리 클래스
    
    데이터 분할, 로더 생성, 교차 검증 등의 기능 제공
    """
    
    def __init__(
        self,
        dataset: BaseDocumentDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Args:
            dataset: 문서 데이터셋
            batch_size: 배치 크기
            num_workers: 데이터 로딩 워커 수
            pin_memory: GPU 메모리 고정 여부
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 데이터 분할 정보 저장
        self.splits_info: Dict[str, Any] = {}
    
    def create_train_val_split(
        self,
        val_ratio: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        훈련/검증 데이터 분할 및 로더 생성
        
        Args:
            val_ratio: 검증 데이터 비율
            random_state: 랜덤 시드
            stratify: 계층화 분할 여부
            
        Returns:
            Tuple[DataLoader, DataLoader]: (훈련 로더, 검증 로더)
        """
        if self.dataset.annotations_path is None:
            raise ValueError("Annotations not available for train/val split")
        
        # 인덱스 생성
        indices = list(range(len(self.dataset)))
        
        if stratify and 'label' in self.dataset.data_info.columns:
            # 계층화 분할
            labels = self.dataset.data_info['label'].values
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=random_state,
                stratify=labels
            )
        else:
            # 랜덤 분할
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=random_state
            )
        
        # 분할 정보 저장
        self.splits_info['train_val'] = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'val_ratio': val_ratio,
            'stratify': stratify,
            'random_state': random_state
        }
        
        # 서브셋 데이터셋 생성
        train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        
        # 데이터 로더 생성
        train_loader = self._create_loader(train_dataset, shuffle=True)
        val_loader = self._create_loader(val_dataset, shuffle=False)
        
        return train_loader, val_loader
    
    def create_kfold_splits(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        stratify: bool = True
    ) -> List[Tuple[DataLoader, DataLoader]]:
        """
        K-Fold 교차 검증용 데이터 분할
        
        Args:
            n_splits: 분할 수
            random_state: 랜덤 시드
            stratify: 계층화 분할 여부
            
        Returns:
            List[Tuple[DataLoader, DataLoader]]: K개의 (훈련, 검증) 로더 쌍
        """
        if self.dataset.annotations_path is None:
            raise ValueError("Annotations not available for k-fold split")
        
        # K-Fold 분할기 초기화
        if stratify and 'label' in self.dataset.data_info.columns:
            kfold = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
            labels = self.dataset.data_info['label'].values
            split_generator = kfold.split(range(len(self.dataset)), labels)
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
            split_generator = kfold.split(range(len(self.dataset)))
        
        # 각 fold에 대한 로더 생성
        fold_loaders = []
        fold_info = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(split_generator):
            # 서브셋 생성
            train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
            val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
            
            # 로더 생성
            train_loader = self._create_loader(train_dataset, shuffle=True)
            val_loader = self._create_loader(val_dataset, shuffle=False)
            
            fold_loaders.append((train_loader, val_loader))
            
            # 분할 정보 저장
            fold_info.append({
                'fold': fold_idx,
                'train_indices': train_indices.tolist(),
                'val_indices': val_indices.tolist()
            })
        
        # 전체 분할 정보 저장
        self.splits_info['kfold'] = {
            'n_splits': n_splits,
            'stratify': stratify,
            'random_state': random_state,
            'folds': fold_info
        }
        
        return fold_loaders
    
    def create_test_loader(self, shuffle: bool = False) -> DataLoader:
        """
        테스트 데이터 로더 생성
        
        Args:
            shuffle: 데이터 셔플 여부
            
        Returns:
            DataLoader: 테스트 데이터 로더
        """
        return self._create_loader(self.dataset, shuffle=shuffle)
    
    def _create_loader(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True
    ) -> DataLoader:
        """
        데이터 로더 생성 헬퍼 메서드
        
        Args:
            dataset: 데이터셋
            shuffle: 셔플 여부
            
        Returns:
            DataLoader: 생성된 데이터 로더
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle  # 훈련시에만 마지막 배치 드롭
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """
        클래스 불균형 해결을 위한 가중치 계산
        
        Returns:
            torch.Tensor: 클래스별 가중치
        """
        if self.dataset.annotations_path is None:
            raise ValueError("Annotations not available for class weights calculation")
        
        # 클래스별 빈도 계산
        class_counts = self.dataset.data_info['label'].value_counts().sort_index()
        total_samples = len(self.dataset)
        
        # 역빈도 가중치 계산
        weights = total_samples / (len(class_counts) * class_counts.values)
        
        return torch.FloatTensor(weights)
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        데이터셋 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 데이터셋 통계
        """
        stats = {
            'total_samples': len(self.dataset),
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'class_names': self.dataset.class_names
        }
        
        # 클래스 분포 추가 (라벨이 있는 경우)
        if self.dataset.annotations_path:
            distribution = self.dataset.get_class_distribution()
            stats['class_distribution'] = distribution
            stats['class_balance_ratio'] = self._calculate_balance_ratio(distribution)
        
        # 분할 정보 추가
        if self.splits_info:
            stats['splits_info'] = self.splits_info
        
        return stats
    
    def _calculate_balance_ratio(self, distribution: Dict[str, int]) -> float:
        """
        클래스 균형 비율 계산 (가장 적은 클래스 / 가장 많은 클래스)
        
        Args:
            distribution: 클래스별 분포
            
        Returns:
            float: 균형 비율 (0~1, 1에 가까울수록 균형)
        """
        if not distribution:
            return 1.0
        
        counts = list(distribution.values())
        return min(counts) / max(counts) if max(counts) > 0 else 1.0

"""
Base Dataset Class

모든 멤버가 상속받아 사용할 수 있는 기본 데이터셋 클래스
Single Responsibility Principle과 Open/Closed Principle 적용
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image


class BaseDocumentDataset(Dataset, ABC):
    """
    문서 분류를 위한 기본 데이터셋 클래스
    
    각 멤버는 이 클래스를 상속받아 자신만의 데이터 처리 로직을 구현할 수 있음
    """
    
    def __init__(
        self, 
        data_path: Path,
        annotations_path: Optional[Path] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None
    ):
        """
        Args:
            data_path: 데이터 폴더 경로
            annotations_path: 라벨 파일 경로 (train용)
            transform: 이미지 변환 함수
            target_transform: 타겟 변환 함수
        """
        self.data_path = Path(data_path)
        self.annotations_path = annotations_path
        self.transform = transform
        self.target_transform = target_transform
        
        # 데이터 로딩 및 검증
        self._validate_paths()
        self.data_info = self._load_data_info()
        self.class_names = self._get_class_names()
        
    def _validate_paths(self) -> None:
        """경로 유효성 검증"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
            
        if self.annotations_path and not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations path not found: {self.annotations_path}")
    
    @abstractmethod
    def _load_data_info(self) -> pd.DataFrame:
        """
        데이터 정보를 로딩하는 추상 메서드
        각 멤버가 자신의 방식으로 구현
        
        Returns:
            DataFrame: 파일명, 라벨 등의 정보를 담은 데이터프레임
        """
        pass
    
    @abstractmethod
    def _get_class_names(self) -> List[str]:
        """
        클래스 이름 목록을 반환하는 추상 메서드
        
        Returns:
            List[str]: 클래스 이름 목록
        """
        pass
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.data_info)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        데이터 아이템 반환
        
        Args:
            idx: 인덱스
            
        Returns:
            Tuple[Any, Any]: (이미지, 라벨) 튜플
        """
        # 데이터 정보 가져오기
        item_info = self.data_info.iloc[idx]
        
        # 이미지 로딩
        image = self._load_image(item_info)
        
        # 라벨 로딩 (train 데이터인 경우)
        label = self._load_label(item_info) if self.annotations_path else None
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform and label is not None:
            label = self.target_transform(label)
            
        return image, label
    
    def _load_image(self, item_info: pd.Series) -> np.ndarray:
        """
        이미지 로딩
        
        Args:
            item_info: 데이터 아이템 정보
            
        Returns:
            np.ndarray: 로딩된 이미지
        """
        image_path = self.data_path / item_info['filename']
        
        # OpenCV로 이미지 로딩 (BGR -> RGB 변환)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_label(self, item_info: pd.Series) -> int:
        """
        라벨 로딩
        
        Args:
            item_info: 데이터 아이템 정보
            
        Returns:
            int: 클래스 인덱스
        """
        if 'label' in item_info:
            return int(item_info['label'])
        else:
            raise KeyError("Label information not found in data")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        클래스별 데이터 분포 반환
        
        Returns:
            Dict[str, int]: 클래스별 데이터 개수
        """
        if self.annotations_path is None:
            return {}
            
        distribution = {}
        for class_name in self.class_names:
            count = (self.data_info['label'] == self.class_names.index(class_name)).sum()
            distribution[class_name] = count
            
        return distribution
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        특정 샘플의 상세 정보 반환
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            Dict[str, Any]: 샘플 정보
        """
        item_info = self.data_info.iloc[idx]
        
        info = {
            'index': idx,
            'filename': item_info['filename'],
            'class_name': self.class_names[item_info['label']] if 'label' in item_info else 'unknown',
            'image_path': str(self.data_path / item_info['filename'])
        }
        
        return info

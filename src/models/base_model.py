"""
Base Model Class

모든 문서 분류 모델의 기본 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class BaseDocumentClassifier(nn.Module, ABC):
    """
    문서 분류를 위한 기본 모델 클래스
    
    모든 멤버가 상속받아 사용할 수 있는 공통 인터페이스 제공
    """
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "base_model",
        dropout_rate: float = 0.1
    ):
        """
        Args:
            num_classes: 분류할 클래스 수
            model_name: 모델 이름
            dropout_rate: 드롭아웃 비율
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        # 모델 구성 요소들 (서브클래스에서 정의)
        self.backbone = None
        self.classifier = None
        
        # 모델 메타데이터
        self.model_info = {
            'name': model_name,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'created_at': None,
            'total_params': None,
            'trainable_params': None
        }
    
    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        """
        백본 네트워크 구성 (추상 메서드)
        각 멤버가 자신만의 아키텍처로 구현
        
        Returns:
            nn.Module: 백본 네트워크
        """
        pass
    
    @abstractmethod
    def _build_classifier(self, feature_dim: int) -> nn.Module:
        """
        분류기 헤드 구성 (추상 메서드)
        
        Args:
            feature_dim: 백본에서 나오는 특성 차원
            
        Returns:
            nn.Module: 분류기
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 클래스 로짓 [batch_size, num_classes]
        """
        # 백본을 통한 특성 추출
        features = self.extract_features(x)
        
        # 분류
        logits = self.classify(features)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        특성 추출 (백본 통과)
        
        Args:
            x: 입력 텐서
            
        Returns:
            torch.Tensor: 추출된 특성
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not initialized. Call _build_backbone() first.")
        
        return self.backbone(x)
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """
        분류 (분류기 헤드 통과)
        
        Args:
            features: 백본에서 추출된 특성
            
        Returns:
            torch.Tensor: 클래스 로짓
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not initialized. Call _build_classifier() first.")
        
        return self.classifier(features)
    
    def predict(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        예측 수행
        
        Args:
            x: 입력 텐서
            return_probs: 확률값 반환 여부
            
        Returns:
            torch.Tensor: 예측 결과 (클래스 인덱스 또는 확률)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            if return_probs:
                return F.softmax(logits, dim=1)
            else:
                return torch.argmax(logits, dim=1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 메타데이터
        """
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.model_info.update({
            'total_params': total_params,
            'trainable_params': trainable_params
        })
        
        return self.model_info.copy()
    
    def freeze_backbone(self) -> None:
        """백본 네트워크 동결 (전이학습용)"""
        if self.backbone is None:
            raise RuntimeError("Backbone not initialized.")
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        print("🧊 백본 네트워크가 동결되었습니다.")
    
    def unfreeze_backbone(self) -> None:
        """백본 네트워크 동결 해제"""
        if self.backbone is None:
            raise RuntimeError("Backbone not initialized.")
        
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        print("🔥 백본 네트워크 동결이 해제되었습니다.")
    
    def get_layer_names(self) -> List[str]:
        """모델의 모든 레이어 이름 반환"""
        return [name for name, _ in self.named_modules()]
    
    def get_feature_map_size(self, input_size: Tuple[int, int, int]) -> Tuple[int, ...]:
        """
        주어진 입력 크기에 대한 특성 맵 크기 계산
        
        Args:
            input_size: 입력 크기 (C, H, W)
            
        Returns:
            Tuple[int, ...]: 특성 맵 크기
        """
        self.eval()
        with torch.no_grad():
            x = torch.randn(1, *input_size)
            features = self.extract_features(x)
            return features.shape[1:]  # 배치 차원 제외
    
    def save_model(self, save_path: Path, save_config: bool = True) -> None:
        """
        모델 저장
        
        Args:
            save_path: 저장 경로
            save_config: 모델 설정도 함께 저장할지 여부
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 모델 상태 저장
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'model_class': self.__class__.__name__
        }
        
        torch.save(checkpoint, save_path)
        
        # 설정 파일 별도 저장
        if save_config:
            config_path = save_path.with_suffix('.json')
            import json
            with open(config_path, 'w') as f:
                json.dump(self.model_info, f, indent=2)
        
        print(f"💾 모델 저장 완료: {save_path}")
    
    @classmethod
    def load_model(cls, load_path: Path, **kwargs) -> 'BaseDocumentClassifier':
        """
        모델 로드
        
        Args:
            load_path: 로드 경로
            **kwargs: 모델 초기화 파라미터
            
        Returns:
            BaseDocumentClassifier: 로드된 모델
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        
        # 모델 정보에서 파라미터 추출
        model_info = checkpoint['model_info']
        
        # 모델 인스턴스 생성
        model = cls(
            num_classes=model_info['num_classes'],
            model_name=model_info['name'],
            dropout_rate=model_info.get('dropout_rate', 0.1),
            **kwargs
        )
        
        # 상태 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.model_info = model_info
        
        print(f"📂 모델 로드 완료: {load_path}")
        return model
    
    def summary(self, input_size: Tuple[int, int, int] = (3, 224, 224)) -> str:
        """
        모델 요약 정보 생성
        
        Args:
            input_size: 입력 크기
            
        Returns:
            str: 모델 요약
        """
        lines = []
        lines.append(f"모델: {self.model_name}")
        lines.append(f"클래스 수: {self.num_classes}")
        lines.append(f"드롭아웃 비율: {self.dropout_rate}")
        lines.append("-" * 50)
        
        # 파라미터 정보
        info = self.get_model_info()
        lines.append(f"총 파라미터 수: {info['total_params']:,}")
        lines.append(f"학습 가능한 파라미터 수: {info['trainable_params']:,}")
        
        # 특성 맵 크기
        try:
            feature_size = self.get_feature_map_size(input_size)
            lines.append(f"특성 맵 크기: {feature_size}")
        except:
            lines.append("특성 맵 크기: 계산 불가")
        
        return "\n".join(lines)

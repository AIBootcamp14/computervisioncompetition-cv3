"""
🧠 Improved Model Factory
클린 코드와 클린 아키텍처를 적용한 개선된 모델 팩토리

Features:
- 더 강력한 모델 아키텍처 지원
- 고급 손실 함수 (Focal Loss, Label Smoothing 등)
- 어텐션 메커니즘 통합
- 앙상블을 위한 다중 모델 지원
- 메모리 효율적인 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict, Any, Union, List
import math
import numpy as np
from abc import ABC, abstractmethod


class BaseDocumentClassifier(nn.Module, ABC):
    """
    문서 분류 모델의 추상 기반 클래스
    클린 아키텍처 원칙: 의존성 역전
    """
    
    def __init__(self, num_classes: int = 17):
        super().__init__()
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파 추상 메서드"""
        pass
    
    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """특징 벡터 추출 추상 메서드"""
        pass


class AttentionModule(nn.Module):
    """
    채널 어텐션과 공간 어텐션을 결합한 모듈
    성능 향상을 위한 핵심 컴포넌트
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        # 채널 어텐션 (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 공간 어텐션 (Spatial Attention)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 채널 어텐션 적용
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        
        # 공간 어텐션을 위한 통계 계산
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        
        # 공간 어텐션 적용
        sa_weight = self.spatial_attention(spatial_input)
        x = x * sa_weight
        
        return x


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling
    Global Average Pooling보다 우수한 성능
    """
    
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 
            (x.size(-2), x.size(-1))
        ).pow(1. / self.p)


class ImprovedDocumentClassifier(BaseDocumentClassifier):
    """
    개선된 문서 분류 모델
    더 강력한 아키텍처와 어텐션 메커니즘 통합
    """
    
    def __init__(
        self,
        architecture: str = "efficientnet_b3",
        num_classes: int = 17,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        use_gem_pooling: bool = True,
        use_attention: bool = True,
        use_mixup: bool = True,
        image_size: int = 512
    ):
        """
        Args:
            architecture: 백본 아키텍처 (더 강력한 모델들)
            num_classes: 클래스 수
            pretrained: 사전 훈련된 가중치 사용
            dropout_rate: 드롭아웃 비율
            use_gem_pooling: GeM 풀링 사용 여부
            use_attention: 어텐션 메커니즘 사용 여부
            use_mixup: MixUp 지원 여부
            image_size: 입력 이미지 크기
        """
        super().__init__(num_classes)
        
        self.architecture = architecture
        self.use_attention = use_attention
        self.use_mixup = use_mixup
        self.image_size = image_size
        
        # 백본 모델 생성
        self.backbone = self._create_backbone(architecture, pretrained)
        
        # 특징 차원 동적 계산
        self.feature_dim = self._get_feature_dim()
        
        # 어텐션 모듈 (선택적)
        if use_attention:
            self.attention = AttentionModule(self.feature_dim)
        
        # 풀링 레이어
        if use_gem_pooling:
            self.global_pool = GeMPooling()
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 향상된 분류 헤드
        self.classifier = self._create_classifier_head(dropout_rate)
        
        # 가중치 초기화
        self._init_weights()
    
    def _create_backbone(self, architecture: str, pretrained: bool) -> nn.Module:
        """
        백본 모델 생성
        더 강력한 아키텍처들 지원
        """
        try:
            backbone = timm.create_model(
                architecture,
                pretrained=pretrained,
                num_classes=0,  # 분류 헤드 제거
                global_pool='',  # 풀링 제거
                drop_rate=0.0   # 백본의 드롭아웃 제거 (분류 헤드에서 관리)
            )
            
            print(f"✅ 백본 모델 생성 성공: {architecture}")
            return backbone
            
        except Exception as e:
            print(f"⚠️ {architecture} 생성 실패: {e}")
            print("   기본 모델(efficientnet_b0)로 대체")
            
            return timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
                drop_rate=0.0
            )
    
    def _get_feature_dim(self) -> int:
        """특징 차원 동적 계산"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.image_size, self.image_size)
            features = self.backbone(dummy_input)
            return features.shape[1]
    
    def _create_classifier_head(self, dropout_rate: float) -> nn.Module:
        """
        향상된 분류 헤드 생성
        더 깊은 구조로 성능 향상
        """
        # 중간 차원 계산 (특징 차원에 따라 적응적 조정)
        if self.feature_dim >= 2048:
            hidden_dim = 1024
        elif self.feature_dim >= 1280:
            hidden_dim = 640
        else:
            hidden_dim = 512
        
        return nn.Sequential(
            # 첫 번째 레이어
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # 두 번째 레이어
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # 출력 레이어
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, self.num_classes)
        )
    
    def _init_weights(self):
        """가중치 초기화 (Xavier/He 초기화)"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                # He 초기화 (ReLU와 함께 사용)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # 백본을 통한 특징 추출
        features = self.backbone(x)
        
        # 어텐션 적용 (선택적)
        if self.use_attention:
            features = self.attention(features)
        
        # 글로벌 풀링
        features = self.global_pool(features)
        features = features.flatten(1)
        
        # 분류
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """특징 벡터 추출 (앙상블용)"""
        with torch.no_grad():
            features = self.backbone(x)
            
            if self.use_attention:
                features = self.attention(features)
            
            features = self.global_pool(features)
            features = features.flatten(1)
            
        return features


class AdvancedFocalLoss(nn.Module):
    """
    개선된 Focal Loss
    클래스 불균형과 어려운 샘플에 더 효과적
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 3.0,  # 더 강한 focusing
        reduction: str = 'mean',
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing 적용
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            smoothed_targets = targets * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal weight 계산
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weight 적용
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha.gather(0, targets)
            focal_loss = alpha_weight * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MixUpCutMixLoss(nn.Module):
    """
    MixUp과 CutMix를 위한 손실 함수
    데이터 증강과 함께 사용
    """
    
    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.criterion = criterion
    
    def forward(
        self,
        outputs: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        MixUp/CutMix 손실 계산
        두 타겟에 대한 가중 평균
        """
        return lam * self.criterion(outputs, targets_a) + \
               (1 - lam) * self.criterion(outputs, targets_b)


class ImprovedModelFactory:
    """
    개선된 모델 팩토리
    더 강력한 모델들과 고급 기법 지원
    """
    
    # 지원하는 강력한 아키텍처들
    POWERFUL_ARCHITECTURES = [
        # EfficientNet 계열 (균형잡힌 성능)
        'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
        'efficientnetv2_m', 'efficientnetv2_l',
        
        # ConvNeXt 계열 (최신 CNN)
        'convnext_tiny', 'convnext_small', 'convnext_base',
        
        # Swin Transformer 계열 (Vision Transformer)
        'swin_tiny_patch4_window7_224', 
        'swin_small_patch4_window7_224',
        'swin_base_patch4_window7_224',
        
        # RegNet 계열 (효율적인 CNN)
        'regnetx_032', 'regnetx_040', 'regnetx_064',
        'regnety_032', 'regnety_040', 'regnety_064',
        
        # ResNet 계열 (안정적인 기본)
        'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
    ]
    
    @staticmethod
    def create_model(
        architecture: str,
        num_classes: int = 17,
        pretrained: bool = True,
        **kwargs
    ) -> ImprovedDocumentClassifier:
        """
        개선된 모델 생성
        더 강력한 아키텍처 우선 사용
        """
        # 아키텍처 검증 및 추천
        if architecture not in ImprovedModelFactory.POWERFUL_ARCHITECTURES:
            print(f"⚠️ {architecture}는 최적화되지 않은 아키텍처입니다.")
            
            # 성능 기반 추천
            if 'efficientnet' in architecture.lower():
                recommended = 'efficientnet_b3'
            elif 'convnext' in architecture.lower():
                recommended = 'convnext_base'
            elif 'swin' in architecture.lower():
                recommended = 'swin_base_patch4_window7_224'
            else:
                recommended = 'efficientnet_b3'  # 기본 추천
            
            print(f"   추천 아키텍처: {recommended}")
            architecture = recommended
        
        # 모델 생성
        model = ImprovedDocumentClassifier(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
        
        print(f"🧠 개선된 모델 생성 완료:")
        print(f"   아키텍처: {architecture}")
        print(f"   파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   특징 차원: {model.feature_dim}")
        
        return model
    
    @staticmethod
    def create_ensemble_models(
        architectures: List[str],
        num_classes: int = 17,
        pretrained: bool = True,
        **kwargs
    ) -> List[ImprovedDocumentClassifier]:
        """
        앙상블용 다중 모델 생성
        다양한 아키텍처 조합
        """
        models = []
        
        print(f"🎭 앙상블 모델 생성 중... ({len(architectures)}개)")
        
        for i, arch in enumerate(architectures):
            print(f"   모델 {i+1}/{len(architectures)}: {arch}")
            
            model = ImprovedModelFactory.create_model(
                architecture=arch,
                num_classes=num_classes,
                pretrained=pretrained,
                **kwargs
            )
            models.append(model)
        
        print(f"✅ 앙상블 모델 생성 완료!")
        return models
    
    @staticmethod
    def create_loss_function(
        loss_type: str = "advanced_focal",
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> nn.Module:
        """
        개선된 손실 함수 생성
        클래스 불균형에 더 효과적
        """
        if loss_type == "advanced_focal":
            return AdvancedFocalLoss(
                alpha=class_weights,
                gamma=kwargs.get('focal_gamma', 3.0),  # 더 강한 focusing
                label_smoothing=kwargs.get('label_smoothing', 0.1)
            )
        
        elif loss_type == "focal":
            # 기존 Focal Loss와 호환성 - 간단한 Focal Loss 구현
            class FocalLoss(nn.Module):
                def __init__(self, alpha=None, gamma=2.0):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                
                def forward(self, inputs, targets):
                    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_loss = (1 - pt) ** self.gamma * ce_loss
                    
                    if self.alpha is not None:
                        if self.alpha.device != inputs.device:
                            self.alpha = self.alpha.to(inputs.device)
                        at = self.alpha.gather(0, targets)
                        focal_loss = at * focal_loss
                    
                    return focal_loss.mean()
            
            return FocalLoss(
                alpha=class_weights,
                gamma=kwargs.get('focal_gamma', 2.0)
            )
        
        elif loss_type == "label_smoothing":
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=kwargs.get('label_smoothing', 0.1)
            )
        
        elif loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(weight=class_weights)
        
        else:
            raise ValueError(f"지원하지 않는 손실 함수: {loss_type}")
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        개선된 옵티마이저 생성
        더 나은 기본값 사용
        """
        # 파라미터 그룹 분리 (백본 vs 분류 헤드)
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        if optimizer_type == "adamw":
            return torch.optim.AdamW([
                {'params': backbone_params, 'lr': learning_rate * 0.1},  # 백본은 낮은 학습률
                {'params': classifier_params, 'lr': learning_rate}        # 분류 헤드는 높은 학습률
            ], weight_decay=weight_decay)
        
        elif optimizer_type == "adam":
            return torch.optim.Adam([
                {'params': backbone_params, 'lr': learning_rate * 0.1},
                {'params': classifier_params, 'lr': learning_rate}
            ], weight_decay=weight_decay)
        
        elif optimizer_type == "sgd":
            return torch.optim.SGD([
                {'params': backbone_params, 'lr': learning_rate * 0.1},
                {'params': classifier_params, 'lr': learning_rate}
            ], weight_decay=weight_decay, momentum=0.9, nesterov=True)
        
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_type}")
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine_warm_restarts",
        epochs: int = 30,
        **kwargs
    ):
        """
        개선된 스케줄러 생성
        더 효과적인 학습률 스케줄링
        """
        if scheduler_type == "cosine_warm_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get('T_0', 10),
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        
        elif scheduler_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                verbose=True
            )
        
        else:
            return None


# 사용 예시 및 테스트
if __name__ == "__main__":
    print("🧠 개선된 모델 팩토리 테스트")
    
    # 단일 강력한 모델 생성
    model = ImprovedModelFactory.create_model(
        architecture="efficientnet_b3",
        num_classes=17,
        pretrained=True,
        dropout_rate=0.4,
        use_attention=True
    )
    
    # 앙상블 모델들 생성
    ensemble_architectures = [
        "efficientnet_b3",
        "efficientnet_b4", 
        "convnext_base"
    ]
    
    ensemble_models = ImprovedModelFactory.create_ensemble_models(
        ensemble_architectures,
        num_classes=17
    )
    
    # 개선된 손실 함수 생성
    criterion = ImprovedModelFactory.create_loss_function(
        loss_type="advanced_focal",
        focal_gamma=3.0,
        label_smoothing=0.1
    )
    
    # 개선된 옵티마이저 생성
    optimizer = ImprovedModelFactory.create_optimizer(
        model,
        optimizer_type="adamw",
        learning_rate=1e-4
    )
    
    print(f"✅ 개선된 모델 시스템 테스트 완료!")
    print(f"   단일 모델: {model.architecture}")
    print(f"   앙상블 모델 수: {len(ensemble_models)}개")
    print(f"   손실 함수: {type(criterion).__name__}")
    print(f"   옵티마이저: {type(optimizer).__name__}")
    
    # 테스트 입력
    test_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(test_input)
        features = model.get_features(test_input)
        print(f"   출력 형태: {output.shape}")
        print(f"   특징 형태: {features.shape}")

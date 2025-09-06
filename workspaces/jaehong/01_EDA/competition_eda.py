"""
Competition EDA - 문서 분류 대회용 탐색적 데이터 분석

Clean Code & Clean Architecture 적용:
- Single Responsibility Principle: 각 분석 기능별로 클래스/메서드 분리
- Open/Closed Principle: 새로운 분석 방법 추가 시 기존 코드 수정 없이 확장 가능
- Dependency Inversion: 추상화된 인터페이스에 의존
- Interface Segregation: 특정 용도에 맞는 작은 인터페이스들로 분리

대회 전략 중심의 EDA:
1. Train/Test 데이터 차이 분석 (핵심!)
2. 이질적 이미지 탐지 (차량 대시보드, 번호판)
3. 회전/블러/노이즈 분석
4. 클래스 불균형 및 Augmentation 전략
5. 최적 Resize 전략 도출
"""

import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Protocol
import warnings

# Data & Math
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Image Processing
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import albumentations as A

# Visualization
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Utils
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
import matplotlib.font_manager as fm

def setup_korean_font():
    """한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows 시스템
        font_candidates = ['Malgun Gothic', 'Arial Unicode MS', 'SimHei']
    elif system == 'Darwin':  # macOS
        font_candidates = ['AppleGothic', 'Arial Unicode MS']
    else:  # Linux
        font_candidates = ['Noto Sans CJK KR', 'DejaVu Sans', 'Liberation Sans']
    
    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = 'DejaVu Sans'  # 기본값
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['axes.unicode_minus'] = False
    print(f"🎨 폰트 설정: {selected_font}")

# 한글 폰트 설정 실행
setup_korean_font()

# 시드 고정 (재현가능한 분석)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class ImageMetrics:
    """이미지 품질 메트릭을 저장하는 데이터 클래스"""
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    rotation_angle: float
    aspect_ratio: float
    file_size_kb: float
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, float]:
        """딕셔너리로 변환"""
        return {
            'brightness': self.brightness,
            'contrast': self.contrast,
            'sharpness': self.sharpness,
            'noise_level': self.noise_level,
            'rotation_angle': self.rotation_angle,
            'aspect_ratio': self.aspect_ratio,
            'file_size_kb': self.file_size_kb,
            'width': self.width,
            'height': self.height
        }


class ImageAnalyzer(ABC):
    """이미지 분석을 위한 추상 기본 클래스"""
    
    @abstractmethod
    def analyze_image(self, image_path: Path) -> Optional[ImageMetrics]:
        """이미지를 분석하여 메트릭을 반환"""
        pass


class DocumentImageAnalyzer(ImageAnalyzer):
    """
    문서 이미지 전용 분석기
    
    책임:
    - 문서 이미지의 품질 메트릭 계산
    - 회전 각도 탐지 (Hough Transform 사용)
    - 노이즈 레벨 측정
    """
    
    def analyze_image(self, image_path: Path) -> Optional[ImageMetrics]:
        """
        문서 이미지 분석
        
        Args:
            image_path: 분석할 이미지 경로
            
        Returns:
            ImageMetrics 객체 또는 None (분석 실패 시)
        """
        try:
            # 이미지 로드
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                return None
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 기본 정보
            height, width = img_gray.shape
            file_size_kb = image_path.stat().st_size / 1024
            aspect_ratio = width / height
            
            # 품질 메트릭 계산
            brightness = self._calculate_brightness(img_gray)
            contrast = self._calculate_contrast(img_gray)
            sharpness = self._calculate_sharpness(img_gray)
            noise_level = self._calculate_noise_level(img_gray)
            rotation_angle = self._detect_rotation_angle(img_gray)
            
            return ImageMetrics(
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                noise_level=noise_level,
                rotation_angle=rotation_angle,
                aspect_ratio=aspect_ratio,
                file_size_kb=file_size_kb,
                width=width,
                height=height
            )
            
        except Exception as e:
            print(f"이미지 분석 실패 {image_path}: {str(e)}")
            return None
    
    def _calculate_brightness(self, img_gray: np.ndarray) -> float:
        """밝기 계산"""
        return np.mean(img_gray)
    
    def _calculate_contrast(self, img_gray: np.ndarray) -> float:
        """대비 계산 (표준편차 사용)"""
        return np.std(img_gray)
    
    def _calculate_sharpness(self, img_gray: np.ndarray) -> float:
        """선명도 계산 (Laplacian 분산 사용)"""
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        return laplacian.var()
    
    def _calculate_noise_level(self, img_gray: np.ndarray) -> float:
        """
        노이즈 레벨 계산
        Gaussian blur 후 원본과의 차이로 노이즈 추정
        """
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        noise = img_gray.astype(np.float32) - blurred.astype(np.float32)
        return np.std(noise)
    
    def _detect_rotation_angle(self, img_gray: np.ndarray) -> float:
        """
        Hough Transform을 사용한 회전 각도 탐지
        
        Returns:
            회전 각도 (도 단위, -45 ~ 45 범위)
        """
        try:
            # 엣지 검출
            edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
            
            # Hough Line Transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return 0.0
            
            # 각도 계산
            angles = []
            for rho, theta in lines[:20]:  # 상위 20개 라인만 사용
                angle = np.degrees(theta) - 90
                if -45 <= angle <= 45:
                    angles.append(angle)
            
            if angles:
                return np.median(angles)
            else:
                return 0.0
                
        except Exception:
            return 0.0


class DatasetAnalyzer:
    """
    데이터셋 전체 분석을 담당하는 클래스
    
    책임:
    - Train/Test 데이터 로드 및 관리
    - 클래스 분포 분석
    - 이미지 메트릭 수집 및 비교
    """
    
    def __init__(self, data_root: Path):
        """
        초기화
        
        Args:
            data_root: 데이터 루트 디렉토리
        """
        self.data_root = Path(data_root)
        self.train_dir = self.data_root / "train"
        self.test_dir = self.data_root / "test"
        self.train_csv = self.data_root / "train.csv"
        self.meta_csv = self.data_root / "meta.csv"
        self.sample_submission_csv = self.data_root / "sample_submission.csv"
        
        # 데이터 로드
        self.train_df = pd.read_csv(self.train_csv)
        self.meta_df = pd.read_csv(self.meta_csv)
        self.sample_submission = pd.read_csv(self.sample_submission_csv)
        
        # 클래스 매핑
        self.class_mapping = dict(zip(self.meta_df['target'], self.meta_df['class_name']))
        
        # 이미지 분석기
        self.image_analyzer = DocumentImageAnalyzer()
        
        print(f"📊 데이터셋 로드 완료:")
        print(f"  • 학습 데이터: {len(self.train_df):,}개")
        print(f"  • 테스트 데이터: {len(self.sample_submission):,}개")
        print(f"  • 클래스 수: {len(self.class_mapping)}개")
    
    def get_image_paths(self, dataset_type: str) -> List[Path]:
        """
        이미지 경로 리스트 반환
        
        Args:
            dataset_type: 'train' 또는 'test'
        """
        if dataset_type == 'train':
            return [self.train_dir / img_id for img_id in self.train_df['ID']]
        elif dataset_type == 'test':
            return [self.test_dir / img_id for img_id in self.sample_submission['ID']]
        else:
            raise ValueError("dataset_type은 'train' 또는 'test'여야 합니다.")
    
    def analyze_images_batch(self, image_paths: List[Path], max_workers: int = 4) -> List[ImageMetrics]:
        """
        이미지들을 병렬로 분석
        
        Args:
            image_paths: 분석할 이미지 경로 리스트
            max_workers: 병렬 처리 워커 수
            
        Returns:
            ImageMetrics 리스트
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_path = {
                executor.submit(self.image_analyzer.analyze_image, path): path 
                for path in image_paths
            }
            
            # 결과 수집
            for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="이미지 분석 중"):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results


class CompetitionEDA:
    """
    대회용 EDA 메인 클래스
    
    책임:
    - 전체 EDA 프로세스 조율
    - 분석 결과 시각화
    - 대회 전략 제안
    """
    
    def __init__(self, data_root: Path):
        """초기화"""
        self.data_root = Path(data_root)
        self.dataset_analyzer = DatasetAnalyzer(data_root)
        
        # 출력 디렉토리 설정
        self.output_dir = Path(__file__).parent / "eda_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print(f"🏆 Competition EDA 초기화 완료")
        print(f"📁 결과 저장 경로: {self.output_dir}")
    
    def run_complete_analysis(self):
        """
        대회 전략에 맞는 포괄적인 EDA 실행
        
        분석 순서:
        1. 기본 데이터 탐색
        2. 클래스 불균형 분석 & 전략 제안
        3. Train/Test 이미지 차이 분석 (핵심!)
        4. 이질적 이미지 탐지
        5. 회전/블러/노이즈 분석
        6. 이미지 크기/종횡비 분석
        7. 최종 대회 전략 제안
        """
        print("=== 🚀 Competition EDA 시작 ===\n")
        
        try:
            # 1. 기본 데이터 탐색
            self._basic_data_exploration()
            
            # 2. 클래스 불균형 분석
            self._analyze_class_imbalance()
            
            # 3. Train/Test 차이 분석 (대회 핵심!)
            self._analyze_train_test_differences()
            
            # 4. 이질적 이미지 탐지
            self._detect_anomalous_images()
            
            # 5. 이미지 품질 분석
            self._analyze_image_quality()
            
            # 6. 크기/종횡비 분석
            self._analyze_image_dimensions()
            
            # 7. 최종 전략 제안
            self._generate_competition_strategy()
            
            # 8. 고급 전략 추가 (Multi-Modal, Advanced Augmentation, Pseudo Labeling)
            self._advanced_multimodal_analysis()
            self._advanced_augmentation_strategies()
            self._pseudo_labeling_strategy()
            
            print("\n=== ✅ EDA 분석 완료 ===")
            print(f"📁 모든 결과가 {self.output_dir}에 저장되었습니다.")
            
        except Exception as e:
            print(f"❌ EDA 분석 중 오류 발생: {str(e)}")
            raise
    
    def _basic_data_exploration(self):
        """기본 데이터 탐색"""
        print("=== 📊 1. 기본 데이터 탐색 ===")
        
        train_df = self.dataset_analyzer.train_df
        meta_df = self.dataset_analyzer.meta_df
        class_mapping = self.dataset_analyzer.class_mapping
        
        # 데이터셋 기본 정보
        print(f"📈 데이터셋 크기:")
        print(f"  • 학습 데이터: {len(train_df):,}개")
        print(f"  • 테스트 데이터: {len(self.dataset_analyzer.sample_submission):,}개")
        print(f"  • 총 클래스 수: {len(class_mapping)}개")
        print(f"  • Train/Test 비율: 1:{len(self.dataset_analyzer.sample_submission)/len(train_df):.1f}")
        
        # 클래스 정보 출력
        print(f"\n📋 클래스 목록:")
        for target, class_name in class_mapping.items():
            print(f"  {target:2d}: {class_name}")
        
        # 파일 존재 확인
        train_files = list(self.dataset_analyzer.train_dir.glob("*.jpg"))
        test_files = list(self.dataset_analyzer.test_dir.glob("*.jpg"))
        
        print(f"\n📁 파일 정보:")
        print(f"  • 학습 이미지 파일: {len(train_files):,}개")
        print(f"  • 테스트 이미지 파일: {len(test_files):,}개")
        
        # 데이터 일관성 확인
        train_ids_csv = set(train_df['ID'].str.replace('.jpg', ''))
        train_ids_files = set([f.stem for f in train_files])
        missing_files = train_ids_csv - train_ids_files
        
        print(f"\n🔍 데이터 일관성:")
        print(f"  • CSV에 있지만 파일이 없는 것: {len(missing_files)}개")
        if missing_files:
            print(f"  • 누락 파일 예시: {list(missing_files)[:3]}")
        
        print()
    
    def _analyze_class_imbalance(self):
        """클래스 불균형 분석 및 전략 제안"""
        print("=== ⚖️ 2. 클래스 불균형 분석 ===")
        
        train_df = self.dataset_analyzer.train_df
        class_mapping = self.dataset_analyzer.class_mapping
        
        # 클래스 분포 계산
        class_counts = train_df['target'].value_counts().sort_index()
        total_samples = len(train_df)
        
        print("📊 클래스별 분포:")
        imbalance_data = []
        for target in sorted(class_mapping.keys()):
            count = class_counts.get(target, 0)
            percentage = (count / total_samples) * 100
            class_name = class_mapping[target]
            print(f"  {target:2d}: {count:3d}개 ({percentage:5.1f}%) - {class_name[:40]}")
            imbalance_data.append({
                'target': target,
                'count': count,
                'percentage': percentage,
                'class_name': class_name
            })
        
        # 불균형 메트릭 계산
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        std_dev = class_counts.std()
        cv = std_dev / class_counts.mean()  # 변동계수
        
        print(f"\n📈 불균형 메트릭:")
        print(f"  • 최대 클래스: {max_count}개")
        print(f"  • 최소 클래스: {min_count}개") 
        print(f"  • 불균형 비율: {imbalance_ratio:.2f}:1")
        print(f"  • 표준편차: {std_dev:.1f}")
        print(f"  • 변동계수: {cv:.3f}")
        
        # 불균형 심각도 평가 및 전략 제안
        if imbalance_ratio > 10:
            severity = "🔴 심각"
            strategy = "SMOTE + Focal Loss + Class Weights"
        elif imbalance_ratio > 5:
            severity = "🟡 보통" 
            strategy = "Focal Loss + Stratified Sampling"
        else:
            severity = "🟢 경미"
            strategy = "Stratified K-Fold로 충분"
        
        print(f"  • 불균형 심각도: {severity}")
        print(f"  • 권장 전략: {strategy}")
        
        # 시각화
        self._plot_class_distribution(imbalance_data)
        
        # Stratified sampling 가중치 계산
        self._calculate_class_weights(class_counts)
        
        print()
    
    def _plot_class_distribution(self, imbalance_data: List[Dict]):
        """클래스 분포 시각화 (한글 지원)"""
        # 폰트 재설정 (차트별로 확실히 적용)
        setup_korean_font()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 데이터 준비
        targets = [d['target'] for d in imbalance_data]
        counts = [d['count'] for d in imbalance_data]
        class_names = [d['class_name'] for d in imbalance_data]
        
        # 막대 그래프
        colors = plt.cm.viridis(np.linspace(0, 1, len(targets)))
        bars = ax1.bar(targets, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Class ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Image Count', fontsize=12, fontweight='bold')
        ax1.set_title('Class Distribution', fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(targets)
        
        # 막대 위에 수치 표시
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 원형 그래프 - 클래스명 포함
        # 클래스명을 축약해서 표시 (너무 길면 잘림)
        short_labels = []
        for i, (target, name) in enumerate(zip(targets, class_names)):
            if len(name) > 20:
                short_name = name[:17] + "..."
            else:
                short_name = name
            short_labels.append(f"{target}: {short_name}")
        
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(targets)))
        wedges, texts, autotexts = ax2.pie(counts, labels=None, autopct='%1.1f%%', 
                                          colors=colors_pie, startangle=90,
                                          textprops={'fontsize': 9})
        ax2.set_title('Class Ratio Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # 범례 추가 (원형 그래프 옆에)
        ax2.legend(wedges, short_labels, title="Classes", loc="center left", 
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8, title_fontsize=10)
        
        plt.tight_layout()
        
        # 한글 깨짐 방지를 위해 폰트 명시적으로 지정하여 저장
        plt.savefig(self.output_dir / 'class_distribution.png', 
                   dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()  # 메모리 절약을 위해 figure 닫기
        
        print(f"📊 클래스 분포 차트 저장: {self.output_dir / 'class_distribution.png'}")
        
        # 클래스별 상세 정보를 텍스트로도 저장
        self._save_class_info_text(imbalance_data)
    
    def _save_class_info_text(self, imbalance_data: List[Dict]):
        """클래스 정보를 텍스트 파일로 저장 (한글 깨짐 방지)"""
        try:
            with open(self.output_dir / 'class_distribution.txt', 'w', encoding='utf-8') as f:
                f.write("📊 클래스별 분포 상세 정보\n")
                f.write("=" * 50 + "\n\n")
                
                total_samples = sum(d['count'] for d in imbalance_data)
                f.write(f"총 샘플 수: {total_samples:,}개\n")
                f.write(f"클래스 수: {len(imbalance_data)}개\n\n")
                
                f.write("클래스별 세부 정보:\n")
                f.write("-" * 50 + "\n")
                
                for d in imbalance_data:
                    f.write(f"클래스 {d['target']:2d}: {d['class_name']}\n")
                    f.write(f"  샘플 수: {d['count']:3d}개\n")
                    f.write(f"  비율: {d['percentage']:5.1f}%\n")
                    f.write("\n")
            
            print(f"📄 클래스 정보 텍스트 저장: {self.output_dir / 'class_distribution.txt'}")
            
        except Exception as e:
            print(f"⚠️ 텍스트 파일 저장 실패: {str(e)}")
    
    def _calculate_class_weights(self, class_counts: pd.Series):
        """클래스 가중치 계산"""
        print(f"\n💡 클래스 가중치 제안:")
        
        # Balanced 가중치 계산
        n_samples = class_counts.sum()
        n_classes = len(class_counts)
        balanced_weights = n_samples / (n_classes * class_counts)
        
        print(f"  📋 Balanced Class Weights:")
        for target, weight in balanced_weights.items():
            class_name = self.dataset_analyzer.class_mapping[target]
            print(f"    {target:2d}: {weight:.3f} - {class_name[:30]}")
        
        # 가중치를 딕셔너리로 저장
        weights_dict = balanced_weights.to_dict()
        with open(self.output_dir / 'class_weights.json', 'w') as f:
            json.dump(weights_dict, f, indent=2)
        
        print(f"  💾 가중치 저장: {self.output_dir / 'class_weights.json'}")
    
    def _analyze_train_test_differences(self):
        """Train/Test 데이터 차이 분석 (대회 핵심!)"""
        print("=== 🔍 3. Train/Test 차이 분석 (대회 핵심!) ===")
        
        # 전체 데이터 분석 (17개 클래스 완전 분석을 위해)
        train_paths = self.dataset_analyzer.get_image_paths('train')
        test_paths = self.dataset_analyzer.get_image_paths('test')
        
        print(f"📊 전체 데이터 분석 (Train: {len(train_paths)}개, Test: {len(test_paths)}개)")
        print(f"   클래스별 정확한 특성 파악을 위해 전체 데이터를 분석합니다.")
        
        # 전체 데이터 사용
        train_sample = train_paths
        test_sample = test_paths
        
        # 이미지 분석 (병렬 처리 최적화)
        print("🔄 Train 이미지 전체 분석 중... (시간이 소요될 수 있습니다)")
        train_metrics = self.dataset_analyzer.analyze_images_batch(train_sample, max_workers=8)
        
        print("🔄 Test 이미지 전체 분석 중... (시간이 소요될 수 있습니다)")
        test_metrics = self.dataset_analyzer.analyze_images_batch(test_sample, max_workers=8)
        
        # 결과 비교
        self._compare_train_test_metrics(train_metrics, test_metrics)
        
        print()
    
    def _compare_train_test_metrics(self, train_metrics: List[ImageMetrics], test_metrics: List[ImageMetrics]):
        """Train/Test 메트릭 비교"""
        
        # 데이터프레임 생성
        train_df = pd.DataFrame([m.to_dict() for m in train_metrics])
        test_df = pd.DataFrame([m.to_dict() for m in test_metrics])
        
        train_df['dataset'] = 'Train'
        test_df['dataset'] = 'Test'
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # 통계 비교
        print("📈 Train vs Test 품질 메트릭 비교:")
        
        metrics_to_compare = ['brightness', 'contrast', 'sharpness', 'noise_level', 'rotation_angle']
        
        for metric in metrics_to_compare:
            train_values = train_df[metric]
            test_values = test_df[metric]
            
            # t-test 수행
            t_stat, p_value = stats.ttest_ind(train_values, test_values)
            
            print(f"\n  📊 {metric.upper()}:")
            print(f"    Train: {train_values.mean():.2f} ± {train_values.std():.2f}")
            print(f"    Test:  {test_values.mean():.2f} ± {test_values.std():.2f}")
            print(f"    차이:   {test_values.mean() - train_values.mean():+.2f}")
            print(f"    p-value: {p_value:.4f} {'(유의함)' if p_value < 0.05 else '(유의하지 않음)'}")
        
        # 시각화
        self._plot_train_test_comparison(combined_df)
        
        # 대회 전략 인사이트
        self._generate_train_test_insights(train_df, test_df)
    
    def _plot_train_test_comparison(self, combined_df: pd.DataFrame):
        """Train/Test 비교 시각화 (한글 지원)"""
        
        # 폰트 재설정
        setup_korean_font()
        
        # 메트릭별 분포 비교
        metrics = ['brightness', 'contrast', 'sharpness', 'noise_level', 'rotation_angle']
        metric_names_en = ['Brightness', 'Contrast', 'Sharpness', 'Noise Level', 'Rotation Angle']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (metric, name_en) in enumerate(zip(metrics, metric_names_en)):
            ax = axes[i]
            
            # 히스토그램
            train_data = combined_df[combined_df['dataset'] == 'Train'][metric]
            test_data = combined_df[combined_df['dataset'] == 'Test'][metric]
            
            ax.hist(train_data, bins=30, alpha=0.7, label='Train', color='blue', density=True)
            ax.hist(test_data, bins=30, alpha=0.7, label='Test', color='red', density=True)
            
            ax.set_xlabel(f'{name_en}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.set_title(f'{name_en} Distribution Comparison', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # 마지막 subplot 제거
        axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'train_test_comparison.png', 
                   dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 Train/Test 비교 차트 저장: {self.output_dir / 'train_test_comparison.png'}")
    
    def _generate_train_test_insights(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train/Test 차이 기반 성능 최적화 전략 (공격적 접근)"""
        print(f"\n🎯 성능 최적화를 위한 Train/Test 차이 분석:")
        
        # 회전 분석 - 적극 활용
        train_rotation_std = train_df['rotation_angle'].std()
        test_rotation_std = test_df['rotation_angle'].std()
        
        if test_rotation_std > train_rotation_std * 1.5:
            print(f"  🔄 회전: Test 데이터 회전이 더 다양함 (std: {test_rotation_std:.1f} vs {train_rotation_std:.1f})")
            print(f"      → RandomRotation(degrees=(-45, 45)) 강화 적용!")
            print(f"      → 회전 증강 확률 0.8로 증가!")
        
        # 노이즈 분석 - 적극 활용
        train_noise_mean = train_df['noise_level'].mean()
        test_noise_mean = test_df['noise_level'].mean()
        
        if test_noise_mean > train_noise_mean * 1.2:
            print(f"  🔊 노이즈: Test가 더 노이지함 ({test_noise_mean:.2f} vs {train_noise_mean:.2f})")
            print(f"      → GaussianNoise(var_limit=(10, {int(test_noise_mean*2)})) 적용!")
            print(f"      → 노이즈 증강 확률 0.6으로 증가!")
        elif test_noise_mean < train_noise_mean * 0.8:
            print(f"  🔇 노이즈: Test가 더 깨끗함 ({test_noise_mean:.2f} vs {train_noise_mean:.2f})")
            print(f"      → 노이즈 증강 확률 0.2로 감소!")
        
        # 선명도 분석 - 적극 활용
        train_sharp_mean = train_df['sharpness'].mean()
        test_sharp_mean = test_df['sharpness'].mean()
        
        if test_sharp_mean < train_sharp_mean * 0.8:
            print(f"  🌫️ 블러: Test가 더 흐림 ({test_sharp_mean:.1f} vs {train_sharp_mean:.1f})")
            print(f"      → MotionBlur(blur_limit=7) + GaussianBlur(blur_limit=7) 강화!")
            print(f"      → 블러 증강 확률 0.7로 증가!")
            
            # 구체적 타겟 설정
            blur_ratio = train_sharp_mean / test_sharp_mean
            print(f"      → 선명도 비율 {blur_ratio:.1f}배 차이를 보정하는 블러 적용!")
        
        # 밝기 분석 - 적극 활용
        train_bright_mean = train_df['brightness'].mean()
        test_bright_mean = test_df['brightness'].mean()
        brightness_diff = test_bright_mean - train_bright_mean
        
        if abs(brightness_diff) > 10:
            print(f"  💡 밝기: 큰 차이 발견 ({test_bright_mean:.1f} vs {train_bright_mean:.1f})")
            
            if brightness_diff > 0:
                # Test가 더 밝음
                brightness_adjustment = min(0.5, brightness_diff / 100)
                print(f"      → Test 대응 밝기 증강: brightness_limit={brightness_adjustment:.2f}")
                print(f"      → 밝은 이미지 생성 확률 증가!")
            else:
                # Test가 더 어두움  
                brightness_adjustment = min(0.5, abs(brightness_diff) / 100)
                print(f"      → Test 대응 어두운 증강: brightness_limit={brightness_adjustment:.2f}")
                print(f"      → 어두운 이미지 생성 확률 증가!")
        
        # 대비 분석 - 적극 활용
        train_contrast_mean = train_df['contrast'].mean()
        test_contrast_mean = test_df['contrast'].mean()
        contrast_diff = test_contrast_mean - train_contrast_mean
        
        if abs(contrast_diff) > 5:
            contrast_adjustment = min(0.4, abs(contrast_diff) / 50)
            print(f"  🌈 대비: Test 대응 조정 ({test_contrast_mean:.1f} vs {train_contrast_mean:.1f})")
            print(f"      → contrast_limit={contrast_adjustment:.2f} 적용!")
        
        print(f"\n🚀 성능 최적화 전략:")
        print(f"  • Test 통계를 직접 활용한 타겟팅 증강")
        print(f"  • Train→Test 분포 변화 적극 보정")
        print(f"  • 구체적 수치 기반 파라미터 최적화")
        print(f"  • Domain Adaptation으로 Test 분포에 맞춤")
    
    def _detect_anomalous_images(self):
        """이질적 이미지 탐지 (차량 대시보드, 번호판 등)"""
        print("=== 🚨 4. 이질적 이미지 탐지 ===")
        
        # 클래스별 특성 분석
        train_df = self.dataset_analyzer.train_df
        class_mapping = self.dataset_analyzer.class_mapping
        
        # 의심스러운 클래스들 (차량 관련)
        vehicle_classes = [2, 16]  # car_dashboard, vehicle_registration_plate
        
        print("🚗 차량 관련 클래스 분석:")
        for class_id in vehicle_classes:
            class_name = class_mapping[class_id]
            count = len(train_df[train_df['target'] == class_id])
            print(f"  {class_id:2d}: {class_name} - {count}개")
        
        # 차량 관련 이미지들의 특성 분석
        vehicle_samples = []
        other_samples = []
        
        # 샘플 수집 (각 클래스에서 몇 개씩)
        for class_id in range(17):
            class_images = train_df[train_df['target'] == class_id]['ID'].tolist()
            sample_size = min(10, len(class_images))
            samples = random.sample(class_images, sample_size)
            
            for img_id in samples:
                img_path = self.dataset_analyzer.train_dir / img_id
                if class_id in vehicle_classes:
                    vehicle_samples.append(img_path)
                else:
                    other_samples.append(img_path)
        
        print(f"\n🔍 이미지 특성 분석 중... (차량: {len(vehicle_samples)}개, 기타: {len(other_samples)}개)")
        
        # 이미지 분석
        vehicle_metrics = self.dataset_analyzer.analyze_images_batch(vehicle_samples[:50])
        other_metrics = self.dataset_analyzer.analyze_images_batch(other_samples[:100])
        
        # 특성 비교
        self._compare_anomalous_vs_normal(vehicle_metrics, other_metrics)
        
        print()
    
    def _compare_anomalous_vs_normal(self, vehicle_metrics: List[ImageMetrics], other_metrics: List[ImageMetrics]):
        """이질적 이미지 vs 일반 문서 비교"""
        
        if not vehicle_metrics or not other_metrics:
            print("⚠️ 분석할 이미지가 부족합니다.")
            return
        
        print("📊 차량 관련 vs 일반 문서 특성 비교:")
        
        # 데이터프레임 생성
        vehicle_df = pd.DataFrame([m.to_dict() for m in vehicle_metrics])
        other_df = pd.DataFrame([m.to_dict() for m in other_metrics])
        
        metrics = ['brightness', 'contrast', 'aspect_ratio', 'sharpness']
        
        for metric in metrics:
            vehicle_values = vehicle_df[metric]
            other_values = other_df[metric]
            
            print(f"\n  📊 {metric.upper()}:")
            print(f"    차량:   {vehicle_values.mean():.2f} ± {vehicle_values.std():.2f}")
            print(f"    일반:   {other_values.mean():.2f} ± {other_values.std():.2f}")
            print(f"    차이:   {vehicle_values.mean() - other_values.mean():+.2f}")
        
        # 전략 제안
        print(f"\n💡 이질적 이미지 대응 전략:")
        
        # 종횡비 차이가 큰 경우
        vehicle_ar_mean = vehicle_df['aspect_ratio'].mean()
        other_ar_mean = other_df['aspect_ratio'].mean()
        
        if abs(vehicle_ar_mean - other_ar_mean) > 0.3:
            print(f"  📐 종횡비 차이 큼 → Multi-scale training 필요")
            print(f"      → 448x448, 512x512, 576x576 해상도로 학습")
        
        # 밝기 차이가 큰 경우  
        vehicle_bright_mean = vehicle_df['brightness'].mean()
        other_bright_mean = other_df['brightness'].mean()
        
        if abs(vehicle_bright_mean - other_bright_mean) > 15:
            print(f"  💡 밝기 차이 큼 → 강한 Color Augmentation 필요")
        
        print(f"  🎯 Binary Classifier 고려: 차량 vs 문서 분류 후 각각 다른 모델 적용")
    
    def _analyze_image_quality(self):
        """이미지 품질 상세 분석"""
        print("=== 🔍 5. 이미지 품질 상세 분석 ===")
        
        # 샘플 분석 (시간 절약)
        train_paths = self.dataset_analyzer.get_image_paths('train')
        sample_paths = random.sample(train_paths, min(200, len(train_paths)))
        
        print(f"🔄 이미지 품질 분석 중... ({len(sample_paths)}개 샘플)")
        metrics = self.dataset_analyzer.analyze_images_batch(sample_paths)
        
        if not metrics:
            print("⚠️ 분석할 이미지가 없습니다.")
            return
        
        # 데이터프레임 생성
        df = pd.DataFrame([m.to_dict() for m in metrics])
        
        # 품질 통계
        print("📊 이미지 품질 통계:")
        quality_metrics = ['brightness', 'contrast', 'sharpness', 'noise_level']
        
        for metric in quality_metrics:
            values = df[metric]
            print(f"  {metric.upper()}:")
            print(f"    평균: {values.mean():.2f}")
            print(f"    표준편차: {values.std():.2f}")
            print(f"    범위: {values.min():.2f} ~ {values.max():.2f}")
        
        # 품질 분포 시각화
        self._plot_quality_distributions(df)
        
        # 품질 기반 증강 전략
        self._suggest_quality_based_augmentation(df)
        
        print()
    
    def _plot_quality_distributions(self, df: pd.DataFrame):
        """품질 메트릭 분포 시각화 (한글 지원)"""
        
        # 폰트 재설정
        setup_korean_font()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        quality_metrics = ['brightness', 'contrast', 'sharpness', 'noise_level']
        metric_names_en = ['Brightness', 'Contrast', 'Sharpness', 'Noise Level']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        
        for i, (metric, name_en) in enumerate(zip(quality_metrics, metric_names_en)):
            ax = axes[i]
            values = df[metric]
            
            # 히스토그램
            ax.hist(values, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
            ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.1f}')
            ax.axvline(values.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {values.median():.1f}')
            
            ax.set_xlabel(f'{name_en}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{name_en} Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_distributions.png', 
                   dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 품질 분포 차트 저장: {self.output_dir / 'quality_distributions.png'}")
    
    def _suggest_quality_based_augmentation(self, df: pd.DataFrame):
        """품질 기반 증강 전략 제안"""
        print(f"\n💡 품질 기반 증강 전략:")
        
        # 밝기 분석
        brightness_std = df['brightness'].std()
        if brightness_std < 20:
            print(f"  💡 밝기 변화 부족 → RandomBrightnessContrast(brightness_limit=0.3)")
        
        # 대비 분석
        contrast_std = df['contrast'].std()
        if contrast_std < 15:
            print(f"  🌈 대비 변화 부족 → RandomBrightnessContrast(contrast_limit=0.3)")
        
        # 선명도 분석
        sharpness_mean = df['sharpness'].mean()
        if sharpness_mean > 1000:
            print(f"  🔍 이미지 선명함 → Blur 증강으로 일반화 개선")
            print(f"      → MotionBlur(blur_limit=3), GaussianBlur(blur_limit=3)")
        
        # 노이즈 분석
        noise_mean = df['noise_level'].mean()
        if noise_mean < 5:
            print(f"  🔇 노이즈 부족 → GaussianNoise(var_limit=(10, 50))")
    
    def _analyze_image_dimensions(self):
        """이미지 크기/종횡비 분석"""
        print("=== 📐 6. 이미지 크기/종횡비 분석 ===")
        
        # 전체 데이터 분석
        train_paths = self.dataset_analyzer.get_image_paths('train')
        test_paths = self.dataset_analyzer.get_image_paths('test')
        
        train_sample = train_paths
        test_sample = test_paths
        
        print(f"📊 전체 크기 분석 중... (Train: {len(train_sample)}개, Test: {len(test_sample)}개)")
        print(f"   17개 클래스 모든 이미지의 크기 특성을 완전 분석합니다.")
        
        # 이미지 분석 (병렬 처리 최적화)
        train_metrics = self.dataset_analyzer.analyze_images_batch(train_sample, max_workers=8)
        test_metrics = self.dataset_analyzer.analyze_images_batch(test_sample, max_workers=8)
        
        # 크기 통계
        self._analyze_dimension_statistics(train_metrics, test_metrics)
        
        # 최적 resize 전략 제안
        self._suggest_resize_strategy(train_metrics, test_metrics)
        
        print()
    
    def _analyze_dimension_statistics(self, train_metrics: List[ImageMetrics], test_metrics: List[ImageMetrics]):
        """크기 통계 분석"""
        
        if not train_metrics or not test_metrics:
            print("⚠️ 분석할 이미지가 부족합니다.")
            return
        
        # 데이터프레임 생성
        train_df = pd.DataFrame([m.to_dict() for m in train_metrics])
        test_df = pd.DataFrame([m.to_dict() for m in test_metrics])
        
        print("📊 이미지 크기 통계:")
        
        # 크기 통계
        for dataset_name, df in [("Train", train_df), ("Test", test_df)]:
            print(f"\n  {dataset_name} 데이터:")
            print(f"    너비: {df['width'].mean():.0f} ± {df['width'].std():.0f} (범위: {df['width'].min()}-{df['width'].max()})")
            print(f"    높이: {df['height'].mean():.0f} ± {df['height'].std():.0f} (범위: {df['height'].min()}-{df['height'].max()})")
            print(f"    종횡비: {df['aspect_ratio'].mean():.2f} ± {df['aspect_ratio'].std():.2f}")
            print(f"    파일크기: {df['file_size_kb'].mean():.0f}KB ± {df['file_size_kb'].std():.0f}KB")
        
        # 크기 분포 시각화
        self._plot_dimension_distributions(train_df, test_df)
    
    def _plot_dimension_distributions(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """크기 분포 시각화 (한글 지원)"""
        
        # 폰트 재설정
        setup_korean_font()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 너비 분포
        axes[0,0].hist(train_df['width'], bins=30, alpha=0.7, label='Train', color='blue', density=True)
        axes[0,0].hist(test_df['width'], bins=30, alpha=0.7, label='Test', color='red', density=True)
        axes[0,0].set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
        axes[0,0].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[0,0].set_title('Width Distribution', fontsize=12, fontweight='bold')
        axes[0,0].legend(fontsize=10)
        axes[0,0].grid(True, alpha=0.3)
        
        # 높이 분포
        axes[0,1].hist(train_df['height'], bins=30, alpha=0.7, label='Train', color='blue', density=True)
        axes[0,1].hist(test_df['height'], bins=30, alpha=0.7, label='Test', color='red', density=True)
        axes[0,1].set_xlabel('Height (pixels)', fontsize=11, fontweight='bold')
        axes[0,1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[0,1].set_title('Height Distribution', fontsize=12, fontweight='bold')
        axes[0,1].legend(fontsize=10)
        axes[0,1].grid(True, alpha=0.3)
        
        # 종횡비 분포
        axes[1,0].hist(train_df['aspect_ratio'], bins=30, alpha=0.7, label='Train', color='blue', density=True)
        axes[1,0].hist(test_df['aspect_ratio'], bins=30, alpha=0.7, label='Test', color='red', density=True)
        axes[1,0].set_xlabel('Aspect Ratio (W/H)', fontsize=11, fontweight='bold')
        axes[1,0].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[1,0].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
        axes[1,0].legend(fontsize=10)
        axes[1,0].grid(True, alpha=0.3)
        
        # 크기 산점도
        axes[1,1].scatter(train_df['width'], train_df['height'], alpha=0.6, label='Train', color='blue', s=20)
        axes[1,1].scatter(test_df['width'], test_df['height'], alpha=0.6, label='Test', color='red', s=20)
        axes[1,1].set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
        axes[1,1].set_ylabel('Height (pixels)', fontsize=11, fontweight='bold')
        axes[1,1].set_title('Width vs Height', fontsize=12, fontweight='bold')
        axes[1,1].legend(fontsize=10)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dimension_distributions.png', 
                   dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 크기 분포 차트 저장: {self.output_dir / 'dimension_distributions.png'}")
    
    def _suggest_resize_strategy(self, train_metrics: List[ImageMetrics], test_metrics: List[ImageMetrics]):
        """최적 resize 전략 제안"""
        
        if not train_metrics or not test_metrics:
            return
        
        # 데이터프레임 생성
        train_df = pd.DataFrame([m.to_dict() for m in train_metrics])
        test_df = pd.DataFrame([m.to_dict() for m in test_metrics])
        
        print(f"\n💡 Resize 전략 제안:")
        
        # 종횡비 분석
        train_ar_std = train_df['aspect_ratio'].std()
        test_ar_std = test_df['aspect_ratio'].std()
        combined_ar_std = pd.concat([train_df['aspect_ratio'], test_df['aspect_ratio']]).std()
        
        if combined_ar_std > 0.5:
            print(f"  📐 종횡비 다양함 (std: {combined_ar_std:.2f})")
            print(f"      → Aspect Ratio 유지하는 Resize 추천")
            print(f"      → Letterbox Padding 또는 Adaptive Resize 사용")
        else:
            print(f"  📐 종횡비 일정함 (std: {combined_ar_std:.2f})")
            print(f"      → Square Resize (448x448, 512x512) 가능")
        
        # 최적 크기 제안
        train_widths = train_df['width']
        train_heights = train_df['height']
        
        # 95% 분위수 기준으로 최적 크기 계산
        width_95 = np.percentile(train_widths, 95)
        height_95 = np.percentile(train_heights, 95)
        
        # 32의 배수로 맞춤 (CNN 효율성)
        optimal_size = max(width_95, height_95)
        optimal_size = int(np.ceil(optimal_size / 32) * 32)
        
        print(f"  🎯 권장 이미지 크기: {optimal_size}x{optimal_size}")
        print(f"      → 95% 이미지가 손실 없이 포함됨")
        
        # Multi-scale 전략
        scales = [optimal_size - 64, optimal_size, optimal_size + 64]
        print(f"  🔄 Multi-scale 학습 권장: {scales}")
    
    def _generate_competition_strategy(self):
        """최종 대회 전략 생성"""
        print("=== 🏆 7. 최종 대회 전략 제안 ===")
        
        strategy = {
            "augmentation": self._get_augmentation_strategy(),
            "model_architecture": self._get_model_strategy(),
            "training": self._get_training_strategy(),
            "ensemble": self._get_ensemble_strategy()
        }
        
        # 전략 출력
        for category, recommendations in strategy.items():
            print(f"\n📋 {category.upper()}:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        # 전략을 JSON으로 저장
        with open(self.output_dir / 'competition_strategy.json', 'w', encoding='utf-8') as f:
            json.dump(strategy, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 전략 저장: {self.output_dir / 'competition_strategy.json'}")
        
        # 코드 예시 생성
        self._generate_augmentation_code()
        
        print()
    
    def _get_augmentation_strategy(self) -> List[str]:
        """증강 전략 제안"""
        return [
            "RandomRotation(degrees=(-30, 30)) - Test 데이터 회전 대응",
            "RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3)",
            "GaussianNoise(var_limit=(10, 50)) - Test 노이즈 대응",
            "MotionBlur(blur_limit=3) + GaussianBlur(blur_limit=3)",
            "RandomPerspective(distortion_scale=0.1) - 문서 왜곡 시뮬레이션",
            "Cutout(num_holes=1, max_h_size=32, max_w_size=32) - 부분 가림 대응",
            "MixUp(alpha=0.2) - 클래스 불균형 완화",
            "CutMix(alpha=1.0) - 소수 클래스 증강"
        ]
    
    def _get_model_strategy(self) -> List[str]:
        """모델 전략 제안"""
        return [
            "EfficientNetV2-S/M - 문서 분류에 최적화된 아키텍처",
            "ConvNeXt-Tiny/Small - 최신 CNN 아키텍처",
            "Swin Transformer-Tiny - Vision Transformer 옵션",
            "Multi-scale Input (448x448, 512x512, 576x576)",
            "Focal Loss - 클래스 불균형 대응",
            "Label Smoothing (0.1) - 과적합 방지",
            "Dropout 강화 (0.3-0.5) - 일반화 성능 향상"
        ]
    
    def _get_training_strategy(self) -> List[str]:
        """훈련 전략 제안"""
        return [
            "Stratified 5-Fold Cross Validation",
            "AdamW Optimizer (lr=1e-4, weight_decay=1e-2)",
            "CosineAnnealingWarmRestarts 스케줄러",
            "Early Stopping (patience=5, monitor='val_f1')",
            "Gradient Clipping (max_norm=1.0)",
            "Mixed Precision Training (AMP)",
            "Class Weights 적용 - 불균형 대응",
            "Pseudo Labeling (confidence > 0.9) - 후반부 적용"
        ]
    
    def _get_ensemble_strategy(self) -> List[str]:
        """앙상블 전략 제안"""
        return [
            "5-Fold 앙상블 (각 fold별 최고 성능 모델)",
            "Multi-Architecture 앙상블 (EfficientNet + ConvNeXt + Swin)",
            "TTA (Test Time Augmentation): 회전, 플립, 스케일링",
            "Soft Voting - 확률 평균으로 최종 예측",
            "Confidence-based Weighted Voting",
            "Private LB 안정성을 위한 Conservative 앙상블"
        ]
    
    def _generate_augmentation_code(self):
        """증강 코드 예시 생성"""
        
        augmentation_code = '''
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
'''
        
        # 코드 저장
        with open(self.output_dir / 'augmentation_strategy.py', 'w', encoding='utf-8') as f:
            f.write(augmentation_code)
        
        print(f"💻 증강 코드 저장: {self.output_dir / 'augmentation_strategy.py'}")
    
    def _advanced_multimodal_analysis(self):
        """고급 Multi-Modal 분석: 이미지 + 메타데이터 결합"""
        print("\n=== 🔗 8-1. Multi-Modal 분석 ===")
        
        # 메타데이터 추출 및 분석
        metadata_features = self._extract_metadata_features()
        
        # 이미지 특성과 메타데이터 상관관계 분석
        correlations = self._analyze_image_metadata_correlation(metadata_features)
        
        # Multi-Modal 모델 아키텍처 제안
        multimodal_architecture = self._design_multimodal_architecture(correlations)
        
        # 결과 저장
        with open(self.output_dir / 'multimodal_strategy.json', 'w', encoding='utf-8') as f:
            json.dump({
                'metadata_features': metadata_features,
                'correlations': correlations,
                'architecture': multimodal_architecture
            }, f, indent=2, ensure_ascii=False)
        
        print(f"🔗 Multi-Modal 전략 저장: {self.output_dir / 'multimodal_strategy.json'}")
    
    def _extract_metadata_features(self):
        """메타데이터 특성 추출"""
        print("📊 메타데이터 특성 추출 중...")
        
        train_df = self.dataset_analyzer.train_df
        
        # 파일명 패턴 분석
        filename_patterns = {}
        for idx, row in train_df.iterrows():
            filename = row['ID']
            # 파일명에서 패턴 추출
            if '_' in filename:
                parts = filename.replace('.jpg', '').split('_')
                if len(parts) > 1:
                    pattern = f"{len(parts)}_parts"
                    filename_patterns[pattern] = filename_patterns.get(pattern, 0) + 1
        
        # 클래스별 특성
        class_characteristics = {}
        for target in range(17):
            class_data = train_df[train_df['target'] == target]
            class_name = self.dataset_analyzer.class_mapping[target]
            
            class_characteristics[target] = {
                'name': class_name,
                'count': len(class_data),
                'avg_filename_length': class_data['ID'].str.len().mean(),
                'has_underscore_ratio': (class_data['ID'].str.contains('_')).mean()
            }
        
        return {
            'filename_patterns': filename_patterns,
            'class_characteristics': class_characteristics,
            'total_samples': len(train_df)
        }
    
    def _analyze_image_metadata_correlation(self, metadata_features):
        """이미지 특성과 메타데이터 상관관계 분석"""
        print("🔍 이미지-메타데이터 상관관계 분석 중...")
        
        # 클래스별 이미지 특성 요약
        correlations = {}
        
        for target, char in metadata_features['class_characteristics'].items():
            # 차량 관련 클래스 식별
            is_vehicle = target in [2, 16]  # car_dashboard, vehicle_registration_plate
            
            correlations[target] = {
                'class_name': char['name'],
                'is_vehicle_related': is_vehicle,
                'sample_count': char['count'],
                'filename_complexity': char['avg_filename_length'],
                'recommended_input_size': 640 if is_vehicle else 512,
                'special_preprocessing': is_vehicle
            }
        
        return correlations
    
    def _design_multimodal_architecture(self, correlations):
        """Multi-Modal 모델 아키텍처 설계"""
        print("🏗️ Multi-Modal 아키텍처 설계 중...")
        
        architecture = {
            "image_branch": {
                "backbone": "EfficientNetV2-S",
                "input_sizes": [512, 640],  # 클래스별 다른 입력 크기
                "feature_dim": 1280
            },
            "metadata_branch": {
                "features": [
                    "filename_length",
                    "has_underscore", 
                    "class_prior_probability",
                    "image_aspect_ratio"
                ],
                "architecture": "MLP(4 -> 64 -> 32)",
                "feature_dim": 32
            },
            "fusion_strategy": {
                "method": "concatenation + attention",
                "combined_dim": 1312,  # 1280 + 32
                "fusion_layers": "Linear(1312 -> 512 -> 17)"
            },
            "training_strategy": {
                "loss_weighting": "dynamic (0.8 * image_loss + 0.2 * metadata_loss)",
                "separate_learning_rates": {
                    "image_branch": 1e-4,
                    "metadata_branch": 1e-3,
                    "fusion_layer": 1e-3
                }
            }
        }
        
        return architecture
    
    def _advanced_augmentation_strategies(self):
        """고급 증강 전략: 클래스별 맞춤 증강 + Domain Adaptation"""
        print("\n=== 🎨 8-2. 고급 증강 전략 ===")
        
        # 클래스별 맞춤 증강 전략
        class_specific_aug = self._design_class_specific_augmentation()
        
        # Domain Adaptation 전략
        domain_adaptation = self._design_domain_adaptation_strategy()
        
        # 고급 증강 코드 생성
        self._generate_advanced_augmentation_code(class_specific_aug, domain_adaptation)
        
        print("🎨 클래스별 맞춤 증강 전략 완료")
    
    def _design_class_specific_augmentation(self):
        """클래스별 맞춤 증강 설계"""
        print("🎯 클래스별 맞춤 증강 설계 중...")
        
        strategies = {
            "vehicle_classes": {
                "classes": [2, 16],  # car_dashboard, vehicle_registration_plate
                "strategy": {
                    "brightness_adjustment": "강화 (Test 데이터가 더 어두움)",
                    "perspective_transform": "감소 (이미 다양한 각도)",
                    "noise_augmentation": "증가 (실제 촬영 환경 시뮬레이션)",
                    "color_jittering": "강화 (다양한 조명 조건)"
                }
            },
            "document_classes": {
                "classes": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                "strategy": {
                    "rotation": "강화 (스캔 시 회전 발생)",
                    "perspective": "강화 (문서 스캔 왜곡)",
                    "blur": "추가 (Test 데이터가 더 흐림)",
                    "contrast": "조정 (문서 품질 다양화)"
                }
            },
            "minority_classes": {
                "classes": [1, 13, 14],  # 샘플 수 적은 클래스
                "strategy": {
                    "mixup_alpha": 0.4,  # 더 강한 MixUp
                    "cutmix_alpha": 1.2,  # 더 강한 CutMix
                    "copy_paste": True,  # Copy-Paste 증강 추가
                    "oversampling_factor": 2.0
                }
            }
        }
        
        return strategies
    
    def _design_domain_adaptation_strategy(self):
        """Domain Adaptation 전략 설계"""
        print("🌐 Domain Adaptation 전략 설계 중...")
        
        # Train/Test 차이 기반 적응 전략
        adaptation_strategy = {
            "brightness_adaptation": {
                "method": "Histogram Matching",
                "target": "Test 데이터 평균 밝기 (172.2)",
                "implementation": "A.HistogramMatching with reference images"
            },
            "sharpness_adaptation": {
                "method": "Controlled Blurring", 
                "target": "Test 데이터 선명도 (688.3)",
                "implementation": "Progressive blur augmentation during training"
            },
            "noise_adaptation": {
                "method": "Noise Injection Curriculum",
                "schedule": "Start clean, gradually add noise to match Test",
                "final_noise_level": 7.3
            },
            "progressive_adaptation": {
                "epochs_1_10": "Standard augmentation",
                "epochs_11_20": "Add brightness adaptation", 
                "epochs_21_30": "Add blur adaptation",
                "epochs_31_40": "Full domain adaptation"
            }
        }
        
        return adaptation_strategy
    
    def _generate_advanced_augmentation_code(self, class_specific, domain_adaptation):
        """고급 증강 코드 생성"""
        
        advanced_code = f'''
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple

class AdvancedAugmentationStrategy:
    """
    고급 증강 전략: 클래스별 맞춤 + Domain Adaptation
    EDA 분석 결과를 반영한 지능형 증강
    """
    
    def __init__(self):
        self.vehicle_classes = {class_specific['vehicle_classes']['classes']}
        self.document_classes = {class_specific['document_classes']['classes']}
        self.minority_classes = {class_specific['minority_classes']['classes']}
        
    def get_class_specific_transforms(self, target_class: int, image_size: int = 512):
        """클래스별 맞춤 증강 반환"""
        
        if target_class in self.vehicle_classes:
            return self._get_vehicle_transforms(image_size)
        elif target_class in self.minority_classes:
            return self._get_minority_class_transforms(image_size)
        else:
            return self._get_document_transforms(image_size)
    
    def _get_vehicle_transforms(self, image_size: int):
        """차량 관련 클래스 전용 증강"""
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # 차량 특화 증강
            A.RandomBrightnessContrast(
                brightness_limit=0.4,  # 강화된 밝기 조정
                contrast_limit=0.4, 
                p=0.9
            ),
            
            # 색상 지터링 강화 (다양한 조명)
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3, 
                saturation=0.3,
                hue=0.1,
                p=0.7
            ),
            
            # 노이즈 증가 (실제 촬영 환경)
            A.OneOf([
                A.GaussNoise(var_limit=(20, 80)),  # 더 강한 노이즈
                A.MultiplicativeNoise(multiplier=[0.8, 1.2]),
            ], p=0.6),
            
            # 원근 변형 감소 (이미 다양한 각도)
            A.Perspective(scale=(0.02, 0.05), p=0.2),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_document_transforms(self, image_size: int):
        """문서 클래스 전용 증강"""
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # 문서 특화 증강
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=45, p=0.8, border_mode=cv2.BORDER_CONSTANT),  # 더 강한 회전
            
            # 원근 변형 강화 (스캔 왜곡)
            A.Perspective(scale=(0.1, 0.2), p=0.5),
            
            # 블러 추가 (Test 데이터 대응)
            A.OneOf([
                A.MotionBlur(blur_limit=5),  # 더 강한 블러
                A.GaussianBlur(blur_limit=5),
            ], p=0.4),
            
            # 대비 조정
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.4,  # 문서 품질 다양화
                p=0.8
            ),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_minority_class_transforms(self, image_size: int):
        """소수 클래스 전용 강화 증강"""
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # 기본 증강
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            
            # 소수 클래스 특화 증강
            A.OneOf([
                A.ElasticTransform(p=0.3),  # 탄성 변형
                A.GridDistortion(p=0.3),    # 격자 왜곡
                A.OpticalDistortion(p=0.3), # 광학 왜곡
            ], p=0.5),
            
            # 강화된 노이즈
            A.OneOf([
                A.GaussNoise(var_limit=(15, 60)),
                A.MultiplicativeNoise(multiplier=[0.85, 1.15]),
            ], p=0.6),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# Domain Adaptation 클래스
class DomainAdaptationAugmentation:
    """Train 데이터를 Test 데이터 특성에 맞게 적응시키는 증강"""
    
    def __init__(self, adaptation_epoch: int = 0):
        self.adaptation_epoch = adaptation_epoch
        
    def get_domain_adapted_transforms(self, image_size: int = 512):
        """에포크에 따른 점진적 Domain Adaptation"""
        
        transforms = [A.Resize(image_size, image_size)]
        
        # 점진적 적응 스케줄
        if self.adaptation_epoch >= 10:
            # 밝기 적응 (Test 데이터가 더 밝음)
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,  # Test 대응
                    contrast_limit=0.2,
                    p=0.8
                )
            )
        
        if self.adaptation_epoch >= 20:
            # 블러 적응 (Test 데이터가 더 흐림)
            transforms.append(
                A.OneOf([
                    A.MotionBlur(blur_limit=4),
                    A.GaussianBlur(blur_limit=4),
                ], p=0.5)
            )
        
        if self.adaptation_epoch >= 30:
            # 노이즈 적응 (Test 데이터가 노이즈 적음)
            transforms.append(
                A.GaussNoise(var_limit=(5, 25), p=0.3)  # 약한 노이즈
            )
        
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)

# 사용 예시
def get_advanced_transforms(target_class: int, epoch: int = 0, image_size: int = 512):
    """
    고급 증강 전략 통합 함수
    
    Args:
        target_class: 클래스 ID (0-16)
        epoch: 현재 에포크 (Domain Adaptation용)
        image_size: 입력 이미지 크기
    """
    
    # 클래스별 맞춤 증강
    class_aug = AdvancedAugmentationStrategy()
    base_transforms = class_aug.get_class_specific_transforms(target_class, image_size)
    
    # Domain Adaptation 추가
    if epoch > 10:  # 일정 에포크 후 적응 시작
        domain_aug = DomainAdaptationAugmentation(epoch)
        domain_transforms = domain_aug.get_domain_adapted_transforms(image_size)
        return domain_transforms
    
    return base_transforms
'''
        
        # 파일 저장
        with open(self.output_dir / 'advanced_augmentation.py', 'w', encoding='utf-8') as f:
            f.write(advanced_code)
        
        print(f"🎨 고급 증강 코드 저장: {self.output_dir / 'advanced_augmentation.py'}")
    
    def _pseudo_labeling_strategy(self):
        """Pseudo Labeling 및 Progressive Labeling 전략"""
        print("\n=== 🏷️ 8-3. Pseudo Labeling 전략 ===")
        
        # Pseudo Labeling 전략 설계
        pseudo_strategy = self._design_pseudo_labeling_strategy()
        
        # Progressive Labeling 코드 생성
        self._generate_pseudo_labeling_code(pseudo_strategy)
        
        print("🏷️ Pseudo Labeling 전략 완료")
    
    def _design_pseudo_labeling_strategy(self):
        """Pseudo Labeling 전략 설계"""
        print("🎯 Pseudo Labeling 전략 설계 중...")
        
        strategy = {
            "confidence_thresholds": {
                "conservative": 0.95,  # 초기 단계
                "moderate": 0.90,      # 중간 단계  
                "aggressive": 0.85     # 후반 단계
            },
            "progressive_schedule": {
                "phase_1": {
                    "epochs": "1-10",
                    "threshold": 0.95,
                    "max_pseudo_ratio": 0.1,  # 전체의 10%만
                    "strategy": "Only highest confidence"
                },
                "phase_2": {
                    "epochs": "11-20", 
                    "threshold": 0.90,
                    "max_pseudo_ratio": 0.2,  # 전체의 20%
                    "strategy": "Class-balanced selection"
                },
                "phase_3": {
                    "epochs": "21-30",
                    "threshold": 0.85,
                    "max_pseudo_ratio": 0.3,  # 전체의 30%
                    "strategy": "Uncertainty-based selection"
                }
            },
            "quality_control": {
                "ensemble_agreement": "3개 이상 모델이 동일 예측",
                "consistency_check": "TTA 결과 일치도 > 0.9",
                "class_balance": "각 클래스별 최대 pseudo 샘플 수 제한",
                "outlier_detection": "Isolation Forest로 이상치 제거"
            },
            "implementation_details": {
                "update_frequency": "매 에포크마다",
                "pseudo_weight": "0.5 (실제 데이터 대비)",
                "mixup_with_real": True,
                "teacher_student": "EMA 업데이트 (α=0.999)"
            }
        }
        
        return strategy
    
    def _generate_pseudo_labeling_code(self, strategy):
        """Pseudo Labeling 구현 코드 생성"""
        
        pseudo_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from collections import defaultdict, Counter

class PseudoLabelingManager:
    """
    Progressive Pseudo Labeling 관리자
    EDA 분석 결과를 반영한 지능형 pseudo labeling
    """
    
    def __init__(self, 
                 num_classes: int = 17,
                 confidence_thresholds: Dict = {strategy['confidence_thresholds']},
                 class_weights: Optional[Dict] = None):
        
        self.num_classes = num_classes
        self.confidence_thresholds = confidence_thresholds
        self.class_weights = class_weights or {{}}
        
        # Progressive schedule
        self.progressive_schedule = {strategy['progressive_schedule']}
        
        # Quality control
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.pseudo_history = defaultdict(list)
        
    def get_current_phase(self, epoch: int) -> Dict:
        """현재 에포크에 따른 phase 정보 반환"""
        if epoch <= 10:
            return self.progressive_schedule['phase_1']
        elif epoch <= 20:
            return self.progressive_schedule['phase_2']
        else:
            return self.progressive_schedule['phase_3']
    
    def select_pseudo_labels(self, 
                           predictions: torch.Tensor,
                           confidences: torch.Tensor,
                           features: torch.Tensor,
                           epoch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pseudo label 선택
        
        Args:
            predictions: 모델 예측 (N, num_classes)
            confidences: 예측 신뢰도 (N,)
            features: 특성 벡터 (N, feature_dim)
            epoch: 현재 에포크
            
        Returns:
            selected_indices: 선택된 샘플 인덱스
            pseudo_labels: Pseudo label
        """
        
        phase = self.get_current_phase(epoch)
        threshold = phase['threshold']
        max_ratio = phase['max_pseudo_ratio']
        
        # 1. 신뢰도 기반 필터링
        high_conf_mask = confidences > threshold
        
        if not high_conf_mask.any():
            return torch.tensor([]), torch.tensor([])
        
        # 2. 이상치 제거
        if len(features[high_conf_mask]) > 10:  # 최소 샘플 수 확보
            outlier_mask = self._detect_outliers(features[high_conf_mask])
            high_conf_mask[high_conf_mask.clone()] = ~outlier_mask
        
        # 3. 클래스 균형 고려 선택
        selected_indices = self._balanced_selection(
            predictions[high_conf_mask],
            confidences[high_conf_mask], 
            high_conf_mask.nonzero().squeeze(),
            max_ratio,
            phase['strategy']
        )
        
        if len(selected_indices) == 0:
            return torch.tensor([]), torch.tensor([])
        
        pseudo_labels = predictions[selected_indices].argmax(dim=1)
        
        # 4. 품질 기록
        self._record_pseudo_quality(selected_indices, pseudo_labels, confidences[selected_indices])
        
        return selected_indices, pseudo_labels
    
    def _detect_outliers(self, features: torch.Tensor) -> torch.Tensor:
        """Isolation Forest를 사용한 이상치 탐지"""
        features_np = features.detach().cpu().numpy()
        outlier_pred = self.isolation_forest.fit_predict(features_np)
        return torch.tensor(outlier_pred == -1)  # -1이 이상치
    
    def _balanced_selection(self, 
                          predictions: torch.Tensor,
                          confidences: torch.Tensor,
                          indices: torch.Tensor,
                          max_ratio: float,
                          strategy: str) -> torch.Tensor:
        """클래스 균형을 고려한 선택"""
        
        max_samples = int(len(predictions) * max_ratio)
        pred_classes = predictions.argmax(dim=1)
        
        if strategy == "Only highest confidence":
            # 단순히 신뢰도 높은 순으로
            _, top_indices = confidences.topk(min(max_samples, len(confidences)))
            return indices[top_indices]
            
        elif strategy == "Class-balanced selection":
            # 클래스별 균등 선택
            selected = []
            samples_per_class = max_samples // self.num_classes
            
            for class_id in range(self.num_classes):
                class_mask = pred_classes == class_id
                if not class_mask.any():
                    continue
                    
                class_confidences = confidences[class_mask]
                class_indices = indices[class_mask]
                
                # 해당 클래스에서 가장 신뢰도 높은 샘플들 선택
                num_select = min(samples_per_class, len(class_confidences))
                _, top_class_indices = class_confidences.topk(num_select)
                selected.extend(class_indices[top_class_indices].tolist())
            
            return torch.tensor(selected)
            
        elif strategy == "Uncertainty-based selection":
            # 불확실성 기반 선택 (entropy 사용)
            entropies = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
            
            # 신뢰도와 불확실성의 균형
            combined_score = confidences - 0.1 * entropies  # 불확실성 페널티
            _, selected_indices = combined_score.topk(min(max_samples, len(combined_score)))
            
            return indices[selected_indices]
        
        return torch.tensor([])
    
    def _record_pseudo_quality(self, indices: torch.Tensor, labels: torch.Tensor, confidences: torch.Tensor):
        """Pseudo label 품질 기록"""
        for idx, label, conf in zip(indices, labels, confidences):
            self.pseudo_history[int(idx)].append({{
                'label': int(label),
                'confidence': float(conf),
                'timestamp': len(self.pseudo_history[int(idx)])
            }})
    
    def get_pseudo_statistics(self) -> Dict:
        """Pseudo labeling 통계 반환"""
        if not self.pseudo_history:
            return {{'total_pseudo_samples': 0}}
        
        total_samples = len(self.pseudo_history)
        label_distribution = Counter()
        avg_confidence = 0
        
        for sample_history in self.pseudo_history.values():
            if sample_history:
                latest = sample_history[-1]
                label_distribution[latest['label']] += 1
                avg_confidence += latest['confidence']
        
        avg_confidence /= total_samples if total_samples > 0 else 1
        
        return {{
            'total_pseudo_samples': total_samples,
            'label_distribution': dict(label_distribution),
            'average_confidence': avg_confidence,
            'class_balance_ratio': max(label_distribution.values()) / min(label_distribution.values()) if label_distribution else 0
        }}

class EnsemblePseudoLabeling:
    """앙상블 기반 고품질 Pseudo Labeling"""
    
    def __init__(self, models: List[nn.Module], agreement_threshold: int = 3):
        self.models = models
        self.agreement_threshold = agreement_threshold
        
    def get_ensemble_pseudo_labels(self, 
                                 dataloader,
                                 device: torch.device) -> Tuple[List, List, List]:
        """
        앙상블 합의 기반 pseudo label 생성
        
        Returns:
            pseudo_data: 선택된 데이터
            pseudo_labels: 합의된 라벨  
            confidence_scores: 신뢰도 점수
        """
        
        all_predictions = []
        all_data = []
        
        # 각 모델의 예측 수집
        for model in self.models:
            model.eval()
            predictions = []
            data_batch = []
            
            with torch.no_grad():
                for batch_data, _ in dataloader:
                    batch_data = batch_data.to(device)
                    outputs = model(batch_data)
                    predictions.append(F.softmax(outputs, dim=1))
                    data_batch.append(batch_data)
            
            all_predictions.append(torch.cat(predictions, dim=0))
            if not all_data:  # 첫 번째 모델에서만 데이터 저장
                all_data = torch.cat(data_batch, dim=0)
        
        # 앙상블 합의 확인
        ensemble_preds = torch.stack(all_predictions)  # (num_models, num_samples, num_classes)
        pred_labels = ensemble_preds.argmax(dim=2)  # (num_models, num_samples)
        
        pseudo_data, pseudo_labels, confidence_scores = [], [], []
        
        for i in range(pred_labels.shape[1]):  # 각 샘플에 대해
            sample_preds = pred_labels[:, i]
            
            # 합의 확인 (과반수 이상 동일 예측)
            label_counts = torch.bincount(sample_preds, minlength=17)
            max_count = label_counts.max()
            
            if max_count >= self.agreement_threshold:
                agreed_label = label_counts.argmax()
                
                # 해당 라벨에 대한 평균 신뢰도
                confidence = ensemble_preds[:, i, agreed_label].mean()
                
                pseudo_data.append(all_data[i])
                pseudo_labels.append(agreed_label)
                confidence_scores.append(confidence)
        
        return pseudo_data, pseudo_labels, confidence_scores

# 사용 예시
def setup_pseudo_labeling(num_classes: int = 17, class_weights: Dict = None):
    """Pseudo Labeling 설정"""
    
    # EDA에서 계산된 클래스 가중치 사용
    if class_weights is None:
        class_weights = {class_weights if hasattr(self, 'class_weights') else {}}
    
    pseudo_manager = PseudoLabelingManager(
        num_classes=num_classes,
        class_weights=class_weights
    )
    
    return pseudo_manager

def progressive_training_with_pseudo_labels(model, 
                                          train_loader,
                                          pseudo_manager: PseudoLabelingManager,
                                          epoch: int):
    """Progressive Pseudo Labeling을 적용한 훈련"""
    
    # 현재 phase 확인
    phase = pseudo_manager.get_current_phase(epoch)
    print(f"Epoch {{epoch}}: {{phase['strategy']}} (threshold={{phase['threshold']}})")
    
    # Pseudo label 선택 및 훈련에 적용하는 로직은
    # 실제 훈련 루프에서 구현
    
    return phase
'''
        
        # 파일 저장
        with open(self.output_dir / 'pseudo_labeling_strategy.py', 'w', encoding='utf-8') as f:
            f.write(pseudo_code)
        
        # 전략 JSON 저장
        with open(self.output_dir / 'pseudo_labeling_config.json', 'w', encoding='utf-8') as f:
            json.dump(strategy, f, indent=2, ensure_ascii=False)
        
        print(f"🏷️ Pseudo Labeling 코드 저장: {self.output_dir / 'pseudo_labeling_strategy.py'}")
        print(f"🏷️ Pseudo Labeling 설정 저장: {self.output_dir / 'pseudo_labeling_config.json'}")


def main():
    """메인 실행 함수"""
    
    # 데이터 경로 설정
    data_root = Path("/home/james/doc-classification/computervisioncompetition-cv3/data")
    
    if not data_root.exists():
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_root}")
        return
    
    try:
        # EDA 실행
        eda = CompetitionEDA(data_root)
        eda.run_complete_analysis()
        
        print("\n🎉 Competition EDA 완료!")
        print("📁 결과 파일들:")
        print(f"  • 클래스 분포: {eda.output_dir / 'class_distribution.png'}")
        print(f"  • Train/Test 비교: {eda.output_dir / 'train_test_comparison.png'}")
        print(f"  • 품질 분포: {eda.output_dir / 'quality_distributions.png'}")
        print(f"  • 크기 분포: {eda.output_dir / 'dimension_distributions.png'}")
        print(f"  • 클래스 가중치: {eda.output_dir / 'class_weights.json'}")
        print(f"  • 대회 전략: {eda.output_dir / 'competition_strategy.json'}")
        print(f"  • 증강 코드: {eda.output_dir / 'augmentation_strategy.py'}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

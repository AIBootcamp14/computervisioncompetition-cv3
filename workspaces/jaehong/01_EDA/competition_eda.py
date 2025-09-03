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
        
        # 샘플링 (전체 분석은 시간이 오래 걸림)
        train_paths = self.dataset_analyzer.get_image_paths('train')
        test_paths = self.dataset_analyzer.get_image_paths('test')
        
        # 샘플 크기 설정
        train_sample_size = min(300, len(train_paths))
        test_sample_size = min(150, len(test_paths))
        
        print(f"📊 샘플 분석 (Train: {train_sample_size}개, Test: {test_sample_size}개)")
        
        # 랜덤 샘플링
        train_sample = random.sample(train_paths, train_sample_size)
        test_sample = random.sample(test_paths, test_sample_size)
        
        # 이미지 분석
        print("🔄 Train 이미지 분석 중...")
        train_metrics = self.dataset_analyzer.analyze_images_batch(train_sample)
        
        print("🔄 Test 이미지 분석 중...")
        test_metrics = self.dataset_analyzer.analyze_images_batch(test_sample)
        
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
        """Train/Test 차이 기반 대회 전략 인사이트"""
        print(f"\n💡 대회 전략 인사이트:")
        
        # 회전 분석
        train_rotation_std = train_df['rotation_angle'].std()
        test_rotation_std = test_df['rotation_angle'].std()
        
        if test_rotation_std > train_rotation_std * 1.5:
            print(f"  🔄 회전: Test 데이터의 회전이 더 다양함 (std: {test_rotation_std:.1f} vs {train_rotation_std:.1f})")
            print(f"      → RandomRotation(degrees=(-30, 30)) 증강 필수!")
        
        # 노이즈 분석
        train_noise_mean = train_df['noise_level'].mean()
        test_noise_mean = test_df['noise_level'].mean()
        
        if test_noise_mean > train_noise_mean * 1.2:
            print(f"  🔊 노이즈: Test 데이터가 더 노이지함 ({test_noise_mean:.2f} vs {train_noise_mean:.2f})")
            print(f"      → GaussianNoise, ISONoise 증강 추가!")
        
        # 선명도 분석
        train_sharp_mean = train_df['sharpness'].mean()
        test_sharp_mean = test_df['sharpness'].mean()
        
        if test_sharp_mean < train_sharp_mean * 0.8:
            print(f"  🌫️ 블러: Test 데이터가 더 흐림 ({test_sharp_mean:.1f} vs {train_sharp_mean:.1f})")
            print(f"      → MotionBlur, GaussianBlur 증강 추가!")
        
        # 밝기 분석
        train_bright_mean = train_df['brightness'].mean()
        test_bright_mean = test_df['brightness'].mean()
        
        if abs(test_bright_mean - train_bright_mean) > 10:
            print(f"  💡 밝기: 차이 있음 ({test_bright_mean:.1f} vs {train_bright_mean:.1f})")
            print(f"      → RandomBrightnessContrast 증강 필수!")
    
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
        
        # 샘플 분석
        train_paths = self.dataset_analyzer.get_image_paths('train')
        test_paths = self.dataset_analyzer.get_image_paths('test')
        
        train_sample = random.sample(train_paths, min(200, len(train_paths)))
        test_sample = random.sample(test_paths, min(100, len(test_paths)))
        
        print(f"📊 크기 분석 중... (Train: {len(train_sample)}개, Test: {len(test_sample)}개)")
        
        # 이미지 분석
        train_metrics = self.dataset_analyzer.analyze_images_batch(train_sample)
        test_metrics = self.dataset_analyzer.analyze_images_batch(test_sample)
        
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

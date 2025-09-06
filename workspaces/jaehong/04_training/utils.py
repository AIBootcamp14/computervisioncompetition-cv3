"""
🛠️ Training Utilities
시니어 캐글러 수준의 유틸리티 함수들

Features:
- Reproducibility
- Model checkpointing
- Metrics tracking
- GPU optimization
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 시드 설정 완료: {seed}")


def get_gpu_info():
    """GPU 정보 출력"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"🖥️ GPU 정보:")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2
        memory_cached = torch.cuda.memory_reserved(current_device) / 1024**2
        print(f"   현재 메모리: {memory_allocated:.1f} MB allocated, {memory_cached:.1f} MB cached")
        
        return {
            "available": True,
            "device_count": device_count,
            "current_device": current_device,
            "device_name": torch.cuda.get_device_name(current_device),
            "total_memory_gb": torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        }
    else:
        print("⚠️ CUDA를 사용할 수 없습니다. CPU로 훈련합니다.")
        return {"available": False}


class AverageMeter:
    """평균 메트릭 추적기"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """시간 측정기"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
        return self.elapsed_time()
    
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def elapsed_time_str(self) -> str:
        elapsed = self.elapsed_time()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Path,
    is_best: bool = False
):
    """체크포인트 저장"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    
    if is_best:
        best_path = filepath.parent / "model_best.pth"
        torch.save(state, best_path)
    
    print(f"💾 체크포인트 저장: {filepath}")


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """체크포인트 로드"""
    if not filepath.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 옵티마이저 상태 로드
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 스케줄러 상태 로드
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"📂 체크포인트 로드: {filepath}")
    print(f"   에포크: {checkpoint.get('epoch', 'N/A')}")
    print(f"   최고 점수: {checkpoint.get('best_score', 'N/A')}")
    
    return checkpoint


def calculate_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """모델 크기 계산"""
    param_size = 0
    param_sum = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    
    buffer_size = 0
    buffer_sum = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    
    all_size = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "total_params": param_sum,
        "total_buffers": buffer_sum,
        "model_size_mb": all_size,
        "param_size_mb": param_size / 1024 / 1024,
        "buffer_size_mb": buffer_size / 1024 / 1024
    }


def print_model_summary(model: torch.nn.Module, input_size: tuple = (3, 512, 512)):
    """모델 요약 출력"""
    model_stats = calculate_model_size(model)
    
    print(f"\n🧠 모델 요약:")
    print(f"   총 파라미터: {model_stats['total_params']:,}")
    print(f"   모델 크기: {model_stats['model_size_mb']:.1f} MB")
    
    # 입력 크기로 테스트
    try:
        dummy_input = torch.randn(1, *input_size)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   입력 크기: {input_size}")
        print(f"   출력 크기: {output.shape[1:]}")
    except Exception as e:
        print(f"   ⚠️ 모델 테스트 실패: {e}")


def optimize_gpu_settings():
    """GPU 최적화 설정"""
    if torch.cuda.is_available():
        # 메모리 할당 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 메모리 정리
        torch.cuda.empty_cache()
        
        print("⚡ GPU 최적화 설정 완료")
        return True
    return False


def save_experiment_results(
    results: Dict[str, Any],
    output_dir: Path,
    experiment_name: str
):
    """실험 결과 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 형태로 저장
    results_file = output_dir / f"{experiment_name}_results.json"
    
    # Numpy arrays를 리스트로 변환
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            serializable_results[key] = value.cpu().numpy().tolist()
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"💾 실험 결과 저장: {results_file}")


def create_submission(
    predictions: np.ndarray,
    sample_submission_path: Path,
    output_path: Path,
    experiment_name: str
):
    """제출 파일 생성"""
    import pandas as pd
    
    # 샘플 제출 파일 로드
    sample_df = pd.read_csv(sample_submission_path)
    
    # 예측 결과로 업데이트
    sample_df['target'] = predictions
    
    # 저장
    submission_file = output_path / f"{experiment_name}_submission.csv"
    sample_df.to_csv(submission_file, index=False)
    
    print(f"📤 제출 파일 생성: {submission_file}")
    
    # 예측 분포 출력
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"📊 예측 분포:")
    for class_id, count in zip(unique, counts):
        percentage = count / len(predictions) * 100
        print(f"   클래스 {class_id:2d}: {count:4d}개 ({percentage:5.1f}%)")
    
    return submission_file


def log_system_info():
    """시스템 정보 로깅"""
    import platform
    import psutil
    
    print(f"\n💻 시스템 정보:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   PyTorch: {torch.__version__}")
    
    # CPU 정보
    print(f"   CPU: {psutil.cpu_count()} cores")
    print(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    # GPU 정보
    gpu_info = get_gpu_info()
    
    return {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu_cores": psutil.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 1024**3,
        "gpu_info": gpu_info
    }


class EarlyStopping:
    """Early Stopping 헬퍼"""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
    
    def __call__(self, val_score: float) -> bool:
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint = True
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            self.save_checkpoint = False
            
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint = True
            self.counter = 0
        
        return self.early_stop


# 사용 예시
if __name__ == "__main__":
    print("🛠️ 유틸리티 테스트:")
    
    # 시드 설정
    set_seed(42)
    
    # 시스템 정보
    system_info = log_system_info()
    
    # GPU 최적화
    optimize_gpu_settings()
    
    # 타이머 테스트
    timer = Timer()
    timer.start()
    time.sleep(0.1)
    elapsed = timer.stop()
    print(f"⏱️ 타이머 테스트: {elapsed:.3f}초")
    
    print("✅ 유틸리티 테스트 완료")


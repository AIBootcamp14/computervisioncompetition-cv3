"""
📊 Performance Analysis System
시니어 캐글러 수준의 성능 분석 시스템

Features:
- Multi-Model Comparison
- Statistical Analysis
- Visualization
- Report Generation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class PerformanceAnalyzer:
    """
    성능 분석기 클래스
    
    Clean Architecture 적용:
    - Entity: 모델 성능 데이터
    - Use Case: 분석 및 비교 로직
    - Interface: 시각화 및 리포트 생성
    """
    
    def __init__(self, workspace_root: str):
        """
        성능 분석기 초기화
        
        Args:
            workspace_root: 워크스페이스 루트 경로
        """
        self.workspace_root = Path(workspace_root)
        self.results_data = {}
        self.model_metadata = {}
        
        # 결과 디렉토리 생성
        self.output_dir = self.workspace_root / "05_evaluation" / "analysis_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("📊 PerformanceAnalyzer 초기화 완료")
    
    def load_model_results(self, model_name: str, results_path: str) -> bool:
        """
        모델 결과 로드
        
        Args:
            model_name: 모델 이름
            results_path: 결과 파일 경로
            
        Returns:
            로드 성공 여부
        """
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.results_data[model_name] = data
            
            # 메타데이터 추출
            self.model_metadata[model_name] = {
                'architecture': data.get('config', {}).get('model', {}).get('architecture', 'unknown'),
                'image_size': data.get('config', {}).get('model', {}).get('image_size', 512),
                'loss_type': data.get('config', {}).get('model', {}).get('loss_type', 'focal'),
                'epochs_trained': data.get('epochs_trained', 0),
                'total_time': data.get('total_time', 0)
            }
            
            print(f"✅ 모델 결과 로드 완료: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ 모델 결과 로드 실패 ({model_name}): {e}")
            return False
    
    def discover_and_load_all_results(self) -> int:
        """모든 훈련 결과 자동 발견 및 로드"""
        print("🔍 훈련 결과 자동 발견 중...")
        
        loaded_count = 0
        
        # 04_training/outputs 디렉토리 스캔
        training_outputs = self.workspace_root / "04_training" / "outputs"
        
        if training_outputs.exists():
            for exp_dir in training_outputs.iterdir():
                if exp_dir.is_dir():
                    results_file = exp_dir / f"{exp_dir.name}_results.json"
                    
                    if results_file.exists():
                        if self.load_model_results(exp_dir.name, str(results_file)):
                            loaded_count += 1
        
        # 07_ensemble/results 디렉토리 스캔
        ensemble_results = self.workspace_root / "07_ensemble" / "results"
        
        if ensemble_results.exists():
            for results_file in ensemble_results.glob("*_results.json"):
                model_name = results_file.stem.replace("_results", "")
                if self.load_model_results(f"ensemble_{model_name}", str(results_file)):
                    loaded_count += 1
        
        print(f"📋 총 {loaded_count}개 모델 결과 로드 완료")
        return loaded_count
    
    def calculate_performance_metrics(self) -> pd.DataFrame:
        """성능 메트릭 계산"""
        print("📊 성능 메트릭 계산 중...")
        
        metrics_data = []
        
        for model_name, results in self.results_data.items():
            try:
                # 기본 메트릭
                best_f1 = results.get('best_f1', 0.0)
                final_f1 = results.get('final_f1', 0.0)
                final_accuracy = results.get('final_accuracy', 0.0)
                
                # 예측 결과가 있는 경우 추가 메트릭 계산
                if 'final_predictions' in results and 'final_targets' in results:
                    predictions = np.array(results['final_predictions'])
                    targets = np.array(results['final_targets'])
                    
                    # 클래스별 메트릭
                    precision = precision_score(targets, predictions, average='macro', zero_division=0)
                    recall = recall_score(targets, predictions, average='macro', zero_division=0)
                    
                    # 클래스별 F1 점수
                    class_f1_scores = f1_score(targets, predictions, average=None, zero_division=0)
                    
                else:
                    precision = 0.0
                    recall = 0.0
                    class_f1_scores = [0.0] * 17
                
                # 메타데이터
                metadata = self.model_metadata.get(model_name, {})
                
                metrics_data.append({
                    'model_name': model_name,
                    'architecture': metadata.get('architecture', 'unknown'),
                    'image_size': metadata.get('image_size', 512),
                    'loss_type': metadata.get('loss_type', 'focal'),
                    'epochs_trained': metadata.get('epochs_trained', 0),
                    'training_time_minutes': metadata.get('total_time', 0) / 60,
                    'best_f1': best_f1,
                    'final_f1': final_f1,
                    'final_accuracy': final_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_improvement': best_f1 - final_f1,
                    'avg_class_f1': np.mean(class_f1_scores),
                    'std_class_f1': np.std(class_f1_scores),
                    'min_class_f1': np.min(class_f1_scores),
                    'max_class_f1': np.max(class_f1_scores)
                })
                
            except Exception as e:
                print(f"⚠️ 메트릭 계산 실패 ({model_name}): {e}")
                continue
        
        df = pd.DataFrame(metrics_data)
        
        # 정렬 (F1 점수 기준)
        df = df.sort_values('final_f1', ascending=False).reset_index(drop=True)
        
        print(f"✅ {len(df)}개 모델 메트릭 계산 완료")
        return df
    
    def create_performance_comparison_chart(self, metrics_df: pd.DataFrame) -> str:
        """성능 비교 차트 생성"""
        print("📈 성능 비교 차트 생성 중...")
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'F1 Score Comparison',
                'Accuracy vs F1 Score',
                'Training Time vs Performance',
                'Architecture Performance'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. F1 점수 비교 (막대 차트)
        fig.add_trace(
            go.Bar(
                x=metrics_df['model_name'],
                y=metrics_df['final_f1'],
                name='Final F1',
                marker_color='lightblue',
                text=metrics_df['final_f1'].round(4),
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['model_name'],
                y=metrics_df['best_f1'],
                name='Best F1',
                marker_color='darkblue',
                text=metrics_df['best_f1'].round(4),
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Accuracy vs F1 Score (산점도)
        fig.add_trace(
            go.Scatter(
                x=metrics_df['final_accuracy'],
                y=metrics_df['final_f1'],
                mode='markers+text',
                name='Models',
                text=metrics_df['model_name'],
                textposition='top center',
                marker=dict(
                    size=10,
                    color=metrics_df['training_time_minutes'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Training Time (min)")
                )
            ),
            row=1, col=2
        )
        
        # 3. 훈련 시간 vs 성능 (산점도)
        fig.add_trace(
            go.Scatter(
                x=metrics_df['training_time_minutes'],
                y=metrics_df['final_f1'],
                mode='markers+text',
                name='Time vs F1',
                text=metrics_df['model_name'],
                textposition='top center',
                marker=dict(size=12, color='red', opacity=0.7)
            ),
            row=2, col=1
        )
        
        # 4. 아키텍처별 성능 (박스 플롯)
        architectures = metrics_df['architecture'].unique()
        for arch in architectures:
            arch_data = metrics_df[metrics_df['architecture'] == arch]
            fig.add_trace(
                go.Box(
                    y=arch_data['final_f1'],
                    name=arch,
                    boxmean=True
                ),
                row=2, col=2
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=800,
            title_text="Model Performance Comparison Dashboard",
            showlegend=True
        )
        
        # 축 라벨 업데이트
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="F1 Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="F1 Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Training Time (min)", row=2, col=1)
        fig.update_yaxes(title_text="F1 Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Architecture", row=2, col=2)
        fig.update_yaxes(title_text="F1 Score", row=2, col=2)
        
        # 저장
        chart_file = self.output_dir / "performance_comparison.html"
        fig.write_html(str(chart_file))
        
        print(f"✅ 성능 비교 차트 저장: {chart_file}")
        return str(chart_file)
    
    def create_confusion_matrix_comparison(self, metrics_df: pd.DataFrame) -> str:
        """혼동 행렬 비교 생성"""
        print("🔍 혼동 행렬 비교 생성 중...")
        
        # 상위 3개 모델 선택
        top_models = metrics_df.head(3)['model_name'].tolist()
        
        fig, axes = plt.subplots(1, len(top_models), figsize=(5*len(top_models), 4))
        if len(top_models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(top_models):
            try:
                results = self.results_data[model_name]
                
                if 'final_predictions' in results and 'final_targets' in results:
                    predictions = np.array(results['final_predictions'])
                    targets = np.array(results['final_targets'])
                    
                    cm = confusion_matrix(targets, predictions)
                    
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        ax=axes[i],
                        cbar=i == len(top_models)-1  # 마지막 차트에만 컬러바
                    )
                    
                    axes[i].set_title(f'{model_name}\nF1: {metrics_df[metrics_df["model_name"]==model_name]["final_f1"].iloc[0]:.4f}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                
            except Exception as e:
                print(f"⚠️ 혼동 행렬 생성 실패 ({model_name}): {e}")
                axes[i].text(0.5, 0.5, f'No data\nfor {model_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        
        # 저장
        cm_file = self.output_dir / "confusion_matrix_comparison.png"
        plt.savefig(str(cm_file), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 혼동 행렬 비교 저장: {cm_file}")
        return str(cm_file)
    
    def create_class_performance_analysis(self, metrics_df: pd.DataFrame) -> str:
        """클래스별 성능 분석"""
        print("📋 클래스별 성능 분석 중...")
        
        # 최고 성능 모델 선택
        best_model = metrics_df.iloc[0]['model_name']
        best_results = self.results_data[best_model]
        
        if 'final_predictions' not in best_results or 'final_targets' not in best_results:
            print(f"⚠️ {best_model}에 예측 데이터가 없습니다.")
            return ""
        
        predictions = np.array(best_results['final_predictions'])
        targets = np.array(best_results['final_targets'])
        
        # 클래스별 메트릭 계산
        class_names = [f"Class_{i}" for i in range(17)]
        
        # 분류 리포트를 딕셔너리로 변환
        report = classification_report(
            targets, predictions, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # 클래스별 데이터 추출
        class_data = []
        for i, class_name in enumerate(class_names):
            if class_name in report:
                class_info = report[class_name]
                class_data.append({
                    'class_id': i,
                    'class_name': class_name,
                    'precision': class_info['precision'],
                    'recall': class_info['recall'],
                    'f1_score': class_info['f1-score'],
                    'support': class_info['support']
                })
        
        class_df = pd.DataFrame(class_data)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 클래스별 F1 점수
        axes[0, 0].bar(class_df['class_id'], class_df['f1_score'], color='skyblue')
        axes[0, 0].set_title('F1 Score by Class')
        axes[0, 0].set_xlabel('Class ID')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Precision vs Recall
        axes[0, 1].scatter(class_df['recall'], class_df['precision'], 
                          s=class_df['support']/10, alpha=0.6, color='red')
        axes[0, 1].set_title('Precision vs Recall (bubble size = support)')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        
        # 각 점에 클래스 ID 라벨 추가
        for i, row in class_df.iterrows():
            axes[0, 1].annotate(str(row['class_id']), 
                              (row['recall'], row['precision']),
                              xytext=(5, 5), textcoords='offset points')
        
        # 3. 클래스별 지원 수 (Support)
        axes[1, 0].bar(class_df['class_id'], class_df['support'], color='lightgreen')
        axes[1, 0].set_title('Support by Class')
        axes[1, 0].set_xlabel('Class ID')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 성능 히트맵
        metrics_matrix = class_df[['precision', 'recall', 'f1_score']].T
        metrics_matrix.columns = class_df['class_id']
        
        sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Performance Metrics Heatmap')
        axes[1, 1].set_xlabel('Class ID')
        
        plt.suptitle(f'Class-wise Performance Analysis - {best_model}', fontsize=16)
        plt.tight_layout()
        
        # 저장
        class_analysis_file = self.output_dir / "class_performance_analysis.png"
        plt.savefig(str(class_analysis_file), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 클래스별 데이터도 CSV로 저장
        class_csv_file = self.output_dir / "class_performance_data.csv"
        class_df.to_csv(str(class_csv_file), index=False)
        
        print(f"✅ 클래스별 성능 분석 저장: {class_analysis_file}")
        return str(class_analysis_file)
    
    def generate_comprehensive_report(self, metrics_df: pd.DataFrame) -> str:
        """종합 분석 리포트 생성"""
        print("📄 종합 분석 리포트 생성 중...")
        
        report_lines = []
        report_lines.append("# 🏆 Model Performance Analysis Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 1. 개요
        report_lines.append("## 📊 Overview")
        report_lines.append(f"- Total Models Analyzed: {len(metrics_df)}")
        report_lines.append(f"- Best F1 Score: {metrics_df['final_f1'].max():.4f}")
        report_lines.append(f"- Average F1 Score: {metrics_df['final_f1'].mean():.4f}")
        report_lines.append(f"- F1 Score Standard Deviation: {metrics_df['final_f1'].std():.4f}")
        report_lines.append("")
        
        # 2. 상위 모델들
        report_lines.append("## 🥇 Top Performing Models")
        top_models = metrics_df.head(5)
        
        for i, (_, model) in enumerate(top_models.iterrows(), 1):
            report_lines.append(f"### {i}. {model['model_name']}")
            report_lines.append(f"   - Architecture: {model['architecture']}")
            report_lines.append(f"   - F1 Score: {model['final_f1']:.4f}")
            report_lines.append(f"   - Accuracy: {model['final_accuracy']:.4f}")
            report_lines.append(f"   - Training Time: {model['training_time_minutes']:.1f} minutes")
            report_lines.append(f"   - Epochs: {model['epochs_trained']}")
            report_lines.append("")
        
        # 3. 아키텍처별 분석
        report_lines.append("## 🏗️ Architecture Analysis")
        arch_stats = metrics_df.groupby('architecture').agg({
            'final_f1': ['mean', 'std', 'max', 'count'],
            'training_time_minutes': 'mean'
        }).round(4)
        
        for arch in arch_stats.index:
            report_lines.append(f"### {arch}")
            report_lines.append(f"   - Models Count: {int(arch_stats.loc[arch, ('final_f1', 'count')])}")
            report_lines.append(f"   - Average F1: {arch_stats.loc[arch, ('final_f1', 'mean')]:.4f}")
            report_lines.append(f"   - Best F1: {arch_stats.loc[arch, ('final_f1', 'max')]:.4f}")
            report_lines.append(f"   - F1 Std: {arch_stats.loc[arch, ('final_f1', 'std')]:.4f}")
            report_lines.append(f"   - Avg Training Time: {arch_stats.loc[arch, ('training_time_minutes', 'mean')]:.1f} min")
            report_lines.append("")
        
        # 4. 성능 vs 효율성 분석
        report_lines.append("## ⚖️ Performance vs Efficiency Analysis")
        
        # 효율성 점수 계산 (F1 / 훈련시간)
        efficiency_scores = metrics_df['final_f1'] / (metrics_df['training_time_minutes'] + 1e-6)
        most_efficient_idx = efficiency_scores.idxmax()
        most_efficient = metrics_df.loc[most_efficient_idx]
        
        report_lines.append(f"### Most Efficient Model: {most_efficient['model_name']}")
        report_lines.append(f"   - F1 Score: {most_efficient['final_f1']:.4f}")
        report_lines.append(f"   - Training Time: {most_efficient['training_time_minutes']:.1f} minutes")
        report_lines.append(f"   - Efficiency Score: {efficiency_scores[most_efficient_idx]:.6f}")
        report_lines.append("")
        
        # 5. 통계적 유의성 분석
        if len(metrics_df) >= 3:
            report_lines.append("## 📈 Statistical Analysis")
            
            # 상위 3개 모델의 F1 점수
            top_3_f1 = metrics_df.head(3)['final_f1'].values
            
            # 일원분산분석 (ANOVA) - 아키텍처별
            if len(metrics_df['architecture'].unique()) > 1:
                arch_groups = [group['final_f1'].values for name, group in metrics_df.groupby('architecture')]
                try:
                    f_stat, p_value = stats.f_oneway(*arch_groups)
                    report_lines.append(f"   - ANOVA F-statistic: {f_stat:.4f}")
                    report_lines.append(f"   - P-value: {p_value:.4f}")
                    if p_value < 0.05:
                        report_lines.append("   - ✅ Significant difference between architectures")
                    else:
                        report_lines.append("   - ❌ No significant difference between architectures")
                except:
                    report_lines.append("   - ⚠️ Statistical test failed")
            report_lines.append("")
        
        # 6. 권장사항
        report_lines.append("## 💡 Recommendations")
        
        best_model = metrics_df.iloc[0]
        report_lines.append(f"1. **Production Model**: Use {best_model['model_name']} for best performance")
        report_lines.append(f"   - F1 Score: {best_model['final_f1']:.4f}")
        report_lines.append("")
        
        if len(metrics_df) >= 3:
            # 앙상블 권장
            top_3_models = metrics_df.head(3)['model_name'].tolist()
            report_lines.append("2. **Ensemble Strategy**: Consider ensemble of top 3 models")
            for i, model in enumerate(top_3_models, 1):
                f1 = metrics_df[metrics_df['model_name'] == model]['final_f1'].iloc[0]
                report_lines.append(f"   - Model {i}: {model} (F1: {f1:.4f})")
            report_lines.append("")
        
        # 효율성 기반 권장
        if most_efficient['model_name'] != best_model['model_name']:
            report_lines.append(f"3. **For Fast Inference**: Use {most_efficient['model_name']} for efficiency")
            report_lines.append(f"   - Good balance of performance ({most_efficient['final_f1']:.4f}) and speed")
            report_lines.append("")
        
        # 7. 결론
        report_lines.append("## 🎯 Conclusion")
        report_lines.append(f"The analysis of {len(metrics_df)} models shows that:")
        report_lines.append(f"- Best performing architecture: {best_model['architecture']}")
        report_lines.append(f"- Optimal image size: {best_model['image_size']}")
        report_lines.append(f"- Recommended loss function: {best_model['loss_type']}")
        report_lines.append("")
        report_lines.append("This comprehensive analysis provides insights for model selection and deployment strategies.")
        
        # 리포트 저장
        report_file = self.output_dir / "comprehensive_analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ 종합 분석 리포트 저장: {report_file}")
        return str(report_file)
    
    def run_full_analysis(self) -> Dict[str, str]:
        """전체 분석 실행"""
        print("🚀 전체 성능 분석 시작")
        print("=" * 60)
        
        # 1. 결과 로드
        loaded_count = self.discover_and_load_all_results()
        if loaded_count == 0:
            print("❌ 분석할 모델 결과가 없습니다.")
            return {}
        
        # 2. 성능 메트릭 계산
        metrics_df = self.calculate_performance_metrics()
        
        # 메트릭 CSV 저장
        metrics_file = self.output_dir / "performance_metrics.csv"
        metrics_df.to_csv(str(metrics_file), index=False)
        print(f"💾 성능 메트릭 저장: {metrics_file}")
        
        # 3. 시각화 생성
        chart_file = self.create_performance_comparison_chart(metrics_df)
        cm_file = self.create_confusion_matrix_comparison(metrics_df)
        class_file = self.create_class_performance_analysis(metrics_df)
        
        # 4. 종합 리포트 생성
        report_file = self.generate_comprehensive_report(metrics_df)
        
        print("\n🎉 전체 분석 완료!")
        print(f"   분석된 모델: {len(metrics_df)}개")
        print(f"   최고 F1 점수: {metrics_df['final_f1'].max():.4f}")
        print(f"   결과 디렉토리: {self.output_dir}")
        
        return {
            'metrics_file': str(metrics_file),
            'chart_file': chart_file,
            'confusion_matrix_file': cm_file,
            'class_analysis_file': class_file,
            'report_file': report_file,
            'output_directory': str(self.output_dir)
        }


def main():
    """메인 함수"""
    print("📊 Performance Analysis System")
    
    # 워크스페이스 경로
    workspace_root = Path(__file__).parent.parent
    
    # 분석기 생성 및 실행
    analyzer = PerformanceAnalyzer(str(workspace_root))
    results = analyzer.run_full_analysis()
    
    if results:
        print("\n📋 생성된 파일들:")
        for key, file_path in results.items():
            if key != 'output_directory':
                print(f"   - {key}: {file_path}")


if __name__ == "__main__":
    main()

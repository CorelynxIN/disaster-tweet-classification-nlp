"""
재해 트윗 분류를 위한 유틸리티 함수 모듈
Utility Functions Module for Disaster Tweet Classification

이 모듈은 데이터 시각화, 성능 평가, 결과 분석을 위한 유틸리티 함수들을 제공합니다.
This module provides utility functions for data visualization, performance evaluation, and result analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 / Korean font setup
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

def set_plot_style():
    """
    플롯 스타일 설정 / Set plot style
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

def plot_data_distribution(df: pd.DataFrame, target_column: str = 'target', 
                          title: str = "데이터 분포 / Data Distribution"):
    """
    타겟 변수의 분포 시각화 / Visualize target variable distribution
    
    Args:
        df: 데이터프레임 / DataFrame
        target_column: 타겟 컬럼명 / Target column name
        title: 그래프 제목 / Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 카운트 플롯 / Count plot
    sns.countplot(data=df, x=target_column, ax=axes[0])
    axes[0].set_title(f'{title} - 카운트 / Count')
    axes[0].set_xlabel('클래스 / Class')
    axes[0].set_ylabel('개수 / Count')
    
    # 파이 차트 / Pie chart
    value_counts = df[target_column].value_counts()
    axes[1].pie(value_counts.values, labels=['비재해 / Non-disaster', '재해 / Disaster'], 
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'{title} - 비율 / Proportion')
    
    plt.tight_layout()
    plt.show()
    
    # 통계 정보 출력 / Print statistics
    print(f"총 데이터 개수 / Total samples: {len(df)}")
    print(f"클래스 분포 / Class distribution:")
    print(value_counts)
    print(f"클래스 비율 / Class proportion:")
    print(value_counts / len(df))

def plot_text_statistics(df: pd.DataFrame, text_column: str = 'text'):
    """
    텍스트 통계 시각화 / Visualize text statistics
    
    Args:
        df: 데이터프레임 / DataFrame
        text_column: 텍스트 컬럼명 / Text column name
    """
    # 텍스트 길이 계산 / Calculate text lengths
    df_copy = df.copy()
    df_copy['text_length'] = df_copy[text_column].str.len()
    df_copy['word_count'] = df_copy[text_column].str.split().str.len()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 텍스트 길이 분포 / Text length distribution
    sns.histplot(data=df_copy, x='text_length', bins=50, ax=axes[0, 0])
    axes[0, 0].set_title('텍스트 길이 분포 / Text Length Distribution')
    axes[0, 0].set_xlabel('텍스트 길이 / Text Length')
    axes[0, 0].set_ylabel('빈도 / Frequency')
    
    # 단어 수 분포 / Word count distribution
    sns.histplot(data=df_copy, x='word_count', bins=50, ax=axes[0, 1])
    axes[0, 1].set_title('단어 수 분포 / Word Count Distribution')
    axes[0, 1].set_xlabel('단어 수 / Word Count')
    axes[0, 1].set_ylabel('빈도 / Frequency')
    
    # 타겟별 텍스트 길이 / Text length by target
    if 'target' in df_copy.columns:
        sns.boxplot(data=df_copy, x='target', y='text_length', ax=axes[1, 0])
        axes[1, 0].set_title('타겟별 텍스트 길이 / Text Length by Target')
        axes[1, 0].set_xlabel('타겟 / Target')
        axes[1, 0].set_ylabel('텍스트 길이 / Text Length')
        
        # 타겟별 단어 수 / Word count by target
        sns.boxplot(data=df_copy, x='target', y='word_count', ax=axes[1, 1])
        axes[1, 1].set_title('타겟별 단어 수 / Word Count by Target')
        axes[1, 1].set_xlabel('타겟 / Target')
        axes[1, 1].set_ylabel('단어 수 / Word Count')
    
    plt.tight_layout()
    plt.show()
    
    # 통계 정보 출력 / Print statistics
    print("텍스트 통계 / Text Statistics:")
    print(f"평균 텍스트 길이 / Average text length: {df_copy['text_length'].mean():.2f}")
    print(f"평균 단어 수 / Average word count: {df_copy['word_count'].mean():.2f}")
    print(f"최대 텍스트 길이 / Max text length: {df_copy['text_length'].max()}")
    print(f"최대 단어 수 / Max word count: {df_copy['word_count'].max()}")

def create_wordcloud(texts: list, title: str = "워드클라우드 / Word Cloud", 
                     max_words: int = 200, width: int = 800, height: int = 400):
    """
    워드클라우드 생성 / Create word cloud
    
    Args:
        texts: 텍스트 리스트 / List of texts
        title: 제목 / Title
        max_words: 최대 단어 수 / Maximum number of words
        width: 너비 / Width
        height: 높이 / Height
    """
    # 모든 텍스트 결합 / Combine all texts
    all_text = ' '.join(texts)
    
    # 워드클라우드 생성 / Create word cloud
    wordcloud = WordCloud(
        width=width, 
        height=height,
        max_words=max_words,
        background_color='white',
        colormap='viridis'
    ).generate(all_text)
    
    # 시각화 / Visualization
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels: list = None, 
                         title: str = "혼동 행렬 / Confusion Matrix"):
    """
    혼동 행렬 시각화 / Visualize confusion matrix
    
    Args:
        y_true: 실제 레이블 / True labels
        y_pred: 예측 레이블 / Predicted labels
        labels: 클래스 레이블 / Class labels
        title: 제목 / Title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ['비재해 / Non-disaster', '재해 / Disaster']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('예측값 / Predicted')
    plt.ylabel('실제값 / Actual')
    plt.tight_layout()
    plt.show()
    
    # 성능 지표 출력 / Print performance metrics
    print(f"\n분류 리포트 / Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

def plot_roc_curve(y_true, y_pred_proba, title: str = "ROC 곡선 / ROC Curve"):
    """
    ROC 곡선 시각화 / Visualize ROC curve
    
    Args:
        y_true: 실제 레이블 / True labels
        y_pred_proba: 예측 확률 / Predicted probabilities
        title: 제목 / Title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('거짓 양성 비율 / False Positive Rate')
    plt.ylabel('참 양성 비율 / True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, 
                               title: str = "정밀도-재현율 곡선 / Precision-Recall Curve"):
    """
    정밀도-재현율 곡선 시각화 / Visualize precision-recall curve
    
    Args:
        y_true: 실제 레이블 / True labels
        y_pred_proba: 예측 확률 / Predicted probabilities
        title: 제목 / Title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('재현율 / Recall')
    plt.ylabel('정밀도 / Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_learning_curve(estimator, X, y, title: str = "학습 곡선 / Learning Curve",
                       cv: int = 5, n_jobs: int = -1, train_sizes=None):
    """
    학습 곡선 시각화 / Visualize learning curve
    
    Args:
        estimator: 모델 / Model
        X: 특성 / Features
        y: 레이블 / Labels
        title: 제목 / Title
        cv: 교차검증 폴드 수 / Number of CV folds
        n_jobs: 병렬 작업 수 / Number of parallel jobs
        train_sizes: 훈련 크기 / Training sizes
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='red', 
             label='훈련 점수 / Training score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='red')
    
    plt.plot(train_sizes, val_scores_mean, 'o-', color='green',
             label='검증 점수 / Validation score')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color='green')
    
    plt.xlabel('훈련 세트 크기 / Training Set Size')
    plt.ylabel('정확도 점수 / Accuracy Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_model_performance(results_dict: dict, 
                            title: str = "모델 성능 비교 / Model Performance Comparison"):
    """
    여러 모델의 성능 비교 시각화 / Visualize performance comparison of multiple models
    
    Args:
        results_dict: 모델별 결과 딕셔너리 / Dictionary of results for each model
        title: 제목 / Title
    """
    # 데이터프레임으로 변환 / Convert to DataFrame
    df_results = pd.DataFrame(results_dict).T
    
    # 성능 지표 리스트 / Performance metrics list
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    if 'auc' in df_results.columns:
        metrics.append('auc')
    
    # 서브플롯 생성 / Create subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    
    for i, metric in enumerate(metrics):
        if metric in df_results.columns:
            ax = axes[i] if len(metrics) > 1 else axes
            bars = ax.bar(df_results.index, df_results[metric])
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel('점수 / Score')
            ax.tick_params(axis='x', rotation=45)
            
            # 막대 위에 값 표시 / Show values on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 결과 테이블 출력 / Print results table
    print("\n모델 성능 비교 테이블 / Model Performance Comparison Table:")
    print(df_results.round(4))

def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                           title: str = "특성 중요도 / Feature Importance"):
    """
    특성 중요도 시각화 / Visualize feature importance
    
    Args:
        model: 훈련된 모델 / Trained model
        feature_names: 특성명 리스트 / List of feature names
        top_n: 상위 n개 특성 / Top n features
        title: 제목 / Title
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("모델이 특성 중요도를 지원하지 않습니다 / Model doesn't support feature importance")
        return
    
    # 특성 중요도 정렬 / Sort feature importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('중요도 / Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def create_interactive_dashboard(df: pd.DataFrame, predictions: np.array = None):
    """
    인터랙티브 대시보드 생성 / Create interactive dashboard
    
    Args:
        df: 데이터프레임 / DataFrame
        predictions: 예측 결과 / Predictions
    """
    # 서브플롯 생성 / Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('데이터 분포 / Data Distribution', 
                       '텍스트 길이 분포 / Text Length Distribution',
                       '단어 수 분포 / Word Count Distribution',
                       '모델 성능 / Model Performance'),
        specs=[[{"type": "pie"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # 파이 차트 / Pie chart
    if 'target' in df.columns:
        target_counts = df['target'].value_counts()
        fig.add_trace(
            go.Pie(labels=['비재해', '재해'], values=target_counts.values),
            row=1, col=1
        )
    
    # 텍스트 길이 히스토그램 / Text length histogram
    text_lengths = df['text'].str.len()
    fig.add_trace(
        go.Histogram(x=text_lengths, name='텍스트 길이'),
        row=1, col=2
    )
    
    # 단어 수 히스토그램 / Word count histogram
    word_counts = df['text'].str.split().str.len()
    fig.add_trace(
        go.Histogram(x=word_counts, name='단어 수'),
        row=2, col=1
    )
    
    # 모델 성능 바 차트 / Model performance bar chart
    if predictions is not None and 'target' in df.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(df['target'], predictions)
        precision = precision_score(df['target'], predictions)
        recall = recall_score(df['target'], predictions)
        f1 = f1_score(df['target'], predictions)
        
        fig.add_trace(
            go.Bar(x=['정확도', '정밀도', '재현율', 'F1'], 
                   y=[accuracy, precision, recall, f1]),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, 
                      title_text="재해 트윗 분류 대시보드 / Disaster Tweet Classification Dashboard")
    fig.show()

def save_results(results_dict: dict, filepath: str):
    """
    결과를 파일로 저장 / Save results to file
    
    Args:
        results_dict: 결과 딕셔너리 / Results dictionary
        filepath: 저장 경로 / File path
    """
    df_results = pd.DataFrame(results_dict).T
    df_results.to_csv(filepath, index=True)
    print(f"결과가 저장되었습니다 / Results saved: {filepath}")

if __name__ == "__main__":
    # 테스트 코드 / Test code
    print("유틸리티 함수 모듈 테스트 / Testing utility functions module")
    
    # 가상 데이터 생성 / Generate dummy data
    np.random.seed(42)
    sample_data = {
        'text': ['This is a disaster tweet about earthquake', 
                'Just had a great day at the beach',
                'Emergency evacuation due to flood',
                'Love this new movie so much'],
        'target': [1, 0, 1, 0]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # 플롯 스타일 설정 / Set plot style
    set_plot_style()
    
    # 데이터 분포 시각화 / Visualize data distribution
    plot_data_distribution(df_sample)
    
    print("유틸리티 함수 테스트 완료 / Utility functions test completed") 
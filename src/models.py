"""
재해 트윗 분류를 위한 머신러닝 모델 모듈
Machine Learning Models Module for Disaster Tweet Classification

이 모듈은 전통적인 머신러닝 모델부터 딥러닝 모델까지 다양한 분류 모델을 제공합니다.
This module provides various classification models from traditional ML to deep learning models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import warnings
warnings.filterwarnings('ignore')

class TraditionalMLModels:
    """
    전통적인 머신러닝 모델들을 위한 클래스
    Class for traditional machine learning models
    """
    
    def __init__(self):
        """
        모델 초기화 / Initialize models
        """
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        
        self.best_models = {}
        self.best_params = {}
    
    def get_param_grids(self):
        """
        하이퍼파라미터 그리드 반환 / Return hyperparameter grids
        
        Returns:
            각 모델별 하이퍼파라미터 그리드 딕셔너리 / Dictionary of hyperparameter grids for each model
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
        return param_grids
    
    def train_model(self, model_name, X_train, y_train):
        """개별 모델 훈련"""
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """모델 평가"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

class DeepLearningModels:
    """
    딥러닝 모델들을 위한 클래스
    Class for deep learning models
    """
    
    def __init__(self, max_features: int = 10000, max_length: int = 100):
        """
        딥러닝 모델 초기화 / Initialize deep learning models
        
        Args:
            max_features: 최대 단어 수 / Maximum number of words
            max_length: 최대 시퀀스 길이 / Maximum sequence length
        """
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = None
        self.models = {}
    
    def prepare_text_data(self, texts: list, is_training: bool = True):
        """
        텍스트 데이터를 딥러닝 모델용으로 준비 / Prepare text data for deep learning models
        
        Args:
            texts: 텍스트 리스트 / List of texts
            is_training: 훈련 데이터 여부 / Whether it's training data
            
        Returns:
            패딩된 시퀀스 / Padded sequences
        """
        if is_training:
            # 토크나이저 초기화 및 학습 / Initialize and fit tokenizer
            self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        # 텍스트를 시퀀스로 변환 / Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # 패딩 적용 / Apply padding
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        return padded_sequences
    
    def create_lstm_model(self, embedding_dim: int = 100, lstm_units: int = 64, dropout_rate: float = 0.5):
        """
        LSTM 모델 생성 / Create LSTM model
        
        Args:
            embedding_dim: 임베딩 차원 / Embedding dimension
            lstm_units: LSTM 유닛 수 / Number of LSTM units
            dropout_rate: 드롭아웃 비율 / Dropout rate
            
        Returns:
            컴파일된 LSTM 모델 / Compiled LSTM model
        """
        model = Sequential([
            Embedding(self.max_features, embedding_dim, input_length=self.max_length),
            LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_bidirectional_lstm_model(self, embedding_dim: int = 100, lstm_units: int = 64, dropout_rate: float = 0.5):
        """
        양방향 LSTM 모델 생성 / Create Bidirectional LSTM model
        
        Args:
            embedding_dim: 임베딩 차원 / Embedding dimension
            lstm_units: LSTM 유닛 수 / Number of LSTM units
            dropout_rate: 드롭아웃 비율 / Dropout rate
            
        Returns:
            컴파일된 양방향 LSTM 모델 / Compiled Bidirectional LSTM model
        """
        model = Sequential([
            Embedding(self.max_features, embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)),
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_model(self, embedding_dim: int = 100, filters: int = 128, kernel_size: int = 3, dropout_rate: float = 0.5):
        """
        CNN 모델 생성 / Create CNN model
        
        Args:
            embedding_dim: 임베딩 차원 / Embedding dimension
            filters: 필터 수 / Number of filters
            kernel_size: 커널 크기 / Kernel size
            dropout_rate: 드롭아웃 비율 / Dropout rate
            
        Returns:
            컴파일된 CNN 모델 / Compiled CNN model
        """
        model = Sequential([
            Embedding(self.max_features, embedding_dim, input_length=self.max_length),
            Conv1D(filters, kernel_size, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters//2, kernel_size, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_hybrid_model(self, embedding_dim: int = 100, lstm_units: int = 64, 
                           filters: int = 64, dropout_rate: float = 0.5):
        """
        CNN + LSTM 하이브리드 모델 생성 / Create CNN + LSTM hybrid model
        
        Args:
            embedding_dim: 임베딩 차원 / Embedding dimension
            lstm_units: LSTM 유닛 수 / Number of LSTM units
            filters: CNN 필터 수 / Number of CNN filters
            dropout_rate: 드롭아웃 비율 / Dropout rate
            
        Returns:
            컴파일된 하이브리드 모델 / Compiled hybrid model
        """
        # 입력 레이어 / Input layer
        input_layer = Input(shape=(self.max_length,))
        embedding = Embedding(self.max_features, embedding_dim)(input_layer)
        
        # CNN 브랜치 / CNN branch
        cnn_branch = Conv1D(filters, 3, activation='relu')(embedding)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Conv1D(filters//2, 3, activation='relu')(cnn_branch)
        cnn_branch = GlobalMaxPooling1D()(cnn_branch)
        
        # LSTM 브랜치 / LSTM branch
        lstm_branch = LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)(embedding)
        
        # 브랜치 결합 / Combine branches
        combined = Concatenate()([cnn_branch, lstm_branch])
        
        # 출력 레이어 / Output layers
        dense = Dense(64, activation='relu')(combined)
        dropout = Dropout(dropout_rate)(dense)
        output = Dense(1, activation='sigmoid')(dropout)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val=None, y_val=None, 
                   epochs: int = 50, batch_size: int = 32, verbose: int = 1):
        """
        딥러닝 모델 훈련 / Train deep learning model
        
        Args:
            model: 훈련할 모델 / Model to train
            X_train: 훈련 특성 / Training features
            y_train: 훈련 레이블 / Training labels
            X_val: 검증 특성 / Validation features
            y_val: 검증 레이블 / Validation labels
            epochs: 에포크 수 / Number of epochs
            batch_size: 배치 크기 / Batch size
            verbose: 출력 레벨 / Verbosity level
            
        Returns:
            훈련 히스토리 / Training history
        """
        # 콜백 설정 / Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        ]
        
        # 검증 데이터 설정 / Setup validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # 모델 훈련 / Train model
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, model_name: str = "Deep Learning Model"):
        """
        딥러닝 모델 평가 / Evaluate deep learning model
        
        Args:
            model: 평가할 모델 / Model to evaluate
            X_test: 테스트 특성 / Test features
            y_test: 테스트 레이블 / Test labels
            model_name: 모델명 / Model name
            
        Returns:
            평가 결과 딕셔너리 / Dictionary of evaluation results
        """
        # 예측 수행 / Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 성능 지표 계산 / Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        # 결과 출력 / Print results
        print(f"\n=== {model_name} 평가 결과 / Evaluation Results ===")
        print(f"정확도 / Accuracy: {accuracy:.4f}")
        print(f"정밀도 / Precision: {precision:.4f}")
        print(f"재현율 / Recall: {recall:.4f}")
        print(f"F1 점수 / F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return results

def save_model(model, filepath: str, model_type: str = 'sklearn'):
    """
    모델 저장 / Save model
    
    Args:
        model: 저장할 모델 / Model to save
        filepath: 저장 경로 / File path
        model_type: 모델 타입 ('sklearn' 또는 'keras') / Model type ('sklearn' or 'keras')
    """
    if model_type == 'sklearn':
        joblib.dump(model, filepath)
    elif model_type == 'keras':
        model.save(filepath)
    else:
        raise ValueError("model_type must be 'sklearn' or 'keras'")
    
    print(f"모델이 저장되었습니다 / Model saved: {filepath}")

def load_model(filepath: str, model_type: str = 'sklearn'):
    """
    모델 로드 / Load model
    
    Args:
        filepath: 모델 파일 경로 / Model file path
        model_type: 모델 타입 ('sklearn' 또는 'keras') / Model type ('sklearn' or 'keras')
        
    Returns:
        로드된 모델 / Loaded model
    """
    if model_type == 'sklearn':
        model = joblib.load(filepath)
    elif model_type == 'keras':
        model = tf.keras.models.load_model(filepath)
    else:
        raise ValueError("model_type must be 'sklearn' or 'keras'")
    
    print(f"모델이 로드되었습니다 / Model loaded: {filepath}")
    return model

if __name__ == "__main__":
    # 테스트 코드 / Test code
    print("재해 트윗 분류 모델 모듈 테스트 / Testing disaster tweet classification models module")
    
    # 가상 데이터 생성 / Generate dummy data
    np.random.seed(42)
    X_train = np.random.rand(1000, 100)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.rand(200, 100)
    y_test = np.random.randint(0, 2, 200)
    
    # 전통적인 ML 모델 테스트 / Test traditional ML models
    print("\n전통적인 ML 모델 테스트 / Testing traditional ML models")
    ml_models = TraditionalMLModels()
    
    # 로지스틱 회귀 모델 훈련 및 평가 / Train and evaluate logistic regression
    lr_model = ml_models.train_model('logistic_regression', X_train, y_train)
    ml_models.evaluate_model(lr_model, X_test, y_test) 
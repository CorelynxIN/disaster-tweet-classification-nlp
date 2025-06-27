"""
재해 트윗 분류를 위한 텍스트 전처리 모듈
Text Preprocessing Module for Disaster Tweet Classification

이 모듈은 트윗 텍스트를 정제하고 머신러닝 모델에 적합한 형태로 변환하는 함수들을 제공합니다.
This module provides functions to clean tweet text and transform it into a format suitable for machine learning models.
"""

import re
import string
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
from typing import List, Union
import warnings
warnings.filterwarnings('ignore')

# NLTK 데이터 다운로드 / Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

class TextPreprocessor:
    """
    텍스트 전처리를 위한 클래스
    Class for text preprocessing
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 stem: bool = False,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 lowercase: bool = True):
        """
        전처리 옵션 초기화 / Initialize preprocessing options
        
        Args:
            remove_stopwords: 불용어 제거 여부 / Whether to remove stopwords
            lemmatize: 표제어 추출 여부 / Whether to lemmatize words
            stem: 어간 추출 여부 / Whether to stem words
            remove_punctuation: 구두점 제거 여부 / Whether to remove punctuation
            remove_numbers: 숫자 제거 여부 / Whether to remove numbers
            lowercase: 소문자 변환 여부 / Whether to convert to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        
        # NLTK 도구 초기화 / Initialize NLTK tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        if self.stem:
            self.stemmer = PorterStemmer()
    
    def clean_url(self, text: str) -> str:
        """
        URL 링크 제거 / Remove URL links
        
        Args:
            text: 입력 텍스트 / Input text
            
        Returns:
            정제된 텍스트 / Cleaned text
        """
        # HTTP/HTTPS URL 제거 / Remove HTTP/HTTPS URLs
        text = re.sub(r'http\S+|https\S+|www\.\S+', '', text)
        return text
    
    def clean_mentions_hashtags(self, text: str) -> str:
        """
        멘션(@)과 해시태그(#) 정제 / Clean mentions and hashtags
        
        Args:
            text: 입력 텍스트 / Input text
            
        Returns:
            정제된 텍스트 / Cleaned text
        """
        # @ 멘션 제거 / Remove @ mentions
        text = re.sub(r'@\w+', '', text)
        # # from 해시태그 제거하되 단어는 유지 / Remove # from hashtags but keep words
        text = re.sub(r'#(\w+)', r'\1', text)
        return text
    
    def clean_special_chars(self, text: str) -> str:
        """
        특수 문자 정제 / Clean special characters
        
        Args:
            text: 입력 텍스트 / Input text
            
        Returns:
            정제된 텍스트 / Cleaned text
        """
        # HTML 태그 제거 / Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 연속된 공백을 하나로 / Replace consecutive spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # 이모지를 텍스트로 변환 / Convert emojis to text
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        return text.strip()
    
    def remove_noise(self, text: str) -> str:
        """
        노이즈 제거 (반복 문자, 특수 기호 등) / Remove noise (repeated chars, special symbols)
        
        Args:
            text: 입력 텍스트 / Input text
            
        Returns:
            정제된 텍스트 / Cleaned text
        """
        # 연속된 같은 문자 3개 이상을 2개로 줄임 / Reduce 3+ consecutive chars to 2
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 특수 기호 제거 (구두점 제외) / Remove special symbols (except punctuation)
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\'\"]', '', text)
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        전체 전처리 파이프라인 / Complete preprocessing pipeline
        
        Args:
            text: 입력 텍스트 / Input text
            
        Returns:
            전처리된 텍스트 / Preprocessed text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # 기본 정제 / Basic cleaning
        text = self.clean_url(text)
        text = self.clean_mentions_hashtags(text)
        text = self.clean_special_chars(text)
        text = self.remove_noise(text)
        
        # 소문자 변환 / Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # 토큰화 / Tokenization
        tokens = word_tokenize(text)
        
        # 숫자 제거 / Remove numbers
        if self.remove_numbers:
            tokens = [token for token in tokens if not token.isdigit()]
        
        # 구두점 제거 / Remove punctuation
        if self.remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]
        
        # 불용어 제거 / Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # 표제어 추출 / Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # 어간 추출 / Stemming
        if self.stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # 빈 토큰 제거 / Remove empty tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        데이터프레임의 텍스트 컬럼 전처리 / Preprocess text column in dataframe
        
        Args:
            df: 입력 데이터프레임 / Input dataframe
            text_column: 텍스트 컬럼명 / Text column name
            
        Returns:
            전처리된 데이터프레임 / Preprocessed dataframe
        """
        df_copy = df.copy()
        df_copy[f'{text_column}_clean'] = df_copy[text_column].apply(self.preprocess_text)
        return df_copy

def extract_text_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    텍스트에서 추가 특성 추출 / Extract additional features from text
    
    Args:
        df: 입력 데이터프레임 / Input dataframe
        text_column: 텍스트 컬럼명 / Text column name
        
    Returns:
        특성이 추가된 데이터프레임 / Dataframe with additional features
    """
    df_copy = df.copy()
    
    # 텍스트 길이 / Text length
    df_copy['text_length'] = df_copy[text_column].str.len()
    
    # 단어 수 / Word count
    df_copy['word_count'] = df_copy[text_column].str.split().str.len()
    
    # 문장 수 / Sentence count
    df_copy['sentence_count'] = df_copy[text_column].str.count(r'[.!?]+') + 1
    
    # 대문자 비율 / Uppercase ratio
    df_copy['uppercase_ratio'] = df_copy[text_column].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    
    # 구두점 수 / Punctuation count
    df_copy['punctuation_count'] = df_copy[text_column].str.count(r'[^\w\s]')
    
    # 해시태그 수 / Hashtag count
    df_copy['hashtag_count'] = df_copy[text_column].str.count(r'#\w+')
    
    # 멘션 수 / Mention count
    df_copy['mention_count'] = df_copy[text_column].str.count(r'@\w+')
    
    # URL 수 / URL count
    df_copy['url_count'] = df_copy[text_column].str.count(r'http\S+|https\S+|www\.\S+')
    
    # 평균 단어 길이 / Average word length
    df_copy['avg_word_length'] = df_copy[text_column].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
    )
    
    return df_copy

def create_tfidf_features(train_texts: List[str], 
                         test_texts: List[str] = None,
                         max_features: int = 10000,
                         ngram_range: tuple = (1, 2),
                         min_df: int = 2,
                         max_df: float = 0.95) -> tuple:
    """
    TF-IDF 벡터화 수행 / Perform TF-IDF vectorization
    
    Args:
        train_texts: 훈련 텍스트 리스트 / Training text list
        test_texts: 테스트 텍스트 리스트 / Test text list
        max_features: 최대 특성 수 / Maximum number of features
        ngram_range: N-gram 범위 / N-gram range
        min_df: 최소 문서 빈도 / Minimum document frequency
        max_df: 최대 문서 빈도 / Maximum document frequency
        
    Returns:
        (훈련 특성, 테스트 특성, 벡터라이저) / (train features, test features, vectorizer)
    """
    # TF-IDF 벡터라이저 초기화 / Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'
    )
    
    # 훈련 데이터로 벡터라이저 학습 및 변환 / Fit and transform training data
    train_features = vectorizer.fit_transform(train_texts)
    
    # 테스트 데이터 변환 (있는 경우) / Transform test data (if provided)
    test_features = None
    if test_texts is not None:
        test_features = vectorizer.transform(test_texts)
    
    return train_features, test_features, vectorizer

def preprocess_disaster_tweets(df: pd.DataFrame, 
                              text_column: str = 'text',
                              is_training: bool = True) -> pd.DataFrame:
    """
    재해 트윗 데이터셋 전용 전처리 함수 / Preprocessing function specific for disaster tweet dataset
    
    Args:
        df: 입력 데이터프레임 / Input dataframe
        text_column: 텍스트 컬럼명 / Text column name
        is_training: 훈련 데이터 여부 / Whether it's training data
        
    Returns:
        전처리된 데이터프레임 / Preprocessed dataframe
    """
    # 전처리 객체 생성 / Create preprocessor object
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        stem=False,
        remove_punctuation=True,
        remove_numbers=False,
        lowercase=True
    )
    
    # 텍스트 전처리 / Preprocess text
    df_processed = preprocessor.preprocess_dataframe(df, text_column)
    
    # 추가 특성 추출 / Extract additional features
    df_processed = extract_text_features(df_processed, text_column)
    
    # 키워드와 위치 정보 처리 / Process keyword and location info
    if 'keyword' in df_processed.columns:
        df_processed['has_keyword'] = df_processed['keyword'].notna().astype(int)
        df_processed['keyword_clean'] = df_processed['keyword'].fillna('').str.lower()
    
    if 'location' in df_processed.columns:
        df_processed['has_location'] = df_processed['location'].notna().astype(int)
        df_processed['location_clean'] = df_processed['location'].fillna('').str.lower()
    
    return df_processed

if __name__ == "__main__":
    # 테스트 코드 / Test code
    sample_tweets = [
        "URGENT: Forest fire near residential area! Evacuate immediately! #emergency #fire",
        "I'm literally on fire today! Crushed my presentation 🔥 #success #motivated",
        "Breaking: Earthquake hits the city center. Magnitude 6.2 recorded.",
        "This traffic jam is killing me... stuck for 2 hours already"
    ]
    
    # 전처리 테스트 / Test preprocessing
    preprocessor = TextPreprocessor()
    
    print("원본 트윗들 / Original tweets:")
    for i, tweet in enumerate(sample_tweets, 1):
        print(f"{i}. {tweet}")
    
    print("\n전처리된 트윗들 / Preprocessed tweets:")
    for i, tweet in enumerate(sample_tweets, 1):
        cleaned = preprocessor.preprocess_text(tweet)
        print(f"{i}. {cleaned}") 
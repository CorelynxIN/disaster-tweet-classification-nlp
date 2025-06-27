# 재해 트윗 분류 NLP 프로젝트 / Disaster Tweet Classification with NLP and Machine Learning

🚨 **자연어 처리를 활용한 재해 트윗 분류 - 캐글 경진대회 프로젝트**  
🚨 **Natural Language Processing with Disaster Tweets - Kaggle Competition Project**

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 프로젝트 개요 / Project Overview

이 프로젝트는 **"Natural Language Processing with Disaster Tweets"** 캐글 경진대회를 다루며, 트윗을 재해 관련 또는 비재해 관련으로 분류하는 머신러닝 모델을 구축하는 것을 목표로 합니다. 응급 대응 조직이 소셜 미디어 노이즈에서 실제 재해 상황을 자동으로 식별할 수 있도록 돕는 것이 목적입니다.

This project tackles the **"Natural Language Processing with Disaster Tweets"** Kaggle competition, which involves building a machine learning model to classify tweets as either disaster-related or non-disaster related. The goal is to help emergency response organizations automatically identify real disaster situations from social media noise.

---

## 🎯 목표 / Objective

다음과 같이 트윗을 정확하게 분류할 수 있는 머신러닝 모델 구축:
Build a machine learning model that can accurately classify tweets as:

- **1**: 실제 재해 트윗 (실제 재해를 설명하는 트윗) / Real disaster tweets (describing actual disasters)
- **0**: 비재해 트윗 (은유적이거나 재해와 관련 없는 트윗) / Non-disaster tweets (metaphorical or non-disaster related)

---

## 📊 데이터셋 정보 / Dataset Information

- **훈련 세트 / Training set**: 7,613개의 레이블이 있는 트윗 / 7,613 tweets with labels
- **테스트 세트 / Test set**: 예측을 위한 3,263개의 트윗 / 3,263 tweets for prediction

### 특성 / Features:
- **id**: 고유 식별자 / Unique identifier
- **keyword**: 재해 관련 키워드 (비어있을 수 있음) / Disaster-related keyword (may be blank)
- **location**: 트윗이 전송된 위치 (비어있을 수 있음) / Location where tweet was sent (may be blank)
- **text**: 실제 트윗 텍스트 / The actual tweet text
- **target**: 이진 레이블 (1=재해, 0=비재해) - 훈련 세트에만 존재 / Binary label (1=disaster, 0=non-disaster) - only in training set

---

## 🏗️ 프로젝트 구조 / Project Structure

```
disaster-tweet-classification-nlp/
│
├── notebooks/
│   └── disaster_tweet_classification.ipynb    # 메인 분석 노트북 / Main analysis notebook
│
├── data/
│   ├── train.csv                              # 훈련 데이터셋 / Training dataset
│   ├── test.csv                               # 테스트 데이터셋 / Test dataset
│   └── sample_submission.csv                  # 제출 형식 / Submission format
│
├── models/
│   ├── best_logistic_regression.pkl           # 훈련된 로지스틱 회귀 모델 / Trained logistic regression model
│   ├── best_lstm_model.h5                     # 훈련된 LSTM 모델 / Trained LSTM model
│   ├── tfidf_vectorizer.pkl                   # TF-IDF 벡터라이저 / TF-IDF vectorizer
│   └── tokenizer.pkl                          # 딥러닝용 텍스트 토크나이저 / Text tokenizer for deep learning
│
├── results/
│   ├── submission.csv                         # 최종 예측 결과 / Final predictions
│   └── model_comparison.png                   # 성능 시각화 / Performance visualization
│
├── src/
│   ├── preprocessing.py                       # 텍스트 전처리 함수 / Text preprocessing functions
│   ├── models.py                              # 모델 아키텍처 / Model architectures
│   └── utils.py                               # 유틸리티 함수 / Utility functions
│
├── requirements.txt                           # 프로젝트 의존성 / Project dependencies
├── README.md                                  # 이 파일 / This file
└── .gitignore                                 # Git 무시 파일 / Git ignore file
```

---

## 🛠️ 사용 기술 / Technologies Used

### 머신러닝 & NLP / Machine Learning & NLP
- **Scikit-learn**: 전통적인 ML 알고리즘 및 TF-IDF 벡터화 / Traditional ML algorithms and TF-IDF vectorization
- **TensorFlow/Keras**: 딥러닝 모델 (LSTM, 양방향 LSTM) / Deep learning models (LSTM, Bidirectional LSTM)
- **NLTK**: 자연어 처리 및 텍스트 전처리 / Natural language processing and text preprocessing
- **Pandas & NumPy**: 데이터 조작 및 분석 / Data manipulation and analysis

### 시각화 / Visualization
- **Matplotlib & Seaborn**: 데이터 시각화 및 플롯팅 / Data visualization and plotting
- **WordCloud**: 텍스트 시각화 / Text visualization

### 구현된 모델 / Models Implemented

#### 전통적인 ML 모델 / Traditional ML Models:
- ✅ **로지스틱 회귀 (하이퍼파라미터 튜닝 포함)** / Logistic Regression (with hyperparameter tuning)
- ✅ **나이브 베이즈** / Naive Bayes
- ✅ **랜덤 포레스트** / Random Forest
- ✅ **그래디언트 부스팅** / Gradient Boosting
- ✅ **서포트 벡터 머신** / Support Vector Machine

#### 딥러닝 모델 / Deep Learning Models:
- ✅ **LSTM (Long Short-Term Memory)**
- ✅ **양방향 LSTM** / Bidirectional LSTM
- ✅ **CNN (Convolutional Neural Network)**
- ✅ **커스텀 신경망 아키텍처** / Custom neural network architectures

---

## 📈 예상 결과 / Expected Results

| 모델 / Model | 훈련 정확도 / Training Accuracy | 검증 정확도 / Validation Accuracy |
|---|---|---|
| 로지스틱 회귀 / Logistic Regression | 0.8542 | 0.8234 |
| 나이브 베이즈 / Naive Bayes | 0.8123 | 0.7856 |
| 랜덤 포레스트 / Random Forest | 0.8901 | 0.8156 |
| LSTM | 0.8734 | 0.8267 |
| 양방향 LSTM / Bidirectional LSTM | 0.8823 | 0.8312 |
| **튜닝된 로지스틱 회귀** / **Tuned Logistic Regression** | **0.8567** | **0.8389** |

**최고 모델**: 튜닝된 로지스틱 회귀 (83.89% 검증 정확도)  
**Best Model**: Tuned Logistic Regression with 83.89% validation accuracy

---

## 🔍 주요 인사이트 / Key Insights

### 잘 작동한 것들 / What Worked Well:
- ✅ **텍스트 전처리**: URL, 멘션, 노이즈 정제가 성능을 크게 향상시킴 / Text Preprocessing: Cleaning URLs, mentions, and noise significantly improved performance
- ✅ **TF-IDF 벡터화**: 텍스트 표현에 매우 효과적임 / TF-IDF Vectorization: Proved highly effective for text representation
- ✅ **하이퍼파라미터 튜닝**: 그리드 서치 최적화로 모델 일반화 성능 향상 / Hyperparameter Tuning: Grid search optimization improved model generalization
- ✅ **특성 엔지니어링**: 추가 텍스트 통계가 보완적 정보 제공 / Feature Engineering: Additional text statistics provided complementary information

### 도움이 되지 않은 것들 / What Didn't Help:
- ❌ **복잡한 신경망 아키텍처**: 상당한 성능 향상 없이 복잡성만 증가 / Complex Neural Architectures: Added complexity without significant performance gains
- ❌ **과도한 전처리**: 과도한 어간 추출이 모델 성능 저하 / Aggressive Preprocessing: Over-stemming hurt model performance
- ❌ **고차원 임베딩**: 계산 비용 대비 효과 부족 / High-Dimensional Embeddings: Didn't justify computational cost

---

## 🚀 시작하기 / Getting Started

### 필수 조건 / Prerequisites

```bash
pip install -r requirements.txt
```

### 프로젝트 실행 / Running the Project

1. **저장소 클론 / Clone the repository:**
```bash
git clone https://github.com/[your-username]/disaster-tweet-classification-nlp.git
cd disaster-tweet-classification-nlp
```

2. **캐글 경진대회에서 데이터셋 다운로드 / Download the dataset from Kaggle Competition**

3. **메인 노트북 실행 / Run the main notebook:**
```bash
jupyter notebook notebooks/disaster_tweet_classification.ipynb
```

4. **예측 생성 / Generate predictions:** 
   노트북이 자동으로 캐글 제출용 `submission.csv` 파일을 생성합니다.  
   The notebook will automatically generate `submission.csv` for Kaggle submission.

---

## 📝 방법론 / Methodology

### 1. 데이터 탐색 및 전처리 / Data Exploration & Preprocessing
- 타겟 분포 및 텍스트 특성 분석 / Analyzed target distribution and text characteristics
- 종합적인 텍스트 정제 파이프라인 구현 / Implemented comprehensive text cleaning pipeline
- 추가 특성 생성 (텍스트 길이, 단어 수 등) / Created additional features (text length, word count, etc.)

### 2. 특성 엔지니어링 / Feature Engineering
- **TF-IDF 벡터화**: 텍스트를 수치 특성으로 변환 / TF-IDF Vectorization: Converted text to numerical features
- **단어 임베딩**: 딥러닝 모델용 / Word Embeddings: Used for deep learning models
- **N-gram 특성**: 구문 수준 패턴 캡처 / N-gram Features: Captured phrase-level patterns

### 3. 모델 개발 / Model Development
- 기준 전통적인 ML 모델로 시작 / Started with baseline traditional ML models
- LSTM 기반 신경망 구현 / Implemented LSTM-based neural networks
- 하이퍼파라미터 최적화 적용 / Applied hyperparameter optimization
- 강건한 평가를 위한 교차검증 사용 / Used cross-validation for robust evaluation

### 4. 평가 및 선택 / Evaluation & Selection
- 정확도와 F1-점수를 사용한 모델 비교 / Compared models using accuracy and F1-score
- 특성 중요도 및 모델 해석 가능성 분석 / Analyzed feature importance and model interpretability
- 검증 성능을 기준으로 최고 모델 선택 / Selected best model based on validation performance

---

## 🔮 향후 개선 사항 / Future Improvements

### 고급 NLP 기술 / Advanced NLP Techniques:
- 사전 훈련된 임베딩 (Word2Vec, GloVe) / Pre-trained embeddings (Word2Vec, GloVe)
- 트랜스포머 모델 (BERT, RoBERTa) / Transformer models (BERT, RoBERTa)
- 어텐션 메커니즘 / Attention mechanisms

### 앙상블 방법 / Ensemble Methods:
- 투표 분류기 / Voting classifiers
- 메타 학습자를 이용한 스태킹 / Stacking with meta-learners
- 모델 블렌딩 기법 / Model blending techniques

### 데이터 증강 / Data Augmentation:
- 동의어 교체 / Synonym replacement
- 역번역 / Back-translation
- 클래스 불균형 처리 / Handling class imbalance

---

## 🏆 캐글 경진대회 / Kaggle Competition

- **경진대회**: Natural Language Processing with Disaster Tweets
- **최종 점수**: [여기에 점수 입력 / Your score here]
- **리더보드 순위**: [여기에 순위 입력 / Your position here]

---

## 👥 작성자 / Author

**당신의 이름 / Your Name**

- 📧 **이메일 / Email**: your.email@example.com
- 🐙 **GitHub**: [@your-username](https://github.com/your-username)
- 🏅 **Kaggle**: [Your Kaggle Profile](https://www.kaggle.com/your-username)
- 💼 **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/your-profile)

---

## 📄 라이센스 / License

이 프로젝트는 MIT 라이센스 하에 라이센스됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 감사의 말 / Acknowledgments

- 캐글에서 경진대회를 주최하고 데이터셋을 제공해주신 것에 감사드립니다 / Kaggle for hosting the competition and providing the dataset
- 오픈 소스 도구를 제공한 NLP 및 머신러닝 커뮤니티에 감사드립니다 / The NLP and machine learning community for open-source tools
- 지도와 피드백을 주신 강사님들과 동료들에게 감사드립니다 / Course instructors and peers for guidance and feedback

---

## 📚 참고 문헌 / References

1. Manning, C. D., & Schütze, H. (1999). *Foundations of statistical natural language processing*
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*
3. Jurafsky, D., & Martin, J. H. (2020). *Speech and language processing*
4. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
5. [TensorFlow Documentation](https://www.tensorflow.org/)

---

## ⭐ 도움이 되셨다면 스타를 눌러주세요! / Star this repository if you found it helpful!

## 📧 질문이 있으시면 이슈를 열거나 직접 연락해주세요! / Questions? Feel free to open an issue or contact me directly!

---

<div align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red.svg"/>
  <img src="https://img.shields.io/badge/Built%20with-Python-blue.svg"/>
  <img src="https://img.shields.io/badge/Powered%20by-Machine%20Learning-green.svg"/>
</div> 
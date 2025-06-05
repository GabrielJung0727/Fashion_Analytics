# 🛍️ Fashion Price Prediction System

AI 기반 패션 상품 가격대 예측 및 이미지 분류 시스템

## ✨ 주요 기능

### 📊 **데이터 기반 머신러닝**
- 상품명, 브랜드, 색상 등의 메타데이터를 분석하여 가격대 예측
- Low/Medium/High 3단계 가격대 분류
- 실시간 상품 예측 및 확률 분포 시각화

### 🖼️ **이미지 기반 딥러닝**
- Fashion MNIST 데이터셋을 활용한 CNN 이미지 분류
- 10개 의류 카테고리 분류 (T-shirt, Trouser, Dress 등)
- 실시간 학습 진행도 모니터링

### 🔄 **통합 데이터셋 시스템**
- **Myntra**: 기본 패션 상품 데이터
- **H&M**: 글로벌 패션 브랜드 데이터
- **ASOS**: 온라인 패션 플랫폼 데이터
- **Fashion Images**: 이미지 기반 패션 데이터

## 🤖 지원 알고리즘

- 🌳 **RandomForest**: 여러 결정 트리의 앙상블
- 📈 **GradientBoosting**: 순차적 약한 학습기 결합
- 🎯 **SVM**: 서포트 벡터 머신 - 최적 분리 경계 찾기
- 🧠 **MLP**: 다층 퍼셉트론 - 심화 신경망 구조

## 📈 성능 지표

시스템은 다음과 같은 포괄적인 성능 지표를 제공합니다:
- ✅ **정확도 & 오차율**: 기본 예측 성능
- 📊 **정밀도 & 재현율**: 클래스별 세부 성능
- 🎯 **F1 점수**: 정밀도와 재현율의 조화 평균
- 📉 **과적합도**: 학습/테스트 정확도 차이
- 🔍 **클래스별 성능**: Low/Medium/High 가격대별 분석

## 🚀 실행 방법

### 1. 환경 설정
```bash
# Python 3.8+ 필요
pip install streamlit pandas scikit-learn matplotlib seaborn plotly numpy tensorflow
```

### 2. 애플리케이션 실행
```bash
streamlit run app.py
```

### 3. 브라우저에서 접속
```
Local URL: http://localhost:8501
```

## 📊 시스템 구조

```
├── app.py                     # 메인 Streamlit 애플리케이션
├── download_fashion_mnist.py  # Fashion MNIST 데이터 다운로더
├── fashion_mnist_loader.py    # 데이터 로더 유틸리티
├── myntra_products_catalog.csv # 기본 데이터셋 (필요시 생성)
└── fashion_mnist_data/        # Fashion MNIST 데이터 (자동 생성)
```

## 🎮 사용 방법

### 1. **학습 데이터 구성** 메뉴
- 통합 데이터셋 정보 및 구성 비율 확인
- 데이터 소스별 학습률 시각화

### 2. **상품 예측** 메뉴
- 실시간 상품 정보 입력
- AI 예측 결과 및 확률 분포 확인
- 마케팅 전략 추천

### 3. **모델 성능 분석** 메뉴
- 혼동 행렬 및 분류 보고서
- 클래스별 성능 지표 분석
- 특성 중요도 시각화

### 4. **Fashion MNIST** 모드
- 이미지 기반 딥러닝 분류
- CNN 모델 학습 및 평가
- 예측 결과 시각화

## 🔧 기술 스택

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **ML Frameworks**: Scikit-learn, TensorFlow
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **UI Language**: 한국어

## 📝 특징

### 🌟 **사용자 친화적 인터페이스**
- 직관적인 한국어 UI
- 실시간 학습 진행도 모니터링
- 상세한 성능 지표 설명

### 🔬 **과학적 접근**
- 다양한 머신러닝 알고리즘 비교
- 통계적 성능 지표 제공
- 데이터 분포 및 복잡성 시각화

### 🏢 **실무 활용**
- 실제 패션 비즈니스에 적용 가능한 인사이트
- 마케팅 전략 및 운영 개선 제안
- 데이터 기반 의사결정 지원

## 🎯 활용 분야

- **E-commerce**: 상품 가격 최적화
- **패션 브랜드**: 제품 포지셔닝 전략
- **데이터 분석**: 패션 트렌드 분석
- **교육**: 머신러닝 실습 및 데모

## 📚 학습 목적

이 프로젝트는 다음과 같은 학습 목표를 가지고 있습니다:
- 머신러닝 알고리즘의 실제 적용
- 데이터 전처리 및 특성 엔지니어링
- 모델 성능 평가 및 해석
- 사용자 인터페이스 설계
- 실무 적용 가능한 AI 시스템 구축

---

💡 **Made with ❤️ for Fashion AI & Data Science**
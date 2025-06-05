import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Fashion MNIST 데이터셋 직접 로드
def load_fashion_mnist():
    print("Fashion MNIST 데이터셋을 로드합니다...")
    
    # 데이터 저장 디렉토리
    DATA_DIR = 'fashion_mnist_data'
    
    # 데이터가 없는 경우 다운로드 스크립트 실행
    if not (os.path.exists(os.path.join(DATA_DIR, 'train_images.npy')) and 
            os.path.exists(os.path.join(DATA_DIR, 'train_labels.npy')) and
            os.path.exists(os.path.join(DATA_DIR, 'test_images.npy')) and
            os.path.exists(os.path.join(DATA_DIR, 'test_labels.npy'))):
        print("데이터셋을 찾을 수 없습니다. download_fashion_mnist.py를 먼저 실행해주세요.")
        import download_fashion_mnist
        download_fashion_mnist.download_and_process_data()
    
    # 저장된 NumPy 배열 로드
    train_images = np.load(os.path.join(DATA_DIR, 'train_images.npy'))
    train_labels = np.load(os.path.join(DATA_DIR, 'train_labels.npy'))
    test_images = np.load(os.path.join(DATA_DIR, 'test_images.npy'))
    test_labels = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
    
    # 클래스 이름 정의
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print(f"학습 데이터: {train_images.shape[0]}개 이미지, 각 {train_images.shape[1]}x{train_images.shape[2]} 픽셀")
    print(f"테스트 데이터: {test_images.shape[0]}개 이미지, 각 {test_images.shape[1]}x{test_images.shape[2]} 픽셀")
    
    # CSV 파일로 변환하여 저장
    save_as_csv(train_images, train_labels, 'fashion_mnist_train.csv')
    save_as_csv(test_images, test_labels, 'fashion_mnist_test.csv')
    
    return (train_images, train_labels), (test_images, test_labels), class_names

# 이미지와 레이블을 CSV 파일로 저장
def save_as_csv(images, labels, filename):
    # 이미지를 1차원 벡터로 변환
    num_images = len(images)
    images_flat = images.reshape(num_images, -1)
    
    # 레이블과 이미지 픽셀을 합쳐서 DataFrame 생성
    df = pd.DataFrame(images_flat)
    df.insert(0, 'label', labels)
    
    # CSV 파일로 저장
    df.to_csv(filename, index=False)
    print(f"{filename} 파일이 생성되었습니다. 크기: {df.shape}")

# 샘플 이미지 시각화
def plot_sample_images(images, labels, class_names, num_examples=25):
    plt.figure(figsize=(10,10))
    for i in range(min(num_examples, len(images))):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.savefig('fashion_mnist_samples.png')
    plt.close()
    print("샘플 이미지가 'fashion_mnist_samples.png'로 저장되었습니다.")

# 모델 구축 및 학습
def build_and_train_model(train_images, train_labels, test_images, test_labels):
    try:
        import tensorflow as tf
        
        # 모델 구축
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        
        # 모델 컴파일
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        # 모델 요약 출력
        model.summary()
        
        # 모델 학습
        print("\n모델 학습을 시작합니다...")
        history = model.fit(train_images, train_labels, epochs=10, 
                            validation_data=(test_images, test_labels))
        
        # 학습 히스토리 시각화
        plot_training_history(history)
        
        # 모델 평가
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(f'\n테스트 정확도: {test_acc*100:.2f}%')
        
        # 모델 저장
        model.save('fashion_mnist_model.h5')
        print("모델이 'fashion_mnist_model.h5'로 저장되었습니다.")
        
        return model, history
    
    except ImportError:
        print("TensorFlow를 설치하지 않았거나 불러올 수 없습니다.")
        print("pip install tensorflow 명령어로 TensorFlow를 설치하세요.")
        return None, None

# 학습 히스토리 시각화
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('fashion_mnist_training_history.png')
    plt.close()
    print("학습 히스토리가 'fashion_mnist_training_history.png'로 저장되었습니다.")

# 예측 결과 시각화
def plot_predictions(model, test_images, test_labels, class_names, num_examples=25):
    try:
        import tensorflow as tf
        
        # 예측 확률 계산
        probability_model = tf.keras.Sequential([
            model, 
            tf.keras.layers.Softmax()
        ])
        predictions = probability_model.predict(test_images[:num_examples])
        
        # 예측 결과 시각화
        plt.figure(figsize=(10,10))
        for i in range(num_examples):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(test_images[i], cmap=plt.cm.binary)
            
            predicted_label = np.argmax(predictions[i])
            true_label = test_labels[i]
            
            if predicted_label == true_label:
                color = 'blue'
            else:
                color = 'red'
                
            plt.xlabel(f"{class_names[predicted_label]} ({class_names[true_label]})", color=color)
        
        plt.savefig('fashion_mnist_predictions.png')
        plt.close()
        print("예측 결과가 'fashion_mnist_predictions.png'로 저장되었습니다.")
    
    except ImportError:
        print("TensorFlow를 설치하지 않았거나 불러올 수 없습니다.")
        print("pip install tensorflow 명령어로 TensorFlow를 설치하세요.")

# 메인 함수
def main():
    print("Fashion MNIST 데이터셋 학습 시작")
    
    # 데이터 로드
    (train_images, train_labels), (test_images, test_labels), class_names = load_fashion_mnist()
    
    # 샘플 이미지 시각화
    plot_sample_images(train_images, train_labels, class_names)
    
    # 모델 학습 (선택적)
    try_train = input("모델을 학습시키시겠습니까? (y/n): ").lower().strip() == 'y'
    if try_train:
        # 모델 학습
        model, history = build_and_train_model(train_images, train_labels, test_images, test_labels)
        
        # 예측 결과 시각화 (모델이 성공적으로 학습된 경우)
        if model is not None:
            plot_predictions(model, test_images, test_labels, class_names)
    
    print("Fashion MNIST 데이터셋 학습 완료")

if __name__ == "__main__":
    main() 
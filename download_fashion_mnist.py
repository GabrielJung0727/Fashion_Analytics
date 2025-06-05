import requests
import os
import gzip
import numpy as np
import struct
import matplotlib.pyplot as plt

# 데이터 URL
URLS = {
    'train_images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
    'train_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
    'test_images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
}

# 데이터 저장 디렉토리
DATA_DIR = 'fashion_mnist_data'

# 디렉토리 생성
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def download_file(url, filename):
    """파일 다운로드 함수"""
    print(f"다운로드 중: {filename}")
    response = requests.get(url, stream=True)
    file_path = os.path.join(DATA_DIR, filename)
    
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    print(f"다운로드 완료: {filename}")
    return file_path

def extract_images(filename):
    """이미지 파일 압축 해제 및 처리"""
    print(f"이미지 추출 중: {filename}")
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    
    print(f"추출 완료: {num}개 이미지, 크기 {rows}x{cols}")
    return images

def extract_labels(filename):
    """레이블 파일 압축 해제 및 처리"""
    print(f"레이블 추출 중: {filename}")
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    print(f"추출 완료: {num}개 레이블")
    return labels

def download_and_process_data():
    """데이터 다운로드 및 처리 메인 함수"""
    # 파일 다운로드
    file_paths = {}
    for name, url in URLS.items():
        filename = url.split('/')[-1]
        file_paths[name] = download_file(url, filename)
    
    # 이미지 및 레이블 추출
    train_images = extract_images(file_paths['train_images'])
    train_labels = extract_labels(file_paths['train_labels'])
    test_images = extract_images(file_paths['test_images'])
    test_labels = extract_labels(file_paths['test_labels'])
    
    # 데이터 정규화 (0-1 범위로 변환)
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # 데이터 저장
    np.save(os.path.join(DATA_DIR, 'train_images.npy'), train_images)
    np.save(os.path.join(DATA_DIR, 'train_labels.npy'), train_labels)
    np.save(os.path.join(DATA_DIR, 'test_images.npy'), test_images)
    np.save(os.path.join(DATA_DIR, 'test_labels.npy'), test_labels)
    
    print("모든 데이터 처리 완료!")
    
    # 샘플 이미지 시각화
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    
    plt.savefig(os.path.join(DATA_DIR, 'sample_images.png'))
    plt.close()
    
    print(f"샘플 이미지가 '{os.path.join(DATA_DIR, 'sample_images.png')}'에 저장되었습니다.")
    
    return train_images, train_labels, test_images, test_labels, class_names

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels, class_names = download_and_process_data()
    
    # 데이터 크기 확인
    print(f"\n데이터셋 정보:")
    print(f"학습 이미지: {train_images.shape}")
    print(f"학습 레이블: {train_labels.shape}")
    print(f"테스트 이미지: {test_images.shape}")
    print(f"테스트 레이블: {test_labels.shape}") 
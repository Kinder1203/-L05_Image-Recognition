"""
과제 01: 간단한 이미지 분류기 구현
- MNIST 손글씨 숫자 이미지(28x28 흑백)를 이용한 분류기
- Sequential 모델과 Dense 레이어를 활용한 간단한 신경망
"""

# 배열 연산 및 수학 함수를 위한 NumPy 라이브러리를 불러옵니다.
import numpy as np
# 그래프와 이미지 시각화를 위한 Matplotlib 라이브러리를 불러옵니다.
import matplotlib.pyplot as plt
# 파일 경로 조작을 위한 os 모듈을 불러옵니다.
import os

# TensorFlow Keras에서 MNIST 데이터셋을 불러오는 모듈을 임포트합니다.
from tensorflow.keras.datasets import mnist
# 레이어를 순차적으로 쌓아 모델을 구성하는 Sequential 클래스를 임포트합니다.
from tensorflow.keras.models import Sequential
# 완전연결층(Dense)과 다차원 텐서를 1차원으로 펼치는 Flatten 레이어를 임포트합니다.
from tensorflow.keras.layers import Dense, Flatten
# 정수 레이블을 원-핫 인코딩 벡터로 변환하는 유틸리티 함수를 임포트합니다.
from tensorflow.keras.utils import to_categorical

# ──────────────────────────────────────────────
# 결과 저장 경로 설정
# ──────────────────────────────────────────────
# 현재 스크립트 파일이 위치한 디렉터리 경로를 기준으로 result_images 폴더 경로를 생성합니다.
RESULT_DIR = os.path.join(os.path.dirname(__file__), "result_images")
# result_images 폴더가 없으면 새로 생성하고, 이미 존재하면 에러 없이 넘어갑니다.
os.makedirs(RESULT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. MNIST 데이터셋 로드
# ──────────────────────────────────────────────
# Keras가 제공하는 MNIST 데이터셋을 자동으로 다운로드하고 훈련/테스트 세트로 분할하여 로드합니다.
# x_train: 훈련 이미지(60,000장, 28x28), y_train: 훈련 레이블(0~9)
# x_test: 테스트 이미지(10,000장, 28x28), y_test: 테스트 레이블(0~9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 구분선을 출력하여 출력 결과의 가독성을 높입니다.
print("=" * 50)
# 데이터셋 로드 완료 메시지를 출력합니다.
print("MNIST 데이터셋 로드 완료")
# 훈련 데이터의 shape(개수, 가로, 세로)과 레이블 shape을 출력하여 구조를 확인합니다.
print(f"  훈련 데이터: {x_train.shape}, 레이블: {y_train.shape}")
# 테스트 데이터의 shape과 레이블 shape을 출력하여 구조를 확인합니다.
print(f"  테스트 데이터: {x_test.shape}, 레이블: {y_test.shape}")
# 구분선을 출력합니다.
print("=" * 50)

# ──────────────────────────────────────────────
# 2. 데이터 전처리
#    - 픽셀 값을 0~1 범위로 정규화
#    - 레이블을 원-핫 인코딩
# ──────────────────────────────────────────────
# 훈련 이미지의 픽셀 값(0~255)을 float32로 변환한 뒤, 255로 나누어 0~1 범위로 정규화합니다.
# 정규화하면 학습 시 그라디언트가 안정되어 모델의 수렴 속도가 빨라집니다.
x_train = x_train.astype("float32") / 255.0
# 테스트 이미지도 동일하게 정규화합니다.
x_test = x_test.astype("float32") / 255.0

# 훈련 레이블(정수 0~9)을 10차원 원-핫 벡터로 변환합니다. (예: 3 → [0,0,0,1,0,0,0,0,0,0])
# categorical_crossentropy 손실 함수와 함께 사용하기 위해 필요합니다.
y_train_cat = to_categorical(y_train, 10)
# 테스트 레이블도 동일하게 원-핫 인코딩합니다.
y_test_cat = to_categorical(y_test, 10)

# 훈련 데이터 앞 10장을 2행 5열로 시각화하여 데이터가 잘 로드되었는지 확인합니다.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
# axes 배열을 1차원으로 펼친 뒤, 인덱스와 축(ax) 객체를 순회합니다.
for i, ax in enumerate(axes.flat):
    # i번째 훈련 이미지를 그레이스케일 컬러맵으로 해당 축에 그립니다.
    ax.imshow(x_train[i], cmap="gray")
    # 이미지 위에 실제 레이블 값을 제목으로 표시합니다.
    ax.set_title(f"Label: {y_train[i]}", fontsize=11)
    # 축 눈금과 테두리를 숨겨 이미지에 집중할 수 있게 합니다.
    ax.axis("off")
# 전체 figure의 상단에 총괄 제목을 굵은 글씨로 표시합니다.
plt.suptitle("MNIST Sample Images", fontsize=14, fontweight="bold")
# 서브플롯 간 간격을 자동으로 조정하여 겹침을 방지합니다.
plt.tight_layout()
# 결과 이미지를 result_images 폴더에 PNG 파일로 저장합니다. (dpi=150으로 선명하게)
plt.savefig(os.path.join(RESULT_DIR, "01_mnist_samples.png"), dpi=150)
# 현재 figure를 닫아 메모리를 해제합니다. (이후 새 그래프 생성 시 간섭 방지)
plt.close()
# 저장 완료 메시지를 터미널에 출력합니다.
print("[저장] 01_mnist_samples.png")

# ──────────────────────────────────────────────
# 3. 간단한 신경망 모델 구축 (Sequential + Dense)
#    - Flatten: 28x28 → 784
#    - Dense 128 (ReLU)
#    - Dense 64  (ReLU)
#    - Dense 10  (Softmax, 출력)
# ──────────────────────────────────────────────
# Sequential 모델을 생성하고, 리스트로 레이어를 순서대로 전달합니다.
model = Sequential([
    # Flatten: 28x28 크기의 2D 이미지를 784(=28*28)차원의 1D 벡터로 변환합니다.
    # Dense 레이어에 입력하기 위해 반드시 필요한 전처리 단계입니다.
    Flatten(input_shape=(28, 28)),
    # 첫 번째 은닉층: 128개의 뉴런, ReLU 활성화 함수를 사용합니다.
    # ReLU(Rectified Linear Unit)는 음수를 0으로 처리하여 비선형성을 부여합니다.
    Dense(128, activation="relu"),
    # 두 번째 은닉층: 64개의 뉴런, ReLU 활성화 함수를 사용합니다.
    # 점진적으로 뉴런 수를 줄여 특징을 압축합니다.
    Dense(64, activation="relu"),
    # 출력층: 10개의 뉴런(0~9 숫자 클래스에 대응), Softmax 활성화 함수를 사용합니다.
    # Softmax는 출력을 확률 분포(합=1)로 변환하여 각 클래스에 속할 확률을 나타냅니다.
    Dense(10, activation="softmax"),
])

# 모델의 학습 방식을 설정합니다.
model.compile(
    # Adam 옵티마이저: 학습률을 자동 조절하며, SGD보다 빠르게 수렴하는 최적화 알고리즘입니다.
    optimizer="adam",
    # 다중 클래스 분류에 적합한 손실 함수입니다. 원-핫 인코딩 레이블과 함께 사용됩니다.
    loss="categorical_crossentropy",
    # 학습 과정에서 정확도(accuracy)를 함께 추적하도록 설정합니다.
    metrics=["accuracy"],
)

# 모델의 구조(레이어 이름, 출력 shape, 파라미터 수)를 요약하여 터미널에 출력합니다.
model.summary()

# ──────────────────────────────────────────────
# 4. 모델 훈련
# ──────────────────────────────────────────────
# 모델을 훈련 데이터로 학습시킵니다.
history = model.fit(
    # 훈련 이미지와 원-핫 인코딩된 레이블을 전달합니다.
    x_train, y_train_cat,
    # 전체 훈련 데이터를 10번 반복하여 학습합니다.
    epochs=10,
    # 한 번에 128장의 이미지를 묶어서 가중치를 업데이트합니다. (미니배치 학습)
    batch_size=128,
    # 훈련 데이터의 20%를 검증 데이터로 분리하여 과적합 여부를 모니터링합니다.
    validation_split=0.2,
    # 학습 진행 상황을 에폭마다 프로그레스 바로 표시합니다.
    verbose=1,
)

# ──────────────────────────────────────────────
# 5. 모델 평가
# ──────────────────────────────────────────────
# 테스트 데이터에 대한 손실(loss)과 정확도(accuracy)를 계산합니다.
# verbose=0으로 설정하여 평가 중 출력을 숨깁니다.
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
# 테스트 정확도를 소수점 4자리까지 출력합니다.
print(f"\n테스트 정확도: {test_acc:.4f}")
# 테스트 손실을 소수점 4자리까지 출력합니다.
print(f"테스트 손실:   {test_loss:.4f}")

# ──────────────────────────────────────────────
# 6. 결과 시각화
# ──────────────────────────────────────────────

# (a) 학습 곡선 (정확도 & 손실)
# 1행 2열의 서브플롯을 포함하는 figure를 생성합니다.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 왼쪽 그래프: 에폭별 훈련 정확도를 파란색 선으로 그립니다.
ax1.plot(history.history["accuracy"], label="Train Accuracy")
# 에폭별 검증 정확도를 주황색 선으로 그립니다.
ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
# 그래프 제목을 설정합니다.
ax1.set_title("Accuracy")
# x축 라벨을 에폭으로 설정합니다.
ax1.set_xlabel("Epoch")
# y축 라벨을 정확도로 설정합니다.
ax1.set_ylabel("Accuracy")
# 범례를 표시하여 어떤 선이 훈련/검증인지 구분합니다.
ax1.legend()
# 반투명 격자선을 추가하여 값 읽기를 쉽게 합니다.
ax1.grid(True, alpha=0.3)

# 오른쪽 그래프: 에폭별 훈련 손실을 파란색 선으로 그립니다.
ax2.plot(history.history["loss"], label="Train Loss")
# 에폭별 검증 손실을 주황색 선으로 그립니다.
ax2.plot(history.history["val_loss"], label="Val Loss")
# 그래프 제목을 설정합니다.
ax2.set_title("Loss")
# x축 라벨을 에폭으로 설정합니다.
ax2.set_xlabel("Epoch")
# y축 라벨을 손실로 설정합니다.
ax2.set_ylabel("Loss")
# 범례를 표시합니다.
ax2.legend()
# 반투명 격자선을 추가합니다.
ax2.grid(True, alpha=0.3)

# 전체 figure의 상단 제목에 테스트 정확도를 함께 표시합니다.
plt.suptitle(f"MNIST Training Result  (Test Acc: {test_acc:.4f})", fontsize=13, fontweight="bold")
# 서브플롯 간 간격을 자동 조정합니다.
plt.tight_layout()
# 학습 곡선 이미지를 파일로 저장합니다.
plt.savefig(os.path.join(RESULT_DIR, "01_mnist_training_curve.png"), dpi=150)
# figure를 닫아 메모리를 해제합니다.
plt.close()
# 저장 완료 메시지를 출력합니다.
print("[저장] 01_mnist_training_curve.png")

# (b) 테스트 이미지 예측 결과
# 테스트 이미지 앞 10장에 대한 예측 확률을 계산합니다. (shape: (10, 10))
predictions = model.predict(x_test[:10])
# 예측 결과를 시각화할 2행 5열의 서브플롯을 생성합니다.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
# 각 서브플롯에 대해 순회합니다.
for i, ax in enumerate(axes.flat):
    # i번째 테스트 이미지를 그레이스케일로 표시합니다.
    ax.imshow(x_test[i], cmap="gray")
    # 예측 확률이 가장 높은 클래스의 인덱스를 예측 레이블로 선택합니다.
    pred_label = np.argmax(predictions[i])
    # 실제 정답 레이블을 가져옵니다.
    true_label = y_test[i]
    # 예측이 맞으면 초록색, 틀리면 빨간색으로 제목 색상을 설정합니다.
    color = "green" if pred_label == true_label else "red"
    # 예측값(Pred)과 실제값(True)을 제목으로 표시합니다.
    ax.set_title(f"Pred: {pred_label} / True: {true_label}", fontsize=10, color=color)
    # 축 눈금을 숨깁니다.
    ax.axis("off")
# 전체 figure의 상단 제목을 설정합니다.
plt.suptitle("MNIST Prediction Results", fontsize=14, fontweight="bold")
# 서브플롯 간 간격을 자동 조정합니다.
plt.tight_layout()
# 예측 결과 이미지를 파일로 저장합니다.
plt.savefig(os.path.join(RESULT_DIR, "01_mnist_predictions.png"), dpi=150)
# figure를 닫아 메모리를 해제합니다.
plt.close()
# 저장 완료 메시지를 출력합니다.
print("[저장] 01_mnist_predictions.png")

# 과제 01 완료 메시지를 출력합니다.
print("\n과제 01 완료!")

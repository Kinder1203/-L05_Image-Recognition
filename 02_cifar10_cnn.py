# 배열 연산 및 수학 함수를 위한 NumPy 라이브러리를 불러옵니다.
import numpy as np
# 그래프와 이미지 시각화를 위한 Matplotlib 라이브러리를 불러옵니다.
import matplotlib.pyplot as plt
# 파일 경로 조작을 위한 os 모듈을 불러옵니다.
import os
# 외부 이미지(dog.jpg)를 로드하고 리사이즈하기 위한 PIL 라이브러리를 불러옵니다.
from PIL import Image

# TensorFlow Keras에서 CIFAR-10 데이터셋을 불러오는 모듈을 임포트합니다.
from tensorflow.keras.datasets import cifar10
# 레이어를 순차적으로 쌓아 모델을 구성하는 Sequential 클래스를 임포트합니다.
from tensorflow.keras.models import Sequential
# CNN 구성에 필요한 레이어들을 임포트합니다.
# Conv2D: 2D 합성곱 레이어, MaxPooling2D: 최대 풀링 레이어
# Flatten: 다차원 텐서를 1차원으로 변환, Dense: 완전연결층, Dropout: 과적합 방지용 드롭아웃
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# 정수 레이블을 원-핫 인코딩 벡터로 변환하는 유틸리티 함수를 임포트합니다.
from tensorflow.keras.utils import to_categorical

# 현재 스크립트 파일이 위치한 디렉터리 경로를 가져옵니다.
BASE_DIR = os.path.dirname(__file__)
# 입력 이미지(dog.jpg)가 들어있는 images 폴더의 경로를 생성합니다.
IMAGE_DIR = os.path.join(BASE_DIR, "images")
# 결과 이미지를 저장할 result_images 폴더의 경로를 생성합니다.
RESULT_DIR = os.path.join(BASE_DIR, "result_images")
# result_images 폴더가 없으면 새로 생성하고, 이미 존재하면 에러 없이 넘어갑니다.
os.makedirs(RESULT_DIR, exist_ok=True)

# CIFAR-10 데이터셋의 10개 클래스 이름을 인덱스 순서대로 정의합니다.
# 인덱스 0=airplane, 1=automobile, ..., 5=dog, ..., 9=truck
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Keras가 제공하는 CIFAR-10 데이터셋을 자동으로 다운로드하고 훈련/테스트 세트로 분할합니다.
# x_train: 훈련 이미지(50,000장, 32x32x3 컬러), y_train: 훈련 레이블(0~9)
# x_test: 테스트 이미지(10,000장, 32x32x3 컬러), y_test: 테스트 레이블(0~9)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 구분선을 출력하여 출력 결과의 가독성을 높입니다.
print("=" * 50)
# 데이터셋 로드 완료 메시지를 출력합니다.
print("CIFAR-10 데이터셋 로드 완료")
# 훈련 데이터의 shape(개수, 가로, 세로, 채널)과 레이블 shape을 출력합니다.
print(f"  훈련 데이터: {x_train.shape}, 레이블: {y_train.shape}")
# 테스트 데이터의 shape과 레이블 shape을 출력합니다.
print(f"  테스트 데이터: {x_test.shape}, 레이블: {y_test.shape}")
# 구분선을 출력합니다.
print("=" * 50)

# 훈련 이미지의 픽셀 값(0~255)을 float32로 변환한 뒤, 255로 나누어 0~1 범위로 정규화합니다.
# 과제 힌트: "데이터 전처리 시 픽셀 값을 0~1 범위로 정규화하면 모델의 수렴이 빨라질 수 있음"
x_train = x_train.astype("float32") / 255.0
# 테스트 이미지도 동일한 방식으로 정규화합니다.
x_test = x_test.astype("float32") / 255.0

# 훈련 레이블(정수 0~9)을 10차원 원-핫 벡터로 변환합니다. (예: 5(dog) → [0,0,0,0,0,1,0,0,0,0])
y_train_cat = to_categorical(y_train, 10)
# 테스트 레이블도 동일하게 원-핫 인코딩합니다.
y_test_cat = to_categorical(y_test, 10)

# 훈련 데이터 앞 10장을 2행 5열로 시각화하여 데이터가 잘 로드되었는지 확인합니다.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
# axes 배열을 1차원으로 펼친 뒤, 인덱스와 축(ax) 객체를 순회합니다.
for i, ax in enumerate(axes.flat):
    # i번째 훈련 이미지를 해당 축에 RGB 컬러로 그립니다. (CIFAR-10은 컬러 이미지)
    ax.imshow(x_train[i])
    # 해당 이미지의 레이블 인덱스를 정수로 가져옵니다.
    label_idx = int(y_train[i])
    # CLASS_NAMES에서 해당 인덱스에 대응하는 클래스 이름을 제목으로 표시합니다.
    ax.set_title(CLASS_NAMES[label_idx], fontsize=11)
    # 축 눈금과 테두리를 숨겨 이미지에 집중할 수 있게 합니다.
    ax.axis("off")
# 전체 figure의 상단에 총괄 제목을 굵은 글씨로 표시합니다.
plt.suptitle("CIFAR-10 Sample Images", fontsize=14, fontweight="bold")
# 서브플롯 간 간격을 자동으로 조정하여 겹침을 방지합니다.
plt.tight_layout()
# 샘플 이미지를 result_images 폴더에 PNG 파일로 저장합니다.
plt.savefig(os.path.join(RESULT_DIR, "02_cifar10_samples.png"), dpi=150)
# 현재 figure를 닫아 메모리를 해제합니다.
plt.close()
# 저장 완료 메시지를 터미널에 출력합니다.
print("[저장] 02_cifar10_samples.png")

# Sequential 모델을 생성하고, 리스트로 레이어를 순서대로 전달합니다.
model = Sequential([
    # ── Block 1 ──
    # 첫 번째 합성곱 레이어: 32개의 3x3 필터로 이미지에서 저수준 특징(엣지, 색상변화 등)을 추출합니다.
    # padding="same"으로 입출력 크기를 동일하게 유지하고, ReLU 활성화로 비선형성을 부여합니다.
    # input_shape=(32, 32, 3)은 CIFAR-10 이미지 크기(가로 32, 세로 32, RGB 3채널)를 명시합니다.
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
    # 2x2 풀링 윈도우로 특징 맵의 크기를 절반(32x32 → 16x16)으로 줄입니다.
    # 연산량을 감소시키고, 위치 변화에 대한 불변성을 확보합니다.
    MaxPooling2D((2, 2)),

    # ── Block 2 ──
    # 두 번째 합성곱 레이어: 64개의 3x3 필터로 중간 수준의 특징(질감, 패턴 등)을 추출합니다.
    # 필터 수를 32→64로 2배 늘려 더 복잡한 특징을 학습할 수 있게 합니다.
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    # 특징 맵 크기를 다시 절반(16x16 → 8x8)으로 줄입니다.
    MaxPooling2D((2, 2)),

    # ── Block 3 ──
    # 세 번째 합성곱 레이어: 128개의 3x3 필터로 고수준 특징(물체의 형태, 구조 등)을 추출합니다.
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    # 특징 맵 크기를 절반(8x8 → 4x4)으로 줄입니다.
    MaxPooling2D((2, 2)),

    # ── 분류기 (Classifier) ──
    # 3D 특징 맵(4x4x128=2048)을 1D 벡터로 평탄화하여 Dense 레이어에 전달할 수 있게 합니다.
    Flatten(),
    # 128개의 뉴런을 가진 완전연결층으로 고수준 특징을 조합하여 분류에 필요한 패턴을 학습합니다.
    Dense(128, activation="relu"),
    # 학습 시 뉴런의 30%를 랜덤으로 비활성화하여 과적합(Overfitting)을 방지합니다.
    # 특정 뉴런에 대한 의존도를 줄여 모델의 일반화 성능을 향상시킵니다.
    Dropout(0.3),
    # 출력층: 10개의 뉴런(CIFAR-10의 10개 클래스에 대응), Softmax로 확률 분포를 출력합니다.
    Dense(10, activation="softmax"),
])

# 모델의 학습 방식을 설정합니다.
model.compile(
    # Adam 옵티마이저: 학습률을 자동 조절하는 효율적인 최적화 알고리즘입니다.
    optimizer="adam",
    # 다중 클래스 분류에 적합한 손실 함수입니다. 원-핫 인코딩 레이블과 함께 사용됩니다.
    loss="categorical_crossentropy",
    # 학습 과정에서 정확도(accuracy)를 함께 추적하도록 설정합니다.
    metrics=["accuracy"],
)

# 모델의 구조(레이어 이름, 출력 shape, 파라미터 수)를 요약하여 터미널에 출력합니다.
model.summary()

# 모델을 훈련 데이터로 학습시킵니다.
history = model.fit(
    # 훈련 이미지와 원-핫 인코딩된 레이블을 전달합니다.
    x_train, y_train_cat,
    # 전체 훈련 데이터를 20번 반복하여 학습합니다. (CIFAR-10은 MNIST보다 복잡하므로 에폭 수를 늘림)
    epochs=20,
    # 한 번에 64장의 이미지를 묶어서 가중치를 업데이트합니다. (미니배치 학습)
    batch_size=64,
    # 훈련 데이터의 20%를 검증 데이터로 분리하여 과적합 여부를 모니터링합니다.
    validation_split=0.2,
    # 학습 진행 상황을 에폭마다 프로그레스 바로 표시합니다.
    verbose=1,
)

# 테스트 데이터에 대한 손실(loss)과 정확도(accuracy)를 계산합니다.
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
# 테스트 정확도를 소수점 4자리까지 출력합니다.
print(f"\n테스트 정확도: {test_acc:.4f}")
# 테스트 손실을 소수점 4자리까지 출력합니다.
print(f"테스트 손실:   {test_loss:.4f}")

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
plt.suptitle(f"CIFAR-10 CNN Training Result  (Test Acc: {test_acc:.4f})", fontsize=13, fontweight="bold")
# 서브플롯 간 간격을 자동 조정합니다.
plt.tight_layout()
# 학습 곡선 이미지를 파일로 저장합니다.
plt.savefig(os.path.join(RESULT_DIR, "02_cifar10_training_curve.png"), dpi=150)
# figure를 닫아 메모리를 해제합니다.
plt.close()
# 저장 완료 메시지를 출력합니다.
print("[저장] 02_cifar10_training_curve.png")

# 교수님이 제공한 테스트 이미지 dog.jpg의 전체 경로를 생성합니다.
dog_path = os.path.join(IMAGE_DIR, "dog.jpg")
# PIL을 사용하여 이미지를 RGB 모드로 열어옵니다.
dog_img = Image.open(dog_path).convert("RGB")
# 원본 이미지를 NumPy 배열로 변환하여 시각화용으로 보존합니다.
dog_original = np.array(dog_img)

# CIFAR-10 모델의 입력 크기(32x32)에 맞게 이미지를 리사이즈합니다.
dog_resized = dog_img.resize((32, 32))
# 리사이즈된 이미지를 NumPy 배열로 변환하고, float32 타입으로 바꾼 뒤 0~1 범위로 정규화합니다.
dog_array = np.array(dog_resized).astype("float32") / 255.0
# 모델은 배치 단위 입력(4D 텐서)을 기대하므로, 맨 앞에 배치 차원을 추가합니다. (32,32,3) → (1,32,32,3)
dog_input = np.expand_dims(dog_array, axis=0)

# 모델에 전처리된 dog 이미지를 입력하여 10개 클래스에 대한 예측 확률을 얻습니다.
prediction = model.predict(dog_input)
# 예측 확률이 가장 높은 클래스의 인덱스를 가져옵니다.
pred_class = np.argmax(prediction[0])
# 해당 클래스의 확률(신뢰도)을 가져옵니다.
pred_confidence = prediction[0][pred_class]

# 예측 결과(클래스 이름과 확률)를 터미널에 출력합니다.
print(f"\ndog.jpg 예측 결과: {CLASS_NAMES[pred_class]} (확률: {pred_confidence:.4f})")

# dog.jpg 예측 결과를 시각화할 1행 2열의 서브플롯을 생성합니다.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# 왼쪽: 원본 dog.jpg 이미지를 표시합니다.
ax1.imshow(dog_original)
# 예측 클래스가 dog(인덱스 5)이면 초록색, 아니면 빨간색으로 예측 결과를 제목에 표시합니다.
ax1.set_title(f"Prediction: {CLASS_NAMES[pred_class]} ({pred_confidence:.2%})",
              fontsize=12, fontweight="bold", color="green" if pred_class == 5 else "red")
# 축 눈금을 숨깁니다.
ax1.axis("off")

# 오른쪽: 10개 클래스별 예측 확률을 수평 막대 그래프로 표시합니다.
# 예측된 클래스는 초록색, 나머지는 파란색으로 색상을 구분합니다.
colors = ["green" if i == pred_class else "steelblue" for i in range(10)]
# 수평 막대 그래프를 그립니다.
ax2.barh(CLASS_NAMES, prediction[0], color=colors)
# x축 라벨을 확률로 설정합니다.
ax2.set_xlabel("Probability")
# 그래프 제목을 설정합니다.
ax2.set_title("Class Probabilities", fontsize=12, fontweight="bold")
# x축 범위를 0~1로 설정합니다. (확률이므로)
ax2.set_xlim(0, 1)
# x축 방향으로 반투명 격자선을 추가합니다.
ax2.grid(True, axis="x", alpha=0.3)

# 전체 figure의 상단 제목을 설정합니다.
plt.suptitle("CIFAR-10 CNN: dog.jpg Prediction", fontsize=14, fontweight="bold")
# 서브플롯 간 간격을 자동 조정합니다.
plt.tight_layout()
# dog.jpg 예측 결과 이미지를 파일로 저장합니다.
plt.savefig(os.path.join(RESULT_DIR, "02_cifar10_dog_prediction.png"), dpi=150)
# figure를 닫아 메모리를 해제합니다.
plt.close()
# 저장 완료 메시지를 출력합니다.
print("[저장] 02_cifar10_dog_prediction.png")

# 테스트 이미지 앞 10장에 대한 예측 확률을 계산합니다.
predictions = model.predict(x_test[:10])
# 예측 결과를 시각화할 2행 5열의 서브플롯을 생성합니다.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
# 각 서브플롯에 대해 순회합니다.
for i, ax in enumerate(axes.flat):
    # i번째 테스트 이미지를 컬러로 표시합니다.
    ax.imshow(x_test[i])
    # 예측 확률이 가장 높은 클래스의 인덱스를 예측 레이블로 선택합니다.
    pred_label = np.argmax(predictions[i])
    # 실제 정답 레이블을 정수로 가져옵니다.
    true_label = int(y_test[i].item())
    # 예측이 맞으면 초록색, 틀리면 빨간색으로 제목 색상을 설정합니다.
    color = "green" if pred_label == true_label else "red"
    # 예측 클래스 이름(P)과 실제 클래스 이름(T)을 제목으로 표시합니다.
    ax.set_title(f"P:{CLASS_NAMES[pred_label]}\nT:{CLASS_NAMES[true_label]}", fontsize=9, color=color)
    # 축 눈금을 숨깁니다.
    ax.axis("off")
# 전체 figure의 상단 제목을 설정합니다.
plt.suptitle("CIFAR-10 CNN Prediction Results", fontsize=14, fontweight="bold")
# 서브플롯 간 간격을 자동 조정합니다.
plt.tight_layout()
# 예측 결과 이미지를 파일로 저장합니다.
plt.savefig(os.path.join(RESULT_DIR, "02_cifar10_predictions.png"), dpi=150)
# figure를 닫아 메모리를 해제합니다.
plt.close()
# 저장 완료 메시지를 출력합니다.
print("[저장] 02_cifar10_predictions.png")

# 과제 02 완료 메시지를 출력합니다.
print("\n과제 02 완료!")

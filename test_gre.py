import numpy as np
import struct
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# 设置 Matplotlib 字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False
# ----------------------------
# 1. 数据读取函数（保持不变）
# ----------------------------
def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images


def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


# ----------------------------
# 2. 加载并预处理数据
# ----------------------------
data_dir = 'MNIST'
train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

print("加载MNIST数据集...")
x_train = read_mnist_images(train_images_path)
y_train = read_mnist_labels(train_labels_path)
x_test = read_mnist_images(test_images_path)
y_test = read_mnist_labels(test_labels_path)

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ----------------------------
# 3. 数据增强（新增优化点1）
# 目的：通过随机变换生成更多样化的训练样本，防止过拟合
# ----------------------------
datagen = ImageDataGenerator(
    rotation_range=10,  # 随机旋转±10度
    zoom_range=0.1,  # 随机缩放±10%
    width_shift_range=0.1,  # 水平平移±10%
    height_shift_range=0.1  # 垂直平移±10%
)
# 拟合数据生成器以计算数据统计信息
datagen.fit(x_train)

# ----------------------------
# 4. 构建优化后的CNN模型
# ----------------------------
model = models.Sequential([
    # 第一个卷积块：卷积+批归一化+池化（新增优化点2：批归一化）
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),  # 加速训练并提高稳定性
    layers.MaxPooling2D((2, 2)),

    # 第二个卷积块
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # 第三个卷积块
    layers.Conv2D(128, (3, 3), activation='relu'),  # 增加卷积核数量
    layers.BatchNormalization(),

    # 分类器部分：加入Dropout防止过拟合（新增优化点3：Dropout正则化）
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # 随机丢弃50%的神经元，防止过拟合
    layers.Dense(10, activation='softmax')
])

print("\n优化后的模型结构:")
model.summary()

# ----------------------------
# 5. 编译模型（优化优化器参数）
# ----------------------------
# 使用带动量的Adam优化器（新增优化点4：优化器参数调整）
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,  # 动量参数
    beta_2=0.999
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# 6. 训练回调函数（新增优化点5：学习率调度和早停）
# ----------------------------
# 当验证损失不再下降时，降低学习率
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # 监控验证损失
    factor=0.5,  # 学习率变为原来的1/2
    patience=3,  # 3个epoch无改善则调整
    min_lr=1e-6,  # 最小学习率
    verbose=1
)

# 早停策略：防止过拟合，当验证准确率不再提升时停止训练
early_stopper = EarlyStopping(
    monitor='val_accuracy',  # 监控验证准确率
    patience=5,  # 5个epoch无改善则停止
    restore_best_weights=True,  # 恢复最佳权重
    verbose=1
)

# ----------------------------
# 7. 训练模型
# ----------------------------
print("\n开始训练优化后的模型...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),  # 使用数据增强
    epochs=30,  # 增加epochs，但通过早停控制实际训练轮次
    validation_data=(x_test, y_test),
    callbacks=[lr_scheduler, early_stopper]  # 应用回调函数
)

# ----------------------------
# 8. 评估模型性能
# ----------------------------
print("\n在测试集上评估模型...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"优化后模型的测试集准确率: {test_acc:.4f}")

# ----------------------------
# 9. 可视化训练过程
# ----------------------------
plt.figure(figsize=(14, 5))

# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.title('训练与验证准确率')
plt.legend()

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title('训练与验证损失')
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# 10. 可视化预测结果
# ----------------------------
n = 5
indices = np.random.choice(len(x_test), n)
samples = x_test[indices]
predictions = model.predict(samples)

plt.figure(figsize=(10, 4))
for i in range(n):
    plt.subplot(1, n, i + 1)
    plt.imshow(samples[i].reshape(28, 28), cmap='gray')
    pred_label = np.argmax(predictions[i])
    true_label = np.argmax(y_test[indices[i]])
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"预测: {pred_label}\n真实: {true_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()

# 保存优化后的模型
model.save('mnist_optimized_cnn_model.h5')
print("\n优化后的模型已保存为 'mnist_optimized_cnn_model.h5'")

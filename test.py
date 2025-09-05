import numpy as np
import struct
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 设置 Matplotlib 字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False
# ----------------------------
# 1. 定义函数：读取MNIST数据集文件
# ----------------------------

def read_mnist_images(filename):
    """
    读取MNIST图像文件（idx3-ubyte格式）
    参数:
        filename: 图像文件路径
    返回:
        解析后的图像数据，形状为(样本数, 28, 28)
    """
    # 以二进制只读模式打开文件
    with open(filename, 'rb') as f:
        # 解析文件头部信息
        # '>IIII'表示使用大端字节序解析4个无符号整数
        # 分别对应：魔数、图像数量、行数、列数
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        # 读取图像数据：每个像素是0-255的无符号整数
        # 然后重塑为(样本数, 行数, 列数)的三维数组
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

    return images


def read_mnist_labels(filename):
    """
    读取MNIST标签文件（idx1-ubyte格式）
    参数:
        filename: 标签文件路径
    返回:
        解析后的标签数据，形状为(样本数,)
    """
    # 以二进制只读模式打开文件
    with open(filename, 'rb') as f:
        # 解析文件头部信息
        # '>II'表示使用大端字节序解析2个无符号整数
        # 分别对应：魔数、标签数量
        magic, num_labels = struct.unpack('>II', f.read(8))

        # 读取标签数据：每个标签是0-9的无符号整数
        labels = np.fromfile(f, dtype=np.uint8)

    return labels


# ----------------------------
# 2. 加载本地MNIST数据集
# ----------------------------

# 指定本地MNIST数据集文件夹路径
# 请确保该文件夹下包含4个文件：
# train-images.idx3-ubyte, train-labels.idx1-ubyte,
# t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte
data_dir = 'MNIST'

# 构建各文件的完整路径
train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

# 加载训练集和测试集
print("正在加载MNIST数据集...")
x_train = read_mnist_images(train_images_path)  # 训练图像
y_train = read_mnist_labels(train_labels_path)  # 训练标签
x_test = read_mnist_images(test_images_path)  # 测试图像
y_test = read_mnist_labels(test_labels_path)  # 测试标签

# 打印数据集基本信息
print(f"训练集: 图像 {x_train.shape}, 标签 {y_train.shape}")
print(f"测试集: 图像 {x_test.shape}, 标签 {y_test.shape}")

# ----------------------------
# 3. 数据预处理
# ----------------------------

# 归一化：将像素值从0-255范围转换到0-1范围
# 同时将数据类型转换为float32，适合神经网络处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 重塑图像形状，添加通道维度
# CNN通常需要明确的通道信息，MNIST是灰度图，所以通道数为1
# 新形状为(样本数, 28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 将标签转换为独热编码格式
# 例如，数字5会被编码为[0,0,0,0,0,1,0,0,0,0]
# 这是多分类问题的标准处理方式
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ----------------------------
# 4. 构建卷积神经网络(CNN)模型
# ----------------------------

# 创建序贯模型（线性堆叠层）
model = models.Sequential([
    # 第一个卷积层：32个3x3卷积核，ReLU激活函数
    # 输入形状为(28, 28, 1)，对应28x28像素的灰度图
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # 第一个池化层：2x2最大池化，将特征图尺寸缩小一半
    layers.MaxPooling2D((2, 2)),

    # 第二个卷积层：64个3x3卷积核，ReLU激活函数
    layers.Conv2D(64, (3, 3), activation='relu'),

    # 第二个池化层：2x2最大池化
    layers.MaxPooling2D((2, 2)),

    # 第三个卷积层：64个3x3卷积核，ReLU激活函数
    layers.Conv2D(64, (3, 3), activation='relu'),

    # 展平层：将多维特征图转换为一维向量
    layers.Flatten(),

    # 全连接层：64个神经元，ReLU激活函数
    layers.Dense(64, activation='relu'),

    # 输出层：10个神经元（对应0-9数字），softmax激活函数
    # softmax确保输出值总和为1，可解释为概率
    layers.Dense(10, activation='softmax')
])

# 打印模型结构摘要
print("\n模型结构:")
model.summary()

# ----------------------------
# 5. 编译和训练模型
# ----------------------------

# 编译模型
# - 优化器：Adam（一种高效的梯度下降优化算法）
# - 损失函数：categorical_crossentropy（多分类问题常用）
# - 评估指标：accuracy（准确率）
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
# - 训练数据：x_train, y_train
# - 训练轮次：10（将整个训练集迭代10次）
# - 批次大小：64（每次迭代使用64个样本计算梯度）
# - 验证集比例：0.1（用10%的训练数据作为验证集）
print("\n开始训练模型...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1  # 从训练集中划分10%作为验证集
)

# ----------------------------
# 6. 评估模型性能
# ----------------------------

print("\n在测试集上评估模型...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试集准确率: {test_acc:.4f}")  # 通常能达到99%以上

# ----------------------------
# 7. 可视化训练过程
# ----------------------------

# 创建一个图形，包含两个子图
plt.figure(figsize=(12, 4))

# 第一个子图：训练准确率和验证准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.title('训练与验证准确率')
plt.legend()

# 第二个子图：训练损失和验证损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title('训练与验证损失')
plt.legend()

# 调整布局，避免重叠
plt.tight_layout()
plt.show()

# ----------------------------
# 8. 可视化预测结果
# ----------------------------

# 随机选择5个测试样本进行预测
n = 5
indices = np.random.choice(len(x_test), n)  # 随机选择5个索引
samples = x_test[indices]  # 获取对应的图像
predictions = model.predict(samples)  # 预测结果

# 显示预测结果
plt.figure(figsize=(10, 4))
for i in range(n):
    plt.subplot(1, n, i + 1)
    # 显示图像（需要将形状从(28,28,1)转换为(28,28)）
    plt.imshow(samples[i].reshape(28, 28), cmap='gray')
    # 获取预测标签和真实标签
    pred_label = np.argmax(predictions[i])  # 预测概率最大的类别
    true_label = np.argmax(y_test[indices[i]])  # 真实标签
    # 显示标题，正确预测为绿色，错误为红色
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"预测: {pred_label}\n真实: {true_label}", color=color)
    plt.axis('off')  # 关闭坐标轴

plt.tight_layout()
plt.show()

# ----------------------------
# 9. 保存模型
# ----------------------------

model.save('mnist_cnn_model.h5')
print("\n模型已保存为 'mnist_cnn_model.h5'")

# -*- coding: utf-8 -*-
"""
猫狗图像二分类识别项目（MobileNetV2 迁移学习）—— CPU 优化版
@author: Administrator
@date: 2026-06-19

针对本机（i5-8365U 4核8线程 / 16GB / 无独立显卡）的优化点：
1. 主干冻结时权重不变，特征只前向计算一次并缓存，之后只训练分类头，
   将 5 轮 × 1000 张的主干推理压缩成 1 轮，是 CPU 上最大的提速项；
2. 只使用 4 个计算线程（物理核数），避免 8 线程在 15W 处理器上过热降频；
3. JPEG 使用 INTEGER_FAST 解码，并一次性读入训练数据，避免每轮重复解码；
4. 分类头用 numpy 特征训练，Keras 会做全局 shuffle，修复原来 shuffle
   缓冲区(200)小于单类样本数(496)导致前几批全是同一类的问题；
5. 只使用 cat.* / dog.* 原始样本，剔除 dog 目录中约 43% 被误标为狗的猫图；
6. 测试集改为批量推理并输出准确率，所有图最后只调用一次 plt.show()。
"""

import os

# 线程数必须在 import tensorflow 之前设置
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # 减少冗余日志
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")  # 算子内并行线程
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")  # 算子间并行线程

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 全局配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 超参数 ----------------
DATA_ROOT = r"G:\深度学习与开发\Jupyter Notebook\train"
IMG_SIZE = 128            # 输入图片尺寸
BATCH_SIZE = 4            # 训练批次（与实验报告保持一致）
INFER_BATCH_SIZE = 32     # 特征提取/测试推理批次，大批量在 CPU 上吞吐更高
TRAIN_EACH_MAX = 500      # 每类最多训练样本数
TEST_EACH_NUM = 4         # 每类抽取测试样本数
EPOCHS = 5                # 训练轮数（与报告一致）
USE_FEATURE_CACHE = True  # True：缓存主干特征只训分类头（推荐，CPU 提速约 3 倍）
RANDOM_SEED = 42

tf.keras.utils.set_random_seed(RANDOM_SEED)


def configure_tf_runtime():
    """配置 CPU 线程数与设备。

    TensorFlow 的线程池只能在运行时初始化之前修改。Spyder 等交互式环境
    第二次运行脚本时，内核里的 TF 已经初始化，再设置会抛出
    RuntimeError: Intra op parallelism cannot be modified after initialization。
    这里捕获该异常并沿用已有配置；需要让 4 线程设置生效时重启内核即可。
    """
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        print("提示：TF 运行时已初始化，沿用当前线程配置"
              "（重启内核后可应用 4 线程设置）")

    try:
        tf.config.set_visible_devices([], 'GPU')  # 本机无独显，跳过 GPU 探测
    except RuntimeError:
        pass


configure_tf_runtime()

IMG_EXTS = ('.jpg', '.jpeg')


# ---------------- 1. 数据集加载与预处理 ----------------
def list_images(folder, prefix):
    """列出文件夹内指定前缀的图片，按文件名排序保证可复现。

    注意：本地 dog 文件夹中混入了约 12500 张数字命名的图片，抽样统计约
    43% 是猫，属于错误标注。只保留 cat.* / dog.* 的原始 Kaggle 样本，
    避免把猫当成狗训练（这是原实验中猫被误判为狗的主要原因）。
    """
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().startswith(prefix)
                  and f.lower().endswith(IMG_EXTS))


def get_img_paths(data_dir, max_each=None):
    """读取分类文件夹的图片路径"""
    cat_dir = os.path.join(data_dir, 'cat')
    dog_dir = os.path.join(data_dir, 'dog')

    if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
        raise FileNotFoundError(
            f"训练目录缺少 cat 或 dog 子目录：{data_dir}")

    cat_all = list_images(cat_dir, 'cat.')
    dog_all = list_images(dog_dir, 'dog.')

    if max_each:
        cat_all = cat_all[:max_each]
        dog_all = dog_all[:max_each]

    return cat_all, dog_all


def decode_img(file_path, label):
    """TF 数据集图片解码、缩放、归一化"""
    img_raw = tf.io.read_file(file_path)
    # INTEGER_FAST 解码比默认整数算法更快，128×128 下画质差异可忽略
    img = tf.io.decode_jpeg(img_raw, channels=3, dct_method='INTEGER_FAST')
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]) / 255.0
    return img, label


def make_image_ds(paths, labels):
    """图片路径 -> (图片, 标签) 数据集"""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    return ds.map(decode_img, num_parallel_calls=2)  # 4 核机器上 2 个解码线程足够


def make_extract_ds(paths, labels):
    """用于主干特征提取/测试推理：大批量、无缓存"""
    return make_image_ds(paths, labels).batch(INFER_BATCH_SIZE).prefetch(1)


def make_train_ds(paths, labels):
    """端到端训练用数据集：缓存后再打乱，保证每批类别混合"""
    ds = make_image_ds(paths, labels)
    # 缓存放在 shuffle 之前，且缓冲区覆盖全量样本，避免同类样本扎堆
    ds = ds.cache()
    ds = ds.shuffle(min(len(paths), 1000), seed=RANDOM_SEED,
                    reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(1)


# ---------------- 2. 数据集拆分：训练集 + 测试集 ----------------
train_root = DATA_ROOT
cat_full, dog_full = get_img_paths(train_root, TRAIN_EACH_MAX)

if len(cat_full) <= TEST_EACH_NUM or len(dog_full) <= TEST_EACH_NUM:
    raise ValueError("样本数不足，无法划分测试集")

# 从每类末尾抽取测试样本，不参与训练
test_cat = cat_full[-TEST_EACH_NUM:]
test_dog = dog_full[-TEST_EACH_NUM:]
train_cat = cat_full[:-TEST_EACH_NUM]
train_dog = dog_full[:-TEST_EACH_NUM]

train_paths = train_cat + train_dog
train_labels = [0] * len(train_cat) + [1] * len(train_dog)
test_paths = test_cat + test_dog
test_labels = [0] * len(test_cat) + [1] * len(test_dog)


# ---------------- 3. MobileNetV2 迁移学习模型 ----------------
def build_models(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """返回：完整模型（报告/推理用）、主干特征模型、分类头模型。

    三个模型共享同一套层对象，训练分类头后完整模型自动同步权重。
    """
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base.trainable = False  # 冻结主干

    # 分类头（与报告一致：GAP + Dense64 + Dropout0.3 + Dense2）
    gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
    dense = tf.keras.layers.Dense(64, activation='relu', name='head_dense')
    dropout = tf.keras.layers.Dropout(0.3, name='head_dropout')
    output = tf.keras.layers.Dense(2, activation='softmax', name='head_out')

    # 特征输入 -> 分类头（缓存特征后只训练这一部分）
    head_in = tf.keras.layers.Input(shape=(1280,), name='feature_input')
    head_model = tf.keras.Model(head_in, output(dropout(dense(head_in))),
                               name='classifier_head')

    # 完整模型：主干 + GAP + 同一套分类头
    gap_out = gap(base.output)
    full_model = tf.keras.Model(base.input, output(dropout(dense(gap_out))),
                                name='mobilenetv2_dogs_vs_cats')

    # 主干输出特征（4×4×1280 -> GAP 前）用于一次性特征提取
    feature_model = tf.keras.Model(base.input, gap_out, name='feature_extractor')
    return full_model, feature_model, head_model


full_model, feature_model, head_model = build_models()
full_model.summary()


# ---------------- 4. 图1：训练集样本可视化 ----------------
vis_ds = make_image_ds(train_paths, train_labels).shuffle(
    len(train_paths), seed=RANDOM_SEED).batch(BATCH_SIZE)
imgs, labs = next(iter(vis_ds))

plt.figure(figsize=(10, 5))
for i in range(BATCH_SIZE):
    plt.subplot(2, 4, i + 1)
    tag = "猫" if labs.numpy()[i] == 0 else "狗"
    plt.title(f"标签：{tag}")
    plt.imshow(imgs.numpy()[i])
    plt.axis("off")
plt.suptitle("图1 训练集猫狗样本可视化")
plt.tight_layout()


# ---------------- 5. 模型训练 ----------------
train_start = time.time()

if USE_FEATURE_CACHE:
    # 核心提速：主干冻结，特征只算一次（1000 张约 6~10 秒）
    t0 = time.time()
    train_feats = feature_model.predict(
        make_extract_ds(train_paths, train_labels), verbose=0).astype(np.float32)
    print(f"✅ 主干特征提取完成：{train_feats.shape}，"
          f"用时 {time.time() - t0:.1f}s")

    head_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    # 缓存特征后每轮只更新 8 万参数，单轮耗时约 1 秒
    history = head_model.fit(
        train_feats,
        np.asarray(train_labels, dtype=np.int32),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1
    )
else:
    full_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    history = full_model.fit(
        make_train_ds(train_paths, train_labels),
        epochs=EPOCHS,
        verbose=1
    )

print(f"✅ 训练完成，总用时 {time.time() - train_start:.1f}s")

# ---------------- 6. 训练曲线可视化 ----------------
loss_record = history.history['loss']
acc_record = history.history['accuracy']
x_axis = range(1, EPOCHS + 1)

plt.figure(figsize=(10, 4), dpi=150)

# 图2：训练损失变化曲线
plt.subplot(1, 2, 1)
plt.plot(x_axis, loss_record, color='#1f77b4', linewidth=2,
         marker='o', markersize=5, label='训练损失')
for x, y in zip(x_axis, loss_record):
    plt.text(x, y, f'{y:.3f}', fontsize=8, ha='center', va='bottom')
plt.xlabel("训练轮次 Epoch")
plt.ylabel("损失值 Loss")
plt.title("图2 训练损失变化曲线")
plt.xticks(x_axis)
plt.legend()
plt.grid(alpha=0.3, linestyle='--')

# 图3：训练准确率变化曲线
plt.subplot(1, 2, 2)
plt.plot(x_axis, acc_record, color='#ff7f0e', linewidth=2,
         marker='s', markersize=5, label='训练准确率')
for x, y in zip(x_axis, acc_record):
    plt.text(x, y, f'{y:.3f}', fontsize=8, ha='center', va='bottom')
plt.xlabel("训练轮次 Epoch")
plt.ylabel("准确率 Accuracy")
plt.title("图3 训练准确率变化曲线")
plt.xticks(x_axis)
plt.legend()
plt.grid(alpha=0.3, linestyle='--')

plt.tight_layout()


# ---------------- 7. 测试集预测与可视化 ----------------
def load_test_imgs(path_list, label_list):
    """加载测试图片为 numpy 数组（同时过滤损坏文件）"""
    imgs, labels = [], []
    for p, y in zip(path_list, label_list):
        try:
            img = tf.keras.utils.load_img(
                p, target_size=(IMG_SIZE, IMG_SIZE))
            imgs.append(tf.keras.utils.img_to_array(img) / 255.0)
            labels.append(y)
        except Exception:
            continue
    return np.asarray(imgs), np.asarray(labels)


test_imgs, valid_labels = load_test_imgs(test_paths, test_labels)

if len(test_imgs) > 0:
    test_start = time.time()
    # 批量推理（8 张图，几十毫秒）
    preds = full_model.predict(test_imgs, batch_size=INFER_BATCH_SIZE, verbose=0)
    test_acc = float(np.mean(np.argmax(preds, axis=1) == valid_labels))
    print(f"✅ 测试集准确率：{test_acc:.3f} "
          f"（{int(test_acc * len(valid_labels))}/{len(valid_labels)}），"
          f"推理用时 {time.time() - test_start:.2f}s")

    cols = 4
    rows = int(np.ceil(len(test_imgs) / cols))
    plt.figure(figsize=(12, 3 * rows))
    for i in range(len(test_imgs)):
        plt.subplot(rows, cols, i + 1)
        pred_idx = int(np.argmax(preds[i]))
        prob = float(preds[i][pred_idx])
        real_label = "猫" if valid_labels[i] == 0 else "狗"
        pred_label = "猫" if pred_idx == 0 else "狗"
        plt.title(f"真实：{real_label}\n预测：{pred_label}\n置信度：{prob:.3f}",
                  fontsize=9)
        plt.imshow(test_imgs[i])
        plt.axis("off")
    plt.suptitle("图4 测试集预测结果对比")
    plt.tight_layout()
else:
    print("❌ 无有效测试图片，预测可视化跳过")

# 所有图只显示一次，避免中途阻塞（在 Spyder 中可逐张查看）
plt.show()
print("✅ 所有实验图表生成完成，项目运行结束")

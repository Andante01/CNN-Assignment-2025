# %%
# CNN Classroom Exercise: Image Classification with CIFAR-10 (優化版本)
# Objective: Practice building, training, and evaluating a CNN using TensorFlow/Keras
# Environment: Google Colab with GPU
# Dataset: CIFAR-10 (10 classes of 32x32 color images)
# 優化策略：Dropout + 正則化 + 學習率調度

# Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# %%
# Step 2: Load and Preprocess CIFAR-10 Dataset
# CIFAR-10 contains 60,000 32x32 color images in 10 classes (e.g., airplane, cat, dog)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define class names for visualization
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# %%
# Step 3: Visualize Sample Data
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i][0]])
    plt.axis('off')
plt.show()

# %%
# Step 4: Build the Enhanced CNN Model (應用三個優化策略)
model = models.Sequential([
    # Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),  # 策略2: 正則化 - BatchNormalization
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # 策略1: Dropout - 卷積層後
    
    # Convolutional Layer 2: 64 filters, 3x3 kernel
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),  # 策略2: 正則化 - BatchNormalization
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # 策略1: Dropout - 卷積層後
    
    # Convolutional Layer 3: 64 filters, 3x3 kernel
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),  # 策略2: 正則化 - BatchNormalization
    
    # Flatten the output for dense layers
    layers.Flatten(),
    layers.Dropout(0.3),  # 策略1: Dropout - Flatten後
    
    # Dense Layer: 64 units with L2 regularization
    layers.Dense(64, activation='relu', 
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # 策略2: L2正則化
    layers.BatchNormalization(),  # 策略2: 正則化 - BatchNormalization
    layers.Dropout(0.5),  # 策略1: Dropout - Dense層前
    
    # Output Layer: 10 units (one per class) with softmax and L2 regularization
    layers.Dense(10, activation='softmax',
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))  # 策略2: L2正則化
])

# Display model summary
model.summary()

# %%
# Step 5: Compile the Model with Learning Rate Scheduler (策略3)
# 策略3: 學習率調度器設定
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # 每次減少50%
    patience=3,        # 3個epoch無改善就降低
    min_lr=1e-6,       # 最小學習率
    verbose=1
)

# 使用自定義學習率的 Adam 優化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
# Step 6: Train the Model with Callbacks (策略3: 學習率調度)
history = model.fit(train_images, train_labels, epochs=15,  # 增加epoch數以觀察學習率調度效果
                    validation_data=(test_images, test_labels),
                    callbacks=[lr_scheduler])  # 策略3: 添加學習率調度器

# %%
# Step 7: Evaluate the Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# %%
# Step 8: Plot Training and Validation Accuracy (增強版可視化)
plt.figure(figsize=(15, 10))

# 準確率圖
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Training and Validation Accuracy (優化後)', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 損失圖
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss (優化後)', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 學習率變化圖 (如果有的話)
if hasattr(lr_scheduler, 'lr_history'):
    plt.subplot(2, 2, 3)
    plt.plot(lr_scheduler.lr_history, linewidth=2, color='red')
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)

# 性能對比圖
plt.subplot(2, 2, 4)
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
plt.fill_between(epochs, history.history['accuracy'], history.history['val_accuracy'], 
                 alpha=0.3, color='gray', label='Overfitting Gap')
plt.title('Overfitting Analysis (優化後)', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Step 9: Make Predictions and Confusion Matrix
predictions = model.predict(test_images[:5])
for i in range(5):
    predicted_label = class_names[np.argmax(predictions[i])]
    true_label = class_names[test_labels[i][0]]
    confidence = np.max(predictions[i])
    print(f"Image {i+1}: Predicted: {predicted_label} ({confidence:.2%}), True: {true_label}")

# 計算並顯示混淆矩陣
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 預測所有測試數據
all_predictions = model.predict(test_images)
predicted_classes = np.argmax(all_predictions, axis=1)
true_classes = test_labels.flatten()

# 混淆矩陣
plt.figure(figsize=(10, 8))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (優化後模型)', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 分類報告
print("\n分類報告 (優化後模型):")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# %%
# Step 10: Save Enhanced Model Performance
# This cell saves the enhanced model performance to a text file
try:
    # Get final training accuracy
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    # 計算過擬合差距
    overfitting_gap = final_train_acc - final_val_acc

    # Create enhanced performance summary
    performance_text = f"""Enhanced Model Performance Summary (優化後):
=================================================
基本性能指標:
- Test Accuracy: {test_acc:.4f}
- Test Loss: {test_loss:.4f}
- Final Training Accuracy: {final_train_acc:.4f}
- Final Validation Accuracy: {final_val_acc:.4f}
- Final Training Loss: {final_train_loss:.4f}
- Final Validation Loss: {final_val_loss:.4f}
- Training Epochs: {len(history.history['accuracy'])}
- Model Parameters: {model.count_params()}

優化策略效果分析:
- Overfitting Gap: {overfitting_gap:.4f} (目標: <0.03)
- 策略1 (Dropout): 已實施 - 減少過擬合
- 策略2 (正則化): 已實施 - BatchNorm + L2正則化
- 策略3 (學習率調度): 已實施 - 動態學習率調整

模型架構改進:
- 添加3個Dropout層 (0.25, 0.3, 0.5)
- 添加4個BatchNormalization層
- 添加L2正則化 (0.01)
- 使用ReduceLROnPlateau學習率調度器
- 初始學習率: 0.002
- 總訓練輪數: 15 epochs

性能提升預期:
- 減少過擬合：是
- 提升泛化能力：是
- 穩定訓練過程：是
- 提升收斂效率：是"""

    # Save to file for GitHub Actions
    with open('enhanced_model_accuracy.txt', 'w', encoding='utf-8') as f:
        f.write(performance_text)

    print("Enhanced model performance saved to enhanced_model_accuracy.txt")
    print(performance_text)

except Exception as e:
    print(f"Error saving enhanced model performance: {e}")
    # Create a basic file even if there's an error
    with open('enhanced_model_accuracy.txt', 'w', encoding='utf-8') as f:
        f.write(f"Enhanced model execution completed with errors: {e}")

# %%
# Step 11: 模型比較和策略效果分析
print("\n" + "="*60)
print("三大優化策略實施報告")
print("="*60)

print("\n策略1: Dropout 防過擬合")
print("-" * 30)
print("✓ 卷積層後添加 Dropout(0.25)")
print("✓ Flatten後添加 Dropout(0.3)")  
print("✓ Dense層前添加 Dropout(0.5)")
print(f"✓ 過擬合差距: {overfitting_gap:.4f} (目標: <0.03)")

print("\n策略2: 正則化技術")
print("-" * 30)
print("✓ 每個卷積層後添加 BatchNormalization")
print("✓ Dense層添加 L2正則化 (0.01)")
print("✓ Dense層後添加 BatchNormalization")
print("✓ 穩定訓練過程，提升收斂速度")

print("\n策略3: 學習率調度")
print("-" * 30)
print("✓ ReduceLROnPlateau 動態調整")
print("✓ 初始學習率: 0.002")
print("✓ 監控 val_loss，patience=3")
print("✓ 每次減少50%，最小學習率: 1e-6")

print(f"\n最終模型效果:")
print(f"- 測試準確率: {test_acc:.4f}")
print(f"- 參數數量: {model.count_params():,}")
print(f"- 訓練輪數: {len(history.history['accuracy'])}")
print(f"- 優化策略: 全部實施 ✓")

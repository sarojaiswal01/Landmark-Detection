import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import config
import predict

def build_transfer_model(num_classes):
    """
    Builds the model with an EfficientNetB0 base and custom classification head.
    """
    print("[*] Building model with EfficientNetB0 base...")
    # Base model initialized with ImageNet weights, excluding top classification layer
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3)
    )
    
    # Freeze the base model to start training the head only
    base_model.trainable = False

    # Define Data Augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

    inputs = tf.keras.Input(shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)  # keep batchnorm in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def plot_history(history, filename):
    """
    Saves accuracy and loss curves for training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#8b5cf6', linewidth=2)
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#06b6d4', linewidth=2)
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Loss
    ax2.plot(history.history['loss'], label='Train Loss', color='#ec4899', linewidth=2)
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Val Loss', color='#f59e0b', linewidth=2)
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[+] Performance graphs saved to {filename}")

def train_model():
    """
    Main training pipeline covering dataset loading, augmentation, training, fine-tuning, and metrics generation.
    """
    # 1. Load Dataset
    if not os.path.exists(config.DATASET_DIR) or len(os.listdir(config.DATASET_DIR)) < 2:
        print("[!] Error: You need a dataset directory containing at least two landmark folders to train the model.")
        print(f"    Please structure your images as: {config.DATASET_DIR}/<Landmark_Name>/img.jpg")
        return False

    print(f"[*] Scanning dataset in {config.DATASET_DIR}...")
    
    # Create datasets using modern tf.keras.utils API
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[+] Found {num_classes} classes: {class_names}")

    # Optimize datasets for IO performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # 2. Build Model
    model, base_model = build_transfer_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        config.MODEL_PATH,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    # 3. Phase 1: Warmup custom classification head
    print("\n" + "="*50)
    print(f"[*] PHASE 1: Training classifier head (Frozen Base) for up to {config.EPOCHS_PHASE1} epochs...")
    print("="*50)
    
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_PHASE1,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb]
    )

    # 4. Phase 2: Fine-Tuning the Base Model
    print("\n" + "="*50)
    print("[*] PHASE 2: Unfreezing top conv layers of base model for Fine-Tuning...")
    print("="*50)
    
    base_model.trainable = True
    # Freeze bottom layers of base model, keep top ~30 layers un-frozen
    # EfficientNetB0 has about 230 layers. Let's unfreeze layers starting from index 200
    for layer in base_model.layers[:200]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Re-compile saves new training architecture details
    print(f"[*] Continuing training for up to {config.EPOCHS_PHASE2} epochs with a low learning rate...")
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_PHASE2,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb]
    )

    print(f"[+] Training completed. Best model saved to {config.MODEL_PATH}")

    # Combine histories for plotting
    combined_history = {}
    for metric in history_phase1.history.keys():
        combined_history[metric] = history_phase1.history[metric] + history_phase2.history[metric]

    # Create dummy History class wrapper for plotting
    class HistoryWrapper:
        def __init__(self, history_dict):
            self.history = history_dict

    plot_history(HistoryWrapper(combined_history), os.path.join(config.BASE_DIR, 'static', 'images', 'training_performance.png'))

    # 5. Evaluate and generate classification report on Validation set
    print("\n[*] Generating Validation Performance Metrics...")
    val_images = []
    val_labels = []
    
    # Extract images and true labels from val_ds
    for images, labels in val_ds:
        val_images.append(images.numpy())
        val_labels.append(labels.numpy())
        
    val_images = np.vstack(val_images)
    val_labels = np.argmax(np.vstack(val_labels), axis=1)

    # Run predictions
    val_preds = np.argmax(model.predict(val_images), axis=1)

    print("\n--- Classification Report ---")
    all_labels = list(range(len(class_names)))
    print(classification_report(val_labels, val_preds, labels=all_labels, target_names=class_names, zero_division=0))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(val_labels, val_preds, labels=all_labels))
    
    return True

if __name__ == "__main__":
    train_model()

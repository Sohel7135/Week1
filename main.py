import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, classification_report

# === SET YOUR PATHS ===
trainpath = r"C:\\Users\\Edunet Foundation\\Downloads\\project\\E waste data\\modified-dataset\\train"
testpath = r"C:\\Users\\Edunet Foundation\\Downloads\\project\\E waste data\\modified-dataset\\test"
validpath = r"C:\\Users\\Edunet Foundation\\Downloads\\project\\E waste data\\modified-dataset\\val"

# === Load Dataset ===
datatrain = tf.keras.utils.image_dataset_from_directory(trainpath, shuffle=True, image_size=(128, 128), batch_size=32)
datatest = tf.keras.utils.image_dataset_from_directory(testpath, shuffle=False, image_size=(128, 128), batch_size=32)
datavalid = tf.keras.utils.image_dataset_from_directory(validpath, shuffle=True, image_size=(128, 128), batch_size=32)

class_names = datatrain.class_names
print("Class Names:", class_names)

# === Class Distribution ===
def plot_class_distribution(dataset, title):
    class_counts = {}
    for images, labels in dataset:
        for label in labels.numpy():
            class_name = dataset.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_class_distribution(datatrain, "Training Data Distribution")

# === Data Augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# === Base Model (EfficientNetV2B0) ===
base_model = EfficientNetV2B0(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# === Full Model ===
model = Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['Accuracy']
)

# === Callbacks ===
early = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# === Training ===
history = model.fit(datatrain, validation_data=datavalid, epochs=15, callbacks=[early])

# === Evaluation ===
loss, accuracy = model.evaluate(datatest)
print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")

# === Plot Accuracy and Loss ===
acc = history.history['Accuracy']
val_acc = history.history['val_Accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# === Confusion Matrix ===
y_true = np.concatenate([y.numpy() for x, y in datatest], axis=0)
y_pred_probs = model.predict(datatest)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

# === Save Model ===
model.save("Efficient_classify.keras")

# === Grad-CAM Utilities ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img
    return Image.fromarray(np.uint8(superimposed_img))

# === Custom Class Labels ===
custom_class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
                      'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']
model = load_model("Efficient_classify.keras")

# === Inference Function with Grad-CAM ===
def classify_image(img):
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    class_name = custom_class_names[index]
    confidence = prediction[0][index]

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv")
    cam_image = save_and_display_gradcam(img_resized, heatmap)

    return f"Predicted: {class_name} (Confidence: {confidence:.2f})", cam_image

# === Gradio Interface ===
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=["text", "image"]
)
iface.launch()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
from lime.lime_image import LimeImageExplainer, SegmentationAlgorithm
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import shap
import cv2
import pickle

image_counter = 0
temp_folder = "temp_data"
output_folder = "explanations"

# Load the model and extract relevant details
def load_model_details(model_path):
    if model_path.endswith('.keras'):
        print("Loading .keras format model...")
        model = tf.keras.models.load_model(model_path, compile=False)
    elif model_path.endswith('.h5'):
        print("Loading .h5 format model...")
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        print("Loading SavedModel using TFSMLayer...")
        model = tf.keras.Sequential([
            tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        ])

    input_shape = model.input_shape[1:3]
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    print(f"Model loaded with input shape: {input_shape} and last conv layer: {last_conv_layer_name}")
    return model, last_conv_layer_name, input_shape

# Load the label encoder based on the training directory
def load_label_encoder(train_directory):
    labels = sorted(os.listdir(train_directory))
    label_encoder = {i: label for i, label in enumerate(labels)}
    print(f"Label encoder created: {label_encoder}")
    return label_encoder

def load_and_preprocess_image(filename, image_size):
    # Load and preprocess the image for model input
    print(f"Loading and preprocessing image from: {filename}")
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3)
    
    if not tf.executing_eagerly():
        image.set_shape([None, None, 3])

    image = tf.image.resize(image, [image_size[0], image_size[1]])
    image = image / 255.0
    image.set_shape([image_size[0], image_size[1], 3])

    return image

# Create a dataset from the training directory
def create_dataset(data_dir, labels, image_size, batch_size):
    print(f"Creating dataset from directory: {data_dir}")
    image_files = []
    image_labels = []

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(label_dir):
            image_files.append(os.path.join(label_dir, image_file))
            image_labels.append(label)

    label_map = {label: idx for idx, label in enumerate(labels)}
    image_labels = [label_map[label] for label in image_labels]

    dataset = tf.data.Dataset.from_tensor_slices((image_files, image_labels))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_image(x, image_size), y))
    dataset = dataset.shuffle(buffer_size=len(image_files))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Dataset created and batched")
    return dataset

# Save preprocessed data (images and labels) to a file
def save_preprocessed_data(X_train, y_train, file_path):
    print(f"Saving preprocessed data to: {file_path}")
    with open(file_path, 'wb') as file:
        pickle.dump((X_train, y_train), file)


def load_preprocessed_data(file_path):
    print(f"Loading preprocessed data from: {file_path}")
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Generate a Grad-CAM heatmap for the given image and model
    
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        preds = tf.convert_to_tensor(preds)
        class_channel = preds[:, pred_index]
        # if pred_index is None:
        #     pred_index = tf.argmax(preds[0])  # Default to the class with the highest probability
        # pred_index = tf.squeeze(pred_index)  # Ensure pred_index is a scalar tensor
        # if tf.executing_eagerly():
        #     pred_index = pred_index.numpy()  # Convert to a NumPy array
        # pred_index = int(pred_index)  # Convert to a Python integer
        # class_channel = preds[0][pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(array, heatmap, alpha=0.8):
    # Save and display the Grad-CAM heatmap overlaid on the original image
    print("Saving and displaying Grad-CAM result...")
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.jet
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((array.shape[1], array.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + array
    superimposed_img = array_to_img(superimposed_img)
    return superimposed_img

def generate_splime_mask_top_n(img_array, model, explainer, top_n=1, num_features=100, num_samples=300):
    # Generate a SP-LIME mask for the given image and model
    # Use superpixel segmentation for SP-LIME
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)

    explanation_instance = explainer.explain_instance(
        img_array, model.predict, top_labels=top_n, hide_color=0,
        num_samples=num_samples, num_features=num_features, segmentation_fn=segmentation_fn
    )
    explanation_mask = explanation_instance.get_image_and_mask(
        explanation_instance.top_labels[0], positive_only=False,
        num_features=num_features, hide_rest=True
    )[1]

    # Ensure mask is in the same shape as the input image
    mask = np.zeros_like(img_array)  # Create a mask of the same shape as img_array
    mask[explanation_mask == 1] = img_array[explanation_mask == 1]  # Overlay highlighted regions
    
    # Set non-highlighted areas to white
    mask = np.where(explanation_mask[:, :, np.newaxis] == 1, mask, 1.0)
    
    return mask, explanation_instance


def explain_image_shap(img, model, class_names, top_prediction, max_evals=1000, batch_size=50):
    # Generate SHAP explanations for the given image and model
    masker = shap.maskers.Image("inpaint_telea", img[0].shape)  # Update if necessary

    # Define a function to predict probabilities from the model
    def f(X):
        return model.predict(X)

    # Create the SHAP explainer
    explainer = shap.Explainer(f, masker, output_names=class_names)

    # Get SHAP values
    shap_values = explainer(img, max_evals=max_evals, batch_size=batch_size, outputs=shap.Explanation.argsort.flip[:1])

    return shap_values

def classify_image_and_explain(image_path, model_path, train_directory, num_samples, num_features, segmentation_alg, kernel_size, max_dist, ratio, max_evals, batch_size, explainer_types, output_folder):
    # Main function to classify the image and generate explanations
    global image_counter

    if output_folder is None:
        output_folder = "explanations"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    model, last_conv_layer_name, input_shape = load_model_details(model_path)
    label_encoder = load_label_encoder(train_directory)
    labels = list(label_encoder.values())
    
    # Load the image
    image = load_img(image_path, target_size=input_shape)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    array = img_to_array(image)
    img_array = array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class of the image
    predictions = model.predict(img_array)
    top_prediction = np.argmax(predictions[0])
    top_label = label_encoder[top_prediction]
    
    print(f"Prediction: {top_label} with probability {predictions[0][top_prediction]:.4f}")

    # Generate explanations based on user-specified types
    if 'gradcam' in explainer_types:
        model.layers[-1].activation = None
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        gradcam_image = save_and_display_gradcam(img_to_array(image), heatmap)
        gradcam_image.save(os.path.join(output_folder, f"gradcam_{image_counter}.png"))
        
    if 'lime' in explainer_types:
        # SPLIME Explanation
        explainer = LimeImageExplainer()
        splime_mask, explanation_instance = generate_splime_mask_top_n(img_array[0], model, explainer, top_n=1, num_features=num_features, num_samples=num_samples)
        # Ensure splime_mask is in [0, 1] range before saving
        splime_mask = np.clip(splime_mask, 0, 1)
        plt.imsave(os.path.join(output_folder, f"splime_{image_counter}.png"), splime_mask)

    if 'shap' in explainer_types:
        custom_image = img_to_array(image) / 255.0  # Preprocess image for SHAP
        shap_values = explain_image_shap(custom_image.reshape(1, *custom_image.shape), model, labels, top_prediction, max_evals=max_evals, batch_size=batch_size)
        shap.image_plot(shap_values[0], custom_image, labels=[top_label], show=False)
        plt.savefig(os.path.join(output_folder, f"shap_{image_counter}.png"))
        #plt.show()
        plt.close()
    
    print("Image classification and explanation process completed.")
    image_counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification and explanation script")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--train_directory", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of samples for LIME")
    parser.add_argument("--num_features", type=int, default=100, help="Number of features for LIME")
    parser.add_argument("--segmentation_alg", type=str, default='quickshift', help="Segmentation algorithm for LIME (options: quickshift, slic)")
    parser.add_argument("--kernel_size", type=int, default=4, help="Kernel size for segmentation algorithm")
    parser.add_argument("--max_dist", type=int, default=200, help="Max distance for segmentation algorithm")
    parser.add_argument("--ratio", type=float, default=0.2, help="Ratio for segmentation algorithm")
    parser.add_argument("--max_evals", type=int, default=400, help="Maximum evaluations for SHAP")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for SHAP")
    parser.add_argument("--explainer_types", type=str, default='all', help="Comma-separated list of explainers to use (options: lime, shap, gradcam). Use 'all' to include all three.")
    parser.add_argument("--output_folder", type=str, default=None, help="Output folder for explanations")

    args = parser.parse_args()
    
    explainer_types = args.explainer_types.split(',') if args.explainer_types != 'all' else ['lime', 'shap', 'gradcam']

    classify_image_and_explain(
        args.image_path, args.model_path, args.train_directory, args.num_samples,
        args.num_features, args.segmentation_alg, args.kernel_size, args.max_dist,
        args.ratio, args.max_evals, args.batch_size, explainer_types, args.output_folder
    )

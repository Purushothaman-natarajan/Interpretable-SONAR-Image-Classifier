import tensorflow as tf
import os
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import uuid
import random

# Parses command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Data Loader with Augmentation and Splits')
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--dim', type=int, default=224, help='Required image dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--target_folder', type=str, required=True, help='Folder to store the train, test, and val splits')
    parser.add_argument('--augment_data', action='store_true', help='Apply data augmentation')
    parser.add_argument('--balance', action='store_true', help='Balance the dataset')
    parser.add_argument('--split_type', type=str, choices=['random', 'stratified'], default='random',
                        help='Type of data split (random or stratified)')
    return parser.parse_args()

# Process the input images
def process_image(file_path, image_size):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

# Balances the images of a specific class
def balance_class_images(image_paths, labels, target_count, image_size, label, label_to_index, output_folder):
    print(f"Balancing class '{label}'...")
    label_idx = label_to_index.get(label, None)
    if label_idx is None:
        print(f"Label '{label}' not found in label_to_index.")
        return [], []

    image_paths = [img for img, lbl in zip(image_paths, labels) if lbl == label_idx]
    num_images = len(image_paths)

    print(f"Class '{label}' has {num_images} images before balancing.")

    balanced_images = []
    balanced_labels = []

    original_count = num_images
    synthetic_count = 0

    if num_images > target_count:
        balanced_images.extend(random.sample(image_paths, target_count))
        balanced_labels.extend([label_idx] * target_count)
        print(f"Removed {num_images - target_count} images from class '{label}'.")
    elif num_images < target_count:
        balanced_images.extend(image_paths)
        balanced_labels.extend([label_idx] * num_images)

        num_to_add = target_count - num_images
        print(f"Class '{label}' needs {num_to_add} additional images for balancing.")

        while num_to_add > 0:
            img_path = random.choice(image_paths)
            image = process_image(img_path, image_size)

            for _ in range(min(num_to_add, 5)):  # Use up to 5 augmentations per image
                augmented_image = augment_image(image)
                balanced_images.append(augmented_image)
                balanced_labels.append(label_idx)
                num_to_add -= 1
                synthetic_count += 1

        print(f"Added {synthetic_count} augmented images to class '{label}'.")
        print(f"Class '{label}' has {len(balanced_images)} images after balancing.")

    class_folder = os.path.join(output_folder, str(label_idx))
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    for i, img in enumerate(balanced_images):
        file_name = f"{uuid.uuid4()}.png"
        file_path = os.path.join(class_folder, file_name)
        save_image(img, file_path)

    print(f"Saved {len(balanced_images)} images for class '{label}' (Original: {original_count}, Synthetic: {synthetic_count}).")

    return balanced_images, balanced_labels

# Saves an image to a file
def save_image(image, file_path):
    if isinstance(image, str):
        image = process_image(image, image_size)
    if isinstance(image, tf.Tensor):
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image = tf.image.encode_png(image)
    else:
        raise ValueError("Expected image to be a TensorFlow tensor, but got a different type.")

    tf.io.write_file(file_path, image)

# Augments an image with random transformations
def augment_image(image):
    # Apply random augmentations using TensorFlow functions
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.1)
    return image

# Creates a list of data augmentation functions
def create_datagens():
    return [augment_image]

# Balances the entire dataset by balancing each class
def balance_data(images, labels, target_count, image_size, unique_labels, label_to_index, output_folder):
    print(f"Balancing data: Target count per class = {target_count}")

    all_balanced_images = []
    all_balanced_labels = []

    for label in tqdm(unique_labels, desc="Balancing classes"):
        num_images = len([img for img, lbl in zip(images, labels) if lbl == label_to_index.get(label, -1)])
        balanced_images, balanced_labels = balance_class_images(
            images, labels, target_count, image_size, label, label_to_index, output_folder
        )
        all_balanced_images.extend(balanced_images)
        all_balanced_labels.extend(balanced_labels)

    total_original_images = sum(1 for img in all_balanced_images if isinstance(img, str))
    total_synthetic_images = len(all_balanced_images) - total_original_images

    print(f"\nTotal saved images: {len(all_balanced_images)} (Original: {total_original_images}, Synthetic: {total_synthetic_images})")

    return all_balanced_images, all_balanced_labels

# Augments an image using TensorFlow functions
def tf_augment_image(file_path, label):
    image = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(file_path)), [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    augmented_image = augment_image(image)
    return augmented_image, label


def map_fn(file_path, label):
    image, label = tf.py_function(tf_augment_image, [file_path, label], [tf.float32, tf.int32])
    image.set_shape([image_size, image_size, 3])
    label.set_shape([])
    return image, label

# Loads images, splits them into train, validation, and test sets, and saves the splits
def load_and_save_splits(path, image_size, batch_size, balance, datagens, target_folder, split_type):
    all_images = []
    labels = []

    for class_folder in os.listdir(path):
        class_path = os.path.join(path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                all_images.append(img_path)
                labels.append(class_folder)  # Use the folder name as the label

    print(f"Loaded {len(all_images)} images across {len(set(labels))} classes.")
    print(f"Labels found: {set(labels)}")  # Print unique labels

    unique_labels = list(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_to_index[label] for label in labels]

    print(f"Label to index mapping: {label_to_index}")

    if split_type == 'stratified':
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_indices, test_indices = next(sss.split(all_images, encoded_labels))
    else:  # random split
        total_images = len(all_images)
        indices = list(range(total_images))
        random.shuffle(indices)
        train_indices = indices[:int(0.8 * total_images)]
        test_indices = indices[int(0.8 * total_images):]

    train_files = [all_images[i] for i in train_indices]
    train_labels = [encoded_labels[i] for i in train_indices]
    test_files = [all_images[i] for i in test_indices]
    test_labels = [encoded_labels[i] for i in test_indices]

    # Create validation and test sets
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_indices, test_indices = next(sss_val.split(test_files, test_labels))

    val_files = [test_files[i] for i in val_indices]
    val_labels = [test_labels[i] for i in val_indices]
    test_files = [test_files[i] for i in test_indices]
    test_labels = [test_labels[i] for i in test_indices]

    # Save splits
    for split_name, file_list, labels_list in [("train", train_files, train_labels), ("val", val_files, val_labels), ("test", test_files, test_labels)]:
        split_folder = os.path.join(target_folder, split_name)
        os.makedirs(split_folder, exist_ok=True)
        with open(os.path.join(split_folder, f"{split_name}_list.txt"), 'w') as file_list_file:
            for img_path, label in zip(file_list, labels_list):
                label_folder = os.path.join(split_folder, str(label))
                if not os.path.exists(label_folder):
                    os.makedirs(label_folder)
                file_list_file.write(f"{img_path}\n")
                save_image(img_path, os.path.join(label_folder, f"{uuid.uuid4()}.png"))

    print(f"Saved splits: train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}.")

# Main function to run the data loader
def main():
    args = parse_arguments()
    load_and_save_splits(args.path, args.dim, args.batch_size, args.balance, create_datagens(), args.target_folder, args.split_type)

if __name__ == "__main__":
    main()

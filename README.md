
# Interpretable-SONAR-Image-Classifier

Explainable AI for Underwater SONAR Image Classifier to enhance model interpretability with techniques like LIME, SHAP, and Grad-CAM. It employs transfer learning with popular deep learning models (e.g., VGG16, ResNet50, DenseNet121) to classify images, while providing insights into model predictions, making it suitable for use cases requiring transparency and reliability in SONAR image analysis.

The following guide details the requirements, usage, and examples for running the scripts within the project, along with how to generate explanations for model predictions.


## Prerequisites

- **Python 3.9**
- **Conda**: You can create the required environment using `environment.yaml`.

### Setting Up the Environment

1. **Conda Environment**: To ensure all dependencies are correctly installed, create a new environment using the provided `environment.yaml` file.

   ```bash
   conda env create -f environment.yaml
   ```

2. **Activate the Environment**:

   ```bash
   conda activate Interpretable-SONAR-Image-Classifier
   ```

This setup ensures compatibility and includes all necessary packages and versions as defined in the `environment.yaml` file.

### Running the Scripts

The following sections describe each script and its usage. Run these scripts directly from the command line or within a Python script.

#### 1. `data_loader.py`

This script loads, processes, splits datasets (train, val, test), and performs optional data augmentation.

**Command Line Usage:**

```sh
python data_loader.py --path <path_to_data> --target_folder <path_to_target_folder> --dim <dimension> --batch_size <batch_size> --num_workers <num_workers> [--augment_data]
```

**Arguments:**

- `--path`: Path to the raw data.
- `--target_folder`: Directory for processed data.
- `--dim`: Image resize dimension (e.g., 224 for 224x224).
- `--batch_size`: Batch size for data loading.
- `--num_workers`: Number of workers for data loading.
- `--augment_data` (optional): Enables data augmentation.

**Example:**

```sh
python data_loader.py --path "./dataset" --target_folder "./processed_data" --dim 224 --batch_size 32 --num_workers 4 --augment_data
```

**Dataset Structure:**

```plaintext
├── Dataset (Raw)
   ├── class_name_1
   │   └── *.jpg
   ├── class_name_2
   │   └── *.jpg
   ├── class_name_3
   │   └── *.jpg
   └── class_name_4
       └── *.jpg
```

#### 2. `train.py`

This script trains models with options for transfer learning and custom base models.

**Command Line Usage:**

```sh
python train.py --base_models <model_names> --shape <shape> --data_path <data_path> --log_dir <log_dir> --model_dir <model_dir> --epochs <epochs> --optimizer <optimizer> --learning_rate <learning_rate> --batch_size <batch_size>
```

**Arguments:**

- `--base_models`: List of base models (e.g., "VGG16" "DenseNet121").
- `--shape`: Image shape, e.g., `224 224 3`.
- `--data_path`: Path to processed data.
- `--log_dir`: Directory for logs.
- `--model_dir`: Directory to save models.
- `--epochs`: Number of epochs.
- `--optimizer`: Optimizer type (`adam` or `sgd`).
- `--learning_rate`: Learning rate.
- `--batch_size`: Training batch size.
- `--patience`: Early stopping patience.

**Example:**

```sh
python train.py --base_models "VGG16" "DenseNet121" --shape 224 224 3 --data_path "./processed_data" --log_dir "./logs" --model_dir "./models" --epochs 100 --optimizer "adam" --learning_rate 0.0001 --batch_size 32
```

#### 3. `test.py`

Tests the trained models and logs the results.

**Command Line Usage:**

```sh
python test.py --data_path <data_path> --base_model_name <base_model_name> --model_path <model_path> --models_folder_path <models_folder_path> --log_dir <log_dir>
```

**Arguments:**

- `--models_dir` (optional): Path to the models directory.
- `--model_path`: Specific model path (.keras file).
- `--img_path`: Image file path for testing.
- `--test_dir`: Test dataset directory.
- `--train_dir`: Directory for training data.
- `--log_dir`: Directory for logs.

**Example:**

```sh
python test.py --model_path "./models/vgg16_best_model.keras" --test_dir "./processed_data/test" --train_dir "./processed_data/train" --log_dir "./logs"
```

#### 4. `predict.py`

Makes predictions on new images.

**Command Line Usage:**

```sh
python predict.py --model_path <model_path> --img_path <img_path> --train_dir <train_dir>
```

**Arguments:**

- `--model_path`: Model file path.
- `--img_path`: Image file path.
- `--train_dir`: Directory for label decoding.

**Example:**

```sh
python predict.py --model_path "./models/vgg16_best_model.keras" --img_path "./processed_data/test/test_image.jpg" --train_dir "./processed_data/train"
```

#### 5. `classify_image_and_explain.py`

Makes predictions and generates explanations using LIME, SHAP, or Grad-CAM.

**Command Line Usage:**

```sh
python classify_image_and_explain.py --image_path <image_path> --model_path <model_path> --train_directory <train_directory> --num_samples <num_samples> --num_features <num_features> --segmentation_alg <segmentation_alg> --kernel_size <kernel_size> --max_dist <max_dist> --ratio <ratio> --max_evals <max_evals> --batch_size <batch_size> --explainer_types <explainer_types> --output_folder <output_folder>
```

**Arguments:**

- `--image_path`: Path to the image file.
- `--model_path`: Model file path.
- `--train_directory`: Directory of training images for label decoding.
- `--num_samples`: Sample count for LIME.
- `--num_features`: Feature count for LIME.
- `--segmentation_alg`: Segmentation algorithm for LIME.
- `--kernel_size`: Kernel size for segmentation.
- `--max_dist`: Max distance for segmentation.
- `--ratio`: Ratio for segmentation.
- `--max_evals`: Max evaluations for SHAP.
- `--batch_size`: Batch size for SHAP.
- `--explainer_types`: Comma-separated list of explainers (`lime`, `shap`, `gradcam`).
- `--output_folder`: Directory to save explanations.

**Example:**

```sh
python classify_image_and_explain.py --image_path "./images/test_image.jpg" --model_path "./models/model.keras" --train_directory "./processed_data/train" --num_samples 300 --num_features 100 --segmentation_alg "quickshift" --kernel_size 4 --max_dist 200 --ratio 0.2 --max_evals 400 --batch_size 50 --explainer_types "lime, shap, gradcam" --output_folder "./explanations"
```

### Supported Base Models

The following base models are supported for training:
- VGG16
- VGG19
- ResNet50
- ResNet101
- InceptionV3
- DenseNet121
- DenseNet201
- MobileNetV2
- Xception
- InceptionResNetV2
- NASNetLarge
- NASNetMobile
- EfficientNetB0
- EfficientNetB7

### Running Scripts in Jupyter Notebook

You can also run these scripts programmatically using Python's `subprocess` module. Here is an example of how to do this for each script:

```python
import subprocess

# Run data_loader.py
subprocess.run([
    "python", "data_loader.py",
    "--path", "./data",
    "--target_folder", "./processed_data",
    "--dim", "224",
    "--batch_size", "32",
    "--num_workers", "4",
    "--augment_data"
])

# Run train.py
subprocess.run([
    "python", "train.py",
    "--base_models", "VGG16,ResNet50",
    "--shape", "224, 224, 3",
    "--data_path", "./data",
    "--log_dir", "./logs",
    "--model_dir", "./models",
    "--epochs", "100",
    "--optimizer", "adam",
    "--learning_rate", "0.001",
    "--batch_size", "32",
    "--patience", "10"
])

# Run test.py
subprocess.run([
    "python", "test.py",
    "--models_dir", "./models",
    "--img

_path", "./images/test_image.jpg",
    "--train_dir", "./data/train",
    "--log_dir", "./logs"
])

# Run classify_image_and_explain.py
subprocess.run([
    "python", "classify_image_and_explain.py",
    "--image_path", "./images/test_image.jpg",
    "--model_path", "./models/model.h5",
    "--train_directory", "./data/train",
    "--num_samples", "300",
    "--num_features", "100",
    "--segmentation_alg", "quickshift",
    "--kernel_size", "4",
    "--max_dist", "200",
    "--ratio", "0.2",
    "--max_evals", "400",
    "--batch_size", "50",
    "--explainer_types", "lime,gradcam",
    "--output_folder", "./explanations"
])
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citing the part of the project: Under water sonar image classifier with XAI LIME

If you use our SONAR classifier or the explainer in your research, please use the following BibTeX entry.

```
@article{natarajan2024underwater,
  title={Underwater SONAR Image Classification and Analysis using LIME-based Explainable Artificial Intelligence},
  author={Natarajan, Purushothaman and Nambiar, Athira},
  journal={arXiv preprint arXiv:2408.12837},
  year={2024}
}
```
# Traffic Sign Classification Project - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Architecture](#architecture)
4. [Development Workflow](#development-workflow)
5. [Training Process](#training-process)
6. [Evaluation](#evaluation)
7. [Deployment](#deployment)
8. [Technical Deep Dive](#technical-deep-dive)
9. [File Structure](#file-structure)
10. [Configuration](#configuration)
11. [Troubleshooting](#troubleshooting)
12. [Future Improvements](#future-improvements)

---

## Project Overview

**Goal:** Build a deep learning system that can classify traffic signs from images into 43 distinct categories using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

**Models Implemented:**
- **TrafficSignCNN** - Custom Convolutional Neural Network (5Conv layers + fully connected head)
- **Vision Transformer (ViT)** - Transformer-based architecture (defined in demo.ipynb)

**Tech Stack:**
- Python 3.x
- PyTorch 2.0.1 (deep learning framework)
- torchvision 0.15.2 (computer vision transforms)
- OpenCV 4.7.0 (image processing)
- pandas (data handling)
- numpy (numerical operations)
- matplotlib (visualization)
- scikit-learn (metrics)
- Flask (web UI deployment)

---

## Dataset

### GTSRB (German Traffic Sign Recognition Benchmark)

**Source:** Kaggle - https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

**Statistics:**
- **43 classes** of traffic signs
- **Training images:** 39,209 images
- **Test images:** 12,630 images
- **Image sizes:** Variable (original), resized to model-specific dimensions
- **Format:** PNG images with CSV annotations

### Dataset Structure

```
data/
├── Train.csv           # Training annotations
├── Test.csv            # Test annotations
├── Train/              # Training images organized by class
│   ├── 0/             # Class 0 images (Speed limit 20km/h)
│   ├── 1/             # Class 1 images (Speed limit 30km/h)
│   ├── ...
│   └── 42/            # Class 42 images
└── Test/              # Test images (not organized by class)
    ├── 00000.png
    ├── 00001.png
    └── ...
```

**CSV Format:**
```
Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
27,27,6,6,22,22,0,Test/00000.png
```

The important columns:
- `ClassId` (column 7, 0-indexed): Traffic sign class (0-42)
- `Path` (column 8): Relative path to the image file

---

## Architecture

### 1. DatasetLoader.py - Custom PyTorch Dataset

```python
class GTRSBDataset(Dataset):
    def __init__(self, csv_file, root_dir, img_size=32, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 7])
        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        y_label = torch.tensor(int(self.annotations.iloc[index, 6]))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)
```

**Key points:**
- Uses OpenCV for image loading (BGR format → RGB conversion)
- Dynamically reads CSV on initialization
- Configurable image size for different models
- Applies optional transforms

### 2. DataPreparation.py - DataLoader Generation

**Purpose:** Create serialized DataLoader objects to speed up repeated training.

**Creates 6 data loaders:**
1. `train_data_loader` - CNN training (32x32, batch=256)
2. `test_data_loader` - CNN testing (32x32, batch=1)
3. `train_data_loader_vit` - ViT training (128x128, batch=256)
4. `test_data_loader_vit` - ViT testing (128x128, batch=1)
5. `train_data_loader_clahe` - CLAHE-enhanced (32x32, batch=64)
6. `test_data_loader_clahe` - CLAHE-enhanced testing (32x32, batch=1)

**Serialization:** Uses Python pickle to save loaders to `serialized_data/` folder, avoiding CSV parsing overhead on every run.

**CLAHE Transform:**
```python
class CLAHE:
    def __call__(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:,:,0]  # Luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        img = clahe.apply(img)
        img = img.reshape(img.shape + (1,))
        return img
```
- Contrast Limited Adaptive Histogram Equalization
- Operates on YCrCb luminance channel only
- Enhances local contrast while preserving details

### 3. model.py - TrafficSignCNN Architecture

**Complete Architecture:**

```python
TrafficSignCNN(
  conv: Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2)
    (2): ELU(inplace=True)
    (3): Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2)
    (5): ELU(inplace=True)
    (6): Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(256, 320, kernel_size=(3, 3), padding=(1, 1))
    (9): ELU(inplace=True)
    (10): Conv2d(320, 256, kernel_size=(3, 3), padding=(1, 1))
    (11): MaxPool2d(kernel_size=2, stride=2)
    (12): ELU(inplace=True)
  )
  classification: Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=16384, out_features=600, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=600, out_features=256, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=256, out_features=43, bias=True)
  )
)
```

**Why this architecture?**
- **5 convolutional layers**: Extract hierarchical features from images
  - Layer progression: 3 → 64 → 128 → 256 → 320 → 256 filters
  - Increasing filters captures more complex patterns
- **3 MaxPool layers**: Reduce spatial dimensions (32x32 → 16x16 → 8x8 → 4x4)
- **Final conv output**: (batch, 256, 4, 4) = 16*256 = 4,096 features
- **Classification head**:
  - Dropout(0.5) for regularization
  - Linear(4096 → 600) with ReLU
  - Dropout(0.5)
  - Linear(600 → 256) with ReLU
  - Linear(256 → 43) output layer

**Total parameters:** Approximately 5-6 million parameters

**Actations used:**
- ELU (Exponential Linear Unit) - most layers
- ReLU - some conv layers and classification head
- Softmax - inference only (CrossEntropyLoss includes it)

### 4. train.py - Training Loop

**Hyperparameters:**
- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 15 (configurable via `EPOCHS` constant)
- **Batch size:** 256 (from data loader)

**Training Process:**
```python
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")
```

**Metrics Tracked:**
- Average loss per epoch
- Accuracy per epoch (exact matches / total samples)

**Outputs:**
- `serialized_data/model.pt` - Trained model weights
- Training loss and accuracy plots (matplotlib)

### 5. evaluate.py - Model Evaluation

**Performance Metrics:**
- Overall test accuracy
- Per-class precision, recall, and F1-score (classification report)
- Manual count of correct/incorrect predictions

**Process:**
```
Load test data loader → Load trained model → Inference on all test samples
→ Compute metrics → Print results
```

**Output:**
- Accuracy: X.XXXX
- Classification report (from scikit-learn)
- Correctly/incorrect classified counts

### 6. predict.py - Single Image Inference

**Command-line tool:**
```bash
python predict.py <image_path>
```

**Process:**
1. Load label mapping from `DataProfiling/label_names.json`
2. Load model from `serialized_data/model.pt`
3. Load and preprocess image (resize to 32x32, RGB convert)
4. Run inference
5. Print: Class ID, Label, Confidence (percentage)

**Default behavior:** If no argument provided, tries `data/Test/00000.png`

### 7. setup_dataset.py - Dataset Setup Guide

**Interactive script** that:
- Checks if `data/` directory exists
- Verifies presence of `train.csv` and `test.csv`
- Counts number of images found
- Provides download instructions if dataset missing

---

## Development Workflow

### Step 1: Environment Setup

```bash
# 1. Clone the repository
cd Traffic-Sign-Classification-main

# 2. Create virtual environment (if not already)
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# Note: If you get NumPy compatibility errors:
# pip install "numpy<2" "opencv-python==4.7.0.72"
```

### Step 2: Dataset Acquisition

**Option A - Kaggle Download:**
1. Go to https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
2. Download the dataset (requires Kaggle login)
3. Extract the archive
4. Copy `Train.csv`, `Test.csv`, and all image folders to project's `data/` directory

**Option B - Manual Copy:**
```
data/
├── Train.csv
├── Test.csv
├── Train/          # ~39,209 images in subfolders 0-42
└── Test/           # ~12,630 images
```

**Verify dataset:**
```bash
python setup_dataset.py
```

Expected output:
```
'data/' directory already exists
Status:
  train.csv: [OK]
  test.csv:  [OK]
  Images:    51882 found
Dataset appears to be present - ready to create data loaders!
```

### Step 3: Data Preparation

Create serialized DataLoader objects:

```bash
python DataPreparation.py
```

**What happens:**
1. Loads CSV files and images using `GTRSBDataset`
2. Creates 6 DataLoaders with appropriate transforms/batch sizes
3. Serializes (pickles) them to `serialized_data/`
4. Creates CLAHE-enhanced variants

**Output files:**
```
serialized_data/
├── train_data_loader           (CNN, 32x32, batch=256)
├── test_data_loader            (CNN, 32x32, batch=1)
├── train_data_loader_vit       (ViT, 128x128, batch=256)
├── test_data_loader_vit        (ViT, 128x128, batch=1)
├── train_data_loader_clahe     (CLAHE, 32x32, batch=64)
└── test_data_loader_clahe      (CLAHE, 32x32, batch=1)
```

**Why serialization matters:**
- CSV parsing and image loading is slow
- Pickling saves the DataLoader state with pre-loaded references
- Saves ~30-60 seconds on each training run

### Step 4: Model Training

```bash
python train.py
```

**Expected output:**
```
Epoch:  0
0.4567  (loss)
0.8534  (accuracy)
Epoch:  1
0.3123
0.8912
...
[After 15 epochs]
```

**What happens:**
1. Load `train_data_loader` from pickle
2. Initialize fresh `TrafficSignCNN(43)` model
3. For each epoch:
   - Iterate through all batches
   - Forward pass → compute loss → backward pass → optimize
   - Track loss and accuracy
4. Save final model weights to `serialized_data/model.pt`
5. Display matplotlib plots:
   - Loss over time (training curve)
   - Accuracy over time (training curve)

**Training time estimate:** ~30-60 minutes on CPU, ~10-15 minutes on GPU

### Step 5: Model Evaluation

```bash
python evaluate.py
```

**Sample output:**
```
Correctly classified images: 11845
Incorrectly classified images: 785
Final Model Accuracy: 0.9378
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       360
           1       0.97      0.96      0.96       330
...
    accuracy                           0.94     12630
   macro avg       0.94      0.94      0.94     12630
```

**What happens:**
1. Load test data loader
2. Load trained model weights
3. Run inference on entire test set
4. Compare predictions with true labels
5. Compute metrics using scikit-learn

### Step 6: Web Application Deployment

**Install Flask:**
```bash
pip install flask
```

**Start the web app:**
```bash
python app.py
```

**Output:**
```
Starting Enhanced Traffic Sign Classifier Web App...
Open your browser and go to: http://127.0.0.1:5000
```

**Features:**
- Upload traffic sign images via drag-and-drop or file picker
- Real-time prediction with confidence score
- Gallery page showing all 43 classes with sample images
- Responsive design with Tailwind CSS
- Search functionality

**Endpoints:**
- `GET /` - Main page with upload and gallery
- `POST /predict` - API endpoint for predictions
- `GET /sample/<class_id>` - Serve sample image for gallery

---

## Training Process (Deep Dive)

### Forward Pass

1. **Input:** Batch of images (256 × 3 × 32 × 32)
   - 256 images per batch
   - 3 channels (RGB)
   - 32×32 pixels

2. **Convolutional Layers:**
   ```
   Input: (256, 3, 32, 32)
   → Conv1 (3→64, 3×3): (256, 64, 32, 32)
   → MaxPool1: (256, 64, 16, 16)
   → ELU: same shape
   → Conv2 (64→128): (256, 128, 16, 16)
   → MaxPool2: (256, 128, 8, 8)
   → Conv3 (128→256): (256, 256, 8, 8)
   → ReLU
   → Conv4 (256→320): (256, 320, 8, 8)
   → ELU
   → Conv5 (320→256): (256, 256, 8, 8)
   → MaxPool3: (256, 256, 4, 4)
   → Flatten: (256, 4096)
   ```

3. **Classification Head:**
   ```
   Flattened: (256, 4096)
   → Dropout(0.5)
   → Linear1 (4096→600): (256, 600)
   → ReLU
   → Dropout(0.5)
   → Linear2 (600→256): (256, 256)
   → ReLU
   → Linear3 (256→43): (256, 43)  # Logits
   ```

4. **Loss:** CrossEntropyLoss combines LogSoftmax + NLLLoss
   - Softmax converts logits to probabilities
   - Compares with true class label

### Backward Pass

1. **Loss.backward():**
   - Computes gradients via chain rule
   - Gradients flow from output to all parameters

2. **optimizer.step():**
   - Updates weights using Adam: `weight = weight - lr * gradient`
   - Adam maintains per-parameter adaptive learning rates

### Batch Normalization Absence

**Important:** This architecture does NOT use BatchNorm. The designer chose:
- ELU activations (self-normalizing properties)
- Dropout for regularization
- MaxPool for downsampling

This is a legacy design choice; modern CNNs typically include BatchNorm.

---

## Evaluation Deep Dive

### Metrics Explained

**Accuracy:** `correct / total`

```
Total test samples: 12,630
Correctly classified: 11,845
Accuracy: 11,845 / 12,630 = 0.9378 (93.78%)
```

**Classification Report:** (from scikit-learn)
- **Precision:** Of all predicted as class X, how many were correct?
  `TP / (TP + FP)`

- **Recall:** Of all true class X samples, how many did we find?
  `TP / (TP + FN)`

- **F1-score:** Harmonic mean of precision and recall
  `2 * (precision * recall) / (precision + recall)`

- **Support:** Number of true samples in each class

### Confusion Matrix Analysis

To manually analyze errors:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, pred_labels)
# Rows = true class, Columns = predicted class
# Diagonal = correct predictions
# Off-diagonal = misclassifications
```

**Common error patterns in GTSRB:**
- Similar speed limits (e.g., 50 vs 60 km/h)
- Stop vs Yield
- Keep left vs Keep right
- Speed limit inside/outside city (may look similar)

---

## Technical Deep Dive

### Why 32x32 for CNN?

GTSRB images are originally 32×32 or 48×48. Using 32×32:
- Preserves enough detail for classification
- Reduces computation (smaller feature maps)
- Matches dataset's inherent resolution

**However:** Original GTSRB images are 48×48 or larger. This project downscales to 32×32, losing some detail. This is a trade-off for speed.

### Why ELU Activation?

ELU (Exponential Linear Unit):
```
ELU(x) = x                     if x > 0
       = α(exp(x) - 1)         if x ≤ 0
```

**Benefits:**
- Smoother gradient near zero compared to ReLU
- Pushes mean activations closer to zero (faster convergence)
- No dying ReLU problem
- Self-normalizing property

**Used in layers 1, 2, 5** (early and middle layers)

### Why Dropout 0.5?

Dropout randomly zeroes 50% of activations during training.

**Benefits:**
- Prevents overfitting
- Forces network to learn redundant representations
- Acts as model averaging

**Applied:**
- After flatten: before first Linear layer
- After first Linear/ReLU: before second Linear layer

**Not applied after last layer:** Output needs full representation.

### Adam Optimizer Settings

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Default betas: (0.9, 0.999)
# Default eps: 1e-8
# Default weight_decay: 0
```

**Why Adam?**
- Adaptive learning rates per parameter
- Combines benefits of AdaGrad + RMSProp
- Usually converges faster than SGD
- Less hyperparameter tuning needed

**Learning rate 0.001:**
- Standard starting point
- Not too high (causes divergence), not too low (slow convergence)
- Could be refined with learning rate finder

### Data Augmentation

**Current implementation:** None during training.

**Missing opportunity:** Could add:
- Random rotations (±15°)
- Random brightness/contrast adjustments
- Random perspective transforms
- Random crops

This would improve generalization and robustness.

---

## File Structure

```
Traffic-Sign-Classification-main/
├── CLAUDE.md                    # Claude Code guidance
├── README.md                    # Project overview
├── PROJECT.md                   # This file - complete documentation
│
├── Core Python Files:
├── DatasetLoader.py             # Custom Dataset class
├── DataPreparation.py           # DataLoader generation
├── model.py                     # CNN architecture
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Single image prediction
├── setup_dataset.py             # Dataset verification
├── app.py                       # Flask web application
├── generate_samples.py          # Gallery sample generator
│
├── Data Folders:
├── data/
│   ├── Train.csv               # Training annotations
│   ├── Test.csv                # Test annotations
│   ├── Train/                  # 39,209 images organized by class (0-42)
│   └── Test/                   # 12,630 test images
│
├── Serialized Data:
├── serialized_data/
│   ├── train_data_loader       # Pickled CNN training loader
│   ├── test_data_loader        # Pickled CNN test loader
│   ├── train_data_loader_vit   # Pickled ViT training loader
│   ├── test_data_loader_vit    # Pickled ViT test loader
│   ├── train_data_loader_clahe # Pickled CLAHE training loader
│   ├── test_data_loader_clahe  # Pickled CLAHE test loader
│   ├── model.pt                # Trained CNN weights
│   └── transformer.pt          # Trained ViT weights
│
├── Static Resources:
├── static/
│   ├── samples/                # Gallery sample images (0.png, 1.png, ...)
│   └── placeholder.png         # Fallback image
│
├── Templates:
├── templates/
│   └── index.html              # Web UI template
│
├── Supporting Files:
├── DataProfiling/
│   ├── label_names.json        # Class ID to name mapping (0-indexed)
│   ├── dataprofiling.py        # Data analysis scripts
│   └── dataprofiling_notebook.ipynb
│
├── Virtual Environment:
├── venv/                       # Python virtual environment
│   ├── Scripts/
│   │   ├── activate           # Windows activation script
│   │   ├── python.exe
│   │   └── pip.exe
│   └── Lib/
│       └── site-packages/     # Installed dependencies
│
└── Misc:
├── demo.ipynb                  # Jupyter notebook with complete walkthrough
├── index.html                  # Exported Jupyter notebook (1.5MB)
└── requirements.txt            # Python dependencies
```

---

## Configuration

### Hyperparameters Reference

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| Image size (CNN) | 32×32 | DatasetLoader.py | Input resolution |
| Image size (ViT) | 128×128 | DataPreparation.py | Input resolution |
| Batch size (CNN) | 256 | DataPreparation.py | Training batch size |
| Batch size (CLAHE) | 64 | DataPreparation.py | Smaller due to enhanced contrast |
| Learning rate | 0.001 | train.py | Adam optimizer step size |
| Epochs | 15 | train.py | Training iterations |
| CNN Conv filters | [64,128,256,320,256] | model.py | Feature channels per layer |
| Dropout rates | 0.5, 0.5 | model.py | Regularization |
| CLAHE clipLimit | 2.5 | DataPreparation.py | Contrast enhancement |
| CLAHE tileGrid | (4,4) | DataPreparation.py | Tile size for CLAHE |

### Dependency Versions

From `requirements.txt`:
- torch==2.0.1
- torchvision==0.15.2
- pandas>=1.5.0
- numpy>=1.24.0
- matplotlib>=3.6.0
- Pillow>=9.5.0
- scikit-learn>=1.2.0
- opencv-python>=4.7.0

**Important:** PyTorch 2.0.1 requires `numpy<2` for compatibility.

---

## Troubleshooting

### Issue: NumPy Version Conflict

**Error:** `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**Solution:**
```bash
pip install "numpy<2"           # Downgrade to 1.x
pip install "opencv-python==4.7.0.72"  # Install compatible OpenCV
```

### Issue: Data loaders missing or outdated

**Solution:**
```bash
python DataPreparation.py
```
This recreates all pickled loaders. Delete the `serialized_data/` folder first if needed.

### Issue: Model inference gives wrong labels

**Cause:** `label_names.json` uses 1-indexed keys (1-43) but model outputs 0-indexed (0-42).

**Solution:** Ensure `label_names.json` keys are "0" through "42".
Current file (correct):
```json
{
  "0": "Speed limit (20km/h)",
  "1": "Speed limit (30km/h)",
  ...
}
```

### Issue: Web app shows 404 for sample images

**Cause:** `static/samples/` folder missing or empty.

**Solution:**
```bash
python generate_samples.py
```
Or manually copy sample images from `data/Train/` to `static/samples/` using class IDs as filenames.

### Issue: CUDA/GPU not detected

**Solution:** Model is configured for CPU (`device = torch.device('cpu')`). For GPU:
1. Check CUDA availability: `torch.cuda.is_available()`
2. Change device: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
3. Move model: `model.to(device)`
4. Move tensors: `input_tensor = transform(image).unsqueeze(0).to(device)`

### Issue: Out of memory (OOM)

**Cause:** Batch size too large for available RAM/VRAM.

**Solutions:**
1. Reduce batch size in `DataPreparation.py` (e.g., 64 or 128)
2. Regenerate data loaders with smaller batch
3. Close other applications
4. For GPU: clear cache with `torch.cuda.empty_cache()`

---

## Future Improvements

### 1. Model Enhancements
- [ ] Add Batch Normalization layers for faster convergence
- [ ] Implement data augmentation (rotations, brightness, etc.)
- [ ] Try ResNet architectures (transfer learning)
- [ ] Implement Vision Transformer properly (not just in notebook)
- [ ] Model ensemble (CNN + ViT voting)
- [ ] Quantization for faster inference

### 2. Performance Optimization
- [ ] ONNX export for deployment
- [ ] TensorRT optimization (NVIDIA)
- [ ] Mobile-friendly model (MobileNet-like architecture)
- [ ] Batch inference for multiple images

### 3. Web Application
- [ ] Multi-model selection (CNN vs ViT)
- [ ] Batch upload support
- [ ] Confidence threshold adjustment
- [ ] Historical prediction log
- [ ] User feedback for incorrect predictions
- [ ] Dockerize for easy deployment
- [ ] Add HTTPS support
- [ ] Database to store predictions

### 4. Testing & Quality
- [ ] Unit tests for DatasetLoader, model
- [ ] Integration tests for API
- [ ] Performance benchmarks
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model performance monitoring
- [ ] A/B testing framework

### 5. Documentation
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Model card with limitations
- [ ] Performance on diverse datasets
- [ ] Privacy policy for web app
- [ ] Deployment guide (AWS, Azure, GCP)

### 6. Advanced Features
- [ ] Real-time webcam classification
- [ ] Video stream processing
- [ ] Traffic sign detection + classification pipeline
- [ ] Map-based visualization of detected signs
- [ ] Mobile app (React Native/Flutter)
- [ ] Chrome extension for live traffic sign recognition

---

## Complete Usage Examples

### Command-Line Workflow

```bash
# 1. Quick test (if model already trained)
python predict.py data/Test/00042.png

# 2. Evaluate on entire test set
python evaluate.py

# 3. Retrain from scratch
rm -rf serialized_data/
python DataPreparation.py
python train.py
python evaluate.py
```

### Web App Workflow

```bash
# 1. Generate gallery samples (one-time)
python generate_samples.py

# 2. Start web server
python app.py

# 3. Open browser to http://127.0.0.1:5000
#    - Upload an image
#    - See instant prediction
#    - Browse gallery for all 43 classes
```

### Jupyter Notebook Workflow

```bash
jupyter notebook demo.ipynb
```

This contains:
- Data exploration and visualization
- DataLoader usage examples
- Training examples (with early stopping)
- ViT implementation
- Model comparison charts
- Prediction showcases

---

## Performance Expectations

**CNN Model (TrafficSignCNN):**
- Training accuracy: ~98-100% after 15 epochs
- Test accuracy: ~93-96% (depends on initialization/augmentation)
- Inference time: ~5-10ms per image on CPU
- Model size: ~18 MB (model.pt)

**ViT Model:**
- Higher capacity, potentially better accuracy
- Slower training/inference (more parameters)
- Requires 128×128 images (4× more pixels)

---

## Key Design Decisions & Rationale

1. **Pickled DataLoaders:** Avoid CSV parsing overhead (~30 sec saved per run)
2. **32×32 input for CNN:** Speed vs. accuracy trade-off (original GTSRB is 48×48)
3. **No data augmentation:** Simplicity, but limits robustness
4. **ELU activations:** Designer preference over ReLU (smoother gradients)
5. **No BatchNorm:** Unusual but works with ELU + proper initialization
6. **Dropout 0.5:** Heavy regularization for ~5M parameter model
7. **Adam optimizer:** Faster convergence than SGD for this use case
8. **15 epochs:** Sufficient for convergence (early stopping not implemented)
9. **Separate CNN and ViT:** Different architectures, different use cases
10. **Flask for deployment:** Simple, no database needed, serves static files

---

## Lessons Learned

1. **Indexing matters:** Model outputs 0-indexed classes (0-42), but original `label_names.json` was 1-indexed (1-43). Caused confusion.

2. **NumPy compatibility:** PyTorch compiled against NumPy 1.x fails with NumPy 2.x. Always pin versions.

3. **Pickle portability:** DataLoaders pickled on Windows may not load on Linux (absolute paths). Generate data loaders on target system.

4. **Memory management:** 256 batch size with 32×32 images fits in ~4GB RAM. Smaller batches if augmentations added.

5. **Validation set missing:** All evaluation is on test set. No validation set for hyperparameter tuning or early stopping.

6. **Random seed control:** No fixed random seed. Results may vary between runs.

7. **Data splitting:** Train/test split is provided by dataset. No user-controlled validation split.

---

## Credits

**Dataset:** Institut für Neuroinformatik, Ruhr-Universität Bochum, Germany
**Original repository:** https://benchmark.ini.rub.de/gtsrb.html
**Kaggle dataset:** https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

**Project structure inspired by:** Standard PyTorch tutorials and best practices

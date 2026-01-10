

# Qwen3-VL Fine-tuning on EgoLife

This repository provides code for **fine-tuning the Qwen3-VL model on the EgoLife (EgoIT-99K) dataset**.
⚠️ **Note:** The current implementation is **only compatible with EgoLife / EgoIT-style datasets**.

---

## 1. Environment Setup

We recommend using **conda** to manage the environment.

### 1.1 Create Conda Environment

```bash
conda create -n finetune python=3.10 -y
conda activate finetune
```

### 1.2 Install FFmpeg

```bash
conda install conda-forge::ffmpeg -y
```

### 1.3 CUDA Requirement

Please ensure that your **CUDA Toolkit version is >= 12.1.0**.

Check your CUDA version:

```bash
nvcc --version
```

If your CUDA Toolkit is lower than 12.1.0, install the correct version:

```bash
conda install nvidia::cuda-toolkit==12.1.0 -y
```

---

## 2. Install Dependencies

### 2.1 Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2.2 Install Other Requirements

```bash
pip install -r requirements.txt
```

---

## 3. Download Model and Dataset

### 3.1 Download Dataset (EgoIT-99K)

```bash
hf download lmms-lab/EgoIT-99K \
  --repo-type dataset \
  --local-dir /path/to/dataset
```

### 3.2 Download Model (Qwen3-VL-8B-Instruct)

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /path/to/model
```

---

## 4. Configure Dataset Path

Set the dataset path in the following file:

**File:**

```text
./qwenvl/data/__init__.py
```

**Line:** 29

Modify the dataset path to point to your local EgoIT-99K directory.

Example:

```python
DATASET_ROOT = "/absolute/path/to/data/EgoIT-99K"
```

---

## 5. Data Cleaning and Filtering

### 5.1 Set Model Card

In `data_filter_complete.py`, set the **model card** information.

**File:**

```text
data_filter_complete.py
```

**Line:** 15

Example:

```python
MODEL_CARD = "/absolute/path/to/model/Qwen3-VL-8B-Instruct"
```

### 5.2 Run Data Filtering Script

```bash
python data_filter_complete.py
```

This step will:

* Clean invalid samples
* Filter unsupported data formats
* Generate a cleaned dataset suitable for Qwen3-VL fine-tuning

---

## 6. Model Fine-tuning

### 6.1 Update Dataset Path

After data cleaning, **update the dataset path again** (same as Step 4) to point to the **cleaned dataset output directory**.

---

### 6.2 Run Supervised Fine-Tuning (SFT)

```bash
bash sft_7b.sh
```

### 6.3 Training Configuration

* Training parameters (batch size, learning rate, LoRA settings, etc.)
* Multi-GPU / distributed training options

Please refer to the **official Qwen3-VL GitHub repository** for detailed explanations and recommended configurations:

👉 [https://github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

---

## Notes

* This codebase is **specifically designed for EgoLife / EgoIT-style datasets**
* Other datasets may require:

  * Data format conversion
  * Modification of dataset loaders
* Video processing relies on **FFmpeg**, ensure it is correctly installed and available in your PATH

---

## Acknowledgements

* **Qwen3-VL** by QwenLM
* **EgoIT-99K / EgoLife** by LMMS-Lab
* Hugging Face Transformers & Datasets

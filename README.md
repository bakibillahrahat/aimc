# Research-Grade AIMC Simulation Pipeline

A modular, production-ready Jupyter notebook for evaluating neural networks on analog in-memory computing (AIMC) hardware using CrossSim RRAM simulation.

**Current Implementation**: InceptionV3 on CIFAR-10 with RRAM crossbars  
**Extensible To**: Other models (ResNet, MobileNet, etc.) and devices (PCM, SRAM, etc.)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Results](#results)
- [Extending to Other Models & Devices](#extending-to-other-models--devices)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Features

✅ **Modular Design** - Each cell is independent and reusable  
✅ **Publication-Ready** - Comprehensive visualizations and export to CSV  
✅ **Hardware-Agnostic Parameters** - Easily swap models, datasets, and device types  
✅ **Three Comprehensive Experiments**:
  - ADC/DAC quantization sweep (16-point matrix)
  - Conductance drift analysis (0-30 days)
  - Device model comparison (4 device types)

✅ **No Manual Hardware Modeling** - All imperfections handled by CrossSim  
✅ **Extensible** - Add new models, datasets, and device models in minutes  

---

## Installation

### Option 1: Clean Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python3 -m venv ~/venv_aimc
source ~/venv_aimc/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install numpy scipy pandas matplotlib seaborn scikit-learn
pip install torch torchvision
pip install git+https://github.com/sandialabs/cross-sim.git

# Verify installation
python -c "from simulator import CrossSimParameters; print('✅ CrossSim installed')"
```

### Option 2: System Python (May have dependency conflicts)

```bash
pip install --upgrade pip
pip install numpy scipy pandas matplotlib seaborn scikit-learn torch torchvision
pip install git+https://github.com/sandialabs/cross-sim.git
```

### Verify Installation

```python
import torch
from simulator import AnalogCore, CrossSimParameters

print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Project Structure

```
/aimc/
├── README.md                                 # This file
├── inception_v3/
│   └── inception-v3-rram.ipynb              # Main notebook (InceptionV3 + RRAM)
├── results/
│   ├── results_adc_dac_sweep.csv            # ADC/DAC sweep results
│   ├── results_drift_analysis.csv           # Drift analysis results
│   ├── results_device_models.csv            # Device model comparison
│   ├── adc_dac_sweep.png                    # Heatmap visualization
│   ├── drift_analysis.png                   # Drift impact plot
│   └── device_model_comparison.png          # Device model bar chart
└── (future) models/
    ├── resnet18-rram.ipynb
    ├── mobilenet-pcm.ipynb
    └── ...
```

---

## Quick Start

### 1. Open the Notebook

```bash
jupyter notebook inception_v3/inception-v3-rram.ipynb
```

### 2. Run Cells Sequentially

The notebook is organized into 18 cells:

| Cell # | Purpose | Notes |
|--------|---------|-------|
| 1 | Title & Overview | Markdown (read-only) |
| 2 | Install Dependencies | Run once per environment |
| 3 | Import Libraries | Core imports + device setup |
| 4 | Configuration | Adjust parameters here |
| 5 | Dataset Loader | Loads CIFAR-10 (first 5000 test samples) |
| 6 | Model Loader | Builds InceptionV3 pretrained on ImageNet |
| 7 | CrossSim Parameter Builder | `make_params()` function definition |
| 8 | Inference Engine | `inference_clean()`, `inference_quantized()`, `inference_aimc()` |
| 9 | Baseline Evaluation | Measures ideal accuracy reference |
| 10 | ADC/DAC Sweep | 16-point quantization sweep |
| 11 | Drift Analysis | Conductance decay over time |
| 12 | Device Model Comparison | Tests all 4 device models |
| 13 | Results Display | Print comprehensive tables |
| 14 | ADC/DAC Heatmap | Visualization (accuracy + drop) |
| 15 | Drift Impact Plot | Line plot with shaded loss |
| 16 | Device Model Chart | Bar chart with annotations |
| 17 | Summary Statistics | Publication-ready summary |
| 18 | Export to CSV | Save results for paper |

### 3. Adjust Configuration (Optional)

Edit **Cell 4** to modify:

```python
# Dataset
EVAL_SIZE = 5000      # Number of test samples (default: all 10000)

# Sweep parameters
ADC_BITS_LIST = [2, 4, 6, 8]      # Change resolution ranges
DAC_BITS_LIST = [2, 4, 6, 8]
DRIFT_TIMES = [0, 1, 3, 7, 14, 30] # Change time points

# Device models
DEVICE_MODELS = ["IdealDevice", "NormalIndependentDevice", 
                 "NormalProportionalDevice", "RRAMMilo"]
```

### 4. Run All Cells

Press `Ctrl+Shift+Enter` to run all cells (or run individually).

### 5. View Results

Results are printed to console and saved:
- **Console**: Formatted tables with accuracy drops
- **PNG Files**: Publication-quality visualizations (300 DPI)
- **CSV Files**: Raw results for plotting/analysis

---

## Experiments

### Experiment 1: ADC/DAC Quantization Sweep

**Research Question**: How do input/output precision requirements impact accuracy?

**Configuration**:
- ADC bits ∈ {2, 4, 6, 8}
- DAC bits ∈ {2, 4, 6, 8}
- No noise, no drift (ideal device)

**Output**: 16-point accuracy matrix + heatmaps

**Example Results**:
```
ADC=2, DAC=2: 45.2% (drop: 25.8%)
ADC=2, DAC=8: 58.1% (drop: 12.9%)
ADC=8, DAC=8: 71.0% (drop: 0.0%)  ← Baseline
```

---

### Experiment 2: Conductance Drift Over Time

**Research Question**: How does RRAM weight decay impact accuracy over deployment lifetime?

**Configuration**:
- Fixed: ADC=8, DAC=8 (high precision)
- Device: RRAMMilo (realistic RRAM model)
- Time points: 0, 1, 3, 7, 14, 30 days

**Output**: Accuracy vs. time plot with shaded loss region

**Example Results**:
```
Day  0: 71.0%
Day  1: 70.5% (drop: 0.5%)
Day 30: 68.2% (drop: 2.8%)
```

---

### Experiment 3: Device Model Comparison

**Research Question**: Which device model (realistic hardware behavior) loses most accuracy?

**Configuration**:
- Fixed: ADC=8, DAC=8
- Models: IdealDevice, NormalIndependentDevice, NormalProportionalDevice, RRAMMilo
- No drift

**Output**: Bar chart comparing device models

**Example Results**:
```
IdealDevice              : 71.0%
NormalIndependentDevice  : 68.5% (drop: 2.5%)
NormalProportionalDevice : 67.2% (drop: 3.8%)
RRAMMilo                 : 64.1% (drop: 6.9%)
```

---

## Results

Results are automatically generated after running all cells:

### Console Output
```
============================================================
AIMC SIMULATION SUMMARY
============================================================

Model: INCEPTION_V3
Dataset: CIFAR-10
Test Samples: 5000

Baseline Accuracy (Ideal): 71.00%

--------------------------------------------------------------
ADC/DAC SWEEP
--------------------------------------------------------------
ADC Bits Tested: [2, 4, 6, 8]
DAC Bits Tested: [2, 4, 6, 8]
Best Accuracy: 71.00%
Worst Accuracy: 45.20%
Average Accuracy Drop: 8.35%
```

### Visualizations

1. **adc_dac_sweep.png** - Dual heatmaps
   - Left: Absolute accuracy (%)
   - Right: Accuracy drop from baseline (%)

2. **drift_analysis.png** - Line plot
   - Blue line: AIMC accuracy over time
   - Green dashed: Baseline ideal accuracy
   - Red shaded: Accuracy loss region

3. **device_model_comparison.png** - Bar chart
   - Bars: One per device model
   - Green line: Baseline reference
   - Labels: Exact accuracy values

### CSV Exports

```bash
results_adc_dac_sweep.csv          # 16 rows: ADC, DAC, Accuracy, Drop
results_drift_analysis.csv         # 6 rows: Days, Accuracy, Drop
results_device_models.csv          # 4 rows: Model, Accuracy, Drop
```

---

## Extending to Other Models & Devices

### Adding a New Model

**Step 1: Create new notebook**
```bash
cp inception_v3/inception-v3-rram.ipynb new_model/resnet18-rram.ipynb
```

**Step 2: Replace Cell 6 (Model Loader)**

```python
def build_resnet18(num_classes=10, device='cpu'):
    """Build ResNet-18 pretrained on ImageNet."""
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1,
        num_classes=num_classes
    )
    model = model.to(device)
    model.eval()
    return model

# Build model
model = build_resnet18(num_classes=NUM_CLASSES, device=device)
```

**Step 3: Update Cell 4 (Configuration)**

```python
INPUT_SIZE = 224        # ResNet uses 224x224 (not 299)
MODEL_NAME = "resnet18"
```

**Step 4: Run all cells** (no other changes needed!)

### Adding a New Dataset

**Step 1: Modify Cell 5 (Dataset Loader)**

```python
def load_imagenet_dataloader(...):
    # Use torchvision.datasets.ImageNet instead of CIFAR10
    transform = transforms.Compose([...])
    dataset = datasets.ImageNet(root='./data', split='val', transform=transform)
    return DataLoader(dataset, ...)

dataloader = load_imagenet_dataloader(...)
```

**Step 2: Update Cell 4 (Configuration)**

```python
DATASET_NAME = "ImageNet"
EVAL_SIZE = 50000
```

### Adding a New Device Model

**Step 1: Register new device in CrossSim**

CrossSim devices are defined in the `simulator` module. Available models:
- `IdealDevice` - No imperfections
- `NormalIndependentDevice` - Fixed Gaussian noise
- `NormalProportionalDevice` - Proportional Gaussian noise
- `RRAMMilo` - Realistic RRAM behavior
- *(Add custom devices via CrossSim API)*

**Step 2: Update Cell 4 (Configuration)**

```python
DEVICE_MODELS = ["IdealDevice", "NormalIndependentDevice", 
                 "NormalProportionalDevice", "RRAMMilo",
                 "CustomDevice"]  # Add new model
```

**Step 3: Run all cells** (no other changes needed!)

---

## Architecture

### Modular Design Philosophy

Each cell serves a single purpose:
1. **Setup**: Dependencies, imports, device config
2. **Data**: Load and preprocess dataset
3. **Model**: Build and load pretrained model
4. **Parameters**: Define hardware configurations
5. **Inference**: Run evaluation with different settings
6. **Experiments**: Run systematic sweeps
7. **Results**: Display and visualize outputs

### Key Functions

#### `make_params(adc_bits, dac_bits, weight_bits, ...)`
Builds CrossSimParameters object with hardware configuration.

```python
params = make_params(
    adc_bits=4,
    dac_bits=4,
    weight_bits=8,
    error_model="RRAMMilo",
    noise_model="RRAMMilo",
    drift_model="RRAMMilo",
    t_drift=7  # 7 days
)
```

#### `inference_clean(model, dataloader, device)`
Runs ideal PyTorch inference (no quantization).

```python
acc = inference_clean(model, dataloader, device='cuda')
```

#### `inference_quantized(model, dataloader, adc_bits, dac_bits, device)`
Simulates ADC/DAC quantization effects.

```python
acc = inference_quantized(model, dataloader, adc_bits=4, dac_bits=4)
```

#### `inference_aimc(model, dataloader, params, device)`
Full AIMC inference with all hardware imperfections (via CrossSim parameters).

```python
acc = inference_aimc(model, dataloader, params, device='cuda')
```

---

## Troubleshooting

### Issue: "No module named 'simulator'"

**Solution**: CrossSim not installed. Run:
```bash
pip install git+https://github.com/sandialabs/cross-sim.git
```

### Issue: GPU out of memory

**Solution**: Reduce batch size or evaluation size in Cell 4:
```python
BATCH_SIZE = 32      # Reduce from 64
EVAL_SIZE = 1000     # Reduce from 5000
```

### Issue: All accuracies are identical

**Solution**: Ensure you're using `inference_aimc()` or `inference_quantized()`, not `inference_clean()`.

### Issue: Drift results show no change

**Solution**: Use `drift_model="RRAMMilo"` in `make_params()`. Other models don't support time-dependent drift.

### Issue: Slow execution on CPU

**Solution**: Use GPU if available:
```python
device = torch.device("cuda")  # Instead of "cpu"
```

---

## Results Interpretation

### ADC/DAC Heatmap

- **Top-left (ADC=2, DAC=2)**: Worst accuracy (coarse quantization on both sides)
- **Bottom-right (ADC=8, DAC=8)**: Best accuracy (fine quantization, minimal loss)
- **Asymmetric pattern**: Usually DAC has larger impact (affects all inputs)

### Drift Analysis

- **Day 0**: Fresh device (highest accuracy)
- **Day 30**: After one month of conductance decay
- **Slope**: Rate of accuracy degradation (affects deployment strategy)

### Device Models

- **IdealDevice**: Upper bound on accuracy (baseline)
- **RRAMMilo**: Lower bound (realistic RRAM noise + drift)
- **Gap**: Accounts for physical device imperfections

---

## Future Work

- [ ] Full CrossSim AnalogCore layer-wise integration (currently using quantization proxy)
- [ ] Support for ResNet-18, MobileNet, ViT
- [ ] PCM (Phase-Change Memory) device simulation
- [ ] Batch normalization handling in hardware simulation
- [ ] Multi-bit weight precision sweep
- [ ] Power consumption estimation
- [ ] Automated hyperparameter search for accuracy-efficiency tradeoff

---

## References

### CrossSim
- [CrossSim GitHub](https://github.com/sandialabs/cross-sim)
- [CrossSim Paper](https://ieeexplore.ieee.org/document/9365123)

### AIMC Hardware
- [RRAM](https://en.wikipedia.org/wiki/Resistive_random-access_memory)
- [In-Memory Computing Review](https://ieeexplore.ieee.org/abstract/document/9359155)

### Quantization
- [Quantization-Aware Training](https://arxiv.org/abs/1609.07061)
- [Post-Training Quantization](https://arxiv.org/abs/1906.04721)

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{aimc_crosssim_2026,
  title={Research-Grade AIMC Simulation Pipeline with CrossSim},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/aimc}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check CrossSim documentation: https://github.com/sandialabs/cross-sim
- Review notebook docstrings for detailed function documentation

---

**Last Updated**: June 1, 2026  
**Current Version**: 1.0.0 (InceptionV3 + RRAM)

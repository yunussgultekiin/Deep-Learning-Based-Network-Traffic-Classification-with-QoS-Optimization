# Deep Learning-Based Network Traffic Classification with QoS Optimization

> **BLM2006 – Introduction to Computer Networks | Marmara University, Department of Computer Engineering**

A deep learning-powered traffic management system that automatically classifies both regular and VPN-encrypted network traffic using flow statistics, dynamically assigning QoS (Quality of Service) priorities in real time. 

---

## Project Summary

Modern networks simultaneously carry diverse types of traffic — VoIP, video streaming, web browsing, P2P, and file transfers. Traditional QoS configurations rely on static, manually defined port/protocol rules, which fall short in dynamic environments. Furthermore, the widespread use of **VPN tunneling** encrypts packet payloads, rendering traditional Deep Packet Inspection (DPI) obsolete.

This project develops a **Deep Neural Network (DNN)** model built in PyTorch that classifies **14 distinct traffic categories** (including VPN and non-VPN variants) based purely on statistical flow features. The system is designed as an end-to-end machine learning pipeline, assigning appropriate QoS priorities mapped to DSCP values in real-time. The concept is further validated on a network topology simulated in Cisco Packet Tracer.

---

## Key Features

- **End-to-End ML Pipeline:** Clean, maintainable, and scalable Python codebase separating data processing, model training, and evaluation logic.
- **Real-World Data:** Uses the **CIC-VPN2016 (ISCX)** dataset containing ~60,000 network flows.
- **VPN Resilience:** Classifies 14 traffic types including VPN-encrypted flows without relying on payload inspection.
- **Robust Deep Learning:** A 3-hidden-layer Deep Neural Network (DNN) built in **PyTorch** utilizing Batch Normalization and Dropout.
- **Dynamic QoS Mapping:** DSCP-based priority assignment grouping VPN and non-VPN counterparts.
- **Network Simulation:** Enterprise topology implementing CBWFQ via Cisco Packet Tracer.

---

## Project Structure

The repository follows standard machine learning engineering practices for reproducibility and maintainability:

```text
├── data/
│   ├── raw/                             # Original CIC-VPN2016 dataset
│   └── processed/                       # Cleaned, normalized, and encoded data
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # EDA and feature distribution analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py               # Data loading, cleaning, and scaling logic
│   ├── model.py                         # PyTorch nn.Module defining the DNN architecture
│   ├── train.py                         # Training loop, validation, and early stopping
│   ├── evaluate.py                      # Inference, metrics, and QoS mapping logic
│   └── utils.py                         # Helper functions (Confusion Matrix, plots)
├── saved_models/
│   └── best_traffic_dnn.pt              # Trained PyTorch model weights
├── packet_tracer/
│   └── network_topology.pkt             # Cisco Packet Tracer topology file
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Technologies & Libraries

### Python Environment
| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | ≥2.0 | DNN model design, dynamic computation graphs, training loops |
| **Pandas** | ≥1.5 | Dataset management and label encoding |
| **Scikit-learn** | ≥1.1 | StandardScaler (normalization), stratified train/test splitting |
| **Matplotlib / Seaborn** | ≥3.5 | Confusion matrix heatmaps and training curve visualizations |

### Network Simulation
| Tool | Purpose |
|------|---------|
| **Cisco Packet Tracer** | Enterprise network simulation demonstrating Class-Based Weighted Fair Queuing (CBWFQ) |

---

## Dataset: CIC-VPN2016 (ISCX)

To ensure academic and real-world validity, this project utilizes the **CIC-VPN2016** dataset created by the Canadian Institute for Cybersecurity (University of New Brunswick). 

- **Size:** 59,706 flow records
- **Features:** 23 numerical, time-based flow features (e.g., `duration`, `flowPktsPerSecond`, `flowBytesPerSecond`, `mean_fiat`, `std_idle`).
- **Advantage:** Features are based on flow statistics rather than packet payloads, making the model resilient against VPN encryption.

### 14 Traffic Classes
The model classifies traffic into the following application types, distinguishing between regular and VPN-tunneled flows:
`BROWSING`, `VPN-BROWSING`, `VOIP`, `VPN-VOIP`, `STREAMING`, `VPN-STREAMING`, `FT` (File Transfer), `VPN-FT`, `CHAT`, `VPN-CHAT`, `MAIL`, `VPN-MAIL`, `P2P`, `VPN-P2P`.

---

## Model Architecture

The DNN is implemented in **PyTorch** (`src/model.py`), designed to handle the complexity of 23 input features and 14 output classes while preventing overfitting via Batch Normalization and Dropout layers.

```text
Input (23 Flow Features)
      ↓
Linear (23 → 256) + BatchNorm1d + ReLU + Dropout (0.4)
      ↓
Linear (256 → 128) + BatchNorm1d + ReLU + Dropout (0.3)
      ↓
Linear (128 → 64)  + ReLU + Dropout (0.2)
      ↓
Linear (64 → 14)   ← Output (Weighted CrossEntropyLoss applies Softmax internally)
```

**Training Configuration (`src/train.py`):**
- **Optimizer:** Adam (lr = 0.001, weight_decay = 1e-4)
- **Loss Function:** Weighted CrossEntropyLoss (to handle class imbalance)
- **Scheduler:** ReduceLROnPlateau
- **Epochs:** 100 (with Early Stopping patience = 10)
- **Batch Size:** 256

---

## QoS Priority Mapping

After classification, flows are dynamically mapped to QoS queues (`src/evaluate.py`). **Crucially, VPN traffic receives the exact same QoS priority as its non-VPN equivalent** (e.g., `VPN-VOIP` is treated the same as regular `VOIP`).

| Traffic Class (Regular & VPN) | QoS Priority | DSCP Marking | Rationale |
|-------------------------------|-------------|--------------|-----------|
| **VOIP** | 1 (Highest) | EF (46) | Extremely sensitive to delay and jitter (<20ms) |
| **STREAMING** | 2 | AF41 (34) | High bandwidth demand, moderate delay tolerance |
| **BROWSING** | 3 | AF21 (18) | Interactive; requires short wait times |
| **CHAT** | 4 | AF11 (10) | Low bandwidth, low latency text/media |
| **MAIL** | 5 | CS1 (8) | Time-insensitive; easily queued |
| **FT** (File Transfer) | 6 | BE (0) | Best Effort; purely throughput focused |
| **P2P** | 7 (Lowest) | BE (0) | Resource-heavy; restricted priority |

---

## Network Topology (Cisco Packet Tracer)

The project includes a simulated enterprise topology to demonstrate real-world enforcement of the AI's decisions.

| Device | Role | Quantity |
|--------|------|----------|
| **Router-Core** (Cisco 2911) | Central router enforcing CBWFQ policies | 1 |
| **Switch-LAN** (Cisco 2960) | Access layer switches for endpoints | 2 |
| **Generic PCs** | Generating Browsing, FT, Mail, and P2P traffic | 6 |
| **Cisco IP Phones** | Generating VoIP traffic | 2 |

**Implementation:** **CBWFQ** is configured on `Router-Core`. Traffic is matched via `class-map` based on DSCP values predicted by the model (EF, AF41, AF21, etc.), and `policy-map` guarantees specific bandwidth allocations before exiting the WAN interface.

---

## Expected Results

| Metric | Target |
|--------|--------|
| **Overall Test Accuracy** | > 95% |
| **Macro-Average F1-Score** | ≥ 0.93 |
| **VOIP F1-Score** | > 0.95 |
| **STREAMING F1-Score** | > 0.92 |
| **Average F1 for VPN Classes** | > 0.90 |

By leveraging flow-duration and inter-arrival time statistics, the model successfully distinguishes encrypted application traffic, allowing network administrators to maintain strict QoS policies without violating user privacy or breaking VPN tunnels.

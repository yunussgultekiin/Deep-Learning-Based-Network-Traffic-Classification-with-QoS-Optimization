# Deep Learning-Based Network Traffic Classification with QoS Optimization

> **BLM2006 – Introduction to Computer Networks | Marmara University, Department of Computer Engineering**

A deep learning-powered traffic management system that classifies both regular and VPN-encrypted network traffic using flow statistics and maps each class to a QoS (Quality of Service) priority. Since Cisco Packet Tracer cannot run PyTorch models natively, the AI decision is represented by injecting the predicted DSCP markings at the endpoints, which mirrors a real-world pipeline where the model decides and the network enforces QoS, allowing the router to validate CBWFQ behavior based on those AI-generated tags.

---

## Project Summary

Modern networks simultaneously carry diverse types of traffic — VoIP, video streaming, web browsing, P2P, and file transfers. Traditional QoS configurations rely on static, manually defined port/protocol rules, which fall short in dynamic environments. Furthermore, the widespread use of **VPN tunneling** encrypts packet payloads, rendering traditional Deep Packet Inspection (DPI) obsolete.

This project trains **tabular deep learning models** in PyTorch — a **DNN baseline** and an **FT-Transformer** — to classify **14 distinct traffic categories** (including VPN and non-VPN variants) using statistical flow features. The system is designed as an end-to-end deep learning pipeline, mapping each class to QoS priorities and DSCP values, then validating the enforcement on a network topology simulated in Cisco Packet Tracer.

---

## Key Features

- **End-to-End DL Pipeline:** Clean, maintainable, and scalable Python codebase separating data processing, model training, and evaluation logic.
- **Real-World Data:** Uses the **CIC-VPN2016 (ISCX)** dataset containing ~60,000 network flows.
- **VPN Resilience:** Classifies 14 traffic types including VPN-encrypted flows without relying on payload inspection.
- **Tabular Models:** DNN baseline plus FT-Transformer for flow-based tabular classification.
- **Training Enhancements:** Focal loss, class-weighting, EMA, OneCycle scheduling, optional confusion-penalty and MixUp.
- **Threshold Calibration:** Per-class threshold scaling on validation to improve class balance.
- **Dynamic QoS Mapping:** DSCP-based priority assignment grouping VPN and non-VPN counterparts.
- **Network Simulation:** Enterprise topology implementing CBWFQ via Cisco Packet Tracer.

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
- **Features:** Base flow statistics plus engineered ratios/log transforms (see the feature engineering notebook).
- **Advantage:** Features are based on flow statistics rather than packet payloads, making the model resilient against VPN encryption.

### 14 Traffic Classes
The model classifies traffic into the following application types, distinguishing between regular and VPN-tunneled flows:
`BROWSING`, `VPN-BROWSING`, `VOIP`, `VPN-VOIP`, `STREAMING`, `VPN-STREAMING`, `FT` (File Transfer), `VPN-FT`, `CHAT`, `VPN-CHAT`, `MAIL`, `VPN-MAIL`, `P2P`, `VPN-P2P`.

---

## Model Architecture

### FT-Transformer (default training)
The FT-Transformer is implemented in **PyTorch** (`models/ft_transformer.py`) and is the default model used by `models/train.py`.

- **Embedding/Transformer:** Tabular feature embedding + multi-head attention.
- **Default config:** `dim=256`, `depth=8`, `heads=8` with attention/FF/dropout regularisation.

### DNN Baseline
The DNN baseline is implemented in **PyTorch** (`models/model.py`).

```text
Input (Flow Features)
      ↓
Linear → BN → ReLU → Dropout
      ↓
Linear → BN → ReLU → Dropout
      ↓
Linear → ReLU → Dropout
      ↓
Linear → Output (num_classes)
```

**Training Configuration (`models/train.py`):**
- **Optimizer:** AdamW (lr = 1e-3, weight_decay = 1e-4)
- **Loss:** FocalLoss (gamma = 2.0) or CrossEntropy + label smoothing
- **Class Imbalance:** Auto class-weights, optional weighted sampler
- **Regularisation:** EMA, optional confusion-penalty, optional MixUp
- **Scheduler:** OneCycleLR (default) or ReduceLROnPlateau
- **Threshold Calibration:** Per-class scaling fitted on validation
- **Artifacts:** `saved_models/dnn_model.pt`, `.meta.json`, `metrics.json`, `thresholds.json`

## Notebooks

- **01_exploratory_data_analysis.ipynb:** Class distribution, feature summaries, and outlier checks.
- **02_feature_engineering.ipynb:** Derived ratios/log transforms and validation plots.
- **03_data_splitting.ipynb:** Train/val/test split and `StandardScaler` fitted on train only.

---

## QoS Priority Mapping

After classification, flows are dynamically mapped to QoS queues (`models/evaluate.py`). **Crucially, VPN traffic receives the exact same QoS priority as its non-VPN equivalent** (e.g., `VPN-VOIP` is treated the same as regular `VOIP`).

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

## Metrics & Artifacts

The latest run metrics are logged to `saved_models/metrics.json` (train history + test raw/calibrated scores). Per-class threshold scales are saved in `saved_models/thresholds.json`.

---

## Cisco Packet Tracer Simulation & Testing Guide

This section provides a step-by-step guide on how to simulate and validate the QoS decisions made by the AI model within a Cisco Packet Tracer environment. The topology assumes that endpoints receive their IP addresses via DHCP.

### 1. Router QoS Configuration (CLI)
To recognize the DSCP tags (EF, AF41, AF21, etc.) assigned by the AI model and assign them to the correct queues (CBWFQ), the following configuration must be applied to the `Router-Core` (Cisco 2911).

**Router CLI Configuration Block:**
```text
enable
configure terminal
no ip domain-lookup

! Define Traffic Classes (Based on AI-Predicted DSCP Markings)
class-map match-any CLASS-VOIP
 match ip dscp ef
class-map match-any CLASS-STREAMING
 match ip dscp af41
class-map match-any CLASS-BROWSING
 match ip dscp af21
class-map match-any CLASS-CHAT
 match ip dscp af11
class-map match-any CLASS-MAIL
 match ip dscp cs1
class-map match-any CLASS-FT
 match ip dscp default
class-map match-any CLASS-P2P
 match ip dscp default

! Create Policy Map and Assign Bandwidth Percentages
policy-map QOS-OUT
 class CLASS-VOIP
  priority percent 20
 class CLASS-STREAMING
  bandwidth percent 20
 class CLASS-BROWSING
  bandwidth percent 15
 class CLASS-CHAT
  bandwidth percent 5
 class CLASS-MAIL
  bandwidth percent 5
 class CLASS-FT
  bandwidth percent 5
 class CLASS-P2P
  bandwidth percent 5
 class class-default
  fair-queue

! Apply Policy to the WAN Interface
interface GigabitEthernet0/0
 service-policy output QOS-OUT
end
```

### 2. Traffic Generation & Simulating AI Decisions
Since the PyTorch model cannot run natively inside Packet Tracer, we simulate the "AI decision" by manually marking packets with specific **ToS (Type of Service)** values. This mimics the behavior of a real-world pipeline where the DL model labels the flow and the network enforces QoS.

**Simulation Command Table:**
Run these commands from the "Command Prompt" of the Generic PCs in the topology:

| Traffic Type (AI Class) | DSCP Label | ToS Decimal Value | Example Simulation Command |
| :--- | :--- | :--- | :--- |
| **VOIP** | EF | 184 | *(Start an IP Phone call)* |
| **STREAMING** | AF41 | 136 | `ping <target_ip> -v 136 -n 100` |
| **BROWSING** | AF21 | 72 | `ping <target_ip> -v 72 -n 100` |
| **CHAT** | AF11 | 40 | `ping <target_ip> -v 40 -n 100` |
| **MAIL** | CS1 | 32 | `ping <target_ip> -v 32 -n 100` |
| **FT / P2P** | Default | 0 | `ping <target_ip> -v 0 -n 100` |

### 3. Verification and Analysis
After initiating traffic, verify that the Router is correctly classifying the packets and enforcing the QoS policy:

1.  **Simulation Mode:** Switch to "Simulation" mode in the bottom-right corner. Click on an outgoing packet and check the "Outbound PDU Details." Verify that the DSCP field matches your expected value (e.g., `0x28` for AF21).
2.  **CLI Statistics:** Run the following command on the Router-Core terminal to monitor queue usage in real-time:
    ```text
    show policy-map interface GigabitEthernet0/0
    ```
    *If correctly configured, you will see the "packets matched" counter increasing for the specific class corresponding to the ToS value you sent.*
```
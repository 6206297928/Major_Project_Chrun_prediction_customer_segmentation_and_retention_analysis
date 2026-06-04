# Telco Customer Churn Prediction & Retention Analytics Pipeline

An end-to-end machine learning infrastructure utilizing a decoupled microservice architecture. The system consists of a high-performance predictive backend powered by a custom-trained multi-layer Artificial Neural Network (TensorFlow) and an executive-facing analytical interface built with Streamlit.

---

## 📡 Architecture Overview

The application is split cleanly into two independent, decoupled services:
1. **Inference Core (Backend):** A FastAPI service that accepts raw customer profiles, transforms categorical vectors into aligned dummy variables, normalizes numerical features via a serialized scikit-learn `StandardScaler`, and executes forward propagation through a trained TensorFlow model.
2. **Analytical Client (Frontend):** A minimalist Streamlit dashboard that handles user data entry, transmits structured JSON payloads to the inference core over HTTP POST, and maps responses into executive performance metrics and rule-based segment cohorts.

---

## 🧬 Model Architecture & Research Rationale

The predictive engine is built on a custom, deep feedforward Sequential Neural Network optimized to solve high-volatility customer attrition patterns. Standard linear models often fail to capture complex feature cross-products (e.g., how the combination of a short tenure *and* an electronic check payment exponentially multiplies risk). This architecture handles those non-linear relationships through dense hidden layer transformations.

### Network Topology
* **Input Layer:** 30 features (after full encoding alignment).
* **Hidden Layer 1:** 128 Dense Neurons + ReLU Activation + Weights & Biases Optimization.
* **Hidden Layer 2:** 64 Dense Neurons + ReLU Activation.
* **Hidden Layer 3:** 32 Dense Neurons + ReLU Activation.
* **Output Layer:** 1 Dense Neuron + Sigmoid Activation (Compresses output strictly to a range between `0.0` and `1.0` representing Churn Probability).

### Optimization Strategy
* **Skewed Dataset Tuning:** Telco customer churn is a highly imbalanced metric. Using a standard `0.50` decision threshold results in dangerous false negatives—missing customers who are actively showing warning signs. To maximize model recall, **Andrew Ng's optimization strategy** was implemented, lowering the operational decision threshold to **`0.35`**. Any probability score $\ge$ 35% is immediately flagged as a high-priority risk.
* **Feature Calibration:** Numerical variables (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`) are passed through a synchronized `StandardScaler` to normalize features into standard normal distributions (zero mean, unit variance). This prevents massive total charges values from overwhelming the network's gradient steps during backpropagation.

---

## 🏎️ End-to-End Pipeline Execution Flow

When a user triggers an evaluation via the Streamlit interface, data flows sequentially across the network infrastructure:

```text
[Streamlit UI] ──(1. JSON Data)──> [FastAPI Server] ──(2. Preprocessing)──> [TensorFlow Layers]
                                                                                   │ (3. Forward Pass)
[Prism UI Render] <──(5. Risk Class)── [Decision Gate] <──(4. Probability)────────┘

# M.Tech-Final-Project

# Project Proposal: Adversarial Attacks and Structural Defense in Graph Neural Networks for Node Classification

## 1. Introduction and Problem Statement
Graph Neural Networks (GNNs), particularly Graph Convolutional Networks (GCNs), have achieved state-of-the-art performance in various graph-based machine learning tasks such as node classification, link prediction, and graph classification. These tasks are foundational in domains including citation networks, social networks, and recommendation systems. 

Despite their success, recent research has exposed a critical vulnerability: GNNs are highly susceptible to adversarial attacks. Minor, often imperceptible perturbations to node features or graph topology (edges) can drastically degrade model performance, leading to significant misclassification errors.

**Problem Statement:**
> How do various forms of adversarial attacks (poisoning and evasion) impact the performance of GNN-based node classification, and how can structural and feature-based defense mechanisms effectively mitigate these vulnerabilities to restore model robustness?

---

## 2. Objectives
This project systematically investigates the robustness of GNNs and proposes robust defense mechanisms. The core objectives are:
1. **Baseline Implementation:** Implement and train a 2-layer Graph Convolutional Network (GCN) as the baseline model for node classification.
2. **Diverse Dataset Evaluation:** Evaluate the baseline model on both static (e.g., Cora) and dynamic/evolving datasets (e.g., synthetic/real social network).
3. **Adversarial Attack Simulation:** Implement and deploy six distinct adversarial attacks to comprehensively evaluate model vulnerabilities:
   - **Poisoning Attacks (Pre-training):** Nettack, Meta Attack, Random Structure Attack.
   - **Evasion Attacks (Post-training):** Feature Perturbation Attack, Edge Flip Attack, Gradient-Based Attack.
4. **Impact Analysis:** Quantify the impact of each attack using comprehensive evaluation metrics and identify the most critical vulnerabilities.
5. **Structural Defense Mechanism:** Design and implement a robust structural defense pipeline based on edge pruning, feature smoothing, and graph reconstruction to neutralize adversarial perturbations.
6. **Defense Evaluation:** Rigorously evaluate the proposed defense mechanism's ability to restore baseline performance and structural integrity.

---

## 3. System Overview & Architecture
The system follows an end-to-end pipeline encompassing data processing, attack simulation, and defensive reconstruction.

**Pipeline Workflow:**
1. **Data Ingestion:** Load Static and Dynamic Datasets.
2. **Baseline Training:** Train the 2-Layer GCN model on clean data.
3. **Performance Benchmarking:** Evaluate clean accuracy and confidence.
4. **Attack Injection:** Apply poisoning and evasion perturbations to graph structure and features.
5. **Impact Evaluation:** Re-evaluate the model to quantify the attack's degradation effect.
6. **Defense Implementation:** Apply structural defense (Pruning + Smoothing + Reconstruction).
7. **Model Retraining/Inference:** Re-evaluate the GCN on the sanitized, reconstructed graph.
8. **Final Evaluation:** Compare post-defense metrics against the clean baseline.

---

## 4. Datasets
To ensure a comprehensive evaluation, the project utilizes datasets with varying characteristics:

### 4.1 Static Dataset – Cora
- **Nodes:** 2,708 (Research papers)
- **Edges:** 5,429 (Citation links)
- **Features:** 1,433-dimensional bag-of-words
- **Classes:** 7 research topics
- **Purpose:** Serves as the primary benchmark for node classification and a controlled environment for evaluating attack vectors.

### 4.2 Dynamic Dataset – Social Network (Evolving Graph)
- **Characteristics:** Nodes represent users, edges represent dynamic interactions over time.
- **Purpose:** Evaluates model and defense robustness in real-world, evolving network conditions where structural anomalies may mimic organic growth.

---

## 5. Baseline Model: 2-Layer GCN
The baseline model is a standard two-layer Graph Convolutional Network designed for semi-supervised node classification.

### Mathematical Formulation:
- **Layer 1 (Feature Aggregation):**  
  `H^{(1)} = ReLU(A_hat * X * W^{(0)})`
- **Layer 2 (Classification):**  
  `Z = Softmax(A_hat * H^{(1)} * W^{(1)})`

*(Where `A_hat` is the symmetrically normalized adjacency matrix with self-loops, `X` represents node features, and `W^{(i)}` are learnable weight matrices.)*

**Expected Output:** Informative node embeddings, precise class predictions, and well-calibrated probability distributions.

---

## 6. Evaluation Metrics
The framework utilizes dual-axis metrics to evaluate both absolute performance and resilience.

**Classification Metrics:**
- Accuracy, Precision, Recall, F1-Score

**Robustness Metrics:**
- **Accuracy Drop:** Decline in accuracy post-attack.
- **Attack Success Rate (ASR):** Percentage of target nodes successfully misclassified.
- **Recovery Rate:** Percentage of performance restored by the defense mechanism relative to the baseline.

---

## 7. Phase 2: Adversarial Attack Implementation
The project categorizes attacks based on the adversary's capability and phase of intervention.

### 7.1 Poisoning Attacks (Training Phase Vulnerability)
These attacks manipulate the training graph to corrupt the learned embeddings.
1. **Nettack:** A targeted attack optimizing feature and structure perturbations to misclassify specific nodes. Highly effective at altering local neighborhood aggregation.
2. **Meta Attack:** Utilizes meta-learning to treat the graph structure as a hyperparameter, maximizing global model loss. Extremely stealthy and causes widespread degradation.
3. **Random Structure Attack:** Baseline comparison attack that randomly injects or deletes edges.

### 7.2 Evasion Attacks (Inference Phase Vulnerability)
These attacks modify the graph at test time without altering the trained model weights.
4. **Feature Perturbation Attack:** Injects targeted noise into node features. Expected to be highly critical as it directly corrupts the initial embeddings before message passing.
5. **Edge Flip Attack:** Strategically adds or removes edges to manipulate the receptive field of target nodes.
6. **Gradient-Based Attack (FGA/PGD):** Utilizes gradient information of the surrogate loss function to find the most devastating perturbations.

---

## 8. Attack Impact Analysis
Each attack will be systematically applied, followed by a comparative analysis against the baseline. 

**Expected Observation:** The *Feature Perturbation* and *Meta Attack* are anticipated to yield the highest degradation (Accuracy drop > 30%), highlighting the critical need for feature-space sanitization alongside structural defense.

---

## 9. Phase 3: Structural Defense Mechanism
The proposed defense aims to sanitize the graph before message passing occurs, neutralizing adversarial noise while preserving task-relevant information.

### Defense Pipeline Strategy:
1. **Edge Pruning:** Adversaries often connect dissimilar nodes to confuse the GCN. We compute cosine similarity between connected node features.
   - *Rule:* If `sim(i,j) < threshold`, remove edge `(i,j)`.
2. **Feature Smoothing:** Adversarial features contain high-frequency noise. We apply a low-pass filter using the graph Laplacian.
   - *Rule:* `X' = A_hat * X` (Smoothing features over local neighborhoods to dilute isolated perturbations).
3. **Graph Reconstruction:** Rebuilding the structural integrity using a k-Nearest Neighbors (k-NN) approach based on the smoothed features to replace adversarial edges with highly probable, safe connections.

---

## 10. Expected Outcomes & Defense Evaluation
The defense will be evaluated based on its ability to recover performance.

| Stage | Expected Accuracy |
| :--- | :--- |
| **Baseline** | ~80–85% |
| **After Attack** | ~40–60% |
| **After Defense** | ~75–82% |

*(Significant recovery demonstrating the viability of the structural defense).*

---

## 11. Visualization and Reporting
To clearly communicate findings, the project will generate comprehensive visual outputs:
- **Performance Graphs:** Bar charts and line plots illustrating Accuracy vs. Attack and Accuracy vs. Defense.
- **Network Visualizations:** Graph-level plots showing the organic structure, the maliciously perturbed structure, and the sanitized defended structure.
- **Embedding Spaces:** t-SNE or UMAP scatter plots showing the clustering of node embeddings before attack, after attack (clusters dispersed), and after defense (clusters restored).

---

## 12. Novelty and Contributions
- **Comprehensive Scope:** Evaluates a wide taxonomy of 6 different attack vectors rather than focusing on a single attack type.
- **Dynamic Context:** Extends traditional static graph evaluation to dynamic/evolving datasets.
- **Multi-Stage Defense:** Implements a synergistic structural defense pipeline (Pruning + Smoothing + Reconstruction) grounded in spectral graph theory.
- **End-to-End Pipeline:** Delivers a fully reproducible framework for testing GNN robustness.

---

## 13. Real-World Applications
The findings and defenses developed in this project directly apply to securing critical graph-based systems:
- **Social Networks:** Defending against fake account syndicates and engagement manipulation.
- **Financial Systems:** Securing transaction graphs against fraud camouflage.
- **Recommendation Engines:** Preventing malicious actors from manipulating product or content recommendations.

---

## 14. Future Extensions
- Integrating **Ontology-based semantic validation** to reject logically impossible edges.
- Exploring **Adversarial Training** regimens for GNNs.
- Developing **Real-time anomaly detection** systems for dynamic graphs.

---

## 15. Final Summary
> This project systematically studies the vulnerability of Graph Neural Networks to adversarial attacks in node classification tasks and proposes structural defense mechanisms to improve robustness. By evaluating multiple attacks across static and dynamic datasets, the work provides practical insights into secure graph learning systems.

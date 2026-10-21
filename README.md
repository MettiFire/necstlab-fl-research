# Federated Learning Benchmark

## Project

Research project by **NECSTLab - Politecnico di Milano** dedicated to the study of **Federated Learning** strategies for predicting sleep quality from biometric data collected via Garmin devices.

The project compares approaches offering different trade-offs between speed, accuracy, communication, and system complexity:

- **Flower Bagging**: parallel client training and local model aggregation;
- **Flower Cyclic**: sequential training, with the model moving from one client to the next;
- **NVIDIA FLARE**: XGBoost federated aggregation based on sharing gradient statistics and histograms.

The overall goal is to identify an intermediate approach that reduces communication overhead while maintaining a predictive quality close to that achievable with more informative federated training.

---

## Research Objectives

### 1. Compare Federated Learning Strategies

Measure and compare the behavior of the three benchmark approaches while maintaining, as much as possible, the same experimental setup:

- same federated dataset;
- same XGBoost-based regression model;
- same number of clients and federated rounds;
- same split between training and validation data.

The comparison aims to highlight not only which method is faster, but which offers the best compromise between model quality and operational costs.

### 2. Analyze Bottlenecks

Break down the overall training cost into its main components:

- **local training**, executed on individual clients;
- **communication**, including data/model transfer and serialization;
- **aggregation**, performed by the server or federated coordinator;
- **tree building**, tied to gradient computation and boosting iterations;
- **orchestration**, i.e., client configuration, launch, and synchronization.

This analysis helps determine whether performance is primarily driven by local computation, the network, or coordination logic.

### 3. Evaluate Quality, Efficiency, and Scalability

The benchmark considers the following dimensions:

| Dimension | Research Question |
|---|---|
| Accuracy | How well does the model predict sleep quality? |
| Time | How long does full federated training take? |
| Communication | How much data must be exchanged between clients and server? |
| Convergence | How many rounds are required to reach a target quality? |
| Scalability | How do time and costs change as the number of clients increases? |
| Robustness | How stable is the method in the presence of non-IID data, delays, or absent clients? |

The primary metrics are **MAE**, **RMSE**, total time and time per round, bytes sent and received, aggregation times, and resource utilization.

---

## Theoretical Context

### Federated Learning

In Federated Learning, data remains with the clients who own it. The coordinator manages the training and collects only models, updates, or aggregate statistics. This allows a shared model to be trained without transferring raw biometric data to a central server.

The challenge is finding the right balance between:

- amount of shared information;
- accuracy of the global model;
- computation time;
- communication bandwidth and latency;
- protection of local data.

The data distribution in this project is federated across **9 clients**. Each client holds data related to different groups of participants and can therefore introduce heterogeneity into the distribution of observations.

### Flower Bagging

In the Bagging paradigm, clients work in parallel on their local data and send the local training results to the coordinator. This approach limits the amount of information exchanged and makes execution relatively simple and fast.

**Theoretical Advantages:**

- parallelism among clients;
- relatively contained communication;
- simplified implementation and debugging.

**Theoretical Limitations:**

- each client sees only a portion of the data;
- averaging or aggregating models can lead to information loss;
- quality can suffer more significantly from non-IID data distribution.

### Flower Cyclic

In the Cyclic paradigm, the model is updated sequentially by the clients. Each client receives the current model, performs local training, and passes it to the next client or the coordinator.

**Theoretical Advantages:**

- the model can progressively incorporate information from different clients;
- the update flow is direct and lightweight;
- the number of models to aggregate can be reduced.

**Theoretical Limitations:**

- execution is less parallel;
- runtime can depend on the slowest client;
- client ordering can influence the final result.

### NVIDIA FLARE and Histogram-Based Aggregation

NVIDIA FLARE employs a richer approach for Federated XGBoost. Instead of sharing only the final local model, clients can share aggregate statistics, such as gradient histograms, useful for tree construction.

**Theoretical Advantages:**

- greater amount of information available for aggregation;
- behavior closer to centralized training;
- potential for higher accuracy on heterogeneous data;
- raw biometric data is not transferred.

**Theoretical Limitations:**

- higher network traffic and more synchronization operations;
- higher serialization and aggregation costs;
- more complex configuration and orchestration;
- potential increase in setup times and time per round.

---

## Dataset and Predictive Task

The dataset contains biometric data collected from **45 participants** who used a Garmin Vivoactive 5 for approximately one month.

- **999 training samples** distributed across 9 clients;
- **189 test samples**;
- approximately **40 numerical features** derived from biometric time series;
- target: **sleep quality score**, expressed on a 0-100 scale.

The features describe, among other aspects:

- heart rate and heart rate variability;
- respiration during sleep and wakefulness;
- stress levels and peaks;
- activity, steps, calories, and intensity;
- duration and quality of previous sleep.

The problem is framed as a regression task using the XGBoost objective `reg:squarederror`. Evaluation must consider both global quality and error discrepancies across individual clients.

---

## Research Question

> Is there an intermediate federated strategy capable of maintaining predictive quality close to NVIDIA FLARE while significantly reducing execution time and communication overhead?

This question arises from the contrast between:

- Flower approaches, which are simpler and lighter but potentially less informative;
- histogram-based approaches, which are statistically richer but more expensive in terms of communication and orchestration.

---

## Hybrid Approaches to Explore

The project considers several possible intermediate strategies:

1. **Selective histogram sharing**: sharing histograms only for the most important features;
2. **Adaptive communication**: using more communication in early rounds and reducing it as convergence approaches;
3. **Compressed histograms**: reducing the number of bins or quantizing shared statistics;
4. **Progressive histogram refinement**: starting with coarse histograms and progressively increasing precision;
5. **Federated knowledge distillation**: training a richer model and distilling its knowledge to lighter local models;
6. **Hybrid parallel-sequential**: combining groups of clients trained in parallel and sequentially.

Each hybrid approach must be evaluated against the baselines considering at least:

- speedup relative to NVIDIA FLARE;
- change in MAE;
- ratio of communicated bytes to NVIDIA FLARE bytes;
- stability of results across multiple runs;
- behavior as the number of clients and degree of data heterogeneity vary.

---

## Expected Outcome

The final goal is not simply to identify the fastest method or the one with the lowest MAE. The objective is to build a multi-objective evaluation that allows selecting the most suitable strategy based on:

- required accuracy;
- maximum available time;
- network bandwidth and latency;
- number of clients;
- level of data heterogeneity;
- privacy and reliability requirements.

The project aims to provide an experimental foundation for designing efficient Federated Learning systems for biometric data and real-world distributed scenarios.

---

## References

- [Flower Documentation](https://flower.ai/docs/)
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/)
- [Federated XGBoost](https://arxiv.org/abs/1901.08755)
- [SecureBoost: Histogram-based Federated Learning](https://arxiv.org/abs/1901.08755)

**NECSTLab - Politecnico di Milano**  
LS2 Research Project on Federated Learning

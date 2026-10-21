# Federated Learning Benchmark

## Project

Research project by **NECSTLab - Politecnico di Milano** dedicated to studying **Federated Learning** strategies for predicting sleep quality from biometric data collected through Garmin devices.

The project compares approaches with different trade-offs among speed, accuracy, communication, and system complexity:

- **Flower Bagging**: parallel client training and aggregation of local models;
- **Flower Cyclic**: sequential training, with the model passing from one client to the next;
- **NVIDIA FLARE**: federated XGBoost aggregation based on sharing gradient statistics and histograms.

The overall objective is to identify an intermediate approach that reduces communication overhead while maintaining predictive quality close to what can be achieved with more informative federated training.

---

## Research objectives

### 1. Compare Federated Learning strategies

Measure and compare the behavior of the three reference approaches while keeping, as much as possible, the same experimental configuration:

- same federated dataset;
- same XGBoost-based regression model;
- same number of clients and federated rounds;
- same split between training and validation data.

The comparison should highlight not only which method is faster, but which offers the best trade-off between model quality and operational costs.

### 2. Analyze bottlenecks

Separate the overall training cost into its main components:

- **local training**, executed on individual clients;
- **communication**, including data or model transfer and serialization;
- **aggregation**, executed by the federated server or coordinator;
- **tree construction**, related to gradient computation and boosting iterations;
- **orchestration**, i.e., client configuration, startup, and synchronization.

This analysis is meant to understand whether performance mainly depends on local computation, the network, or coordination logic.

### 3. Evaluate quality, efficiency, and scalability

The benchmark considers the following dimensions:

| Dimension | Research question |
|---|---|
| Accuracy | How well does the model predict sleep quality? |
| Time | How long does complete federated training take? |
| Communication | How much data must be exchanged between clients and server? |
| Convergence | How many rounds are required to reach a target quality? |
| Scalability | How do time and costs change as the number of clients increases? |
| Robustness | How stable is the method in the presence of non-IID data, delays, or missing clients? |

The main metrics are **MAE**, **RMSE**, total time and per-round time, bytes sent and received, aggregation times, and resource usage.

---

## Theoretical context

### Federated Learning

In Federated Learning, data remains with the clients that own it. The coordinator organizes training and collects only models, updates, or aggregated statistics. In this way, it is possible to train a shared model without transferring raw biometric data to a central server.

The challenge is finding the right balance among:

- amount of shared information;
- global model accuracy;
- computation time;
- communication bandwidth and latency;
- protection of local data.

The project data distribution is federated across **9 clients**. Each client contains data related to different participant groups and can therefore introduce heterogeneity in the observation distribution.

### Flower Bagging

In the Bagging paradigm, clients work in parallel on their own data and send local training results to the coordinator. This approach limits the amount of exchanged information and makes execution relatively simple and fast.

**Theoretical advantages:**

- parallelism across clients;
- relatively limited communication;
- simplified implementation and debugging.

**Theoretical limitations:**

- each client sees only a portion of the data;
- model averaging or aggregation may lose information;
- quality may be more affected by non-IID distribution.

### Flower Cyclic

In the Cyclic paradigm, the model is updated sequentially by clients. Each client receives the current model, performs its local training, and transfers it to the next client or to the coordinator.

**Theoretical advantages:**

- the model can progressively incorporate information from different clients;
- the update flow is direct and lightweight;
- the number of models to aggregate can be reduced.

**Theoretical limitations:**

- execution is less parallel;
- time can depend on the slowest client;
- client order can influence the final result.

### NVIDIA FLARE and histogram-based aggregation

NVIDIA FLARE uses a richer approach for Federated XGBoost. Instead of sharing only the final local model, clients can share aggregated statistics, such as gradient histograms, which are useful for tree construction.

**Theoretical advantages:**

- more information available for aggregation;
- behavior closer to centralized training;
- possibility of achieving better accuracy on heterogeneous data;
- raw biometric data is not transferred.

**Theoretical limitations:**

- higher network traffic and number of synchronization operations;
- higher serialization and aggregation costs;
- more complex configuration and orchestration;
- possible increase in startup times and per-round time.

---

## Dataset and predictive problem

The dataset contains biometric data collected from **45 participants** who used a Garmin Vivoactive 5 for about one month.

- **999 training samples** distributed across 9 clients;
- **189 test samples**;
- about **40 numerical features** derived from biometric time series;
- target: **sleep quality score**, expressed on a 0-100 scale.

Among other aspects, the features describe:

- heart rate and variability;
- breathing during sleep and wakefulness;
- stress levels and peaks;
- activity, steps, calories, and intensity;
- duration and quality of previous sleep.

The problem is treated as a regression task with XGBoost objective `reg:squarederror`. Evaluation must consider both overall quality and differences in error across individual clients.

---

## Research question

> Is there an intermediate federated strategy capable of maintaining predictive quality close to NVIDIA FLARE while significantly reducing execution time and communication overhead?

The question arises from the contrast between:

- Flower approaches, simpler and lighter but potentially less informative;
- histogram-based approaches, richer from a statistical perspective but more expensive in terms of communication and orchestration.

---

## Hybrid approaches to explore

The project considers several possible intermediate strategies:

1. **Selective histogram sharing**: share histograms only for the most important features;
2. **Adaptive communication**: use more communication in early rounds and reduce it after approaching convergence;
3. **Compressed histograms**: reduce the number of bins or quantize shared statistics;
4. **Progressive histogram refinement**: start with coarse histograms and progressively increase their precision;
5. **Federated knowledge distillation**: train a richer model and transfer its knowledge to lighter local models;
6. **Hybrid parallel-sequential**: combine groups of clients trained in parallel and sequentially.

Each hybrid approach should be evaluated against the baselines considering at least:

- speedup compared to NVIDIA FLARE;
- MAE variation;
- ratio between communicated bytes and NVIDIA FLARE bytes;
- stability of results across multiple runs;
- behavior as the number of clients and degree of data heterogeneity vary.

---

## Expected outcome

The final outcome is not simply to identify the fastest method or the one with the lowest MAE. The objective is to build a multi-objective evaluation that enables choosing the most suitable strategy based on:

- required accuracy;
- maximum available time;
- network bandwidth and latency;
- number of clients;
- level of data heterogeneity;
- privacy and reliability requirements.

The project therefore aims to provide an experimental foundation for designing efficient Federated Learning systems for biometric data and real distributed scenarios.

---

## References

- [Flower Documentation](https://flower.ai/docs/)
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/)
- [Federated XGBoost](https://arxiv.org/abs/1901.08755)
- [SecureBoost: Histogram-based Federated Learning](https://arxiv.org/abs/1901.08755)

**NECSTLab - Politecnico di Milano**  
LS2 research project on Federated Learning

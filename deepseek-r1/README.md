# Setting Up DeepSeek-R1 671B for Distributed Multi-Node, Multi-GPU Inference with Ray

This guide provides step-by-step instructions for setting up **DeepSeek-R1 671B** (or other large vLLM-supported models) for inference on a **distributed, multi-node, multi-GPU** environment using **Ray.io**.

## **1. Overview**
Some large language models (LLMs) require more VRAM than is available on a single GPU. For example:
- **DeepSeek-R1 671B** requires approximately **8 GPUs × 87GB VRAM**.
- A standard **H100 (81GB VRAM) is insufficient**; instead, use **H100 (94GB VRAM) or H200 (142GB VRAM)**.
- **CPU offloading is not supported** for certain models due to weight constraints.
- To work around these limitations, we use **two p5.48xlarge instances** instead of a single **p5e.48xlarge**, deploying DeepSeek-R1 on **16 H100 GPUs** with **FP8 precision**.
- If using **bfloat16**, you will have less available VRAM for KV cache.

---

## **2. VRAM Requirements for DeepSeek-R1 671B**
### **2.1. FP8 Precision**
When running DeepSeek-R1 671B with **FP8 precision**, VRAM consumption is significantly reduced, allowing for a more efficient distribution across GPUs. Approximate requirements:
- **Total VRAM Required:** ~700GB
- **Per GPU (H100 94GB):** ~43GB per GPU (on a 16-GPU setup)
- **Per GPU (H100 80GB):** Not feasible due to limited memory availability
- **Optimal Setup:** 16x **H100 94GB** or **H200 142GB**

### **2.2. BFloat16 Precision**
Using **bfloat16** requires more VRAM due to increased weight storage size, impacting available memory for KV cache. Approximate requirements:
- **Total VRAM Required:** ~1.4TB
- **Per GPU (H100 94GB):** ~87GB per GPU (on a 16-GPU setup)
- **Per GPU (H100 80GB):** Not feasible without extreme memory optimizations
- **Impact on KV Cache:** Less available VRAM for storing activations and KV cache

### **2.3. Estimating VRAM Requirements**
VRAM usage can be estimated based on the model size, precision, and additional overhead:
- **Model Parameters:** Given **671 billion parameters**, the raw storage requirement per precision is:
  - **FP8:** Each parameter is stored in 1 byte.
        For 671 billion parameters, the raw storage is approximately:
        ```
        671 billion * 1 byte ≈ 671 GB
        ```
  - **BF16:** ~2 bytes per parameter → 
        ```
        671×2 bytes ≈ 1.34TB total
        ```
- **PyTorch Overhead:** Additional ~10-20% VRAM usage due to framework-specific optimizations and tensor storage.
- **KV Cache Memory:** Depends on sequence length and batch size, with FP8 allowing more cache storage than BF16.


### **2.4. Summary of VRAM Usage**
| Precision  | Total VRAM Required | Per GPU (16 GPUs) | Recommended GPUs |
|------------|---------------------|-------------------|------------------|
| **FP8**   | ~700GB               | ~43GB             | 8 * H100 94GB, 8 * H200 142GB or (2 Nodes * 8*H100 80GB) |
| **BF16**  | ~1.4TB               | ~87GB             | Same as FP8 but need multi node distributed setup as single node cannot handle it|

Given the substantial VRAM requirements, a **multi-node, multi-GPU setup is mandatory** for DeepSeek-R1 671B inference.

---

## **3. Attempting a Single-Node Setup with CPU Offloading**
### **3.1. Issue with CPU Offloading and Weight Tying**
When I attempted to run **DeepSeek-R1 671B** on a single node with **CPU offloading**, I encountered a major issue with **weight tying**. According to [vLLM issue #12541](https://github.com/vllm-project/vllm/issues/12541), DeepSeek-R1 does not support CPU offloading due to the way its model weights are structured. Specifically:

- **Weight Tying Constraint:** Some large-scale transformer models use a technique called **weight tying**, where model weights are shared across layers to reduce memory consumption.
- **CPU Offloading Conflict:** When attempting CPU offloading, the weight-sharing mechanism fails, causing inconsistencies that prevent proper execution.
- **Error Manifestation:** The error typically presents itself as a failure to allocate memory correctly, leading to crashes or incorrect model behavior.

Below is the way you try to load the model with CPU offloading which results in the above issue
```bash
nohup vllm serve deepseek-ai/DeepSeek-R1   --tensor-parallel-size 8 --trust-remote-code --cpu-offload-gb 10  --gpu-memory-utilization 0.95 &
```

### **3.2. Why a Multi-Node, Multi-GPU Setup is Necessary**
Due to the **weight tying limitation**, attempting to run DeepSeek-R1 on a single node with CPU offloading is **not feasible**. This is why a **multi-node, multi-GPU setup is required**, where model weights are evenly distributed across GPUs without relying on CPU memory. This avoids issues related to weight sharing and ensures efficient inference performance.

For large models requiring **more VRAM than a single node can provide**, a distributed approach using **Ray** enables:
- **Model Parallelism**: Splitting the model across multiple GPUs and nodes.
- **Efficient VRAM Utilization**: Reducing per-GPU memory load by distributing layers.
- **Scalability**: Enabling inference on models that exceed single-node VRAM limits.

---

## **4. AWS EC2 Setup**

### **4.1. EC2 Node Configuration**
- **Instances:** `2 × p5.48xlarge`
- **AMI Name:** `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5.1 (Ubuntu 22.04) 20250202`
- **AMI ID:** `ami-0c87233e00bd17f39`

### **4.2. Basic Checks on EC2 Nodes**
After launching the instances, verify GPU availability:
```bash
nvidia-smi
nvcc --version
```

---

## **4. Setting Up the Ray Cluster**

### **4.1. Install Ray on Head Node**
```bash
pip install -U "ray[default]" "ray[train]" "ray[tune]" "ray[rllib]" "ray[serve]"
ray start --head --num-gpus=8
```

### **4.2. Install Ray on Worker Nodes**
Replace `<head_node_ip>` with the actual head node's IP:
```bash
ray start --address=<head_node_ip>:6379 --num-gpus=8
```

### **4.3. Restarting Ray**
```bash
ray stop
# Use the appropriate command to restart Ray on head or worker nodes
```

---

## **5. Hugging Face Setup**
### **5.1. Authenticate with Hugging Face**
```bash
export HUGGING_FACE_HUB_TOKEN=<your_huggingface_token>
```
### **5.2. (Optional) Change Default Cache Directory**
```bash
export HF_HOME=~/hfcache/
```

---

## **6. Install vLLM**
### **6.1. Install vLLM on Both Head and Worker Nodes**
```bash
pip install vllm
```

### **6.2. Enable Eager Execution to Avoid CUDA Graph Capture Issues**
Without this, Ray workers may get stuck at **100% CPU and GPU utilization** during CUDA graph capture.
```bash
export VLLM_USE_ENFORCE_EAGER=1
```

### **6.3. Launch vLLM with DeepSeek-R1**
```bash
nohup vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 16 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 &
```

---

## **7. Troubleshooting**

### **7.1. Permission Denied: Hugging Face Cache**
If you encounter permission issues on either the **head** or **worker** nodes, check `nohup.out` for errors. To fix:
```bash
sudo chown -R ubuntu:ubuntu ~/.cache/huggingface
```

### **7.2. `MetadataIncompleteBuffer` Error**
If you see errors like:
```bash
safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer
```
This usually means the Hugging Face cache is corrupted, likely due to an interrupted download. **Fix by clearing the cache:**
```bash
sudo rm -rf ~/.cache/huggingface/hub
```

### **7.3. Stuck at "Capturing CUDA Graph Shapes" (100% CPU/GPU Usage)**
If Ray workers appear stuck with **100% CPU/GPU utilization**, enabling **Eager Execution** may help:
```bash
export VLLM_USE_ENFORCE_EAGER=1
```



# Project Workflow Guide

## Complete Pipeline: From Data to Results

This guide walks through the complete workflow of the RAG project, from initial setup to generating evaluation results.

## Phase 1: Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/srijanshk/rag_pw.git
cd rag_pw

# 2. Create environment
conda env create -f environment.yml
conda activate thesis_env

# 3. Set GPU configuration
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

## Phase 2: Data Preparation

### Step 1: Download Benchmark Datasets

```bash
# Download GSM8K and MATH datasets
python download_data.py

# Expected output:
# - data/benchmarks/gsm8k/test.jsonl
# - data/benchmarks/math/*.jsonl
```

### Step 2: Prepare Knowledge Base

#### Option A: OpenMathInstruct2 (Recommended)

```bash
# 1. Download OpenMathInstruct2 dataset
# From: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2

# 2. Chunk the dataset
python chunk_openmath_jsonl.py \
    --input data/openmathinstruct2/train.jsonl \
    --output data/openmath_chunks.tsv \
    --max_chunk_size 512

# Expected output:
# - data/openmath_chunks.tsv (~1M chunks)
```

#### Option B: MathPile

```bash
# 1. Download MathPile dataset
# From: https://huggingface.co/datasets/GAIR/MathPile

# 2. Process and chunk
python mathpile_chunk.py \
    --input /path/to/mathpile \
    --output data/mathpile_chunks.tsv \
    --chunk_size 512

# Expected output:
# - data/mathpile_chunks.tsv
```

### Step 3: Build Embeddings and Index

```bash
# Build FAISS HNSW index with BGE-M3 embeddings
python build_faiss_hnsw_bge_m3.py \
    --tsv data/openmath_chunks.tsv \
    --faiss-index indexes/openmath_bge-m3_hnsw.index \
    --faiss-meta indexes/openmath_bge-m3_metadata.jsonl \
    --M 32 \
    --efC 200 \
    --batch-size 128

# This will:
# 1. Load chunks from TSV
# 2. Generate BGE-M3 embeddings (GPU accelerated)
# 3. Build HNSW index with specified parameters
# 4. Save index and metadata

# Expected duration: 2-4 hours for 1M chunks
# Expected output:
# - indexes/openmath_bge-m3_hnsw.index (~2-4 GB)
# - indexes/openmath_bge-m3_metadata.jsonl (~500 MB)
```

**Index Parameters Guide:**
- `M=32`: Good balance between recall and memory
- `M=64`: Higher recall, more memory (recommended for large-scale)
- `efC=200`: Construction parameter (higher = better quality)
- `efC=400`: Slower build, better index quality

## Phase 3: Evaluation

### Approach 1: Zero-shot Chain-of-Thought (Baseline)

No retrieval, just LLM reasoning:

```bash
# Run on GSM8K
python run_zero_cot.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/benchmarks/gsm8k \
    --split test \
    --batch 8 \
    --max_new 1024 \
    --out_path results/cot/gsm8k_zero_cot.jsonl \
    --wandb_project "RAG_EVAL" \
    --wandb_run_name "gsm8k_zero_cot"

# Run on MATH
python run_zero_cot.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/benchmarks/math \
    --split test \
    --batch 8 \
    --max_new 1024 \
    --out_path results/cot/math_zero_cot.jsonl

# Expected duration: ~30-60 minutes per dataset
```

### Approach 2: Static RAG + CoT

Fixed retrieval before reasoning:

```bash
# Run on GSM8K
python run_static_cot.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/benchmarks/gsm8k \
    --faiss_index indexes/openmath_bge-m3_hnsw.index \
    --faiss_meta indexes/openmath_bge-m3_metadata.jsonl \
    --k_final 5 \
    --batch 2 \
    --max_new 768 \
    --out_path results/static/gsm8k_static_rag.jsonl \
    --wandb_project "RAG_EVAL" \
    --wandb_run_name "gsm8k_static_rag"

# Run on MATH
python run_static_cot.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/benchmarks/math \
    --faiss_index indexes/openmath_bge-m3_hnsw.index \
    --faiss_meta indexes/openmath_bge-m3_metadata.jsonl \
    --k_final 5 \
    --batch 2 \
    --max_new 768 \
    --out_path results/static/math_static_rag.jsonl

# Expected duration: ~1-2 hours per dataset
```

**Parameter Tuning:**
- `k_final=3`: Fewer contexts, faster, less information
- `k_final=5`: Balanced (recommended)
- `k_final=10`: More contexts, slower, potentially more helpful

### Approach 3: Dynamic/Active RAG

Interactive retrieval during reasoning:

```bash
# Run on GSM8K
python run_active_rag.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset gsm8k \
    --faiss_index indexes/openmath_bge-m3_hnsw.index \
    --faiss_meta indexes/openmath_bge-m3_metadata.jsonl \
    --max_search_turns 2 \
    --k_per_search 3 \
    --out_path results/dynamic/gsm8k_active_rag.jsonl \
    --wandb_project "RAG_EVAL" \
    --wandb_run_name "gsm8k_active_rag"

# Run on MATH
python run_active_rag.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset math \
    --faiss_index indexes/openmath_bge-m3_hnsw.index \
    --faiss_meta indexes/openmath_bge-m3_metadata.jsonl \
    --max_search_turns 2 \
    --k_per_search 3 \
    --out_path results/dynamic/math_active_rag.jsonl

# Expected duration: ~2-3 hours per dataset
```

**Parameter Tuning:**
- `max_search_turns=1`: Single retrieval during reasoning
- `max_search_turns=2`: Up to 2 retrievals (recommended)
- `max_search_turns=3`: More retrieval opportunities
- `k_per_search=3`: Documents per retrieval call

### Batch Evaluation with Shell Scripts

```bash
# Run all three approaches on both datasets
cd script

# 1. Zero-shot CoT
bash cot_eval.sh

# 2. Static RAG
bash run_static_cot.sh

# 3. Dynamic RAG
bash dynamic_cot_eval.sh
```

## Phase 4: Analysis

### Analyze Individual Results

```bash
# Analyze accuracy and metrics
python analyze_result.py \
    results/cot/gsm8k_zero_cot.jsonl

# Output:
# - Accuracy: XX.X%
# - Correct: XXX
# - Total: XXX
# - Failed parses: XX
```

### Compare Multiple Approaches

```bash
# Compare all methods
python analysis/evaluate_runs.py \
    --cot results/cot/gsm8k_zero_cot.jsonl \
    --static results/static/gsm8k_static_rag.jsonl \
    --dynamic results/dynamic/gsm8k_active_rag.jsonl \
    --output results/analysis/comparison.json
```

### Visualization in Jupyter

```bash
# Launch Jupyter
jupyter notebook

# Open: notebooks/evaluation.ipynb
# - Visualize accuracy comparisons
# - Analyze retrieval patterns
# - Plot performance metrics
```

## Phase 5: Results Organization

### Expected Results Structure

```
results/
├── cot/
│   ├── gsm8k_zero_cot.jsonl
│   └── math_zero_cot.jsonl
├── static/
│   ├── gsm8k_static_rag.jsonl
│   └── math_static_rag.jsonl
├── dynamic/
│   ├── gsm8k_active_rag.jsonl
│   └── math_active_rag.jsonl
└── analysis/
    ├── comparison.json
    ├── accuracy_plot.png
    └── evaluation_summary.csv
```

## Monitoring and Debugging

### Monitor GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log GPU usage
nvidia-smi dmon -s pucvmet -d 1 > gpu_log.txt &
```

### Check Logs

```bash
# View evaluation logs
tail -f logs/gsm8k_zero_cot_*.log

# Search for errors
grep -i error logs/*.log
```

### Weights & Biases Dashboard

```bash
# View in browser
# Navigate to: https://wandb.ai/your-username/RAG_EVAL
```

## Common Workflows

### Quick Test Run (Small Sample)

```bash
# Test with first 10 examples
python run_zero_cot.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/benchmarks/gsm8k \
    --split test \
    --num_samples 10 \
    --batch 2 \
    --out_path results/test_run.jsonl
```

### Reproduce Paper Results

```bash
# Run complete pipeline
bash script/cot_eval.sh
bash script/run_static_cot.sh
bash script/dynamic_cot_eval.sh

# Analyze results
python analysis/evaluate_runs.py --all
```

### Experiment with Different Retrieval Parameters

```bash
# Try different k values for static RAG
for k in 3 5 7 10; do
    python run_static_cot.py \
        --dataset_path data/benchmarks/gsm8k \
        --faiss_index indexes/openmath_bge-m3_hnsw.index \
        --faiss_meta indexes/openmath_bge-m3_metadata.jsonl \
        --k_final $k \
        --out_path results/static/gsm8k_k${k}.jsonl
done
```

## Time and Resource Estimates

| Phase | Duration | GPU Memory | Disk Space |
|-------|----------|------------|------------|
| Environment Setup | 10-20 min | - | 5 GB |
| Data Download | 30-60 min | - | 20 GB |
| Index Building | 2-4 hours | 16 GB | 5 GB |
| Zero-shot Eval | 30-60 min | 16 GB | 100 MB |
| Static RAG Eval | 1-2 hours | 20 GB | 200 MB |
| Dynamic RAG Eval | 2-3 hours | 24 GB | 300 MB |
| Analysis | 5-10 min | - | 50 MB |

**Total Project**: ~6-12 hours, 24 GB GPU, 30 GB disk

## Best Practices

1. **Start Small**: Test with 10-100 examples before full evaluation
2. **Monitor Resources**: Keep an eye on GPU memory and disk space
3. **Save Frequently**: Results are saved incrementally
4. **Use Checkpoints**: Resume interrupted runs
5. **Log Everything**: Use W&B or local logs for tracking
6. **Version Control**: Commit changes to git regularly

## Troubleshooting Workflow

If something goes wrong:

1. **Check logs**: `tail -f logs/*.log`
2. **Verify GPU**: `nvidia-smi`
3. **Test retriever**: Run retrieval test notebook
4. **Reduce batch size**: If OOM errors occur
5. **Check disk space**: `df -h`
6. **Verify paths**: Ensure all file paths are correct

## Next Steps

After completing the workflow:

1. Analyze results in detail
2. Experiment with different prompts
3. Try different embedding models
4. Tune retrieval parameters
5. Write up findings
6. Share results with team

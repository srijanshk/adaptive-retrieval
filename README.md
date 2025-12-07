# Adaptive Retrieval helps Reasoning in LLMs - but mostly if it's not used

**Master Thesis:** *Enhancing LLMs reasoning capabilities by including retrieval*

**Workshop Paper:** *Adaptive Retrieval helps Reasoning in LLMs - but mostly if it's not used*  
**Authors:** Srijan Shakya, Anamaria-Roberta Hartl, Sepp Hochreiter, Korbinian Pöppel  
**Affiliation:** Institute of Machine Learning, Johannes Kepler University Linz, Austria

This repository contains the implementation and experiments for our paper investigating adaptive retrieval-augmented generation (RAG) for mathematical reasoning. We explore a fundamental principle: **treating retrieval as a form of dynamic in-context learning** where an LLM agent actively decides when to query external knowledge during reasoning.

## 🎯 Key Findings

Our experiments reveal surprising insights about adaptive retrieval:

1. **Adaptive retrieval outperforms static RAG**: +1.1pp on GSM8K, +6.4pp on MATH-500
2. **Static retrieval hurts performance**: -6.3pp on GSM8K compared to baseline CoT
3. **Best performance when retrieval is NOT used**: When the agent chooses not to retrieve, accuracy is +2.1pp (GSM8K) and +19.5pp (MATH-500) higher than baseline
4. **Retrieval scales with difficulty**: 7.0% retrieval rate on GSM8K vs 38.8% on MATH-500
5. **The decision to retrieve is a metacognitive signal**: The agent's choice reflects its confidence and uncertainty

## 🔬 Method Overview

We compare three reasoning strategies:

### 1. **Baseline: Chain-of-Thought (CoT)**
- Zero-shot prompting with "think step-by-step"
- No external knowledge retrieval
- Pure parametric reasoning

### 2. **Static Retrieval-Augmented CoT**
- Single retrieval using the problem as query
- Top-k results prepended to prompt
- No further retrieval during reasoning

### 3. **Adaptive Retrieval-Augmented CoT** (Our Approach)
- LLM decides when to retrieve using `<search>query</search>` tags
- Retrieval executed on-demand during reasoning
- Model continues with injected context
- Iterative until final `<answer>` generated

### System Components

- **Core LLM**: Meta LLaMA 3.1 8B Instruct (zero-shot, greedy decoding)
- **Embedding Model**: BAAI/bge-m3 for dense retrieval
- **Reranker**: BAAI/bge-m3-reranker (cross-encoder)
- **Vector Index**: FAISS HNSW (k_dense=200, k_final=5)
- **Knowledge Corpora**: 
  - OpenMathInstruct-2 (recommended, domain-specific Q&A pairs)
  - MathPile (broad mathematical texts)
- **Benchmarks**: GSM8K (1,319 problems) and MATH-500 (500 problems)

## 📊 Results Summary

### Overall Performance

| Dataset   | LLM Baseline | CoT Baseline | Static RAG+CoT | Adaptive RAG+CoT | Δ vs CoT |
|-----------|--------------|--------------|----------------|------------------|----------|
| GSM8K     | 43.7%        | 82.1%        | 75.8%          | **83.2%**        | +1.1pp   |
| MATH-500  | 29.8%        | 44.2%        | 42.4%          | **50.6%**        | +6.4pp   |

### Performance by MATH-500 Difficulty Level

| Difficulty | LLM    | CoT    | Static RAG | Adaptive RAG | Retrieval Rate |
|------------|--------|--------|------------|--------------|----------------|
| Level 1    | 51.2%  | 72.1%  | 76.7%      | **86.0%**    | 14.0%          |
| Level 2    | 44.4%  | 60.0%  | 63.3%      | **67.8%**    | 21.1%          |
| Level 3    | 37.1%  | 52.4%  | 49.5%      | **61.9%**    | 33.3%          |
| Level 4    | 21.9%  | 38.3%  | 32.0%      | **47.7%**    | 41.4%          |
| Level 5    | 14.9%  | **23.9%** | 21.6%   | 21.6%        | 60.4%          |

### Retrieval Decision Analysis

**When retrieval is NOT used:**
- GSM8K: 84.2% accuracy (+2.1pp vs CoT baseline)
- MATH-500: 63.7% accuracy (+19.5pp vs CoT baseline)

**When retrieval IS used:**
- GSM8K: 70.7% accuracy (-11.4pp vs CoT baseline)  
- MATH-500: 41.6% accuracy (-2.6pp vs CoT baseline)

**Key Insight**: The agent's decision to forgo retrieval is a strong indicator of confidence and correctness.

## 📁 Project Structure

```
rag_pw/
├── README.md
├── environment.yml              # Conda environment configuration
├── requirements.txt             # Python dependencies
│
├── Core RAG Components
│   ├── BGERetriever.py         # Dense retriever using BGE-M3 embeddings
│   ├── BGERetriever_v2.py      # Optimized version with batch processing
│   ├── retrieval_utils.py      # Utility functions for retrieval
│
├── Evaluation Scripts
│   ├── run_zero_cot.py         # Zero-shot CoT evaluation
│   ├── run_static_cot.py       # Static RAG + CoT evaluation
│   ├── run_active_rag.py       # Dynamic/Active RAG (deprecated)
│   ├── dynamic_rag.py          # Adaptive RAG implementation (main)
│
├── Index Building & Data Processing
│   ├── build_faiss_hnsw_bge_m3.py  # Build FAISS HNSW index
│   ├── build_faiss_index_gpu.py    # GPU-accelerated index building
│   ├── chunk_openmath_jsonl.py     # Chunk OpenMath data
│   ├── mathpile_chunk.py           # Chunk MathPile data
│   ├── chunk_jsonl_to_tsv.py       # Convert chunks to TSV format
│   ├── download_data.py            # Download benchmark datasets
│   └── clean_tsv.py                # Clean TSV data
│
├── Analysis & Utilities
│   ├── analyze_result.py           # Result analysis and metrics
│   └── mathpile_analyze.py         # MathPile dataset analysis
│
├── script/                    # Shell scripts for batch evaluation
│   ├── cot_eval.sh           # Zero-shot CoT evaluation
│   ├── run_static_cot.sh     # Static RAG evaluation
│   └── dynamic_cot_eval.sh   # Adaptive RAG evaluation
│
├── misc/                      # Experimental/training scripts
│   ├── grid_search_bm25.py   # BM25 hyperparameter tuning
│   ├── sparse_retrieve_contexts.py  # Sparse retrieval experiments
│   ├── train_retriever.py    # Retriever fine-tuning
│   └── train_generator.py    # Generator fine-tuning
│
├── analysis/                  # Analysis notebooks
│   ├── evaluation.ipynb      # Evaluation analysis
│   └── systematic_evaluation.ipynb
│
├── data/                      # Dataset directory
│   ├── benchmarks/           # GSM8K and MATH datasets
│   │   ├── gsm8k/           # GSM8K test set (1,319 problems)
│   │   └── math/            # MATH dataset (500 problems, 5 difficulty levels)
│   └── openmathinstruct2/    # OpenMath knowledge base
│       └── train.jsonl      # OpenMath training data
│
├── indexes/                   # FAISS vector indexes (generated)
│   ├── openmath_bge-m3_hnsw.index
│   └── openmath_bge-m3_metadata.jsonl
│
└── results/                   # Evaluation results
    ├── cot/                  # Zero-shot CoT results
    ├── static/               # Static RAG results
    ├── dynamic/              # Adaptive RAG results
    └── analysis/             # Analysis and visualizations
```

## 🚀 Complete Workflow: From Data to Results

### Step 1: Environment Setup

#### 1.1 Prerequisites

- **Anaconda** or **Miniconda** ([Download](https://www.anaconda.com/products/distribution))
- **CUDA-capable GPU** (NVIDIA GPU with CUDA support)
- **Python 3.12+**
- **Minimum 24GB GPU RAM** (for LLaMA 3.1 8B in float16)

#### 1.2 Clone Repository

```bash
git clone https://github.com/srijanshk/adaptive-retrieval.git
cd rag_pw
```

#### 1.3 Create Conda Environment

```bash
# Create environment from specification
conda env create -f environment.yml

# Activate environment
conda activate thesis_env
```

#### 1.4 Set GPU Configuration

```bash
# Set visible GPU devices
export CUDA_VISIBLE_DEVICES="0"  # or "0,1" for multi-GPU

# Configure CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

#### 1.5 Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check BGE-M3
python -c "from FlagEmbedding import BGEM3FlagModel; print('BGE-M3: OK')"

# Check FAISS
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"

# Check Transformers
python -c "from transformers import AutoModelForCausalLM; print('Transformers: OK')"
```

### Step 2: Data Preparation

#### 2.1 Download Benchmark Datasets

The benchmarks (GSM8K and MATH) can be downloaded from HuggingFace:

```bash
# Download GSM8K and MATH datasets
python download_data.py
```

**Expected output structure:**
```
data/benchmarks/
├── gsm8k/
│   └── test.jsonl          # 1,319 grade school math problems
└── math/
    ├── algebra/
    ├── counting_and_probability/
    ├── geometry/
    ├── intermediate_algebra/
    ├── number_theory/
    ├── prealgebra/
    └── precalculus/
```

#### 2.2 Prepare Knowledge Base

We provide two knowledge corpus options. **OpenMathInstruct-2 is recommended** for better domain-specific performance.

##### Option A: OpenMathInstruct-2 (Recommended)

OpenMathInstruct-2 is a curated dataset of 14M mathematical question-answer pairs.

**Download:**
```bash
# Download from HuggingFace
# Visit: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
# Or use wget/curl to download train.jsonl

mkdir -p data/openmathinstruct2
cd data/openmathinstruct2

# Download (example using HuggingFace CLI)
huggingface-cli download nvidia/OpenMathInstruct-2 --repo-type dataset --local-dir .
```

**Chunk the dataset:**
```bash
# Chunk OpenMath data into retrievable passages
python chunk_openmath_jsonl.py \
    --input data/openmathinstruct2/train.jsonl \
    --output data/openmath_chunks.tsv \
    --max_chunk_size 512 \
    --max_chunks 1000000

# Expected output: ~1M chunks in TSV format
# Format: id\tproblem\tsolution\tproblem_from\trow_id\tchunk_id
```

**Parameters:**
- `--max_chunk_size`: Maximum tokens per chunk (default: 512)
- `--max_chunks`: Limit total chunks to process (optional)

##### Option B: MathPile

MathPile is a large-scale mathematical text corpus (~10B tokens).

**Download:**
```bash
# Download MathPile from https://huggingface.co/datasets/GAIR/MathPile
mkdir -p data/mathpile
# Follow download instructions from HuggingFace
```

**Chunk the dataset:**
```bash
python mathpile_chunk.py \
    --input data/mathpile \
    --output data/mathpile_chunks.tsv \
    --chunk_size 512 \
    --stride 128
```

#### 2.3 Clean and Prepare TSV Data

```bash
# Optional: Clean the TSV file (remove malformed entries)
python clean_tsv.py \
    --input data/openmath_chunks.tsv \
    --output data/openmath_chunks_clean.tsv
```

### Step 3: Build FAISS Vector Index

Build a FAISS HNSW index with BGE-M3 embeddings for efficient dense retrieval.

#### 3.1 Build Index

```bash
python build_faiss_hnsw_bge_m3.py \
    --tsv data/openmath_chunks.tsv \
    --faiss-index indexes/openmath_bge-m3_hnsw.index \
    --faiss-meta indexes/openmath_bge-m3_metadata.jsonl \
    --M 32 \
    --efC 200 \
    --batch_size 256
```

**Parameters:**
- `--M`: Number of bidirectional links per vector (default: 32)
  - Higher = better recall, more memory
  - Typical range: 16-64
- `--efC`: Construction-time search depth (default: 200)
  - Higher = better quality index, slower build
  - Typical range: 100-500
- `--batch_size`: Embedding batch size (default: 256)
  - Adjust based on GPU memory

**Expected output:**
```
indexes/
├── openmath_bge-m3_hnsw.index      # FAISS HNSW index (~4GB for 1M vectors)
└── openmath_bge-m3_metadata.jsonl  # Metadata (id, text, scores)
```

**Index build time:**
- 1M chunks: ~30-60 minutes on V100/A100 GPU
- Memory: ~16GB GPU RAM during embedding, ~8GB for index construction

#### 3.2 (Optional) GPU-Accelerated Index Building

For very large datasets (>5M chunks):

```bash
python build_faiss_index_gpu.py \
    --tsv data/openmath_chunks.tsv \
    --faiss-index indexes/openmath_bge-m3_gpu.index \
    --faiss-meta indexes/openmath_bge-m3_metadata.jsonl \
    --use_gpu
```

### Step 4: Run Evaluations

All evaluation scripts support the following common parameters:

**Core parameters:**
- `--model_name`: HuggingFace model identifier (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `--benchmark`: Dataset to evaluate (`gsm8k` or `math500`)
- `--num_samples`: Limit evaluation to N samples (optional, for quick tests)
- `--out_path`: Output file path for results

#### 4.1 Zero-Shot Chain-of-Thought (Baseline)

```bash
# Run CoT baseline on GSM8K
python run_zero_cot.py \
    --benchmark gsm8k \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --num_samples 1319 \
    --out_path results/cot/gsm8k_cot.json

# Run CoT baseline on MATH-500
python run_zero_cot.py \
    --benchmark math500 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --num_samples 500 \
    --out_path results/cot/math500_cot.json
```

**Using shell script:**
```bash
bash script/cot_eval.sh
```

#### 4.2 Static Retrieval-Augmented CoT

```bash
# Run Static RAG on GSM8K
python run_static_cot.py \
    --benchmark gsm8k \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --faiss_index indexes/openmath_bge-m3_hnsw.index \
    --faiss_meta indexes/openmath_bge-m3_metadata.jsonl \
    --k_dense 200 \
    --k_final 5 \
    --num_samples 1319 \
    --out_path results/static/gsm8k_static_rag.json

# Run Static RAG on MATH-500
python run_static_cot.py \
    --benchmark math500 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --faiss_index indexes/openmath_bge-m3_hnsw.index \
    --faiss_meta indexes/openmath_bge-m3_metadata.jsonl \
    --k_dense 200 \
    --k_final 5 \
    --num_samples 500 \
    --out_path results/static/math500_static_rag.json
```

**Parameters:**
- `--k_dense`: Initial candidates retrieved from FAISS (default: 200)
- `--k_final`: Final documents after re-ranking (default: 5)

**Using shell script:**
```bash
bash script/run_static_cot.sh
```

#### 4.3 Adaptive Retrieval-Augmented CoT (Our Method)

```bash
# Run Adaptive RAG on GSM8K  
python dynamic_rag.py \
    --benchmark gsm8k \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --faiss_index indexes/openmath_bge-m3_hnsw.index \
    --faiss_meta indexes/openmath_bge-m3_metadata.jsonl \
    --k_dense 200 \
    --k_final 5 \
    --max_tool_calls 3 \
    --tool_gen_tokens 1024 \
    --answer_gen_tokens 1024 \
    --injection_mode summary \
    --num_samples 1319 \
    --out_path results/dynamic/gsm8k_adaptive_rag.json

# Run Adaptive RAG on MATH-500
python dynamic_rag.py \
    --benchmark math500 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --faiss_index indexes/openmath_bge-m3_hnsw.index \
    --faiss_meta indexes/openmath_bge-m3_metadata.jsonl \
    --k_dense 200 \
    --k_final 5 \
    --max_tool_calls 3 \
    --tool_gen_tokens 1024 \
    --answer_gen_tokens 1024 \
    --injection_mode summary \
    --num_samples 500 \
    --out_path results/dynamic/math500_adaptive_rag.json
```

**Parameters:**
- `--max_tool_calls`: Maximum retrieval iterations per problem (default: 3)
- `--tool_gen_tokens`: Max tokens for reasoning between retrievals (default: 1024)
- `--answer_gen_tokens`: Max tokens for final answer generation (default: 1024)
- `--injection_mode`: How to inject retrieved context
  - `summary`: Extract canonical method/formula (recommended)
  - `full`: Inject full retrieved passages

**Using shell script:**
```bash
bash script/dynamic_cot_eval.sh
```

#### 4.4 Weights & Biases Integration (Optional)

Track experiments with W&B:

```bash
python dynamic_rag.py \
    --benchmark gsm8k \
    --wandb_project "adaptive-retrieval" \
    --wandb_run "gsm8k-adaptive-v1" \
    [other parameters...]
```

### Step 5: Analyze Results

#### 5.1 Generate Analysis Report

```bash
python analyze_result.py \
    results/cot/gsm8k_cot.json \
    results/static/gsm8k_static_rag.json \
    results/dynamic/gsm8k_adaptive_rag.json
```

**Output metrics:**
- Overall exact match (EM) accuracy
- Retrieval statistics (triggered %, executed %, avg count)
- Performance with vs without retrieval
- Consistency check statistics
- Per-difficulty breakdown (for MATH-500)

#### 5.2 Jupyter Notebook Analysis

```bash
# Start Jupyter
jupyter notebook

# Open analysis notebooks
# - analysis/evaluation.ipynb
# - analysis/systematic_evaluation.ipynb
```

## 📋 Hyperparameter Configuration

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Core LLM** | `meta-llama/Llama-3.1-8B-Instruct` | Instruction-tuned language model |
| **Precision** | `float16` | Mixed precision for efficiency |
| **Temperature** | `0.0` | Greedy decoding (deterministic) |
| **Max New Tokens** | `1024` | Maximum generation length |

### Retrieval Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Embedding Model** | `BAAI/bge-m3` | Dense embedding model |
| **Reranker** | `BAAI/bge-m3-reranker` | Cross-encoder reranker |
| **Vector Index** | FAISS HNSW | Approximate nearest neighbor search |
| **k_dense** | `200` | Initial candidates from FAISS |
| **k_final** | `5` | Final documents after reranking |
| **M** (HNSW) | `32` | Bidirectional links per vector |
| **efConstruction** | `200` | Construction-time search depth |
| **efSearch** | `700` | Query-time search depth |

## 🔍 Understanding the Prompt Templates

All prompts use Llama 3 chat formatting:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{problem}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
```

### Zero-Shot CoT Prompt

```
You are an expert mathematician.

Think step-by-step.  
Write every reasoning step inside '<think> ... </think>' blocks. 

When you are completely done, produce **exactly one**
    <answer> your final answer </answer>
    
Nothing after `</answer>`.
```

### Adaptive RAG Prompt

```
You are an expert mathematician.

Think step-by-step.  
Write every reasoning step inside '<think> ... </think>' blocks. 

If you need to look up a formula, definition, or problems, 
you can use the <search> tool by writing a search query inside 
the <search> tag like this: 
<search>your search query</search>

After retrieval, you may:
1. Use the information if helpful
2. Explicitly state "Retrieved information not helpful" and continue without it

After the search results are returned, continue your step-by-step thinking.

When you are completely done, produce **exactly one**
    <answer> your final answer </answer>
    
Nothing after `</answer>`.
```
## 📊 Expected Results

Based on our paper experiments:

**GSM8K (1,319 problems):**
- CoT Baseline: 82.1%
- Static RAG: 75.8% (-6.3pp)
- Adaptive RAG: 83.2% (+1.1pp)

**MATH-500 (500 problems):**
- CoT Baseline: 44.2%
- Static RAG: 42.4% (-1.8pp)
- Adaptive RAG: 50.6% (+6.4pp)

**Retrieval Usage:**
- GSM8K: 7.0% of problems trigger retrieval
- MATH-500: 38.8% of problems trigger retrieval
- Correlation with difficulty: Level 1 (14.0%) → Level 5 (60.4%)

## 💡 Key Implementation Details

### Two-Stage Retrieval Pipeline

1. **Stage 1 - Dense Retrieval (FAISS)**
   - Embed query with BGE-M3
   - HNSW approximate nearest neighbor search
   - Retrieve top-200 candidates

2. **Stage 2 - Reranking (Cross-Encoder)**
   - Rerank candidates with BGE-M3-reranker
   - ColBERT scores for fine-grained matching
   - Select top-5 final documents

### Summary-Based Context Injection

For Adaptive RAG, we use a "canonical method extraction" approach:

1. Retrieve passages containing potential solutions
2. Use LLM to extract abstract formula/theorem (not problem-specific)
3. Inject only the canonical method, not raw passages

**Example:**
- **Retrieved**: "To solve (2x+3)^4, we expand using binomial theorem..."
- **Injected**: "The Binomial Theorem states that for positive integer n, (x+a)^n = Σ C(n,k) x^k a^(n-k)"

This reduces noise and prevents copying specific numbers from examples.

## 🔧 Troubleshooting

### Common Issues

**GPU Out of Memory:**
```bash
# Reduce batch size
python dynamic_rag.py --batch_size 1 ...

# Use 4-bit quantization
python dynamic_rag.py --quantize_4bit ...
```

**FAISS Index Issues:**
```python
# Check index loaded correctly
import faiss
index = faiss.read_index("indexes/openmath_bge-m3_hnsw.index")
print(f"Index size: {index.ntotal} vectors")
```

**Slow Generation:**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Use `float16` precision
- Consider using vLLM for faster inference (commented in code)

## 📝 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{shakya2025adaptive,
  title={Adaptive Retrieval helps Reasoning in LLMs - but mostly if it's not used},
  author={Shakya, Srijan and Hartl, Anamaria-Roberta and Hochreiter, Sepp and P{\"o}ppel, Korbinian},
  booktitle={Workshop on Principles of Generative Modeling (PriGM), Eurips 2025},
  year={2025}
}
```

## 🙏 Acknowledgments

- **Meta AI** for LLaMA 3.1
- **BAAI** for BGE-M3 embeddings and reranker
- **Facebook Research** for FAISS
- **NVIDIA** for OpenMathInstruct-2 dataset
- **HuggingFace** for model hosting and transformers library
- **JKU-IML** for GPU

---

For questions or issues, please open a GitHub issue or contact: srijanshakya977@gmail.com

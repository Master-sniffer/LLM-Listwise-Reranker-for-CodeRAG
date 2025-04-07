# Retrieval-Augmented Generation (RAG) System for Code Repositories

## 1. Introduction

This readme presents detailed documentation of a Retrieval-Augmented Generation (RAG) system designed to support natural language queries over software code repositories. The objective of the system is to facilitate intuitive information retrieval by mapping user questions to relevant files in the repository. The `escrcpy` project, a graphical user interface (GUI) for the Android screen mirroring tool `scrcpy`, serves as the primary case study throughout this implementation.

## 2. Datasets, Tools, and Evaluation Metrics

### 2.1 Dataset

- **Target Repository**: [`escrcpy`](https://github.com/viarotel-org/escrcpy)
- **Scope of Indexing**: 294 source code files with pre-defined extensions
- **Evaluation Dataset**: `escrcpy-commits-generated.json`, consisting of 34 manually curated query–file relevance pairs
- **Supported File Types**: JavaScript, TypeScript, Vue, HTML, CSS, Markdown, and other typical frontend development formats

### 2.2 Tools and Libraries

#### Embedding Models

- **Sentence Transformer**: `all-MiniLM-L6-v2`—used for dense semantic representation of file content

#### Reranking Models

- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`—applied to refine candidate results through semantic scoring

#### Retrieval Components

- **BM25**: Lexical-based sparse retrieval algorithm
- **FAISS**: Vector-based dense retrieval framework

#### Supporting Libraries

- `sentence-transformers` – Embedding and cross-encoder utilities  
- `rank_bm25` – Implementation of BM25 ranking algorithm  
- `faiss` – Efficient similarity search in vector spaces  
- `gitpython` – Git operations and repository management  
- `numpy` – Numerical computations  
- `pathlib` – Filesystem path manipulations

### 2.3 Evaluation Metric

- **Primary Metric**: Recall@10 — evaluates the proportion of queries where at least one relevant file appears among the top 10 retrieved results  
- **Current Score**: 0.68 (i.e., 23 out of 34 queries successfully retrieved a relevant document in the top 10)

## 3. System Architecture and Implementation

### 3.1 System Overview

The proposed system adopts a hybrid retrieval architecture that combines both lexical and semantic search techniques, followed by reranking using a cross-encoder model. The process consists of the following key phases:

1. **Repository Cloning and File Extraction**:
   - Git repository is cloned
   - Only files with allowed extensions are retained
   - Content is extracted and preprocessed for indexing

2. **Dual Index Construction**:
   - **BM25 Index**: Built using textual file content for traditional keyword search
   - **Vector Index**: Sentence embeddings are computed and stored using FAISS

3. **Query Execution Pipeline**:
   - User query is passed through both retrieval mechanisms (BM25 and FAISS)
   - Retrieved candidates are merged and reranked using a cross-encoder
   - Top `K` results are returned to the user

### 3.2 Core Components

#### FileLevelIndexer

This component is responsible for creating both BM25 and dense vector indices:

- Extracts and processes file content
- Constructs BM25-based lexical index
- Generates vector embeddings and builds a FAISS index
- Provides an interface for querying across both indices

#### CrossEncoderReranker

This module applies listwise reranking to refine the retrieved candidates:

- Scores each query–document pair using a cross-encoder
- Reranks results based on semantic relevance
- Notably improves retrieval precision and contextual alignment

#### Evaluation Methodology

The evaluation is performed using the `evaluate_recall10` function:

- Iterates over the evaluation queries
- Collects top-`K` results from the system
- Computes Recall@10 as the percentage of queries with at least one correct retrieval in the top 10

## 4. Experimental Results

The current configuration of the system demonstrates a Recall@10 score of **0.68** on the evaluation dataset. Below is an example illustrating the quality of results:

**Query**: *"How does the application manage scrcpy for a device?"*  
**Top retrieved files**:
- `docs/en/reference/scrcpy/develop.md`
- `docs/en/reference/scrcpy/otg.md`
- `docs/en/reference/scrcpy/device.md`
- `docs/en/reference/scrcpy/audio.md`
- `develop.md`
- `docs/en/reference/scrcpy/control.md`
- `docs/en/reference/scrcpy/camera.md`
- `docs/en/reference/scrcpy/index.md`
- `docs/en/help/scrcpy.md`
- `docs/en/reference/scrcpy/tunnels.md`

This result illustrates that the system successfully identifies and ranks relevant documentation files, thereby validating its capacity to interpret semantic intent from natural language queries.

## 5. Technical Challenges

The implementation of the RAG system revealed several technical challenges:

1. **Lexical–Semantic Trade-off**: Balancing the contributions of BM25 and vector-based retrieval required empirical tuning, as different query types benefit from different strategies.
2. **Large File Handling**: Very large files posed challenges during embedding. Content truncation methods were implemented to maintain system performance.
3. **Code-Aware Tokenization**: Generic NLP tokenizers often fail to handle code-specific syntax. Modifications were made to better tokenize source code.
4. **Reranking Overhead**: The cross-encoder reranker introduces computational overhead. Consequently, reranking was limited to a subset of the top candidates.
5. **Output Interference**: Progress bar libraries such as `tqdm` caused display issues in Jupyter environments, which necessitated disabling progress bars.

## 6. Running the System

### 6.1 Prerequisites

  ```bash
  pip install sentence-transformers rank_bm25 faiss-cpu gitpython numpy tqdm
  ```

### 6.2 Execution Steps


1. **Prepare the Evaluation Dataset**:
   - Ensure `escrcpy-commits-generated.json` is located in the working directory

2. **Run the Notebook**:
   - Open the Jupyter Notebook
   - Execute all cells sequentially to initialize the pipeline

### 6.3 Configuration Parameters

Key configurable variables within the notebook:

- `GITHUB_REPO_URL`: Repository to be indexed
- `REPO_LOCAL_PATH`: Local clone destination
- `VECTOR_MODEL_NAME`: Embedding model identifier
- `CROSS_ENCODER_MODEL_NAME`: Reranking model identifier
- `TOP_K`: Number of top results to return
- `ALLOWED_EXTENSIONS`: Permissible source file types

### 6.4 Reproducing Evaluation

To replicate evaluation results:

1. Place the file `escrcpy-commits-generated.json` in the working directory
2. Execute the notebook without altering default parameters
3. Recall@10 will be computed and printed at the end of evaluation


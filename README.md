# PyTorch Two-Tower Recommender with Hybrid Cold-Start Strategy

A production-ready candidate-generation recommendation system built with PyTorch, FastAPI, and Docker Compose. The system pairs a Two-Tower deep learning retrieval model with semantic content similarity and popularity fallback mechanisms to reliably serve recommendations for both warm and cold-start users on the MovieLens 100K dataset.

## Objective

In modern recommendation systems, candidate generation models trained on interaction data struggle when serving users with sparse or zero interaction history (the cold-start problem). This project addresses candidate generation by combining neural retrieval with fallback strategies:

- **Two-Tower Neural Retrieval**: Efficient candidate scoring using learned user and item embedding representations for active users.
- **Content-Based Similarity**: High-dimensional semantic fallback for low-history users based on item text embeddings.
- **Popularity Baseline**: Non-personalized interaction fallback for zero-history users.
- **Warm/Cold Slice Evaluation**: Offline benchmark evaluating Recall@10 and NDCG@10 independently on warm and cold user partitions.
- **Production Serving**: API deployment via FastAPI and Docker Compose with health monitoring and real-time routing.

---

## Solution

The recommendation system implements a dynamic hybrid routing framework that selects the optimal recommendation strategy at inference time based on the user's historical interaction count:

1. **Two-Tower Neural Retrieval**: For users with interaction history exceeding threshold $N=10$, a PyTorch Two-Tower model computes dot-product scores between learned user and item embeddings to select top candidates.
2. **Content-Based Similarity**: For users with $1 \le \text{history count} \le 10$ (or unknown user IDs with item history), a content fallback computes a user profile vector by averaging the precomputed SentenceTransformer embeddings of history items, returning items with highest cosine similarity.
3. **Popularity-Based Fallback**: For zero-history users ($\text{history count} = 0$), a non-personalized baseline ranks items based on interaction volume weighted by average rating.

Combining these strategies ensures that the system provides personalized recommendations for active users while maintaining high retrieval quality for cold-start and anonymous sessions without service degradation or index crashes.

---

## Architecture

```
                       MovieLens 100K Dataset
                                 |
                                 v
                        Data Preparation
                                 |
                                 v
                     Train / Warm / Cold Split
                                 |
             +-------------------+-------------------+
             |                   |                   |
             v                   v                   v
      Two-Tower Model    Content Similarity   Popularity Baseline
       (PyTorch)       (MiniLM Embeddings)   (Weighted Ratings)
             |                   |                   |
             +-------------------+-------------------+
                                 |
                                 v
                           Hybrid Router
                              (N = 10)
                                 |
                                 v
                        FastAPI Serving Layer
                                 |
                                 v
                        Docker Compose (Port 8000)
```

---

## Model Architecture

The Two-Tower model (`models/two_tower.py`) consists of two distinct embedding towers:

- **User Tower**: Maps user IDs to a low-dimensional embedding space ($\mathbb{R}^{64}$). Out-Of-Vocabulary (OOV) and unknown users map to index `0`, initialized to zero vectors and frozen against gradient updates.
- **Item Tower**: Maps item IDs to the same shared embedding space ($\mathbb{R}^{64}$). Index `0` is similarly reserved for OOV/padding items.
- **Scoring**: Prediction score between user $u$ and item $i$ is calculated via dot product: $\text{Score}(u, i) = \vec{u} \cdot \vec{v}_i$. All-item candidate scoring efficiently computes matrix multiplication against all valid item embeddings ($1 \dots 1682$).

### In-Batch Negative Training
The model is trained using in-batch negative contrastive learning (`src/train.py`):
- **Batch Size ($B$)**: 256
- **Loss Computation**: For a batch of $B$ positive $(user, item)$ pairs, a score matrix of shape $B \times B$ is computed. The diagonal elements represent positive pairs, while off-diagonal entries act as negative samples.
- **Loss Function**: `torch.nn.CrossEntropyLoss()` optimizes targets $[0, 1, \dots, B-1]$ across epochs using the Adam optimizer.

---

## Cold-Start Strategy

The online router (`src/router.py`) enforces strict, deterministic routing rules based on the user's interaction count:

| History Count | Strategy Selected | Description |
|---:|---|---|
| $> 10$ | `two_tower` | Active user strategy using Two-Tower model inference. |
| $1 \le \text{count} \le 10$ | `content_fallback` | Low-history fallback using cosine similarity on item text embeddings. |
| $0$ | `popularity_fallback` | Zero-history fallback using global popularity ranking. |

### Unknown User Handling
- Unknown user IDs (e.g., `user_id = 99999`) with non-empty history safely route to `content_fallback`, avoiding embedding index out-of-bounds exceptions.
- Unknown user IDs with empty history route directly to `popularity_fallback`.

---

## Data

The system uses the canonical **MovieLens 100K** dataset:

- **Total Interactions**: 100,000 ratings (1–5 scale)
- **Users**: 943 unique users
- **Items**: 1,682 unique movies

### Train / Test Partitioning
A strict split (`src/split_data.py`) prevents data leakage and segregates cold users:

- `train.csv`: 73,727 interaction rows (used exclusively for model training and popularity scoring).
- `test_warm.csv`: 17,758 interaction rows (warm user evaluation slice; $\text{warm users} \subseteq \text{train users}$).
- `test_cold.csv`: 8,515 interaction rows (cold user evaluation slice; $\text{cold users} \cap \text{train users} = \emptyset$).

---

## Content Embeddings

Item text descriptions (combining movie title and genre metadata) are embedded using the local HuggingFace model `sentence-transformers/all-MiniLM-L6-v2` (`src/build_embeddings.py`):

- **Embedding Matrix File**: `data/processed/item_embeddings.npy`
- **Matrix Dimensions**: $1682 \times 384$
- **Data Type**: `float32`

Semantic embeddings capture genre and title relationships, enabling item similarity retrieval for users with limited interaction history.

---

## Offline Evaluation

Evaluation metrics are computed independently on warm and cold user slices using Recall@10 and NDCG@10 (`src/evaluate.py`, `src/metrics.py`):

| Strategy | Warm Recall@10 | Warm NDCG@10 | Cold Recall@10 | Cold NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.0551 | 0.0575 | 0.0787 | 0.0751 |
| Content | 0.0123 | 0.0117 | 0.0787 | 0.0751 |
| Two-Tower | 0.0497 | 0.0493 | 0.0245 | 0.0336 |
| **Hybrid** | **0.0498** | **0.0490** | **0.0787** | **0.0751** |

### Key Evaluation Insights
- **Cold Regression**: The learned Two-Tower model performance drops on cold users ($\text{Recall@10} = 0.0245$), as its user tower lacks learned representation for unseen users.
- **Hybrid Superiority**: The Hybrid router dynamic fallback mechanism raises cold user Recall@10 to **0.0787**, outperforming the pure Two-Tower model on cold-start users.

---

## Project Structure

```
two-tower-recommender/
│
├── .env                          # Local environment variables
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git ignore specifications
├── README.md                     # Project documentation
├── requirements.txt              # Production dependencies
├── submission.json               # Final hyperparameter metadata
├── Dockerfile                    # Container build configuration
├── docker-compose.yml            # Container orchestration specification
├── pyrightconfig.json            # Static type checker configuration
│
├── data/
│   ├── raw/                      # Raw MovieLens 100K files
│   │   ├── u.data
│   │   ├── u.item
│   │   ├── u.user
│   │   └── README
│   └── processed/                # Preprocessed data & embeddings
│       ├── train.csv
│       ├── test_warm.csv
│       ├── test_cold.csv
│       ├── items.csv
│       ├── item_embeddings.npy
│       ├── user_mapping.json
│       └── item_mapping.json
│
├── models/
│   ├── __init__.py
│   ├── two_tower.py              # PyTorch Two-Tower model definition
│   └── baselines.py              # Content & Popularity baseline algorithms
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration loader
│   ├── data_loader.py            # MovieLens data ingestion
│   ├── split_data.py             # Strict train/warm/cold splitter
│   ├── build_embeddings.py       # SentenceTransformer embedding pipeline
│   ├── train.py                  # PyTorch model training loop
│   ├── evaluate.py               # Offline evaluator & threshold sweeper
│   ├── metrics.py                # Recall@K and NDCG@K implementations
│   ├── router.py                 # Hybrid recommendation router
│   └── utils.py                  # Utility functions
│
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application & REST endpoints
│   └── schemas.py                # Pydantic request/response schemas
│
├── artifacts/
│   └── two_tower.pt              # Trained PyTorch checkpoint
│
├── results/
│   ├── metrics.json              # Evaluator benchmark results
│   ├── threshold_sweep.json      # Threshold optimization results
│   └── evaluation_report.json   # Full evaluation summary report
│
├── tests/
│   ├── __init__.py
│   ├── test_data_split.py        # Data splitting tests
│   ├── test_embeddings.py        # Embedding matrix tests
│   ├── test_metrics.py           # Metric calculation tests
│   ├── test_router.py            # Hybrid router rule tests
│   ├── test_api.py               # FastAPI endpoint tests
│   └── test_audit.py             # End-to-end verification audit suite
│
├── scripts/
│   ├── setup_data.py             # Dataset downloader
│   ├── run_pipeline.py           # End-to-end pipeline runner
│   └── smoke_test.py             # Quick verification script
│
└── .vscode/                      # Editor configuration
    ├── settings.json
    ├── launch.json
    └── extensions.json
```

---

## Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.11+ | Core programming language |
| **Framework** | PyTorch | Deep learning model development |
| **Embeddings** | Sentence-Transformers | Pretrained MiniLM-L6-v2 text encoder |
| **Data Processing** | Pandas, NumPy, Scikit-Learn | Data manipulation and similarity computation |
| **API Web Framework** | FastAPI, Uvicorn, Pydantic | High-performance asynchronous REST API |
| **Containerization** | Docker, Docker Compose | Containerized service deployment |
| **Testing** | Pytest | Automated test suite |

---

## Getting Started

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git
- Docker Desktop (optional for local non-containerized execution)

### 1. Clone the Repository
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd two-tower-recommender
```

### 2. Create Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment Configuration
Copy the configuration template to `.env`:
- **Windows**:
  ```powershell
  Copy-Item .env.example .env
  ```
- **Linux / macOS**:
  ```bash
  cp .env.example .env
  ```
*Note: `.env` is ignored by Git and requires no third-party API keys.*

---

## Running the Project

### A. Serve API Using Pre-existing Artifacts
If the repository already contains processed data and trained artifacts:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### B. Reproduce the End-to-End Pipeline
To re-run data downloading, pre-processing, embedding generation, training, and evaluation from scratch:
```bash
python scripts/run_pipeline.py
```

---

## Train the Model

To run model training independently:
```bash
python src/train.py
```
The trained PyTorch model checkpoint will be saved to `artifacts/two_tower.pt`.

---

## Run Evaluation

To view CLI documentation for offline evaluation:
```bash
python src/evaluate.py --help
```

To run offline evaluation and generate metric outputs:
```bash
python src/evaluate.py
```
Outputs are saved to:
- `results/metrics.json`
- `results/threshold_sweep.json`
- `results/evaluation_report.json`

---

## Run Tests

Execute the full automated test suite:
```bash
pytest -q
```
*Current test status: 99 passed.*

To run bytecode compilation verification:
```bash
python -m compileall app src models tests
```

---

## Recommendation API

Start the FastAPI application:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Health Check Endpoint
- **GET** `/health`
- **Response**: `{"status": "ok"}`

### Recommendation Endpoint
- **POST** `/api/recommend`

#### Request Schema
```json
{
  "user_id": 10,
  "history": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}
```

#### Response Schema
```json
{
  "user_id": 10,
  "strategy_used": "two_tower",
  "recommendations": [127, 288, 257, 181, 50, 100, 1, 286, 121, 174]
}
```
*Guarantees: Returns exactly 10 unique integer item IDs.*

---

## API Examples

### 1. Warm User (>10 items history)
```bash
curl -X POST "http://localhost:8000/api/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 10, "history": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}'
```
**Response**: `strategy_used: "two_tower"`

### 2. Low-History User (1–10 items history)
```bash
curl -X POST "http://localhost:8000/api/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 99999, "history": [101]}'
```
**Response**: `strategy_used: "content_fallback"`

### 3. Zero-History User (0 items history)
```bash
curl -X POST "http://localhost:8000/api/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 99999, "history": []}'
```
**Response**: `strategy_used: "popularity_fallback"`

---

## Docker Deployment

Build and run the containerized API using Docker Compose:

```bash
docker compose up -d --build
```

Check container status and health:
```bash
docker compose ps
```
*Expected status: `api → Up (healthy)`*

View container logs:
```bash
docker compose logs --tail=100 api
```

Stop container service:
```bash
docker compose down
```

---

## Key Configuration Parameters

System defaults defined in `src/config.py` and `submission.json`:

- `THRESHOLD_N`: `10`
- `EMBEDDING_DIM`: `64`
- `BATCH_SIZE`: `256`
- `LEARNING_RATE`: `0.001`
- `EPOCHS`: `10`
- `RANDOM_SEED`: `42`

---

## Key Features

- **Candidate Generation**: Two-Tower architecture scoring candidate items efficiently via matrix multiplication.
- **In-Batch Negatives**: Contrastive learning formulation operating over $B \times B$ score matrices.
- **Semantic Text Embeddings**: Precomputed SentenceTransformer MiniLM vectors for item metadata representation.
- **Dynamic Hybrid Routing**: Online history-based router serving Two-Tower, Content, or Popularity strategies.
- **Leak-Free Data Splitting**: Strict partition ensuring zero overlap between training and cold test users.
- **Out-of-Vocabulary Safety**: Index `0` handling for unobserved users and items prevents runtime exceptions.
- **Production Containerization**: Docker Compose orchestration with automated health checks on port 8000.

---

## Reproducibility

The repository guarantees end-to-end reproducibility:
- Fixed random seed (`RANDOM_SEED=42`) across data splits, model initialization, and training loops.
- Serialized item embeddings (`data/processed/item_embeddings.npy`) and trained PyTorch checkpoint (`artifacts/two_tower.pt`).
- One-step automated execution script via `python scripts/run_pipeline.py`.

---

## Submission Artifacts

- `artifacts/two_tower.pt`: Trained PyTorch Two-Tower model weights.
- `data/processed/item_embeddings.npy`: Pre-generated $1682 \times 384$ item text embeddings.
- `results/metrics.json`: Standardized offline evaluation metrics.
- `results/threshold_sweep.json`: Decision matrix for interaction threshold selection ($N=10$).
- `results/evaluation_report.json`: Detailed evaluation log.
- `submission.json`: Submission configuration file.

---

## License

License information can be added according to the repository owner's requirements.

---

## Author

**Chinni Rakesh**  
B.Tech — Computer Science & Engineering (Artificial Intelligence & Machine Learning)

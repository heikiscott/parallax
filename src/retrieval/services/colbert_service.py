"""ColBERT v2 Service for Late Interaction Retrieval.

This module provides ColBERT model loading and inference using Jina ColBERT v2.
ColBERT uses "late interaction" - computing token-level embeddings and MaxSim scoring.

Core Features:
- Query encoding to multi-vector representation
- Document encoding to multi-vector representation
- MaxSim scoring between query and document embeddings

Usage:
    service = ColBERTService()
    await service.initialize()

    query_emb = await service.encode_query("What is the capital of France?")
    doc_emb = await service.encode_document("Paris is the capital of France.")
    score = service.compute_maxsim(query_emb, doc_emb)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Optional dependency injection support
try:
    from config import load_config
    from core.di import get_bean, service
    _HAS_DI = True
except ImportError:
    _HAS_DI = False
    # Create no-op decorator when core.di is not available
    def service(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator


def _get_colbert_config():
    """Load ColBERT configuration from YAML."""
    if _HAS_DI:
        return load_config("src/colbert")
    return None


@dataclass
@service(name="colbert_config", primary=True)
class ColBERTConfig:
    """ColBERT model configuration.

    Configuration is loaded from config/src/colbert.yaml
    """
    model_name: str = "jinaai/jina-colbert-v2"
    device: str = "cpu"
    max_query_length: int = 128
    max_doc_length: int = 512
    batch_size: int = 8
    normalize_embeddings: bool = True
    trust_remote_code: bool = True

    def __post_init__(self):
        """Load configuration from YAML file."""
        try:
            cfg = _get_colbert_config()
            if cfg is None:
                logger.info("ColBERT config not available (DI not loaded), using defaults")
                return

            colbert_cfg = cfg.colbert

            self.model_name = getattr(colbert_cfg, 'model_name', self.model_name)
            self.device = getattr(colbert_cfg, 'device', self.device)
            self.max_query_length = int(getattr(colbert_cfg, 'max_query_length', self.max_query_length))
            self.max_doc_length = int(getattr(colbert_cfg, 'max_doc_length', self.max_doc_length))
            self.batch_size = int(getattr(colbert_cfg, 'batch_size', self.batch_size))
            self.normalize_embeddings = getattr(colbert_cfg, 'normalize_embeddings', self.normalize_embeddings)
            self.trust_remote_code = getattr(colbert_cfg, 'trust_remote_code', self.trust_remote_code)

        except Exception as e:
            logger.warning(f"Failed to load ColBERT config, using defaults: {e}")


class ColBERTService:
    """ColBERT v2 model service for late interaction retrieval.

    This service provides:
    - Lazy model initialization (only loads when first used)
    - Query/document encoding to multi-vector representations
    - MaxSim scoring for retrieval ranking

    The service is designed to be reusable across different retrieval pipelines.
    """

    def __init__(self, config: Optional[ColBERTConfig] = None):
        """Initialize ColBERT service.

        Args:
            config: Optional configuration. If not provided, loads from YAML.
        """
        self.config = config or ColBERTConfig()
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._torch = None
        self._F = None

    async def initialize(self) -> None:
        """Lazy initialization of the ColBERT model.

        This method is called automatically on first use.
        Model loading can take 10-30 seconds on CPU.
        """
        if self._initialized:
            return

        logger.info(f"Loading ColBERT model: {self.config.model_name}")
        logger.info(f"Device: {self.config.device}")

        try:
            # Import here to avoid loading torch at module import time
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self._F = F

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )

            # Load model
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )

            # Move to device
            if self.config.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on CUDA")
            else:
                self.model = self.model.cpu()
                if self.config.device == "cuda":
                    logger.warning("CUDA requested but not available, using CPU")
                logger.info("Model loaded on CPU")

            self.model.eval()
            self._initialized = True
            logger.info("ColBERT model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ColBERT model: {e}")
            raise

    def _ensure_initialized(self) -> None:
        """Ensure model is initialized (sync check)."""
        if not self._initialized:
            raise RuntimeError(
                "ColBERT service not initialized. Call 'await service.initialize()' first."
            )

    async def encode_query(self, query: str) -> np.ndarray:
        """Encode query to multi-vector representation.

        Args:
            query: The query text to encode.

        Returns:
            np.ndarray: Token embeddings of shape [seq_len, embedding_dim].
                       For Jina ColBERT v2, embedding_dim is 128.
        """
        await self.initialize()

        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=self.config.max_query_length,
            truncation=True,
            padding=True
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[0]  # [seq_len, dim]

            if self.config.normalize_embeddings:
                embeddings = self._F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

    async def encode_document(self, document: str) -> np.ndarray:
        """Encode a single document to multi-vector representation.

        Args:
            document: The document text to encode.

        Returns:
            np.ndarray: Token embeddings of shape [seq_len, embedding_dim].
        """
        await self.initialize()

        inputs = self.tokenizer(
            document,
            return_tensors="pt",
            max_length=self.config.max_doc_length,
            truncation=True,
            padding=True
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[0]  # [seq_len, dim]

            if self.config.normalize_embeddings:
                embeddings = self._F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

    async def encode_documents(self, documents: List[str]) -> List[np.ndarray]:
        """Batch encode documents to multi-vector representations.

        Args:
            documents: List of document texts to encode.

        Returns:
            List of np.ndarray, each of shape [seq_len, embedding_dim].
        """
        await self.initialize()

        all_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.config.max_doc_length,
                truncation=True,
                padding=True
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self.model(**inputs)
                batch_embs = outputs.last_hidden_state  # [batch, seq_len, dim]

                if self.config.normalize_embeddings:
                    batch_embs = self._F.normalize(batch_embs, p=2, dim=-1)

            # Convert each document's embeddings to numpy
            for j in range(batch_embs.shape[0]):
                emb = batch_embs[j].cpu().numpy()
                all_embeddings.append(emb)

            if i > 0 and i % (batch_size * 10) == 0:
                logger.info(f"Encoded {i}/{len(documents)} documents")

        return all_embeddings

    def compute_maxsim(
        self,
        query_emb: np.ndarray,
        doc_emb: np.ndarray
    ) -> float:
        """Compute MaxSim score between query and document embeddings.

        MaxSim is the core scoring function in ColBERT:
        - For each query token, find the maximum similarity with any document token
        - Sum these maximum similarities across all query tokens

        This captures fine-grained semantic matching at the token level.

        Args:
            query_emb: Query token embeddings [q_len, dim]
            doc_emb: Document token embeddings [d_len, dim]

        Returns:
            float: MaxSim score (higher is better)
        """
        # Compute similarity matrix: [q_len, d_len]
        sim_matrix = np.dot(query_emb, doc_emb.T)

        # For each query token, find max similarity with any doc token
        max_sim_per_query_token = np.max(sim_matrix, axis=1)

        # Sum across all query tokens
        return float(np.sum(max_sim_per_query_token))

    def compute_maxsim_batch(
        self,
        query_emb: np.ndarray,
        doc_embs: List[np.ndarray]
    ) -> List[float]:
        """Compute MaxSim scores for query against multiple documents.

        Args:
            query_emb: Query token embeddings [q_len, dim]
            doc_embs: List of document token embeddings

        Returns:
            List of MaxSim scores
        """
        return [self.compute_maxsim(query_emb, doc_emb) for doc_emb in doc_embs]

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension (128 for Jina ColBERT v2)."""
        return 128  # Jina ColBERT v2 uses 128-dim embeddings


# Singleton instance
_colbert_service: Optional[ColBERTService] = None


def get_colbert_service() -> ColBERTService:
    """Get the singleton ColBERT service instance.

    Returns:
        ColBERTService: The ColBERT service singleton.
    """
    global _colbert_service
    if _colbert_service is None:
        _colbert_service = ColBERTService()
    return _colbert_service


def reset_colbert_service() -> None:
    """Reset the ColBERT service singleton.

    Useful for testing or reconfiguration.
    """
    global _colbert_service
    _colbert_service = None

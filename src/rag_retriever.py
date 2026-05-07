"""
RAG Retriever — PubMed Central OA + DermNet NZ knowledge base.
Embedding model runs on CPU (no VRAM impact).
IMPORTANT: ISIC is explicitly excluded as a source (data leakage risk).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_ISIC_WARNING = (
    "⚠️ ISIC excluded from RAG sources: data used for StyleGAN + "
    "EfficientNet-B0 training → data leakage + Fitzpatrick I-III skin type bias amplification."
)


class RAGRetriever:
    """Retrieves relevant dermatology context from PubMed + DermNet knowledge base.

    Uses ChromaDB as local vector store and S-PubMedBert-MS-MARCO for embeddings.
    All embedding computation runs on CPU to preserve GPU VRAM for LLM.

    Sources:
        - PubMed Central Open Access: melanoma/dermoscopy abstracts
        - DermNet NZ: clinical descriptions and ABCDE criteria
        - FORBIDDEN: ISIC Archive (training data overlap)

    Args:
        config: Configuration dict from config.yaml (rag section).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.chroma_path = Path(config.get("chroma_path", "rag_kb/chroma_db"))
        self.embedding_model_name = config.get(
            "embedding_model", "pritamdeka/S-PubMedBert-MS-MARCO"
        )
        self.top_k = config.get("top_k", 5)
        self._embedding_model = None
        self._collection = None
        logger.info(_ISIC_WARNING)
        logger.info(
            f"RAGRetriever initialized | "
            f"store={self.chroma_path} | "
            f"top_k={self.top_k}"
        )

    def _load_embedding_model(self):
        """Lazy-load the sentence transformer on CPU.

        Returns:
            Loaded SentenceTransformer model.
        """
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(
                f"Loading embedding model: {self.embedding_model_name} (CPU)"
            )
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device="cpu",
            )
            logger.info("Embedding model loaded on CPU (~400MB RAM)")
        return self._embedding_model

    def _get_collection(self):
        """Lazy-load ChromaDB collection.

        Returns:
            ChromaDB collection object.

        Raises:
            RuntimeError: If knowledge base has not been built yet.
        """
        if self._collection is None:
            import chromadb

            if not self.chroma_path.exists() or not any(self.chroma_path.iterdir()):
                raise RuntimeError(
                    f"Knowledge base not found at {self.chroma_path}. "
                    "Run: python -m src.pipeline --build-rag-kb"
                )
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = client.get_collection("dermatology_kb")
            logger.info(
                f"ChromaDB collection loaded | "
                f"documents={self._collection.count()}"
            )
        return self._collection

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[dict]:
        """Retrieve top-k relevant chunks from the knowledge base.

        Args:
            query: Search query string. Typically built from
                classification + gradcam focus_regions.
            top_k: Number of results to return. Defaults to config value.

        Returns:
            List of dicts with keys:
                - text (str): Chunk text.
                - source (str): Source name (pubmed/dermnet).
                - score (float): Cosine similarity score.
                - reference (str): Citation/reference string.
        """
        k = top_k or self.top_k

        try:
            embedding_model = self._load_embedding_model()
            collection = self._get_collection()
        except RuntimeError as e:
            logger.warning(f"RAG unavailable: {e}")
            return []

        query_embedding = embedding_model.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB distance → similarity (cosine: 1 - distance)
            similarity = max(0.0, 1.0 - dist)
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": round(similarity, 4),
                "reference": meta.get("reference", ""),
            })

        logger.info(
            f"RAG retrieved {len(chunks)} chunks for query: '{query[:60]}...'"
        )
        return chunks

    def build_query(
        self, classification_result: dict, gradcam_result: dict
    ) -> str:
        """Build retrieval query from classification and GradCAM results.

        Args:
            classification_result: Output from MelanomaClassifier.predict().
            gradcam_result: Output from GradCAMExplainer.generate().

        Returns:
            Query string for embedding lookup.
        """
        cls = classification_result["class"]
        focus = gradcam_result.get("focus_regions", "central region")

        query = (
            f"{cls} dermoscopy ABCDE criteria with activation on "
            f"{focus} regions melanoma diagnosis"
        )
        return query

    def format_rag_context(
        self, retrieved_chunks: list[dict]
    ) -> tuple[str, str]:
        """Format retrieved chunks into prompt context and references.

        Args:
            retrieved_chunks: Output from retrieve().

        Returns:
            Tuple of (rag_context_text, rag_references_text) for ABCDE template.
        """
        if not retrieved_chunks:
            return (
                "No RAG context available.",
                "No references available.",
            )

        context_lines = []
        reference_lines = []

        for i, chunk in enumerate(retrieved_chunks, 1):
            source_tag = chunk["source"].upper()
            score = chunk["score"]
            context_lines.append(
                f"[{i}] ({source_tag}, similarità: {score:.3f})\n{chunk['text']}"
            )
            ref = chunk.get("reference", "")
            if ref:
                reference_lines.append(f"[{i}] {ref}")

        context_text = "\n\n".join(context_lines)
        references_text = (
            "\n".join(reference_lines) if reference_lines else "See RAG sources."
        )

        return context_text, references_text

    def is_available(self) -> bool:
        """Check if the knowledge base is built and accessible.

        Returns:
            True if ChromaDB collection exists and has documents.
        """
        try:
            collection = self._get_collection()
            return collection.count() > 0
        except Exception:
            return False

#!/usr/bin/env python3
"""
NVIDIA NeMo Retriever Embedding Client
Uses NVIDIA's NV-EmbedQA model via build.nvidia.com API
"""

import os
import requests
from typing import List, Optional
import numpy as np


class NVIDIAEmbeddings:
    """Client for NVIDIA NeMo Retriever embedding models"""

    def __init__(self, api_key: Optional[str] = None, model: str = "NV-Embed-QA"):
        """
        Initialize NVIDIA embeddings client

        Args:
            api_key: NVIDIA API key from build.nvidia.com
            model: Model name (default: NV-Embed-QA)
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.model = model
        self.base_url = "https://integrate.api.nvidia.com/v1/embeddings"

        # Always initialize local model as fallback
        from sentence_transformers import SentenceTransformer
        self.local_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Check if we should try NVIDIA API
        self.use_local = not self.api_key or self.api_key == "demo-key-for-hackathon"

        if self.use_local:
            print("⚠️  No NVIDIA API key - using local sentence-transformers fallback")
        else:
            print(f"✅ NVIDIA Embeddings initialized: {model} (with local fallback)")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if self.use_local:
            return self._embed_local(texts)

        return self._embed_nvidia(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if self.use_local:
            return self._embed_local([text])[0]

        return self._embed_nvidia([text])[0]

    def _embed_nvidia(self, texts: List[str]) -> List[List[float]]:
        """Call NVIDIA embedding API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Use correct model name for NVIDIA API
        model_name = "nvidia/nv-embedqa-e5-v5"  # Correct model name

        payload = {
            "input": texts,
            "model": model_name,
            "input_type": "passage",  # Required for NV-Embed models
            "encoding_format": "float",
            "truncate": "NONE"
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            # Extract embeddings in order
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

        except Exception as e:
            print(f"⚠️  NVIDIA API error, falling back to local: {e}")
            return self._embed_local(texts)

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Fallback to local sentence-transformers"""
        embeddings = self.local_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

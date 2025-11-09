#!/usr/bin/env python3
"""
Clinical Knowledge RAG System
Stores and retrieves Parkinson's clinical guidelines, treatment protocols, etc.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from .nvidia_embeddings import NVIDIAEmbeddings


class ClinicalKnowledgeRAG:
    """RAG system for clinical guidelines and medical knowledge"""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize Clinical Knowledge RAG

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        # Set up persistence directory
        if persist_directory is None:
            persist_directory = str(Path(__file__).parent.parent.parent / "data" / "clinical_knowledge_db")

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize NVIDIA embeddings
        self.embeddings = NVIDIAEmbeddings()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="clinical_guidelines",
            metadata={"description": "Parkinson's disease clinical guidelines and protocols"}
        )

        print(f"📚 Clinical Knowledge RAG initialized")
        print(f"   → Database: {persist_directory}")
        print(f"   → Documents: {self.collection.count()}")

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the knowledge base

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
        """
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(documents)

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Added {len(documents)} documents to Clinical Knowledge base")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Query the knowledge base with citation tracking

        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Dict with 'documents', 'metadatas', 'distances', 'citations'
        """
        # Embed query
        query_embedding = self.embeddings.embed_query(query_text)

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        # Build citations for explainability
        citations = self._build_citations(
            results.get('metadatas', [[]])[0],
            results.get('distances', [[]])[0],
            results.get('ids', [[]])[0]
        )

        # Format results
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'ids': results['ids'][0] if results['ids'] else [],
            'citations': citations  # Add citation tracking
        }

    def _build_citations(
        self,
        metadatas: List[Dict],
        distances: List[float],
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Build citation objects for explainability

        Args:
            metadatas: List of metadata dicts
            distances: List of similarity distances
            ids: List of document IDs

        Returns:
            List of citation dicts
        """
        citations = []

        for metadata, distance, doc_id in zip(metadatas, distances, ids):
            relevance_score = 1 / (1 + distance)  # Convert distance to score
            relevance_level = "HIGH" if distance < 0.5 else "MEDIUM" if distance < 0.8 else "LOW"

            citation = {
                'id': doc_id,
                'source': metadata.get('source', 'Unknown'),
                'category': metadata.get('category', 'General'),
                'relevance_score': round(relevance_score, 3),
                'relevance_level': relevance_level,
                'distance': round(distance, 3)
            }

            # Add page number if available
            if 'page' in metadata:
                citation['page'] = metadata['page']

            # Add publication year if available
            if 'year' in metadata:
                citation['year'] = metadata['year']

            citations.append(citation)

        return citations

    def get_formatted_context(
        self,
        query_text: str,
        n_results: int = 3,
        include_metadata: bool = True,
        include_citations: bool = True
    ) -> str:
        """
        Get formatted context string for LLM prompts

        Args:
            query_text: Query string
            n_results: Number of results to retrieve
            include_metadata: Whether to include source metadata
            include_citations: Whether to add citation markers

        Returns:
            Formatted context string (use get_context_with_citations for structured data)
        """
        results = self.query(query_text, n_results=n_results)

        if not results['documents']:
            return "No relevant clinical guidelines found."

        context_parts = []
        for i, (doc, metadata, distance) in enumerate(
            zip(results['documents'], results['metadatas'], results['distances'])
        ):
            relevance = "HIGH" if distance < 0.5 else "MEDIUM" if distance < 0.8 else "LOW"

            context = f"\n**Source {i+1}** (Relevance: {relevance})\n"
            if include_metadata and metadata:
                if 'source' in metadata:
                    context += f"Source: {metadata['source']}\n"
                if 'category' in metadata:
                    context += f"Category: {metadata['category']}\n"
            context += f"{doc}\n"

            # Add citation marker for tracking
            if include_citations:
                context += f"[Citation: {results['citations'][i]['id']}]\n"

            context_parts.append(context)

        return "\n".join(context_parts)

    def get_context_with_citations(
        self,
        query_text: str,
        n_results: int = 3
    ) -> Dict[str, Any]:
        """
        Get context AND citations for full explainability

        Args:
            query_text: Query string
            n_results: Number of results

        Returns:
            Dict with 'context' (string) and 'citations' (list)
        """
        results = self.query(query_text, n_results=n_results)

        context = self.get_formatted_context(query_text, n_results, include_citations=True)

        return {
            'context': context,
            'citations': results.get('citations', []),
            'query': query_text,
            'num_sources': len(results.get('documents', []))
        }

    def delete_all(self):
        """Delete all documents from the collection"""
        # Delete and recreate collection
        self.client.delete_collection("clinical_guidelines")
        self.collection = self.client.get_or_create_collection(
            name="clinical_guidelines",
            metadata={"description": "Parkinson's disease clinical guidelines and protocols"}
        )
        print("🗑️  Cleared all documents from Clinical Knowledge base")

    def count(self) -> int:
        """Get count of documents in knowledge base"""
        return self.collection.count()

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
        Query the knowledge base

        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Dict with 'documents', 'metadatas', 'distances'
        """
        # Embed query
        query_embedding = self.embeddings.embed_query(query_text)

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        # Format results
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }

    def get_formatted_context(
        self,
        query_text: str,
        n_results: int = 3,
        include_metadata: bool = True
    ) -> str:
        """
        Get formatted context string for LLM prompts

        Args:
            query_text: Query string
            n_results: Number of results to retrieve
            include_metadata: Whether to include source metadata

        Returns:
            Formatted context string
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
            context_parts.append(context)

        return "\n".join(context_parts)

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

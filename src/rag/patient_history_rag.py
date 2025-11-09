#!/usr/bin/env python3
"""
Patient History RAG System
- Stores patient visit history in SQLite
- Creates FAISS vector embeddings of voice features for similarity search
- Enables longitudinal trend detection and similar patient matching
"""

import sqlite3
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import faiss


class PatientHistoryRAG:
    """RAG system for patient visit history and similarity search"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Patient History RAG

        Args:
            db_path: Path to SQLite database
        """
        # Set up database path
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "patient_history.db")

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dicts

        # Initialize database schema
        self._init_database()

        # Initialize FAISS index for voice feature similarity
        self.feature_dim = 44  # Number of voice features
        self.faiss_index = None
        self.visit_id_map = []  # Maps FAISS index to visit_id

        # Load existing FAISS index if available
        self._load_faiss_index()

        print(f"📊 Patient History RAG initialized")
        print(f"   → Database: {db_path}")
        print(f"   → Visits: {self.count_visits()}")
        print(f"   → FAISS index: {self.faiss_index.ntotal if self.faiss_index else 0} vectors")

    def _init_database(self):
        """Create database schema"""
        cursor = self.conn.cursor()

        # Patient visits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_visits (
                visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                visit_date TEXT NOT NULL,

                -- Clinical features (44 total)
                jitter REAL,
                shimmer REAL,
                hnr REAL,
                voice_features TEXT,  -- JSON of all 44 features

                -- ML predictions
                pd_probability REAL,
                healthy_probability REAL,
                risk_level TEXT,
                prediction INTEGER,

                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')

        # Trend analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trend_analysis (
                trend_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                current_value REAL,
                previous_value REAL,
                percent_change REAL,
                trend_direction TEXT,  -- 'improving', 'worsening', 'stable'
                alert_triggered INTEGER DEFAULT 0,
                analysis_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patient_id ON patient_visits(patient_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_visit_date ON patient_visits(visit_date)')

        self.conn.commit()

    def add_visit(
        self,
        patient_id: str,
        visit_date: str,
        clinical_features: Dict[str, float],
        ml_result: Dict[str, Any],
        voice_features: Optional[np.ndarray] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Add a patient visit to the database

        Args:
            patient_id: Unique patient identifier
            visit_date: Visit date (ISO format)
            clinical_features: Dict with jitter, shimmer, hnr
            ml_result: ML prediction results
            voice_features: Full 44-dimensional feature vector (for FAISS)
            notes: Optional notes

        Returns:
            visit_id: ID of the inserted visit
        """
        cursor = self.conn.cursor()

        # Convert voice features to JSON if provided
        import json
        voice_features_json = json.dumps(voice_features.tolist()) if voice_features is not None else None

        cursor.execute('''
            INSERT INTO patient_visits (
                patient_id, visit_date,
                jitter, shimmer, hnr, voice_features,
                pd_probability, healthy_probability, risk_level, prediction,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, visit_date,
            clinical_features.get('jitter'), clinical_features.get('shimmer'), clinical_features.get('hnr'),
            voice_features_json,
            ml_result.get('pd_probability'), ml_result.get('healthy_probability'),
            ml_result.get('risk_level'), ml_result.get('prediction'),
            notes
        ))

        self.conn.commit()
        visit_id = cursor.lastrowid

        # Add to FAISS index if voice features provided
        if voice_features is not None:
            self._add_to_faiss(visit_id, voice_features)

        print(f"   ✅ Added visit for patient {patient_id} (visit_id: {visit_id})")

        return visit_id

    def _add_to_faiss(self, visit_id: int, features: np.ndarray):
        """Add feature vector to FAISS index"""
        # Initialize FAISS index if not exists
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatL2(self.feature_dim)

        # Ensure features are 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Add to index
        self.faiss_index.add(features.astype('float32'))
        self.visit_id_map.append(visit_id)

        # Save updated index
        self._save_faiss_index()

    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        if self.faiss_index is None:
            return

        index_path = Path(self.db_path).parent / "faiss_index.bin"
        map_path = Path(self.db_path).parent / "visit_id_map.pkl"

        faiss.write_index(self.faiss_index, str(index_path))

        with open(map_path, 'wb') as f:
            pickle.dump(self.visit_id_map, f)

    def _load_faiss_index(self):
        """Load FAISS index from disk"""
        index_path = Path(self.db_path).parent / "faiss_index.bin"
        map_path = Path(self.db_path).parent / "visit_id_map.pkl"

        if index_path.exists() and map_path.exists():
            self.faiss_index = faiss.read_index(str(index_path))

            with open(map_path, 'rb') as f:
                self.visit_id_map = pickle.load(f)

    def get_patient_history(self, patient_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get visit history for a patient

        Args:
            patient_id: Patient ID
            limit: Optional limit on number of visits

        Returns:
            List of visit dicts
        """
        cursor = self.conn.cursor()

        query = 'SELECT * FROM patient_visits WHERE patient_id = ? ORDER BY visit_date DESC'
        if limit:
            query += f' LIMIT {limit}'

        cursor.execute(query, (patient_id,))

        return [dict(row) for row in cursor.fetchall()]

    def detect_trends(self, patient_id: str) -> Dict[str, Any]:
        """
        Detect trends in patient's voice features over time

        Args:
            patient_id: Patient ID

        Returns:
            Dict with trend analysis
        """
        visits = self.get_patient_history(patient_id)

        if len(visits) < 2:
            return {'message': 'Need at least 2 visits for trend analysis', 'visits_count': len(visits)}

        # Analyze trends for key features
        trends = {}
        features = ['jitter', 'shimmer', 'hnr', 'pd_probability']

        for feature in features:
            values = [v[feature] for v in visits if v[feature] is not None]

            if len(values) >= 2:
                current = values[0]  # Most recent
                previous = values[1]  # Previous visit
                percent_change = ((current - previous) / previous * 100) if previous != 0 else 0

                # Determine trend direction
                if feature == 'hnr':
                    # Higher HNR is better
                    direction = 'improving' if percent_change > 5 else 'worsening' if percent_change < -5 else 'stable'
                else:
                    # Lower jitter, shimmer, pd_prob is better
                    direction = 'improving' if percent_change < -5 else 'worsening' if percent_change > 5 else 'stable'

                trends[feature] = {
                    'current': current,
                    'previous': previous,
                    'percent_change': round(percent_change, 2),
                    'direction': direction,
                    'alert': direction == 'worsening'
                }

                # Save to trend_analysis table
                self._save_trend(patient_id, feature, current, previous, percent_change, direction)

        return {
            'patient_id': patient_id,
            'visits_analyzed': len(visits),
            'trends': trends,
            'overall_assessment': self._assess_overall_trend(trends)
        }

    def _save_trend(self, patient_id: str, feature_name: str, current: float, previous: float,
                    percent_change: float, direction: str):
        """Save trend analysis to database"""
        cursor = self.conn.cursor()

        alert_triggered = 1 if direction == 'worsening' else 0

        cursor.execute('''
            INSERT INTO trend_analysis (
                patient_id, feature_name, current_value, previous_value,
                percent_change, trend_direction, alert_triggered
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (patient_id, feature_name, current, previous, percent_change, direction, alert_triggered))

        self.conn.commit()

    def _assess_overall_trend(self, trends: Dict) -> str:
        """Assess overall trend direction"""
        worsening_count = sum(1 for t in trends.values() if t['direction'] == 'worsening')
        improving_count = sum(1 for t in trends.values() if t['direction'] == 'improving')

        if worsening_count >= 2:
            return 'CONCERNING - Multiple features worsening'
        elif worsening_count == 1:
            return 'WATCH - One feature worsening'
        elif improving_count >= 2:
            return 'POSITIVE - Multiple features improving'
        else:
            return 'STABLE - No significant changes'

    def find_similar_patients(self, features: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Find similar patients using FAISS similarity search

        Args:
            features: 44-dimensional feature vector
            k: Number of similar patients to return

        Returns:
            List of similar visit dicts with distances
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        # Ensure features are 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Search FAISS index
        distances, indices = self.faiss_index.search(features.astype('float32'), k)

        # Retrieve visit details
        similar_visits = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.visit_id_map):
                visit_id = self.visit_id_map[idx]
                visit = self.get_visit_by_id(visit_id)
                if visit:
                    visit['similarity_distance'] = float(dist)
                    visit['similarity_score'] = 1 / (1 + dist)  # Convert distance to similarity score
                    similar_visits.append(visit)

        return similar_visits

    def get_visit_by_id(self, visit_id: int) -> Optional[Dict]:
        """Get a visit by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM patient_visits WHERE visit_id = ?', (visit_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def count_visits(self) -> int:
        """Count total visits in database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM patient_visits')
        return cursor.fetchone()[0]

    def close(self):
        """Close database connection"""
        self.conn.close()

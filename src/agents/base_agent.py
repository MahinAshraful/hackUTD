#!/usr/bin/env python3
"""
Base Agent class for all specialized agents
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json


class BaseAgent:
    """Base class for all AI agents in the system"""

    def __init__(self, name: str, nemotron_client):
        """
        Initialize base agent

        Args:
            name: Agent name
            nemotron_client: NemotronClient instance
        """
        self.name = name
        self.nemotron = nemotron_client
        self.status = 'initialized'
        self.start_time = None
        self.end_time = None
        self.result = None

    def start(self):
        """Mark agent as started"""
        self.status = 'running'
        self.start_time = datetime.now()

    def complete(self, result: Dict[str, Any]):
        """
        Mark agent as complete

        Args:
            result: Agent execution result
        """
        self.status = 'completed'
        self.end_time = datetime.now()
        self.result = result

    def fail(self, error: str):
        """
        Mark agent as failed

        Args:
            error: Error message
        """
        self.status = 'failed'
        self.end_time = datetime.now()
        self.result = {'error': error}

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status

        Returns:
            Status dict with timing and result info
        """
        duration = None
        if self.start_time:
            end = self.end_time or datetime.now()
            duration = (end - self.start_time).total_seconds()

        return {
            'agent': self.name,
            'status': self.status,
            'duration_seconds': duration,
            'result': self.result
        }

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent logic (to be implemented by subclasses)

        Args:
            context: Execution context with input data

        Returns:
            Agent result dict
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def _create_result(self, success: bool, **kwargs) -> Dict[str, Any]:
        """
        Create standardized result dict

        Args:
            success: Whether execution succeeded
            **kwargs: Additional result fields

        Returns:
            Standardized result dict
        """
        return {
            'success': success,
            'agent': self.name,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

#!/usr/bin/env python3
"""
Base Agent class for all specialized agents
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import sys
from pathlib import Path

# Import validator and messenger
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from validators.medical_validator import validate_agent_output
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    print("⚠️  Medical validator not available")

try:
    from messaging.agent_messenger import messenger, MessageTypes
    MESSENGER_AVAILABLE = True
except ImportError:
    MESSENGER_AVAILABLE = False
    print("⚠️  Agent messenger not available")


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
        self.context = {}  # Store context for validation
        self.validation_result = None

        # Register with messaging system
        if MESSENGER_AVAILABLE:
            messenger.register_agent(self.name)
            self.messenger = messenger
        else:
            self.messenger = None

    def start(self):
        """Mark agent as started"""
        self.status = 'running'
        self.start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"🤖 {self.name} Agent - STARTING")
        print(f"{'='*70}")

    def complete(self, result: Dict[str, Any]):
        """
        Mark agent as complete with validation

        Args:
            result: Agent execution result
        """
        # Validate output if validator available
        if VALIDATOR_AVAILABLE:
            validation = validate_agent_output(self.name, result, self.context)
            self.validation_result = validation

            if not validation['is_valid']:
                print(f"⚠️  VALIDATION FAILED: {validation['message']}")
                # Add validation info to result
                result['validation'] = validation
            elif validation.get('warnings'):
                print(f"⚠️  VALIDATION WARNINGS:")
                for warning in validation['warnings']:
                    print(f"     - {warning}")
                result['validation'] = validation
            else:
                print(f"✅ Output validated successfully")
                result['validation'] = validation

        self.status = 'completed'
        self.end_time = datetime.now()
        self.result = result

        duration = (self.end_time - self.start_time).total_seconds()
        print(f"✅ {self.name} Agent - COMPLETED ({duration:.2f}s)")

        # Show result summary
        if 'content' in result:
            content_preview = result['content'][:150] if len(result['content']) > 150 else result['content']
            print(f"   📄 Output: {content_preview}...")
        elif 'plan' in result:
            plan_preview = result['plan'][:150] if len(result['plan']) > 150 else result['plan']
            print(f"   📋 Plan: {plan_preview}...")
        elif 'explanation' in result:
            exp_preview = result['explanation'][:150] if len(result['explanation']) > 150 else result['explanation']
            print(f"   💡 Explanation: {exp_preview}...")
        elif 'treatment_plan' in result:
            plan_preview = result['treatment_plan'][:150] if len(result['treatment_plan']) > 150 else result['treatment_plan']
            print(f"   💊 Treatment: {plan_preview}...")
        elif 'trajectory_analysis' in result:
            traj_preview = result['trajectory_analysis'][:150] if len(result['trajectory_analysis']) > 150 else result['trajectory_analysis']
            print(f"   📊 Risk: {traj_preview}...")

        print(f"{'='*70}\n")

    def fail(self, error: str):
        """
        Mark agent as failed

        Args:
            error: Error message
        """
        self.status = 'failed'
        self.end_time = datetime.now()
        self.result = {'error': error}
        print(f"❌ {self.name} Agent - FAILED: {error}")
        print(f"{'='*70}\n")

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

    # Messaging methods
    def send_message(
        self,
        to_agent: str,
        message_type: str,
        content: Any,
        priority: int = 0,
        requires_response: bool = False
    ) -> Optional[str]:
        """
        Send a message to another agent

        Args:
            to_agent: Recipient agent name
            message_type: Type of message
            content: Message content
            priority: Priority (higher = more urgent)
            requires_response: Whether response is required

        Returns:
            message_id or None if messenger unavailable
        """
        if not self.messenger:
            return None

        return self.messenger.send_message(
            from_agent=self.name,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            priority=priority,
            requires_response=requires_response
        )

    def receive_messages(self, max_messages: int = 10) -> List[Any]:
        """
        Process pending messages

        Args:
            max_messages: Maximum messages to process

        Returns:
            List of handler responses
        """
        if not self.messenger:
            return []

        return self.messenger.process_messages(self.name, max_messages)

    def register_message_handler(self, message_type: str, handler: callable):
        """
        Register a handler for a specific message type

        Args:
            message_type: Type of message to handle
            handler: Function to handle the message
        """
        if self.messenger:
            self.messenger.register_handler(self.name, message_type, handler)

    def has_pending_messages(self) -> bool:
        """Check if agent has pending messages"""
        if not self.messenger:
            return False

        return self.messenger.has_pending_messages(self.name)

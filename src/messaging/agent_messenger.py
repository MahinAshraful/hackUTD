#!/usr/bin/env python3
"""
Agent-to-Agent Messaging System
Enables collaborative workflows and dynamic agent communication
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from queue import Queue, Empty
import threading


class Message:
    """Message object for agent communication"""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        content: Any,
        priority: int = 0,
        requires_response: bool = False
    ):
        self.id = f"msg_{datetime.now().timestamp()}"
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.requires_response = requires_response
        self.timestamp = datetime.now().isoformat()
        self.response = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'from': self.from_agent,
            'to': self.to_agent,
            'type': self.message_type,
            'content': self.content,
            'priority': self.priority,
            'requires_response': self.requires_response,
            'timestamp': self.timestamp,
            'response': self.response
        }


class AgentMessenger:
    """Message broker for agent-to-agent communication"""

    def __init__(self):
        self.message_queues: Dict[str, Queue] = {}
        self.message_history: List[Message] = []
        self.handlers: Dict[str, Dict[str, Callable]] = {}  # agent -> {msg_type: handler}
        self.lock = threading.Lock()

    def register_agent(self, agent_name: str):
        """Register an agent with the messaging system"""
        with self.lock:
            if agent_name not in self.message_queues:
                self.message_queues[agent_name] = Queue()
                self.handlers[agent_name] = {}
                print(f"📬 Agent '{agent_name}' registered for messaging")

    def register_handler(
        self,
        agent_name: str,
        message_type: str,
        handler: Callable[[Message], Any]
    ):
        """
        Register a message handler for an agent

        Args:
            agent_name: Name of the agent
            message_type: Type of message to handle
            handler: Function to handle the message
        """
        with self.lock:
            if agent_name not in self.handlers:
                self.handlers[agent_name] = {}

            self.handlers[agent_name][message_type] = handler
            print(f"   📝 Handler registered: {agent_name}.{message_type}")

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        content: Any,
        priority: int = 0,
        requires_response: bool = False
    ) -> str:
        """
        Send a message from one agent to another

        Args:
            from_agent: Sending agent name
            to_agent: Receiving agent name
            message_type: Type of message
            content: Message content
            priority: Priority (higher = more urgent)
            requires_response: Whether response is required

        Returns:
            message_id: ID of the sent message
        """
        message = Message(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            priority=priority,
            requires_response=requires_response
        )

        with self.lock:
            # Ensure recipient is registered
            if to_agent not in self.message_queues:
                self.register_agent(to_agent)

            # Add to queue
            self.message_queues[to_agent].put((priority, message))
            self.message_history.append(message)

        print(f"   📨 Message sent: {from_agent} → {to_agent} [{message_type}]")

        return message.id

    def receive_message(self, agent_name: str, timeout: float = 0.1) -> Optional[Message]:
        """
        Receive a message for an agent

        Args:
            agent_name: Agent name
            timeout: Timeout in seconds

        Returns:
            Message or None if queue is empty
        """
        if agent_name not in self.message_queues:
            return None

        try:
            priority, message = self.message_queues[agent_name].get(timeout=timeout)
            return message
        except Empty:
            return None

    def process_messages(self, agent_name: str, max_messages: int = 10) -> List[Any]:
        """
        Process all pending messages for an agent

        Args:
            agent_name: Agent name
            max_messages: Maximum messages to process

        Returns:
            List of handler responses
        """
        responses = []

        for _ in range(max_messages):
            message = self.receive_message(agent_name)

            if message is None:
                break

            # Check if handler exists
            if agent_name in self.handlers and message.message_type in self.handlers[agent_name]:
                handler = self.handlers[agent_name][message.message_type]

                print(f"   📥 Processing message: {message.from_agent} → {agent_name} [{message.message_type}]")

                # Execute handler
                try:
                    response = handler(message)
                    message.response = response

                    # Send response back if required
                    if message.requires_response:
                        self.send_message(
                            from_agent=agent_name,
                            to_agent=message.from_agent,
                            message_type=f"{message.message_type}_RESPONSE",
                            content=response,
                            priority=message.priority
                        )

                    responses.append(response)

                except Exception as e:
                    print(f"   ⚠️  Handler error: {e}")
                    responses.append({'error': str(e)})
            else:
                print(f"   ⚠️  No handler for {message.message_type} on {agent_name}")

        return responses

    def broadcast_message(
        self,
        from_agent: str,
        message_type: str,
        content: Any,
        exclude: Optional[List[str]] = None
    ):
        """
        Broadcast a message to all agents

        Args:
            from_agent: Sending agent
            message_type: Message type
            content: Content
            exclude: List of agents to exclude
        """
        exclude = exclude or []

        for agent_name in self.message_queues.keys():
            if agent_name not in exclude and agent_name != from_agent:
                self.send_message(
                    from_agent=from_agent,
                    to_agent=agent_name,
                    message_type=message_type,
                    content=content
                )

    def get_message_history(
        self,
        agent_name: Optional[str] = None,
        message_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get message history

        Args:
            agent_name: Filter by agent (sender or receiver)
            message_type: Filter by message type

        Returns:
            List of message dicts
        """
        filtered = self.message_history

        if agent_name:
            filtered = [
                m for m in filtered
                if m.from_agent == agent_name or m.to_agent == agent_name
            ]

        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]

        return [m.to_dict() for m in filtered]

    def has_pending_messages(self, agent_name: str) -> bool:
        """Check if agent has pending messages"""
        if agent_name not in self.message_queues:
            return False

        return not self.message_queues[agent_name].empty()


# Global messenger instance
messenger = AgentMessenger()


# Common message types
class MessageTypes:
    """Standard message types for agent communication"""

    # Request types
    REQUEST_DATA = "request_data"
    REQUEST_ANALYSIS = "request_analysis"
    REQUEST_RE_EVALUATION = "request_re_evaluation"

    # Response types
    DATA_RESPONSE = "data_response"
    ANALYSIS_RESPONSE = "analysis_response"

    # Notifications
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    ALERT = "alert"

    # Collaboration
    COLLABORATE = "collaborate"
    DELEGATE = "delegate"
    CONSENSUS_REQUEST = "consensus_request"

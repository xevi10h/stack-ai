import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class NodeRole(Enum):
    """Node role in the cluster"""

    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"  # During election


class NodeStatus(Enum):
    """Node health status"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ReplicationLog:
    """Represents a replicated operation"""

    log_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    operation: str = ""  # create, update, delete
    entity_type: str = ""  # library, document, chunk
    entity_id: UUID = None
    data: Dict[str, Any] = field(default_factory=dict)
    committed: bool = False


@dataclass
class Node:
    """Represents a database node in the cluster"""

    node_id: str
    host: str
    port: int
    role: NodeRole = NodeRole.FOLLOWER
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    replication_lag: int = 0  # Number of ops behind leader
    term: int = 0  # Election term (Raft-inspired)


class ReplicationManager:
    """
    Manages replication from leader to followers

    Features:
    - Async replication (eventual consistency)
    - Replication log for ordered operations
    - Configurable replication factor
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.replication_log: List[ReplicationLog] = []
        self.current_leader: Optional[str] = None
        self.term = 0

    def register_node(self, node: Node) -> None:
        """Register a new node in the cluster"""
        self.nodes[node.node_id] = node

        # First node becomes leader
        if len(self.nodes) == 1:
            node.role = NodeRole.LEADER
            self.current_leader = node.node_id

    def get_leader(self) -> Optional[Node]:
        """Get the current leader node"""
        if self.current_leader:
            return self.nodes.get(self.current_leader)
        return None

    def get_followers(self) -> List[Node]:
        """Get all follower nodes"""
        return [node for node in self.nodes.values() if node.role == NodeRole.FOLLOWER]

    async def replicate(
        self, operation: str, entity_type: str, entity_id: UUID, data: Dict[str, Any]
    ) -> ReplicationLog:
        """
        Replicate an operation to all followers

        Flow:
        1. Append to replication log
        2. Send to all healthy followers asynchronously
        3. Return immediately (async replication)
        """
        log_entry = ReplicationLog(
            operation=operation, entity_type=entity_type, entity_id=entity_id, data=data
        )

        self.replication_log.append(log_entry)

        # Replicate to followers asynchronously
        followers = self.get_followers()

        # Fire and forget (async replication)
        for follower in followers:
            asyncio.create_task(self._send_to_follower(follower, log_entry))

        return log_entry

    async def _send_to_follower(
        self, follower: Node, log_entry: ReplicationLog
    ) -> bool:
        """Send a log entry to a follower node"""
        # In a real implementation, this would use HTTP/gRPC
        # For this demo, we simulate the replication
        try:
            await asyncio.sleep(0.01)  # Simulate network delay
            follower.replication_lag = len(self.replication_log) - 1
            return True
        except Exception:
            follower.status = NodeStatus.UNHEALTHY
            return False

    def elect_leader(self) -> Optional[Node]:
        """
        Elect a new leader using simplified Raft algorithm

        Election process:
        1. Increment term
        2. Find healthy followers
        3. Select follower with lowest replication lag
        4. Promote to leader
        """
        self.term += 1

        # Find best candidate (healthy, lowest lag)
        candidates = [
            node
            for node in self.nodes.values()
            if node.status == NodeStatus.HEALTHY and node.role == NodeRole.FOLLOWER
        ]

        if not candidates:
            return None

        # Select candidate with lowest replication lag
        new_leader = min(candidates, key=lambda n: n.replication_lag)
        new_leader.role = NodeRole.LEADER
        new_leader.term = self.term

        # Demote old leader if exists
        if self.current_leader and self.current_leader in self.nodes:
            old_leader = self.nodes[self.current_leader]
            if old_leader.node_id != new_leader.node_id:
                old_leader.role = NodeRole.FOLLOWER

        self.current_leader = new_leader.node_id
        return new_leader


class HealthChecker:
    """
    Monitors node health via heartbeats

    Features:
    - Periodic heartbeat checks
    - Automatic failover on leader failure
    - Configurable timeout
    """

    def __init__(
        self, replication_manager: ReplicationManager, heartbeat_timeout: float = 5.0
    ):
        self.replication_manager = replication_manager
        self.heartbeat_timeout = heartbeat_timeout
        self._running = False

    async def start(self):
        """Start health checking loop"""
        self._running = True
        while self._running:
            await self._check_health()
            await asyncio.sleep(1)

    async def _check_health(self):
        """Check health of all nodes"""
        current_time = time.time()

        for node in self.replication_manager.nodes.values():
            # Check if node is responsive
            if current_time - node.last_heartbeat > self.heartbeat_timeout:
                node.status = NodeStatus.UNHEALTHY

                # If leader is unhealthy, trigger election
                if (
                    node.role == NodeRole.LEADER
                    and node.node_id == self.replication_manager.current_leader
                ):
                    await self._handle_leader_failure()

    async def _handle_leader_failure(self):
        """Handle leader failure by electing new leader"""
        new_leader = self.replication_manager.elect_leader()
        if new_leader:
            print(f"Leader failover: New leader elected - {new_leader.node_id}")

    def heartbeat(self, node_id: str):
        """Receive heartbeat from a node"""
        if node_id in self.replication_manager.nodes:
            node = self.replication_manager.nodes[node_id]
            node.last_heartbeat = time.time()
            node.status = NodeStatus.HEALTHY

    def stop(self):
        """Stop health checking"""
        self._running = False

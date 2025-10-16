"""
Tests for Leader-Follower replication architecture
"""

import asyncio
import time

import pytest

from app.infrastructure.replication.node import (
    HealthChecker,
    Node,
    NodeRole,
    NodeStatus,
    ReplicationManager,
)


def test_register_first_node_becomes_leader():
    """First registered node should automatically become the leader"""
    manager = ReplicationManager()

    node1 = Node(node_id="node1", host="localhost", port=8001)
    manager.register_node(node1)

    assert node1.role == NodeRole.LEADER
    assert manager.current_leader == "node1"
    assert manager.get_leader() == node1


def test_register_subsequent_nodes_become_followers():
    """Subsequent nodes should become followers"""
    manager = ReplicationManager()

    node1 = Node(node_id="node1", host="localhost", port=8001)
    node2 = Node(node_id="node2", host="localhost", port=8002)
    node3 = Node(node_id="node3", host="localhost", port=8003)

    manager.register_node(node1)
    manager.register_node(node2)
    manager.register_node(node3)

    assert node1.role == NodeRole.LEADER
    assert node2.role == NodeRole.FOLLOWER
    assert node3.role == NodeRole.FOLLOWER

    followers = manager.get_followers()
    assert len(followers) == 2
    assert node2 in followers
    assert node3 in followers


@pytest.mark.asyncio
async def test_async_replication():
    """Test async replication to followers"""
    manager = ReplicationManager()

    node1 = Node(node_id="node1", host="localhost", port=8001)
    node2 = Node(node_id="node2", host="localhost", port=8002)
    node3 = Node(node_id="node3", host="localhost", port=8003)

    manager.register_node(node1)
    manager.register_node(node2)
    manager.register_node(node3)

    log_entry = await manager.replicate(
        operation="create",
        entity_type="library",
        entity_id="test-uuid",
        data={"name": "Test Library"},
    )

    await asyncio.sleep(0.1)

    assert log_entry in manager.replication_log
    assert log_entry.operation == "create"
    assert log_entry.entity_type == "library"


def test_leader_election_on_failure():
    """Test that a new leader is elected when current leader fails"""
    manager = ReplicationManager()

    node1 = Node(node_id="node1", host="localhost", port=8001)
    node2 = Node(node_id="node2", host="localhost", port=8002)
    node3 = Node(node_id="node3", host="localhost", port=8003)

    manager.register_node(node1)
    manager.register_node(node2)
    manager.register_node(node3)

    assert node1.role == NodeRole.LEADER

    node1.status = NodeStatus.UNHEALTHY

    new_leader = manager.elect_leader()

    assert new_leader is not None
    assert new_leader.role == NodeRole.LEADER
    assert new_leader.status == NodeStatus.HEALTHY
    assert new_leader.node_id in ["node2", "node3"]
    assert manager.current_leader == new_leader.node_id


def test_election_selects_node_with_lowest_lag():
    """Test that leader election prioritizes nodes with lowest replication lag"""
    manager = ReplicationManager()

    node1 = Node(node_id="node1", host="localhost", port=8001)
    node2 = Node(node_id="node2", host="localhost", port=8002)
    node3 = Node(node_id="node3", host="localhost", port=8003)

    manager.register_node(node1)
    manager.register_node(node2)
    manager.register_node(node3)

    node2.replication_lag = 10
    node3.replication_lag = 5

    node1.status = NodeStatus.UNHEALTHY

    new_leader = manager.elect_leader()

    assert new_leader == node3


def test_no_election_without_healthy_followers():
    """Test that election fails when no healthy followers are available"""
    manager = ReplicationManager()

    node1 = Node(node_id="node1", host="localhost", port=8001)
    node2 = Node(node_id="node2", host="localhost", port=8002)

    manager.register_node(node1)
    manager.register_node(node2)

    node1.status = NodeStatus.UNHEALTHY
    node2.status = NodeStatus.UNHEALTHY

    new_leader = manager.elect_leader()

    assert new_leader is None


@pytest.mark.asyncio
async def test_health_checker_detects_unhealthy_nodes():
    """Test that health checker marks nodes as unhealthy after timeout"""
    manager = ReplicationManager()
    health_checker = HealthChecker(manager, heartbeat_timeout=0.5)

    node1 = Node(node_id="node1", host="localhost", port=8001)
    node2 = Node(node_id="node2", host="localhost", port=8002)

    manager.register_node(node1)
    manager.register_node(node2)

    node2.last_heartbeat = time.time() - 1.0

    await health_checker._check_health()

    assert node1.status == NodeStatus.HEALTHY
    assert node2.status == NodeStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_heartbeat_updates_node_status():
    """Test that heartbeat updates restore node to healthy"""
    manager = ReplicationManager()
    health_checker = HealthChecker(manager)

    node1 = Node(node_id="node1", host="localhost", port=8001)
    manager.register_node(node1)

    node1.status = NodeStatus.UNHEALTHY
    node1.last_heartbeat = time.time() - 10

    health_checker.heartbeat("node1")

    assert node1.status == NodeStatus.HEALTHY
    assert node1.last_heartbeat > time.time() - 1


def test_term_increments_on_election():
    """Test that term number increments with each election"""
    manager = ReplicationManager()

    node1 = Node(node_id="node1", host="localhost", port=8001)
    node2 = Node(node_id="node2", host="localhost", port=8002)

    manager.register_node(node1)
    manager.register_node(node2)

    initial_term = manager.term

    node1.status = NodeStatus.UNHEALTHY
    new_leader = manager.elect_leader()

    assert manager.term == initial_term + 1
    assert new_leader.term == manager.term

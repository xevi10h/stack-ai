"""
Node Management API

Endpoints for managing cluster nodes, replication, and health checking.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_replication_manager
from app.infrastructure.replication.node import (
    Node,
    NodeRole,
    NodeStatus,
    ReplicationManager,
)

router = APIRouter(prefix="/nodes", tags=["Nodes"])


class RegisterNodeRequest(BaseModel):
    node_id: str = Field(..., description="Unique node identifier")
    host: str = Field(..., description="Node hostname or IP address")
    port: int = Field(..., description="Node port", gt=0, le=65535)


class NodeResponse(BaseModel):
    node_id: str
    host: str
    port: int
    role: str
    status: str
    last_heartbeat: float
    replication_lag: int
    term: int


class ClusterStatusResponse(BaseModel):
    total_nodes: int
    leader_id: str | None
    healthy_nodes: int
    unhealthy_nodes: int
    current_term: int


@router.post(
    "/register",
    response_model=NodeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new node",
)
def register_node(
    request: RegisterNodeRequest,
    replication_manager: ReplicationManager = Depends(get_replication_manager),
):
    """
    Register a new node in the cluster.

    The first registered node becomes the leader automatically.
    Subsequent nodes are added as followers.
    """
    node = Node(
        node_id=request.node_id,
        host=request.host,
        port=request.port,
    )

    replication_manager.register_node(node)

    return NodeResponse(
        node_id=node.node_id,
        host=node.host,
        port=node.port,
        role=node.role.value,
        status=node.status.value,
        last_heartbeat=node.last_heartbeat,
        replication_lag=node.replication_lag,
        term=node.term,
    )


@router.get(
    "/",
    response_model=List[NodeResponse],
    summary="List all nodes",
)
def list_nodes(
    replication_manager: ReplicationManager = Depends(get_replication_manager),
):
    """
    Get a list of all registered nodes in the cluster.
    """
    nodes = replication_manager.nodes.values()

    return [
        NodeResponse(
            node_id=node.node_id,
            host=node.host,
            port=node.port,
            role=node.role.value,
            status=node.status.value,
            last_heartbeat=node.last_heartbeat,
            replication_lag=node.replication_lag,
            term=node.term,
        )
        for node in nodes
    ]


@router.get(
    "/leader",
    response_model=NodeResponse,
    summary="Get current leader",
)
def get_leader(
    replication_manager: ReplicationManager = Depends(get_replication_manager),
):
    """
    Get the current leader node.

    Returns 404 if no leader is elected.
    """
    leader = replication_manager.get_leader()

    if not leader:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No leader currently elected",
        )

    return NodeResponse(
        node_id=leader.node_id,
        host=leader.host,
        port=leader.port,
        role=leader.role.value,
        status=leader.status.value,
        last_heartbeat=leader.last_heartbeat,
        replication_lag=leader.replication_lag,
        term=leader.term,
    )


@router.get(
    "/followers",
    response_model=List[NodeResponse],
    summary="Get all followers",
)
def get_followers(
    replication_manager: ReplicationManager = Depends(get_replication_manager),
):
    """
    Get a list of all follower nodes.
    """
    followers = replication_manager.get_followers()

    return [
        NodeResponse(
            node_id=node.node_id,
            host=node.host,
            port=node.port,
            role=node.role.value,
            status=node.status.value,
            last_heartbeat=node.last_heartbeat,
            replication_lag=node.replication_lag,
            term=node.term,
        )
        for node in followers
    ]


@router.get(
    "/status",
    response_model=ClusterStatusResponse,
    summary="Get cluster status",
)
def get_cluster_status(
    replication_manager: ReplicationManager = Depends(get_replication_manager),
):
    """
    Get overall cluster health and status.
    """
    nodes = list(replication_manager.nodes.values())
    healthy_nodes = sum(1 for node in nodes if node.status == NodeStatus.HEALTHY)
    unhealthy_nodes = sum(1 for node in nodes if node.status == NodeStatus.UNHEALTHY)

    return ClusterStatusResponse(
        total_nodes=len(nodes),
        leader_id=replication_manager.current_leader,
        healthy_nodes=healthy_nodes,
        unhealthy_nodes=unhealthy_nodes,
        current_term=replication_manager.term,
    )


@router.post(
    "/{node_id}/heartbeat",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Send heartbeat",
)
def send_heartbeat(
    node_id: str,
    replication_manager: ReplicationManager = Depends(get_replication_manager),
):
    """
    Send a heartbeat for a specific node.

    This updates the node's last_heartbeat timestamp and marks it as healthy.
    """
    if node_id not in replication_manager.nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node {node_id} not found",
        )

    node = replication_manager.nodes[node_id]
    import time

    node.last_heartbeat = time.time()
    node.status = NodeStatus.HEALTHY


@router.post(
    "/elect-leader",
    response_model=NodeResponse,
    summary="Trigger leader election",
)
def trigger_election(
    replication_manager: ReplicationManager = Depends(get_replication_manager),
):
    """
    Manually trigger a leader election.

    This is useful for testing or forcing a failover.
    """
    new_leader = replication_manager.elect_leader()

    if not new_leader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No healthy followers available for election",
        )

    return NodeResponse(
        node_id=new_leader.node_id,
        host=new_leader.host,
        port=new_leader.port,
        role=new_leader.role.value,
        status=new_leader.status.value,
        last_heartbeat=new_leader.last_heartbeat,
        replication_lag=new_leader.replication_lag,
        term=new_leader.term,
    )

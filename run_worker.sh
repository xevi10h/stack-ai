#!/bin/bash
# Run Temporal worker with sandbox disabled for demo purposes

export TEMPORAL_WORKFLOW_SANDBOX_UNRESTRICTED=true
python3 -m app.infrastructure.temporal.worker

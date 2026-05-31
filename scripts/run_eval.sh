#!/bin/bash
# Run MCP evaluation with Claude Haiku
# Usage: ./scripts/run_eval.sh [--limit N] [--model MODEL]

export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-sk-e384f4c9c526c31fe4a4a96c05b3fd3cbc2cb1c07a83c4402d0f3f69ae95c733}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://vip.claudible.io}"

uv run python scripts/eval_mcp_claude.py "$@"

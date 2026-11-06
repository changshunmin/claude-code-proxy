#!/bin/bash
# Claude Code 启动脚本（使用代理）

PROXY_URL="${ANTHROPIC_BASE_URL:-http://localhost:8082}"
AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-dummy-token}"

echo "正在启动 Claude Code（通过代理）..."
echo "代理地址: $PROXY_URL"

export ANTHROPIC_BASE_URL="$PROXY_URL"
export ANTHROPIC_AUTH_TOKEN="$AUTH_TOKEN"

claude "$@"

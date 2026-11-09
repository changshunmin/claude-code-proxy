#!/bin/bash
# Claude Code Proxy 启动脚本

cd "$(dirname "$0")"

# 检查 .env 文件是否存在
if [ ! -f ".env" ]; then
    echo "错误：.env 文件不存在，请先配置！"
    exit 1
fi

# 加载环境变量
export $(grep -v '^#' .env | xargs)

# 验证必要配置
if [ "$OPENAI_API_KEY" = "你的API密钥" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "错误：请先在 .env 文件中设置 OPENAI_API_KEY"
    exit 1
fi

if [ "$OPENAI_BASE_URL" = "你的API基础URL" ] || [ -z "$OPENAI_BASE_URL" ]; then
    echo "错误：请先在 .env 文件中设置 OPENAI_BASE_URL"
    exit 1
fi

echo "==================================="
echo "Claude Code Proxy 启动中..."
echo "==================================="
echo "代理地址: http://${HOST:-0.0.0.0}:${PORT:-8082}"
echo "上游 API: $OPENAI_BASE_URL"
echo "BIG 模型: $BIG_MODEL"
echo "MIDDLE 模型: $MIDDLE_MODEL"
echo "SMALL 模型: $SMALL_MODEL"
echo "==================================="
echo ""
echo "在新终端运行以下命令使用 Claude Code:"
echo "  export ANTHROPIC_BASE_URL=\"http://api.deepseek.ai/anthropic\""
echo "  export ANTHROPIC_AUTH_TOKEN=\"<DeepSeek API key from .env>\""
echo "  export ANTHROPIC_MODEL=\"deepseek-flash\""
echo "  claude"
echo ""
echo "（http://api.deepseek.ai 需在 /etc/hosts 指向 127.0.0.1）"
echo ""

python3 start_proxy.py

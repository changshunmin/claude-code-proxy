#!/bin/bash
cd /opt/proxy/claude-code-proxy || exit 1

# 激活虚拟环境
source venv/bin/activate

# 执行你自己的启动命令（就是你原来的 start.sh 逻辑）
# 这里直接运行你的启动脚本，后台挂起 + 日志输出
nohup ./start.sh >> app.log 2>&1 &

echo $! > app.pid
echo "启动成功，PID：$(cat app.pid)"
echo "日志：app.log"

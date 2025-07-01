#!/bin/bash
echo "程序开始运行了"
CMD="python S2A-SAM/tools/train.py"  # 这里填入你启动模型训练时的命令，比如这里我用python run.py指令启动模型
 
$CMD &
sleep 10

while pgrep -f "$CMD" > /dev/null; do
    sleep 60  # 每隔1min检查一次
done
 
echo "程序已结束，正在关机..."
/usr/bin/shutdown -h now

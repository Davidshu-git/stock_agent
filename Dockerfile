# 使用轻量级 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统编译依赖 (FAISS 和某些科学计算库需要)
RUN apt-get update && apt-get install -y \
    gcc build-essential tzdata \
    && rm -rf /var/lib/apt/lists/*

# ⏰ 极其重要：强制设定时区为上海，否则 daily_job.py 的 15:30 调度会错乱！
ENV TZ=Asia/Shanghai

# 优先拷贝并安装依赖（利用 Docker 缓存层加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝项目源码
COPY . .

# 默认启动盘后调度器
CMD ["python", "daily_job.py"]

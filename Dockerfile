FROM stock-base:latest
WORKDIR /app
ENV TZ=Asia/Shanghai
COPY . .
CMD ["python", "daily_job.py"]
#!/usr/bin/env python3
"""
独立进程启动器：供 trigger_daily_report 跨进程调用

本脚本作为独立进程的入口点，负责：
1. 初始化独立进程的日志系统
2. 执行 job_routine() 盘后任务
3. 更新任务状态文件
4. 捕获异常并持久化

使用方式:
    python spawn_job.py --job-id job_20260313_143052_abc123
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Literal

# 确保能找到 daily_job 模块
sys.path.insert(0, str(Path(__file__).parent))

from daily_job import job_routine

JobStatus = Literal["pending", "running", "completed", "failed"]


class JobState(TypedDict, total=False):
    """任务状态数据结构"""
    job_id: str
    status: JobStatus
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    log_path: str | None


def update_status(job_id: str, status: JobStatus, **kwargs) -> None:
    """
    原子更新任务状态文件
    
    Args:
        job_id: 任务唯一 ID
        status: 任务状态（优先级最高，不会被覆盖）
        **kwargs: 其他需要更新的字段
    """
    status_dir = Path("./jobs/status").resolve()
    status_dir.mkdir(parents=True, exist_ok=True)
    
    status_file = status_dir / f"{job_id}.json"
    
    # 如果文件已存在，先读取现有数据（保留历史字段）
    existing_data = {}
    if status_file.exists():
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # 合并数据：existing_data 优先，但 status 字段必须使用传入的新值
    merged_data: JobState = {
        "job_id": job_id,
        **existing_data,
        "status": status,  # status 字段始终使用新值，确保状态正确更新
        **kwargs
    }
    
    # 原子写入
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)


def main() -> None:
    """独立进程主函数"""
    parser = argparse.ArgumentParser(description="研报任务独立进程启动器")
    parser.add_argument("--job-id", required=True, help="任务唯一 ID")
    args = parser.parse_args()
    
    job_id = args.job_id
    
    # 准备日志路径
    log_dir = Path("./jobs/logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"
    
    # 初始化独立日志系统（固定为 INFO 级别）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("spawn_job")
    
    logger.info("=" * 60)
    logger.info(f"🚀 独立进程启动 | Job ID: {job_id} | PID: {__import__('os').getpid()}")
    logger.info(f"日志路径：{log_path}")
    logger.info("=" * 60)
    
    # 更新状态为 running
    update_status(
        job_id=job_id,
        status="running",
        started_at=datetime.now().isoformat(),
        log_path=str(log_path)
    )
    
    try:
        logger.info("开始执行盘后研报生成任务...")
        job_routine()
        
        # 执行成功
        update_status(
            job_id=job_id,
            status="completed",
            completed_at=datetime.now().isoformat()
        )
        logger.info(f"✅ 任务执行完成 | Job ID: {job_id}")
        logger.info("=" * 60)
        
    except Exception as e:
        # 执行失败
        error_msg = f"{type(e).__name__}: {str(e)}"
        update_status(
            job_id=job_id,
            status="failed",
            completed_at=datetime.now().isoformat(),
            error=error_msg
        )
        logger.exception(f"❌ 任务执行失败 | Job ID: {job_id} | Error: {error_msg}")
        logger.info("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

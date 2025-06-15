"""
预训练模块
实现各种预训练语言模型的训练任务

作者: shihom_wu
版本: 1.0.0
"""

from .tasks import (
    MLMTask,
    NSPTask,
    SOPTask,
    CLMTask,
    PretrainingDataProcessor
)

__all__ = [
    'MLMTask',
    'NSPTask', 
    'SOPTask',
    'CLMTask',
    'PretrainingDataProcessor'
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DEM融合脚本（分块掩模替换法）
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入待测试模块
# 注意：等脚本创建后再导入
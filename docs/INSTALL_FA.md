# FlashAttention 安装指南

## 当前状态

运行 benchmark 时会显示安装状态：
```
  Installation Status:
    FA2: ✗ not installed
    FA3: ✗ not installed
    FA4: ✓ installed
```

## 安装方法

### FlashAttention 2

```bash
# 方法 1: PyPI (推荐)
pip install flash-attn --no-build-isolation

# 方法 2: 从源码编译
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install -e . --no-build-isolation
```

**要求：**
- CUDA 11.6+
- PyTorch 2.0+
- GPU: SM80+ (A100, H100, B200)

### FlashAttention 3 (Hopper+)

```bash
# FA3 需要 H100/H200
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install -e . --no-build-isolation

# 或使用特定版本
pip install flash-attn==3.0.0 --no-build-isolation
```

**要求：**
- CUDA 12.0+
- GPU: SM90 (H100/H200)

### FlashAttention 4 (Blackwell)

```bash
# FA4 仅支持 B200
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install -e "flash_attn/cute[cu13]" --no-build-isolation
```

**要求：**
- CUDA 12.4+
- GPU: SM100 (B200)

## 快速验证

```python
# 检查安装
python -c "
try:
    from flash_attn import flash_attn_func
    print('FA2: ✓')
except:
    print('FA2: ✗')

try:
    from flash_attn_interface import flash_attn_func
    print('FA3: ✓')
except:
    print('FA3: ✗')

try:
    from flash_attn.cute.interface import flash_attn_func
    print('FA4: ✓')
except:
    print('FA4: ✗')
"
```

## Docker 环境（推荐）

```dockerfile
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# 安装 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 安装 FlashAttention
pip install flash-attn --no-build-isolation

# 对于 B200 + FA4
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install -e "flash_attn/cute[cu13]" --no-build-isolation
```

## 已知问题

### 1. 编译时间长
FA2/3 编译需要 10-30 分钟，请耐心等待。

### 2. CUDA 版本不匹配
```bash
# 检查 CUDA 版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 确保两者一致
```

### 3. 内存不足
编译时可能需要 16GB+ RAM，建议使用 swap：
```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 参考链接

- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [FA2 Paper](https://arxiv.org/abs/2307.08691)
- [FA3 Paper](https://arxiv.org/abs/2408.04268)

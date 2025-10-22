import os
from matplotlib import pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
import torch
import json
import numpy as np
import psutil


def cluster_output_handler(listResult):
    clusterIndexList = []
    for i in range(len(set(listResult))):
        clusterIndexList.append([])
    for i in range(len(listResult)):
        clusterIndexList[listResult[i]].append(i)

    return clusterIndexList

def edgeList2edgeIndex(edgeList):
    result=[[i[0],i[1]] for i in edgeList]
    return result

def csr_sub_to_edge_index(csr_mat, dev):
    """scipy.csr -> edge_index (2×E) torch.LongTensor for GAT"""
    coo = csr_mat.tocoo()
    if coo.nnz == 0:
        return torch.zeros((2,0), dtype=torch.long, device=dev)
    edge_idx = np.vstack([coo.row, coo.col]).astype(np.int64)
    return torch.from_numpy(edge_idx).long().to(dev)

def csr_sub_to_torch_sparse(csr_mat, dev):
    """scipy.csr -> torch.sparse_coo_tensor"""
    coo = csr_mat.tocoo()
    if coo.nnz == 0:
        # 防止空图引发下游算子崩溃：构造一个 0 元素的稀疏张量
        idx = torch.zeros((2,0), dtype=torch.long, device=dev)
        val = torch.zeros((0,), dtype=torch.float32, device=dev)
        return torch.sparse_coo_tensor(idx, val, torch.Size(coo.shape), device=dev)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)).to(dev)
    values  = torch.from_numpy(coo.data.astype(np.float32)).to(dev)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape), device=dev).coalesce()

def subgraph_posweight_norm(csr_mat, dev):
    """为子图计算 pos_weight / norm（标量）"""
    m = csr_mat.shape[0]
    total = float(m * m)
    ones = float(csr_mat.sum())
    # 防止除零
    pos_w = (total - ones) / max(ones, 1.0)
    norm  = total / max((total - ones) * 2.0, 1.0)
    return (torch.tensor(pos_w, dtype=torch.float32, device=dev),
            torch.tensor(norm,  dtype=torch.float32, device=dev))

def plot(y, xlabel='epochs', ylabel='', hline=None, output_dir='', suffix=''):
    plt.plot(range(len(y)), y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y=hline, color='green', linestyle='-') if hline else None
    plt.savefig(os.path.join(output_dir, f"{ylabel.replace(' ', '_')}{suffix}.png"), dpi=200)
    plt.clf()

def log_gpu_memory(device, note=""):
    """记录当前 GPU 显存使用情况"""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        tag = f" [{note}]" if note else ""
        logging.info(f"[GPU MEM{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {peak:.2f} MB | Total: {total:.2f} MB")

def log_system_usage(device, note=""):
    """记录 GPU/CPU 占用情况"""
    # GPU 内存
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        logging.info(
            f"[GPU{note}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB "
            f"| Peak: {peak:.2f} MB | Total: {total:.2f} MB"
        )

    # CPU 内存
    cpu_mem = psutil.virtual_memory()
    logging.info(
        f"[CPU{note}] Used: {cpu_mem.used/1024**2:.2f} MB | "
        f"Available: {cpu_mem.available/1024**2:.2f} MB | "
        f"Percent: {cpu_mem.percent:.1f}%"
    )

def save_run_stats(save_dir, runtime, peak_mem=None, epoch_peaks=None, total_mem=None):
    """将运行时间、显存峰值等写入文件"""
    os.makedirs(save_dir, exist_ok=True)
    stats_path = os.path.join(save_dir, "run_stats.json")

    stats = {
        "runtime_seconds": round(runtime, 2),
        "runtime_minutes": round(runtime / 60, 2),
    }
    if peak_mem is not None:
        stats["peak_memory_MB"] = round(peak_mem, 2)
    if total_mem is not None:
        stats["gpu_total_MB"] = round(total_mem, 2)
    if epoch_peaks is not None:
        stats["epoch_peak_memory_MB"] = [round(x, 2) for x in epoch_peaks]

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logging.info(f"--------> Saved run stats to {stats_path}")


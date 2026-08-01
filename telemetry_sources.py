# telemetry_sources.py  —  health スレッドから使うテレメトリ取得ヘルパー。
#   依存は任意。無ければ空 dict を返すだけでアプリは動く。
#   ホットパスからは絶対に呼ばない（コストがあるため health スレッドが ≈1Hz で叩く）。


def _nvml_handle():
    """NVMLを一度だけ初期化し、(pynvmlモジュール, GPU0ハンドル) を返す。
    gpu_stats/gpu_static_info の両方が使う初期化処理を一本化する。"""
    import pynvml
    if not getattr(_nvml_handle, "_init", False):
        pynvml.nvmlInit()
        _nvml_handle._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        _nvml_handle._init = True
    return pynvml, _nvml_handle._h


def gpu_stats():
    """nvidia-ml-py (pip install nvidia-ml-py) があれば GPU 情報を返す。無ければ空。
    VRAM使用量・消費電力は HWダウングレード判断（下位カードへ下げられるか）の決定打。"""
    try:
        pynvml, h = _nvml_handle()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "gpu_temp": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "gpu_util[%]": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
            "gpu_clock[mhz]": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "gpu_mem_used[mb]": mem.used / 1e6,                          # VRAM 使用量
            "gpu_power[w]": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,  # 消費電力（TDP/電力枠）
        }
    except Exception:
        return {}


def gpu_static_info():
    """GPU 名・総 VRAM 等の静的情報。startup イベントに載せ、比較の前提を固定する。
    総 VRAM は定数なので毎サンプルではなく startup のみで十分。"""
    try:
        pynvml, h = _nvml_handle()
        name = pynvml.nvmlDeviceGetName(h)
        return {
            "gpu_name": name.decode() if isinstance(name, bytes) else name,
            "gpu_mem_total_mb": pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e6,
        }
    except Exception:
        return {}


def proc_mem():
    """自プロセスの RSS（常駐メモリ）。システム全体より正確にリークを追える。無ければ空。"""
    try:
        import psutil
        return {"proc_rss[mb]": psutil.Process().memory_info().rss / 1e6}
    except Exception:
        return {}


def torch_vram():
    """torch の VRAM 内訳。alloc=生きているテンソル（本物のリーク）、
    reserved=確保プール（これだけ増えるのは無害）。CUDA 不在時は空。"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "torch_vram_alloc[mb]": torch.cuda.memory_allocated() / 1e6,
                "torch_vram_reserved[mb]": torch.cuda.memory_reserved() / 1e6,
            }
    except Exception:
        pass
    return {}


def sys_stats():
    """psutil があれば CPU/メモリ/ディスクを返す。無ければ空。"""
    try:
        import psutil, shutil
        return {
            "cpu_util[%]": psutil.cpu_percent(interval=None),
            "ram_used[mb]": psutil.virtual_memory().used / 1e6,
            "disk_free[gb]": shutil.disk_usage(".").free / 1e9,
        }
    except Exception:
        return {}


def _lhm_find(node, text):
    """LHM JSON ツリーを再帰探索して Text が一致するノードの Value を返す。"""
    if node.get("Text") == text:
        return node.get("Value")
    for child in node.get("Children", []):
        result = _lhm_find(child, text)
        if result is not None:
            return result
    return None


def cpu_temp_c():
    """CPU パッケージ温度を返す。取得できなければ None（アプリは無停止で続行）。
    優先順位: LibreHardwareMonitor Web Server → psutil（Linux/Mac 向け）
    LHM は管理者権限で起動し Options→Remote Web Server→Run を有効にする必要がある。"""
    # --- LHM Web Server 経由（Windows + LHM 常駐時）---
    try:
        import urllib.request, json
        with urllib.request.urlopen("http://localhost:8085/data.json", timeout=0.5) as r:
            data = json.loads(r.read())
        # "CPU Package" を優先、なければ "Core Max"（AMD等）
        for label in ("CPU Package", "CPU Tdie", "Core Max"):
            raw = _lhm_find(data, label)
            if raw:
                return float(raw.split()[0])
    except Exception:
        pass

    # --- psutil フォールバック（Linux/Mac 環境向け）---
    try:
        import psutil
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures() or {}
            for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                if key in temps and temps[key]:
                    return temps[key][0].current
    except Exception:
        pass

    return None


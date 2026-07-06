"""
setup_check.py — 环境检测

运行：python setup_check.py
"""

import sys
import importlib
from pathlib import Path


def check(name: str, min_version: str = None) -> bool:
    try:
        mod = importlib.import_module(name.replace("-", "_"))
        ver = getattr(mod, "__version__", "unknown")
        if min_version and ver != "unknown":
            from packaging.version import Version
            ok = Version(ver) >= Version(min_version)
        else:
            ok = True
        status = "✅" if ok else "⚠️ "
        print(f"  {status} {name:<20} {ver}")
        return ok
    except ImportError:
        print(f"  ❌ {name:<20} 未安装")
        return False


def main():
    print(f"Python {sys.version}")
    print(f"项目目录：{Path(__file__).parent.absolute()}")

    # Python 版本检测
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"⚠️  Python 版本过低（当前 {major}.{minor}，建议 3.10+）")
    else:
        print(f"✅ Python 版本：{major}.{minor}（建议 3.10+）")
    print()

    print("依赖检测：")
    ok = all([
        check("torch", "2.0.0"),
        check("numpy", "1.24.0"),
        check("sklearn"),
        check("tqdm"),
        check("matplotlib"),
    ])

    print("\n数据文件检测：")
    data_dir = Path(__file__).parent / "data"
    files = {
        "trajectory_data.npz": "轨迹数据",
        "dataset.npz": "训练数据集",
    }
    data_ok = True
    for fname, desc in files.items():
        p = data_dir / fname
        if p.exists():
            size = p.stat().st_size / 1024 / 1024
            print(f"  ✅ {fname:<30} {size:.1f} MB  ({desc})")
        else:
            print(f"  ❌ {fname:<30} 不存在（{desc}）")
            data_ok = False

    print("\n模型文件检测：")
    model_dir = Path(__file__).parent / "outputs" / "models"
    models = ["GRU_best.pt", "TCN_best.pt", "Transformer_best.pt"]
    model_ok = True
    for fname in models:
        p = model_dir / fname
        if p.exists():
            size = p.stat().st_size / 1024
            print(f"  ✅ {fname:<30} {size:.0f} KB")
        else:
            print(f"  ⚠️  {fname:<30} 不存在（将跳过该模型评估）")
            model_ok = False

    print()
    if ok and data_ok:
        print("✅ 环境就绪，可以运行：python run.py")
        if not model_ok:
            print("⚠️  部分模型文件缺失，仅评估 A3 算法和已有模型")
    else:
        if not ok:
            print("❌ 请先安装依赖：pip install -r requirements.txt")
        if not data_ok:
            print("❌ 请将数据文件放入 data/ 目录")


if __name__ == "__main__":
    main()
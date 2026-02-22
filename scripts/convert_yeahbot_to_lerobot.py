#!/usr/bin/env python3
"""
将 YeahBot-C1 原始数据（local_data_raw）转换为 LeRobot Dataset v3 格式。
每个 session_* 文件夹视为一个 episode；相机图像与 joint_states 按时间戳对齐。
用法（在 AutoDL 或本机，先安装 lerobot）:
  pip install "lerobot[smolvla]"
  python scripts/convert_yeahbot_to_lerobot.py --raw-dir local_data_raw --output-dir ./lerobot_yeahbot_c1 [--push-to-hub --repo-id USER/yeahbot-c1]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# 延迟导入，兼容不同 lerobot 版本（0.1.x 可能用 common.datasets，0.4.x 用 datasets）
# #region agent log
def _debug_log(msg: str, data: dict | None = None):
    import json
    log_path = Path(__file__).resolve().parents[1] / ".cursor" / "debug.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"message": msg, "data": data or {}, "timestamp": __import__("time").time()}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
# #endregion

def _ensure_lerobot():
    err = None
    tried = []
    # #region agent log
    _debug_log("_ensure_lerobot entry", {"hypothesisId": "H1"})
    # #endregion
    for module_path in (
        "lerobot.datasets.lerobot_dataset",  # 0.4.x
        "lerobot.common.datasets.lerobot_dataset",  # 部分旧版
    ):
        # #region agent log
        _debug_log("import attempt", {"module_path": module_path, "hypothesisId": "H2"})
        # #endregion
        try:
            mod = __import__(module_path, fromlist=["LeRobotDataset"])
            LeRobotDataset = getattr(mod, "LeRobotDataset")
            try:
                from lerobot.datasets.utils import DEFAULT_FEATURES
            except ImportError:
                from lerobot.common.datasets.utils import DEFAULT_FEATURES
            # #region agent log
            _debug_log("import success", {"module_path": module_path, "hypothesisId": "H2"})
            # #endregion
            return LeRobotDataset, DEFAULT_FEATURES
        except ImportError as e:
            err = e
            tried.append({"path": module_path, "err": str(e)})
            continue
    # #region agent log
    import lerobot
    lb_file = getattr(lerobot, "__file__", None)
    lb_dir = sorted([x for x in dir(lerobot) if not x.startswith("_")])
    try:
        import pkgutil
        submodules = [m.name for m in pkgutil.iter_modules(lerobot.__path__)] if getattr(lerobot, "__path__", None) else []
    except Exception as e2:
        submodules = [str(e2)]
    _debug_log("lerobot structure before raise", {
        "lerobot.__file__": lb_file,
        "lerobot dir (public)": lb_dir,
        "lerobot submodules": submodules,
        "tried": tried,
        "hypothesisId": "H3",
    })
    # #endregion
    raise ImportError(
        "未找到 LeRobot 的 LeRobotDataset。当前环境可能安装了旧版 lerobot（0.1.x 等），"
        "其包结构与 v3 数据集不兼容。请安装 0.4.x：pip install \"lerobot[smolvla]>=0.4\" ，"
        "或在该环境执行：pip show lerobot 查看版本。"
    ) from err


# YeahBot 关节名（与 joint_states_data.csv 中 positions 顺序一致）
STATE_DIM = 15  # 4 wheel + 11 arm
CAMERA_KEY = "top"  # 在 LeRobot 里作为 observation.images.top
IMAGE_SHAPE = (480, 640, 3)  # H, W, C（与 image_index 一致）


def load_session_data(session_dir: Path):
    """加载一个 session 的 image_index 和 joint_states，按时间戳对齐。"""
    session_dir = Path(session_dir)
    camera_dir = session_dir / "camera"
    joint_path = session_dir / "joint_states" / "joint_states_data.csv"
    image_index_path = camera_dir / "image_index.csv"

    if not image_index_path.exists() or not joint_path.exists():
        return None, None

    # 图像索引: timestamp -> filename
    img_df = pd.read_csv(image_index_path)
    img_df = img_df.sort_values("timestamp").reset_index(drop=True)

    # 关节: timestamp, positions (分号分隔)
    joint_df = pd.read_csv(joint_path)
    joint_df["positions"] = joint_df["positions"].apply(
        lambda s: np.array([float(x) for x in str(s).split(";")], dtype=np.float32)
    )
    joint_ts = joint_df["timestamp"].values
    joint_pos = np.stack(joint_df["positions"].values)

    if joint_pos.shape[1] != STATE_DIM:
        # 若列数不对，尝试截断或填充
        if joint_pos.shape[1] > STATE_DIM:
            joint_pos = joint_pos[:, :STATE_DIM]
        else:
            pad = np.zeros((joint_pos.shape[0], STATE_DIM - joint_pos.shape[1]), dtype=np.float32)
            joint_pos = np.hstack([joint_pos, pad])

    return img_df, (joint_ts, joint_pos)


def interpolate_state_at_timestamp(ts: float, joint_ts: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
    """对给定时间戳插值得到关节状态。"""
    if ts <= joint_ts[0]:
        return joint_pos[0]
    if ts >= joint_ts[-1]:
        return joint_pos[-1]
    idx = np.searchsorted(joint_ts, ts, side="right") - 1
    t0, t1 = joint_ts[idx], joint_ts[idx + 1]
    w = (ts - t0) / (t1 - t0) if t1 > t0 else 0.0
    return (1 - w) * joint_pos[idx] + w * joint_pos[idx + 1]


def build_lerobot_features(fps: int, state_dim: int = STATE_DIM):
    """构建 LeRobot features 字典（与 create 一致）。"""
    from lerobot.datasets.utils import DEFAULT_FEATURES

    features = {
        **DEFAULT_FEATURES,
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"joint_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"joint_{i}" for i in range(state_dim)],
        },
        f"observation.images.{CAMERA_KEY}": {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }
    return features


def convert_session_to_episode(
    session_dir: Path,
    episode_index: int,
    task_str: str,
    joint_ts: np.ndarray,
    joint_pos: np.ndarray,
    img_df: pd.DataFrame,
    camera_dir: Path,
):
    """将一个 session 转为 (frames, task_str) 供 add_frame 使用。"""
    frames = []
    total_rows = len(img_df)
    # #region agent log
    _debug_log("convert_session_to_episode start", {"episode_index": episode_index, "total_rows": total_rows, "hypothesisId": "OOM6"})
    # #endregion
    for i, row in img_df.iterrows():
        ts = float(row["timestamp"])
        state = interpolate_state_at_timestamp(ts, joint_ts, joint_pos)
        # 下一帧的状态作为 action；若是最后一帧则 action = state
        if i + 1 < len(img_df):
            next_ts = float(img_df.iloc[i + 1]["timestamp"])
            action = interpolate_state_at_timestamp(next_ts, joint_ts, joint_pos)
        else:
            action = state.copy()

        img_path = camera_dir / row["filename"]
        if not img_path.exists():
            continue
        # #region agent log
        if len(frames) % 100 == 0 and len(frames) > 0:
            _debug_log("convert_session progress", {"episode_index": episode_index, "frames_so_far": len(frames), "hypothesisId": "OOM6"})
        # #endregion
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        frame = {
            "observation.state": state,
            "action": action,
            "observation.images.top": img,
            "timestamp": np.array([ts], dtype=np.float32),
            "next.reward": np.array([0.0], dtype=np.float32),
            "next.success": np.array([i == len(img_df) - 1], dtype=bool),
        }
        frames.append(frame)
    return frames, task_str


def main():
    # #region agent log
    _debug_log("main entry", {"hypothesisId": "OOM1"})
    # #endregion
    parser = argparse.ArgumentParser(description="YeahBot 原始数据 → LeRobot v3 数据集")
    parser.add_argument("--raw-dir", type=str, default="local_data_raw", help="原始数据目录（含 session_*）")
    parser.add_argument("--output-dir", type=str, default="./lerobot_yeahbot_c1", help="输出 LeRobot 数据集根目录")
    parser.add_argument("--repo-id", type=str, default=None, help="HuggingFace 数据集 repo，例如 USER/yeahbot-c1")
    parser.add_argument("--push-to-hub", action="store_true", help="转换完成后 push 到 Hub")
    parser.add_argument("--fps", type=int, default=10, help="数据集 FPS（用于对齐与存储）")
    parser.add_argument("--state-dim", type=int, default=STATE_DIM, help="状态/动作维度")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_dir}")
    # #region agent log
    _debug_log("before _ensure_lerobot", {"hypothesisId": "OOM1"})
    # #endregion
    LeRobotDataset, _ = _ensure_lerobot()
    # #region agent log
    _debug_log("after _ensure_lerobot", {"hypothesisId": "OOM1"})
    # #endregion
    features = build_lerobot_features(args.fps, args.state_dim)

    # 本地 repo_id：用 output_dir 名或指定 repo_id
    repo_id = args.repo_id or f"local/{output_dir.name}"
    print(f"创建 LeRobot 数据集: repo_id={repo_id}, root={output_dir}")

    # LeRobotDataset.create 要求 root 目录不存在（内部会 mkdir）
    # #region agent log
    _debug_log("before rmtree", {"output_dir": str(output_dir), "exists": output_dir.exists(), "hypothesisId": "OOM2"})
    # #endregion
    if output_dir.exists():
        shutil.rmtree(output_dir)
    # #region agent log
    _debug_log("after rmtree", {"hypothesisId": "OOM2"})
    # #endregion

    # #region agent log
    _debug_log("before LeRobotDataset.create", {"hypothesisId": "OOM3"})
    # #endregion
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        features=features,
        robot_type="yeahbot_c1",
        root=str(output_dir),
        use_videos=True,
    )
    # #region agent log
    _debug_log("after LeRobotDataset.create", {"hypothesisId": "OOM3"})
    # #endregion

    session_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("session_")])
    # #region agent log
    _debug_log("session_dirs listed", {"num_sessions": len(session_dirs), "hypothesisId": "OOM4"})
    # #endregion
    if not session_dirs:
        print(f"在 {raw_dir} 下未找到 session_* 目录")
        return

    default_task = "完成 YeahBot 示教任务"
    all_tasks = []

    for ep_idx, session_dir in enumerate(session_dirs):
        meta_path = session_dir / "metadata.json"
        task_str = default_task
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta.get("robot"), str):
                    task_str = meta["robot"] + " 示教"
            except Exception:
                pass

        # #region agent log
        _debug_log("episode start", {"ep_idx": ep_idx, "session": session_dir.name, "hypothesisId": "OOM5"})
        # #endregion
        img_df, joint_data = load_session_data(session_dir)
        if img_df is None or joint_data is None or len(img_df) < 2:
            print(f"跳过无效或过短 session: {session_dir.name}")
            continue
        # #region agent log
        _debug_log("after load_session_data", {"ep_idx": ep_idx, "n_imgs": len(img_df), "hypothesisId": "OOM5"})
        # #endregion

        joint_ts, joint_pos = joint_data
        camera_dir = session_dir / "camera"
        frames, _ = convert_session_to_episode(
            session_dir, ep_idx, task_str, joint_ts, joint_pos, img_df, camera_dir
        )
        # #region agent log
        _debug_log("after convert_session_to_episode", {"ep_idx": ep_idx, "n_frames": len(frames) if frames else 0, "hypothesisId": "OOM6"})
        # #endregion
        if not frames:
            continue

        all_tasks.append(task_str)
        for fi, frame in enumerate(frames):
            # 只传 LeRobot 期望的 feature 键；index/episode_index/frame_index/task_index/timestamp 由 add_frame 内部填充
            # 确保 observation.state / action 为 float32（插值结果为 float64）
            frame_for_add = {
                "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
                "action": np.asarray(frame["action"], dtype=np.float32),
                "observation.images.top": frame["observation.images.top"],
                "next.reward": frame["next.reward"],
                "next.success": frame["next.success"],
                "task": task_str,
            }
            dataset.add_frame(frame_for_add)

        # 归一化用 mean/std；std 过小会导致 (x-mean)/std 爆炸、loss/grdn 极大，故对 std 做下限（若已转换过需重跑本脚本以更新 meta）
        episode_stats = {}
        STD_FLOOR = 1e-2  # 最小 std，避免某维几乎不动时归一化爆炸
        for key in ["observation.state", "action"]:
            arr = np.stack([f[key] for f in frames])
            mean = arr.mean(axis=0).astype(np.float32)
            std = np.std(arr, axis=0).astype(np.float32)
            std = np.maximum(std, STD_FLOOR)
            episode_stats[key] = {"min": arr.min(axis=0), "max": arr.max(axis=0), "mean": mean, "std": std}
        # 仅传入当前 LeRobot 版本 save_episode 实际接受的参数（不同版本 API 不同）
        import inspect
        sig = inspect.signature(dataset.save_episode)
        kwargs = {}
        if "episode_index" in sig.parameters:
            kwargs["episode_index"] = ep_idx
        if "episode_length" in sig.parameters:
            kwargs["episode_length"] = len(frames)
        if "episode_tasks" in sig.parameters:
            kwargs["episode_tasks"] = [task_str]
        if "episode_stats" in sig.parameters:
            kwargs["episode_stats"] = episode_stats
        if "episode_metadata" in sig.parameters:
            kwargs["episode_metadata"] = {}
        dataset.save_episode(**kwargs)
        # #region agent log
        _debug_log("after save_episode", {"ep_idx": ep_idx, "n_frames": len(frames), "hypothesisId": "OOM7"})
        # #endregion
        print(f"  episode {ep_idx}: {session_dir.name} -> {len(frames)} frames")

    if dataset.meta.total_frames == 0:
        print("没有有效帧，退出")
        return

    # LeRobot 0.4 可能已移除 save_episode_tasks（任务已通过 save_episode 的 episode_tasks 写入）
    try:
        dataset._save_episode_data(list(dict.fromkeys(all_tasks)) if all_tasks else [default_task])
    except AttributeError:
        pass
    dataset.finalize()
    print(f"已写入: {output_dir}, episodes={dataset.meta.total_episodes}, frames={dataset.meta.total_frames}")

    if args.push_to_hub and args.repo_id:
        print("正在 push 到 Hugging Face Hub...")
        dataset.push_to_hub()
        print("完成")
    elif args.push_to_hub and not args.repo_id:
        print("未指定 --repo-id，跳过 push。指定 --repo-id USER/yeahbot-c1 可上传到 Hub。")


if __name__ == "__main__":
    main()

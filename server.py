#!/usr/bin/env python3
"""FastAPI demo server for the cylinder-parts counting prototype."""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web_demo"
RUNS_DIR = BASE_DIR / "demo_runs"
MPL_DIR = RUNS_DIR / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from count_cylinder_parts_demo import sample_and_count


ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="物资绕拍视频计数 Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/count")
async def count_video(video: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="请上传 mp4、mov、avi、mkv 或 webm 视频文件。")

    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    output_dir = run_dir / "output"
    run_dir.mkdir(parents=True, exist_ok=False)

    input_path = run_dir / f"input{suffix}"
    try:
        with input_path.open("wb") as file_obj:
            shutil.copyfileobj(video.file, file_obj)

        args = argparse.Namespace(
            video=str(input_path),
            output_dir=str(output_dir),
            frame_stride=12,
            max_frames=0,
            cluster_eps=0.0,
            include_background_piles=False,
            save_every_annotated_frame=False,
            max_center_std=43.0,
            min_ring_bamboo=0.17,
        )
        result = sample_and_count(args)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"视频计数失败：{exc}") from exc
    finally:
        await video.close()

    return build_response(run_id, result, output_dir)


def build_response(run_id: str, result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    per_frame = result.get("per_frame", [])
    annotated_dir = output_dir / "annotated"
    annotated_images = sorted(annotated_dir.glob("*.jpg"))

    return {
        "run_id": run_id,
        "estimated_count": result.get("estimated_count"),
        "best_view_sanity_count": result.get("best_view_sanity_count"),
        "global_projection_cluster_count_diagnostic": result.get("global_projection_cluster_count_diagnostic"),
        "total_raw_detections": result.get("total_raw_detections"),
        "sampled_frame_count": result.get("sampled_frame_count"),
        "pose_ok_count": result.get("pose_ok_count"),
        "video_info": result.get("video_info", {}),
        "config": result.get("config", {}),
        "per_frame": per_frame,
        "annotated_images": [to_run_url(path) for path in annotated_images[:12]],
        "projection_plot": optional_run_url(output_dir / "projected_clusters.png"),
        "result_json": optional_run_url(output_dir / "count_result.json"),
    }


def optional_run_url(path: Path) -> str | None:
    if not path.exists():
        return None
    return to_run_url(path)


def to_run_url(path: Path) -> str:
    relative_path = path.resolve().relative_to(RUNS_DIR.resolve())
    return "/runs/" + relative_path.as_posix()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8004, reload=False)

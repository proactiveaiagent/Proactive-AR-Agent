from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
from PIL import Image
from ..models.base import MultimodalModel
from ..schema.scene_schema import Face
from insightface.app import FaceAnalysis
import numpy as np
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0, det_size=(640, 640))

@dataclass
class FrameSamplingConfig:
    fps: float = 1.0
    max_frames: int = 16


def sample_video_frames(video_path: str, cfg: FrameSamplingConfig) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(video_fps / max(cfg.fps, 0.001))), 1)

    frames: List[Image.Image] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            if len(frames) >= cfg.max_frames:
                break
        idx += 1

    cap.release()
    return frames


def transcribe_audio_stub(audio_path: str, asr_model: MultimodalModel) -> Optional[str]:
    """Placeholder for ASR.

    Implement with Whisper or your ASR service, returning a text transcript.
    """

    transcription = asr_model.generate(audio_path)
    return transcription


def draw_on(img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bounding_box
            color = (0, 0, 255)
            text = 'Index: %d' % i
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.7
            thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x = box[0]  # 左对齐
            y = box[1] + text_height + 2  # 上边框 + 文字高度 + 2像素间距
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 4)
            cv2.putText(dimg, text, (x, y), font, font_scale, (0,255,0), thickness)
        return dimg


def face_detect(sampled_frames: List[Image.Image]) -> (List[Image.Image], List[Face], Image.Image):
    """Placeholder for face detection.

    Implement with your face detection model/service, returning a list of face info dicts.
    Each dict should contain at least 'bounding_box' and 'face_emb' keys.
    """
    index = len(sampled_frames) - 1
    img = np.array(sampled_frames[index])
    faces = []  # 初始化结果列表
    try:
        # 将base64解码为图片

        detected_faces = face_app.get(img)
        frame_faces = []

        for face in detected_faces:
            bbox = [int(x) for x in face.bbox.astype(int).tolist()]
            dscore = face.det_score
            embedding = [float(x) for x in face.normed_embedding.tolist()]
            embedding_np = np.array(face.embedding)
            qscore = np.linalg.norm(embedding_np, ord=2)

            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            aspect_ratio = height / width

            face_type = "ortho" if 1 < aspect_ratio < 1.5 else "side"

            face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # _, buffer = cv2.imencode('.jpg', face_img)
            face_info = {
                "bounding_box": bbox,
                "face_emb": embedding,
                "cluster_id": -1,
                "extra_data": {
                    "face_type": face_type,
                    "face_detection_score": str(dscore),
                    "face_quality_score": str(qscore)
                },
            }
            # Face(frame_id=frame_idx, bounding_box=bbox, face_emb=embedding, cluster_id=-1, extra_data={"face_type": face_type})
            
            frame_faces.append(face_info)
            
        faces = [Face(bounding_box=f['bounding_box'], face_emb=f['face_emb'], cluster_id=f['cluster_id'], extra_data=f['extra_data']) for f in frame_faces]
        faces.sort(key=lambda f: (f.bounding_box[0], f.bounding_box[1]))
        img=draw_on(img, faces)
        return sampled_frames, faces, Image.fromarray(img)

    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return []

    return faces
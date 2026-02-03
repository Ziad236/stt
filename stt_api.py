import os
import tempfile
import torch
import librosa
import soundfile as sf
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware
import datetime
import shutil
import warnings

warnings.filterwarnings("ignore")

# ===============================================================
# CUDA/cuDNN CONFIGURATION
# ===============================================================
def setup_cuda_environment():
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix:
        cudnn_path = os.path.join(conda_prefix, 'lib/python3.10/site-packages/nvidia/cudnn/lib')
        ctranslate_path = os.path.join(conda_prefix, 'lib/python3.10/site-packages/ctranslate2.libs')
        cuda_lib_path = os.path.join(conda_prefix, 'lib')
        os.environ['LD_LIBRARY_PATH'] = f"{cudnn_path}:{ctranslate_path}:{cuda_lib_path}:{os.getenv('LD_LIBRARY_PATH', '')}"
        return True
    return False


def initialize_device():
    try:
        setup_cuda_environment()
        if torch.cuda.is_available():
            torch.cuda.init()
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


# ===============================================================
# CONFIG
# ===============================================================
MODEL_NAME = "Systran/faster-whisper-large-v3"
CHUNK_SECONDS = 30
OVERLAP_SECONDS = 2
TARGET_SR = 16000
SAVE_DIR = "uploads"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE, COMPUTE_TYPE = initialize_device()

model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=os.path.join(os.path.dirname(__file__), "models")
)

# ===============================================================
# NOISE DETECTOR (INTEGRATED)
# ===============================================================
class AudioNoiseDetector:
    def __init__(self):
        self.thresholds = {
            'mean_rms_db': -35,
            'rms_std_low': 2,
            'rms_std_high': 12,
            'silence_ratio': 0.7,
            'spectral_centroid_low': 500,
            'spectral_centroid_high': 4000,
            'zcr_high': 0.15,
            'noise_floor_db': -30
        }

    def analyze(self, audio, sr):
        rms = librosa.feature.rms(y=audio)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]

        silence_ratio = np.sum(rms_db < np.percentile(rms_db, 20)) / len(rms_db)
        noise_floor = np.percentile(rms_db, 10)

        failures = {
            "Low RMS": np.mean(rms_db) < self.thresholds["mean_rms_db"],
            "Flat RMS": np.std(rms_db) < self.thresholds["rms_std_low"],
            "Chaotic RMS": np.std(rms_db) > self.thresholds["rms_std_high"],
            "High Silence": silence_ratio > self.thresholds["silence_ratio"],
            "High Spectral Centroid": np.mean(centroid) > self.thresholds["spectral_centroid_high"],
            "Low Spectral Centroid": np.mean(centroid) < self.thresholds["spectral_centroid_low"],
            "High ZCR": np.mean(zcr) > self.thresholds["zcr_high"],
            "High Noise Floor": noise_floor > self.thresholds["noise_floor_db"]
        }

        failed_metrics = [k for k, v in failures.items() if v]

        return {
            "is_noisy": len(failed_metrics) >= 3,
            "failed_metrics": failed_metrics
        }


noise_detector = AudioNoiseDetector()

# ===============================================================
# AUDIO CHUNKING
# ===============================================================
def chunk_audio(y, sr, chunk_length_s=30, overlap_s=2):
    chunk_size = int(chunk_length_s * sr)
    overlap_size = int(overlap_s * sr)
    step = chunk_size - overlap_size

    chunks = []
    for start in range(0, len(y), step):
        end = min(start + chunk_size, len(y))
        chunks.append(y[start:end])
        if end == len(y):
            break
    return chunks


# ===============================================================
# FASTAPI APP
# ===============================================================
app = FastAPI(title="STT with Noise Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================================================
# ENDPOINT
# ===============================================================
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load once → used for BOTH noise detection & STT
        audio, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)

        # ---------------- NOISE CHECK ----------------
        noise_result = noise_detector.analyze(audio, sr)
        if noise_result["is_noisy"]:
            os.remove(tmp_path)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Audio is too noisy. Please move to a quiet place and try again.",
                    "failed_metrics": noise_result["failed_metrics"]
                }
            )

        # Save copy for debugging
        filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        shutil.copy(tmp_path, os.path.join(SAVE_DIR, filename))

        chunks = chunk_audio(audio, sr, CHUNK_SECONDS, OVERLAP_SECONDS)

        all_text = []
        language = None
        language_prob = None

        for i, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as cf:
                sf.write(cf.name, chunk, sr)
                segments, info = model.transcribe(
                    cf.name,
                    beam_size=1,
                    temperature=0,
                    vad_filter=True,
                    condition_on_previous_text=True,
                )
                os.remove(cf.name)

            all_text.append(" ".join(seg.text.strip() for seg in segments))

            if i == 0:
                language = info.language
                language_prob = round(info.language_probability, 3)

        os.remove(tmp_path)

        return {
            "filename": file.filename,
            "language": language,
            "language_probability": language_prob,
            "duration_seconds": round(len(audio) / sr, 2),
            "chunks_processed": len(chunks),
            "text": " ".join(all_text).strip()
        }

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    #uvicorn fast_api_v2:app  --host 0.0.0.0 --port 8001

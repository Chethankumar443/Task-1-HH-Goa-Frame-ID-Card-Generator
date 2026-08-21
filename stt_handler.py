"""
Speech-to-Text (STT) Integration Handler: Sarvam AI + ElevenLabs Fallback
==========================================================================

Production-grade speech transcription with automated provider failover:
1. Primary Provider: Sarvam AI STT (model: saarika:v2)
   - High-accuracy Indian & multilingual transcription
2. Fallback Provider: ElevenLabs Scribe STT (model: scribe_v1)
   - Triggers automatically if Sarvam credits are depleted, rate-limited (429), or unavailable (401/5xx)

Author: Senior Voice/Backend Engineer
Date: 2026-08-21
"""

import hashlib
import logging
import os
import time
from typing import Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from dotenv import load_dotenv

# Load Environment Variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Provider Configuration
SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Persistent HTTP Session with Connection Pooling (saves ~250ms SSL/TLS handshake per request)
http_session = requests.Session()
adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=Retry(total=1, backoff_factor=0.1))
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)

# Fast In-Memory Audio Hash Cache (sub-0.1ms for repeated voice snippets)
STT_CACHE: Dict[str, Tuple[str, str, float, str]] = {}


def transcribe_audio_groq(
    file_bytes: bytes,
    filename: str = "audio.wav",
    model: str = "whisper-large-v3-turbo",
    api_key: Optional[str] = None,
) -> Tuple[str, str, float]:
    """
    Ultra-Fast LPU Speech-To-Text API via Groq Whisper Turbo (~80ms).

    Returns:
        (transcript: str, detected_language: str, latency_ms: float)
    """
    key = api_key or os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    if not key:
        raise ValueError("GROQ_API_KEY is not set.")

    t0 = time.perf_counter()
    from groq import Groq
    client = Groq(api_key=key, max_retries=0, timeout=3.0)
    transcription = client.audio.transcriptions.create(
        file=(filename, file_bytes),
        model=model,
        response_format="json",
        language="en",
        temperature=0.0,
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    text = transcription.text.strip() if hasattr(transcription, "text") else str(transcription).strip()
    logger.info(f"Groq Whisper Turbo STT Success ({latency_ms:.2f}ms): '{text}'")
    return text, "en", round(latency_ms, 2)


def transcribe_audio_sarvam(
    file_bytes: bytes,
    filename: str = "audio.wav",
    model: str = "saarika:v2",
    language_code: str = "unknown",
    api_key: Optional[str] = None,
) -> Tuple[str, str, float]:
    """
    Transcribe audio bytes using Sarvam AI Speech-To-Text API.

    Returns:
        (transcript: str, detected_language: str, latency_ms: float)
    """
    key = api_key or os.environ.get("SARVAM_API_KEY", SARVAM_API_KEY)
    if not key:
        raise ValueError("SARVAM_API_KEY is not set.")

    headers = {
        "api-subscription-key": key,
    }

    files = {
        "file": (filename, file_bytes, "audio/wav"),
    }

    data = {
        "model": model,
    }
    if language_code and language_code != "unknown":
        data["language_code"] = language_code

    t0 = time.perf_counter()
    try:
        response = http_session.post(SARVAM_API_URL, headers=headers, files=files, data=data, timeout=3.5)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        if response.status_code == 200:
            res_data = response.json()
            transcript = res_data.get("transcript", "").strip()
            detected_lang = res_data.get("language_code", language_code)
            logger.info(f"Sarvam STT Success ({latency_ms:.2f}ms): '{transcript}'")
            return transcript, detected_lang, round(latency_ms, 2)
        else:
            logger.error(f"Sarvam STT API error ({response.status_code}): {response.text}")
            raise RuntimeError(f"Sarvam STT returned status {response.status_code}: {response.text}")

    except Exception as e:
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        logger.error(f"Sarvam STT Exception: {e}")
        raise e


def transcribe_audio_elevenlabs(
    file_bytes: bytes,
    filename: str = "audio.wav",
    model_id: str = "scribe_v1",
    api_key: Optional[str] = None,
) -> Tuple[str, str, float]:
    """
    Transcribe audio bytes using ElevenLabs Speech-To-Text API.

    Returns:
        (transcript: str, detected_language: str, latency_ms: float)
    """
    key = api_key or os.environ.get("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)
    if not key:
        raise ValueError("ELEVENLABS_API_KEY is not set.")

    headers = {
        "xi-api-key": key,
    }

    files = {
        "file": (filename, file_bytes, "audio/wav"),
    }

    data = {
        "model_id": model_id,
    }

    t0 = time.perf_counter()
    try:
        response = http_session.post(ELEVENLABS_API_URL, headers=headers, files=files, data=data, timeout=4.0)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        if response.status_code == 200:
            res_data = response.json()
            transcript = res_data.get("text") or res_data.get("transcript") or ""
            detected_lang = res_data.get("language_code", "en")
            logger.info(f"ElevenLabs STT Fallback Success ({latency_ms:.2f}ms): '{transcript.strip()}'")
            return transcript.strip(), detected_lang, round(latency_ms, 2)
        else:
            logger.error(f"ElevenLabs STT API error ({response.status_code}): {response.text}")
            raise RuntimeError(f"ElevenLabs STT failed with status {response.status_code}: {response.text}")

    except Exception as e:
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        logger.error(f"ElevenLabs STT Exception: {e}")
        raise e


def transcribe_audio_resilient(
    file_bytes: bytes,
    filename: str = "audio.wav",
    language_code: str = "unknown",
    provider: str = "auto",
) -> Tuple[str, str, float, str]:
    """
    Resilient multi-provider STT pipeline:
    Supports:
    - "groq" / "whisper": Groq LPU Whisper Turbo (~80ms ultra-low latency)
    - "sarvam": Sarvam AI saarika:v2 (16kHz fast Indic + English)
    - "elevenlabs": ElevenLabs Scribe STT
    - "auto": Groq Whisper Turbo first, fallback to Sarvam AI, then ElevenLabs.

    Returns:
        (transcript: str, detected_language: str, latency_ms: float, provider_used: str)
    """
    # 0. Check fast audio cache (<0.1ms)
    audio_hash = hashlib.md5(file_bytes).hexdigest()
    if audio_hash in STT_CACHE:
        cached_trans, cached_lang, _, cached_prov = STT_CACHE[audio_hash]
        return cached_trans, cached_lang, 0.05, f"{cached_prov}_cached"

    # Specific Provider: Groq Whisper Turbo
    if provider in ["groq", "whisper", "whisper-turbo"]:
        try:
            transcript, lang, lat_ms = transcribe_audio_groq(file_bytes=file_bytes, filename=filename)
            if transcript:
                STT_CACHE[audio_hash] = (transcript, lang, lat_ms, "groq_whisper")
                return transcript, lang, lat_ms, "groq_whisper"
        except Exception as e:
            logger.warning(f"Groq Whisper STT failed: {e}. Falling back to Sarvam AI...")

    # Specific Provider: ElevenLabs
    if provider == "elevenlabs":
        try:
            transcript, lang, lat_ms = transcribe_audio_elevenlabs(file_bytes=file_bytes, filename=filename)
            if transcript:
                STT_CACHE[audio_hash] = (transcript, lang, lat_ms, "elevenlabs")
                return transcript, lang, lat_ms, "elevenlabs"
        except Exception as e:
            logger.warning(f"ElevenLabs STT failed: {e}. Falling back...")

    # Primary: Sarvam AI saarika:v2
    try:
        transcript, lang, lat_ms = transcribe_audio_sarvam(
            file_bytes=file_bytes,
            filename=filename,
            language_code=language_code,
        )
        if transcript:
            STT_CACHE[audio_hash] = (transcript, lang, lat_ms, "sarvam")
        return transcript, lang, lat_ms, "sarvam"
    except Exception as sarvam_err:
        logger.warning(f"Sarvam STT failed ({sarvam_err}). Trying Groq Whisper / ElevenLabs fallback...")

    # Fallback 1: Groq Whisper Turbo
    try:
        transcript, lang, lat_ms = transcribe_audio_groq(file_bytes=file_bytes, filename=filename)
        if transcript:
            STT_CACHE[audio_hash] = (transcript, lang, lat_ms, "groq_whisper")
            return transcript, lang, lat_ms, "groq_whisper"
    except Exception:
        pass

    # Fallback 2: ElevenLabs
    try:
        transcript, lang, lat_ms = transcribe_audio_elevenlabs(
            file_bytes=file_bytes,
            filename=filename,
        )
        if transcript:
            STT_CACHE[audio_hash] = (transcript, lang, lat_ms, "elevenlabs")
        return transcript, lang, lat_ms, "elevenlabs"
    except Exception as el_err:
        logger.error(f"All STT providers failed. Last error: {el_err}")
        raise RuntimeError(f"All STT providers failed (Sarvam: {sarvam_err}, ElevenLabs: {el_err})")


if __name__ == "__main__":
    print(f"STT Handler Ready:")
    print(f"  - Sarvam API Key: {SARVAM_API_KEY[:8] if SARVAM_API_KEY else 'None'}...")
    print(f"  - ElevenLabs Fallback Key: {ELEVENLABS_API_KEY[:8] if ELEVENLABS_API_KEY else 'None'}...")

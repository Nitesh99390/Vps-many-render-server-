"""
=============================================================================
 Edge-TTS Render Server  (app.py)  -  v3.0 Professional Edition
=============================================================================
 A production-ready Flask micro-service that converts text to speech using
 Microsoft Edge neural voices (edge-tts).

 Endpoints
 ---------
   GET  /                 -> plain "OK" (keep-alive pingers)
   GET  /health           -> JSON status, uptime, version, limits
   GET  /stats            -> JSON runtime statistics (requests, cache, latency)
   GET  /voices           -> voice catalogue   (?locale=hi-IN | ?lang=hi | ?gender=Male | ?q=madhur)
   GET  /voices/locales   -> list of available locales with counts
   POST /tts              -> JSON {text, voice, rate, pitch, volume} -> audio/mpeg
   GET  /tts              -> same via query string (quick testing)
   POST /tts/stream       -> chunked audio/mpeg streamed while synthesising
   POST /tts/subtitles    -> JSON {duration_ms, srt, vtt, words[], audio_base64?}
   GET  /tts/subtitles    -> same via query string

 Response headers on /tts
 ------------------------
   X-Duration-Ms   spoken duration in milliseconds (from word boundaries)
   X-Char-Count    number of characters synthesised
   X-Voice         voice that was used
   X-Cache         HIT | MISS
   X-Request-ID    unique request id (echoed if client sends one)

 Environment variables (all optional)
 ------------------------------------
   PORT               default 10000
   API_KEY            if set, clients must send header  X-API-Key: <key>
                      (or ?api_key=<key>)
   MAX_TEXT_LENGTH    default 6000 characters per request
   MAX_CONCURRENCY    default 6 simultaneous synth jobs
   DEFAULT_VOICE      default hi-IN-MadhurNeural
   TTS_RETRIES        default 3
   RATE_LIMIT         requests per minute per IP (default 120, 0 = disabled)
   CACHE_MAX_MB       in-memory audio cache size (default 64, 0 = disabled)
   LOG_LEVEL          default INFO
   ENABLE_CORS        default true

 Requirements
 ------------
   pip install flask edge-tts gunicorn
   Run (Render):  gunicorn app:app --workers 1 --threads 8 --timeout 300
=============================================================================
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
from concurrent.futures import TimeoutError as FutureTimeoutError
import logging
import os
import queue
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import edge_tts
from flask import Flask, Response, g, jsonify, request, stream_with_context

try:  # Correct client IPs behind Render / Cloudflare proxies
    from werkzeug.middleware.proxy_fix import ProxyFix
except Exception:  # pragma: no cover
    ProxyFix = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VERSION = "3.1.0"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


PORT = _env_int("PORT", 10000)
API_KEY = os.getenv("API_KEY", "").strip()
MAX_TEXT_LENGTH = max(1, _env_int("MAX_TEXT_LENGTH", 6000))
MAX_CONCURRENCY = max(1, _env_int("MAX_CONCURRENCY", 6))
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "hi-IN-MadhurNeural").strip()
TTS_RETRIES = max(1, _env_int("TTS_RETRIES", 3))
RATE_LIMIT = _env_int("RATE_LIMIT", 120)
CACHE_MAX_BYTES = _env_int("CACHE_MAX_MB", 64) * 1024 * 1024
ENABLE_CORS = _env_bool("ENABLE_CORS", True)
VOICE_CACHE_TTL = 6 * 3600  # seconds
SYNTH_TIMEOUT = max(5, _env_int("SYNTH_TIMEOUT", 180))  # whole request, including queue time
TRUST_PROXY_HOPS = max(0, _env_int("TRUST_PROXY_HOPS", 0))

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("tts-server")

RATE_RE = re.compile(r"^[+-]\d{1,3}%$")
PITCH_RE = re.compile(r"^[+-]\d{1,3}Hz$")
VOLUME_RE = re.compile(r"^[+-]\d{1,3}%$")
VOICE_RE = re.compile(r"^[a-z]{2,3}-[A-Za-z]{2,4}(-[A-Za-z]+)?-[A-Za-z0-9]+Neural$")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

START_TIME = time.time()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB request bodies
app.config["JSON_SORT_KEYS"] = False
app.json.sort_keys = False
if ProxyFix is not None and TRUST_PROXY_HOPS:
    # Enable only behind a trusted reverse proxy that overwrites forwarded headers.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=TRUST_PROXY_HOPS, x_proto=TRUST_PROXY_HOPS)


# ---------------------------------------------------------------------------
# Background asyncio loop (safe with gunicorn threads / Flask threaded mode)
# ---------------------------------------------------------------------------
class AsyncRunner:
    """Runs coroutines on a single dedicated event loop living in a thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="tts-loop", daemon=True)
        self._thread.start()
        # Create the semaphore *inside* the loop for compatibility with all Python versions
        self.semaphore: asyncio.Semaphore = asyncio.run_coroutine_threadsafe(
            self._make_semaphore(), self.loop
        ).result(timeout=10)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @staticmethod
    async def _make_semaphore() -> asyncio.Semaphore:
        return asyncio.Semaphore(MAX_CONCURRENCY)

    def run(self, coro, timeout: float = SYNTH_TIMEOUT):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeoutError:
            fut.cancel()  # Do not leave a timed-out job occupying a synthesis slot.
            raise

    def submit(self, coro):
        """Fire-and-forget schedule (returns concurrent.futures.Future)."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    @property
    def active(self) -> int:
        # Semaphore internal value = remaining permits
        remaining = getattr(self.semaphore, "_value", MAX_CONCURRENCY)
        return MAX_CONCURRENCY - remaining


class LazyRunner:
    """Do not start threads at import time (gunicorn --preload forks afterwards)."""

    def __init__(self):
        self._instance = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._instance is None:
                self._instance = AsyncRunner()
            return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)

    @property
    def active(self):
        return self._instance.active if self._instance is not None else 0


runner = LazyRunner()


# ---------------------------------------------------------------------------
# Runtime statistics
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.success = 0
        self.failed = 0
        self.rejected = 0
        self.cache_hits = 0
        self.chars = 0
        self.audio_bytes = 0
        self.total_latency = 0.0
        self.last_error: Optional[str] = None
        self.last_error_at: Optional[float] = None

    def record(self, ok: bool, chars: int = 0, nbytes: int = 0, latency: float = 0.0,
               error: Optional[str] = None, cached: bool = False) -> None:
        with self._lock:
            self.requests += 1
            if ok:
                self.success += 1
                self.chars += chars
                self.audio_bytes += nbytes
                self.total_latency += latency
                if cached:
                    self.cache_hits += 1
            else:
                self.failed += 1
                self.last_error = error
                self.last_error_at = time.time()

    def reject(self) -> None:
        with self._lock:
            self.rejected += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            avg = (self.total_latency / self.success) if self.success else 0.0
            return {
                "requests_total": self.requests,
                "success": self.success,
                "failed": self.failed,
                "rejected": self.rejected,
                "cache_hits": self.cache_hits,
                "characters_synthesised": self.chars,
                "audio_bytes_served": self.audio_bytes,
                "avg_latency_seconds": round(avg, 3),
                "last_error": self.last_error,
                "last_error_at": self.last_error_at,
            }


stats = Stats()


# ---------------------------------------------------------------------------
# In-memory LRU audio cache (bounded by bytes)
# ---------------------------------------------------------------------------
class AudioCache:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._data: "OrderedDict[str, Tuple[bytes, int]]" = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(text: str, voice: str, rate: str, pitch: str, volume: str) -> str:
        raw = f"{voice}|{rate}|{pitch}|{volume}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, k: str) -> Optional[Tuple[bytes, int]]:
        if self.max_bytes <= 0:
            return None
        with self._lock:
            item = self._data.get(k)
            if item is not None:
                self._data.move_to_end(k)
            return item

    def put(self, k: str, audio: bytes, duration_ms: int) -> None:
        if self.max_bytes <= 0 or len(audio) > self.max_bytes // 4:
            return
        with self._lock:
            if k in self._data:
                self._size -= len(self._data[k][0])
            self._data[k] = (audio, duration_ms)
            self._data.move_to_end(k)
            self._size += len(audio)
            while self._size > self.max_bytes and self._data:
                _, (old, _) = self._data.popitem(last=False)
                self._size -= len(old)

    def info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.max_bytes > 0,
                "entries": len(self._data),
                "size_bytes": self._size,
                "max_bytes": self.max_bytes,
            }


cache = AudioCache(CACHE_MAX_BYTES)


# ---------------------------------------------------------------------------
# Per-IP sliding-window rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, ip: str) -> Tuple[bool, int]:
        if self.per_minute <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 5000:
                for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= 60]:
                    self._hits.pop(key, None)
            dq = self._hits.setdefault(ip, deque())
            while dq and now - dq[0] > 60:
                dq.popleft()
            if len(dq) >= self.per_minute:
                retry = int(60 - (now - dq[0])) + 1
                return False, retry
            dq.append(now)
            # opportunistic cleanup of idle IPs
            if len(self._hits) > 5000:
                for k in [k for k, v in self._hits.items() if not v]:
                    self._hits.pop(k, None)
            return True, 0


limiter = RateLimiter(RATE_LIMIT)


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------
def _make_communicate(text: str, voice: str, rate: str, pitch: str, volume: str):
    """Build an edge_tts.Communicate compatible with old & new edge-tts versions.

    edge-tts >= 7 defaults to SentenceBoundary events; we request WordBoundary so
    that accurate durations and word-level subtitles are available.
    """
    try:
        return edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume, boundary="WordBoundary")
    except TypeError:
        pass
    try:
        return edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    except TypeError:
        # Very old edge-tts versions do not accept `pitch`
        return edge_tts.Communicate(text, voice, rate=rate, volume=volume)


async def _synthesize_once(text: str, voice: str, rate: str, pitch: str, volume: str,
                           collect_words: bool = False):
    communicate = _make_communicate(text, voice, rate, pitch, volume)
    audio = bytearray()
    duration_ms = 0
    words: List[Dict[str, Any]] = []
    async for chunk in communicate.stream():
        ctype = chunk.get("type")
        if ctype == "audio":
            audio.extend(chunk["data"])
        elif ctype in ("WordBoundary", "SentenceBoundary"):
            # offsets are in 100-nanosecond ticks
            start = chunk.get("offset", 0) / 10_000
            dur = chunk.get("duration", 0) / 10_000
            duration_ms = max(duration_ms, int(start + dur))
            if collect_words:
                words.append({"text": chunk.get("text", ""), "start_ms": int(start), "end_ms": int(start + dur)})
    if not audio:
        raise edge_tts.exceptions.NoAudioReceived("Empty audio stream")
    return bytes(audio), duration_ms, words


async def synthesize(text: str, voice: str, rate: str, pitch: str, volume: str,
                     collect_words: bool = False):
    """Synthesize with retry + concurrency limiting."""
    last_err: Optional[Exception] = None
    async with runner.semaphore:
        for attempt in range(1, TTS_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    _synthesize_once(text, voice, rate, pitch, volume, collect_words),
                    timeout=max(1, (SYNTH_TIMEOUT - 3) / TTS_RETRIES),
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("Synthesis attempt %d/%d failed (%s): %s",
                            attempt, TTS_RETRIES, type(exc).__name__, exc)
                if attempt < TTS_RETRIES:
                    await asyncio.sleep(1.0 * attempt)
    raise RuntimeError(f"TTS failed after {TTS_RETRIES} attempts: {type(last_err).__name__}: {last_err}")


# ---------------------------------------------------------------------------
# Voice catalogue (cached)
# ---------------------------------------------------------------------------
_voice_cache: Dict[str, Any] = {"data": None, "ts": 0.0, "names": set()}
_voice_lock = threading.Lock()


def get_voices() -> List[Dict[str, Any]]:
    with _voice_lock:
        if _voice_cache["data"] and time.time() - _voice_cache["ts"] < VOICE_CACHE_TTL:
            return _voice_cache["data"]
    try:
        raw = runner.run(edge_tts.list_voices(), timeout=60)
        voices = []
        for v in raw:
            tags = v.get("VoiceTag", {}) or {}
            voices.append({
                "name": v.get("ShortName"),
                "gender": v.get("Gender"),
                "locale": v.get("Locale"),
                "friendly_name": v.get("FriendlyName", ""),
                "personalities": tags.get("VoicePersonalities", []),
                "content_categories": tags.get("ContentCategories", []),
            })
        voices.sort(key=lambda v: (v["locale"] or "", v["name"] or ""))
        with _voice_lock:
            _voice_cache["data"] = voices
            _voice_cache["ts"] = time.time()
            _voice_cache["names"] = {v["name"] for v in voices if v["name"]}
        return voices
    except Exception as exc:  # noqa: BLE001
        log.error("Could not fetch voice list: %s", exc)
        return _voice_cache["data"] or []


def voice_exists(name: str) -> Optional[bool]:
    """True/False if catalogue is known, None if catalogue unavailable."""
    names = _voice_cache["names"]
    # Validation must not block a request on a remote catalogue refresh.
    if not names or time.time() - _voice_cache["ts"] >= VOICE_CACHE_TTL:
        return None
    return name in names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def require_api_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if API_KEY:
            supplied = (request.headers.get("X-API-Key")
                        or request.args.get("api_key", "")
                        or (request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                            if hasattr(str, "removeprefix") else "")).strip()
            if not secrets.compare_digest(supplied.encode("utf-8"), API_KEY.encode("utf-8")):
                stats.reject()
                return _error("Unauthorized: invalid or missing API key", 401)
        return fn(*args, **kwargs)
    return wrapper


def rate_limited(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        ok, retry = limiter.allow(ip)
        if not ok:
            stats.reject()
            resp = _error(f"Rate limit exceeded ({RATE_LIMIT}/min). Retry in {retry}s", 429)
            resp[0].headers["Retry-After"] = str(retry)
            return resp
        return fn(*args, **kwargs)
    return wrapper


def _error(message: str, status: int = 400):
    payload = {"error": message, "status": status, "request_id": getattr(g, "request_id", None)}
    return jsonify(payload), status


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _payload() -> Optional[dict]:
    if request.method == "POST":
        payload = request.get_json(silent=True)
        if payload is None and request.form:
            payload = request.form.to_dict()
        if payload is None and request.mimetype == "text/plain" and request.data:
            # Never treat malformed JSON as literal speech.
            try:
                payload = {"text": request.data.decode("utf-8")}
            except UnicodeDecodeError:
                payload = None
        if not isinstance(payload, dict):
            return None
        # allow query-string overrides for voice/rate/etc. on POST
        for k in ("voice", "rate", "pitch", "volume"):
            if k not in payload and k in request.args:
                payload[k] = request.args.get(k)
        return payload
    return request.args.to_dict()


def _validate(params: dict):
    """Return (clean_params, error_message)."""
    if not isinstance(params.get("text"), str):
        return None, "Field 'text' must be a string"
    text = normalize_text(params["text"])
    if not text:
        return None, "Field 'text' is required and cannot be empty"
    if len(text) > MAX_TEXT_LENGTH:
        return None, f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_LENGTH}"

    voice = str(params.get("voice") or DEFAULT_VOICE).strip()
    rate = str(params.get("rate") or "+0%").strip().replace(" ", "")
    pitch = str(params.get("pitch") or "+0Hz").strip().replace(" ", "")
    volume = str(params.get("volume") or "+0%").strip().replace(" ", "")

    # Be lenient: "20%" -> "+20%", "10Hz" -> "+10Hz"
    if rate and rate[0] not in "+-":
        rate = "+" + rate
    if pitch and pitch[0] not in "+-":
        pitch = "+" + pitch
    if volume and volume[0] not in "+-":
        volume = "+" + volume

    if not VOICE_RE.match(voice):
        return None, f"Invalid voice name: {voice!r} (example: hi-IN-MadhurNeural)"
    exists = voice_exists(voice)
    if exists is False:
        return None, f"Unknown voice: {voice!r}. Use GET /voices to list available voices"
    if not RATE_RE.fullmatch(rate) or not -99 <= int(rate[:-1]) <= 200:
        return None, f"Invalid rate: {rate!r} (example: +20% or -10%)"
    if not PITCH_RE.fullmatch(pitch) or not -100 <= int(pitch[:-2]) <= 100:
        return None, f"Invalid pitch: {pitch!r} (example: +10Hz or -5Hz)"
    if not VOLUME_RE.fullmatch(volume) or not -100 <= int(volume[:-1]) <= 100:
        return None, f"Invalid volume: {volume!r} (example: +30% or -10%)"

    return {"text": text, "voice": voice, "rate": rate, "pitch": pitch, "volume": volume}, None


def _fmt_srt_time(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms2 = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms2:03d}"


def _fmt_vtt_time(ms: int) -> str:
    return _fmt_srt_time(ms).replace(",", ".")


def build_subtitles(words: List[Dict[str, Any]], max_words: int = 8, max_ms: int = 4000):
    """Group word boundaries into readable cues; returns (srt, vtt)."""
    cues: List[Tuple[int, int, str]] = []
    buf: List[str] = []
    start = 0
    for w in words:
        if not buf:
            start = w["start_ms"]
        buf.append(w["text"])
        end = w["end_ms"]
        ends_sentence = w["text"].rstrip().endswith((".", "!", "?", "।", "؟", "。"))
        if len(buf) >= max_words or (end - start) >= max_ms or ends_sentence:
            cues.append((start, end, " ".join(buf)))
            buf = []
    if buf and words:
        cues.append((start, words[-1]["end_ms"], " ".join(buf)))

    srt_lines, vtt_lines = [], ["WEBVTT", ""]
    for i, (s, e, t) in enumerate(cues, 1):
        srt_lines += [str(i), f"{_fmt_srt_time(s)} --> {_fmt_srt_time(e)}", t, ""]
        vtt_lines += [f"{_fmt_vtt_time(s)} --> {_fmt_vtt_time(e)}", t, ""]
    return "\n".join(srt_lines).strip() + "\n", "\n".join(vtt_lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Request lifecycle hooks
# ---------------------------------------------------------------------------
@app.before_request
def _before():
    supplied_id = request.headers.get("X-Request-ID", "")
    g.request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied_id) else uuid.uuid4().hex[:16]
    _schedule_voice_warmup()
    g.started = time.time()
    if request.method == "OPTIONS" and ENABLE_CORS:
        resp = Response("", 204)
        return resp


@app.after_request
def _after(resp: Response):
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "")
    resp.headers["X-Server-Version"] = VERSION
    if ENABLE_CORS:
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, X-Request-ID, Authorization"
        resp.headers["Access-Control-Expose-Headers"] = (
            "X-Duration-Ms, X-Char-Count, X-Voice, X-Cache, X-Request-ID, Content-Length"
        )
    return resp


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "HEAD"])
def root():
    return "OK", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        status="ok",
        version=VERSION,
        uptime_seconds=int(time.time() - START_TIME),
        max_text_length=MAX_TEXT_LENGTH,
        max_concurrency=MAX_CONCURRENCY,
        active_jobs=runner.active,
        auth_required=bool(API_KEY),
        default_voice=DEFAULT_VOICE,
        rate_limit_per_minute=RATE_LIMIT,
        cache=cache.info(),
        edge_tts_version=getattr(edge_tts, "__version__", "unknown"),
    )


@app.route("/stats", methods=["GET"])
@require_api_key
def stats_route():
    data = stats.snapshot()
    data["uptime_seconds"] = int(time.time() - START_TIME)
    data["active_jobs"] = runner.active
    data["cache"] = cache.info()
    return jsonify(data)


@app.route("/voices", methods=["GET"])
@require_api_key
def voices():
    locale = request.args.get("locale", "").strip().lower()
    lang = request.args.get("lang", "").strip().lower()
    gender = request.args.get("gender", "").strip().lower()
    q = request.args.get("q", "").strip().lower()
    data = get_voices()
    if locale:
        data = [v for v in data if (v["locale"] or "").lower() == locale]
    elif lang:
        data = [v for v in data if (v["locale"] or "").lower().startswith(lang + "-")]
    if gender:
        data = [v for v in data if (v["gender"] or "").lower() == gender]
    if q:
        data = [v for v in data if q in (v["name"] or "").lower() or q in (v["friendly_name"] or "").lower()]
    return jsonify(count=len(data), voices=data)


@app.route("/voices/locales", methods=["GET"])
@require_api_key
def voice_locales():
    counts: Dict[str, int] = {}
    for v in get_voices():
        counts[v["locale"] or "unknown"] = counts.get(v["locale"] or "unknown", 0) + 1
    return jsonify(count=len(counts), locales=[{"locale": k, "voices": n} for k, n in sorted(counts.items())])


@app.route("/tts", methods=["POST", "GET"])
@require_api_key
@rate_limited
def tts():
    payload = _payload()
    if payload is None:
        return _error('Request body must be JSON: {"text": "..."}')

    params, err = _validate(payload)
    if err:
        return _error(err)

    started = time.time()
    ckey = AudioCache.key(**params)
    cached = cache.get(ckey)
    if cached:
        audio, duration_ms = cached
        hit = True
    else:
        hit = False
        try:
            audio, duration_ms, _ = runner.run(synthesize(**params), timeout=SYNTH_TIMEOUT)
        except FutureTimeoutError:
            stats.record(False, error="Synthesis timed out")
            return _error("Synthesis timed out; please retry", 504)
        except RuntimeError as exc:
            log.error("[%s] TTS error: %s", g.request_id, exc)
            stats.record(False, error=str(exc))
            return _error(str(exc), 502)
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] Unexpected TTS failure", g.request_id)
            stats.record(False, error=f"{type(exc).__name__}: {exc}")
            return _error(f"Internal error: {type(exc).__name__}: {exc}", 500)
        if not duration_ms:
            # rough fallback estimate: ~15 characters per second
            duration_ms = int(len(params["text"]) / 15 * 1000)
        cache.put(ckey, audio, duration_ms)

    latency = time.time() - started
    stats.record(True, len(params["text"]), len(audio), latency, cached=hit)
    log.info("[%s] TTS ok | voice=%s chars=%d bytes=%d dur=%.1fs took=%.2fs cache=%s",
             g.request_id, params["voice"], len(params["text"]), len(audio),
             duration_ms / 1000, latency, "HIT" if hit else "MISS")

    filename = request.args.get("filename") or payload.get("filename") or "speech.mp3"
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", str(filename))[:80] or "speech.mp3"
    if not filename.lower().endswith(".mp3"):
        filename += ".mp3"
    disposition = "attachment" if str(payload.get("download", "")).lower() in ("1", "true") else "inline"

    resp = Response(audio, mimetype="audio/mpeg")
    resp.headers["Content-Length"] = str(len(audio))
    resp.headers["Content-Disposition"] = f'{disposition}; filename="{filename}"'
    resp.headers["X-Duration-Ms"] = str(duration_ms)
    resp.headers["X-Char-Count"] = str(len(params["text"]))
    resp.headers["X-Voice"] = params["voice"]
    resp.headers["X-Cache"] = "HIT" if hit else "MISS"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/tts/stream", methods=["POST", "GET"])
@require_api_key
@rate_limited
def tts_stream():
    """Stream audio chunks as soon as they arrive (chunked transfer encoding)."""
    payload = _payload()
    if payload is None:
        return _error('Request body must be JSON: {"text": "..."}')
    params, err = _validate(payload)
    if err:
        return _error(err)

    q: "queue.Queue[Any]" = queue.Queue(maxsize=32)
    sentinel = object()
    req_id = g.request_id
    started = time.monotonic()
    deadline = started + SYNTH_TIMEOUT

    async def enqueue(item):
        # queue.put() must not block the shared asyncio loop.
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                await asyncio.sleep(0.02)

    async def produce_audio():
        async with runner.semaphore:
            communicate = _make_communicate(**params)
            received = False
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    received = True
                    await enqueue(chunk["data"])
            if not received:
                raise RuntimeError("The upstream service returned no audio")

    async def producer():
        try:
            await asyncio.wait_for(produce_audio(), timeout=SYNTH_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("[%s] Stream error: %s", req_id, exc)
            await enqueue(exc)
        else:
            await enqueue(sentinel)

    future = runner.submit(producer())

    def next_item():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise queue.Empty
        return q.get(timeout=remaining)

    # Get the first chunk BEFORE sending HTTP 200, so early failures are real errors.
    try:
        first = next_item()
    except queue.Empty:
        future.cancel()
        stats.record(False, error="Stream timed out")
        return _error("Stream timed out", 504)
    if isinstance(first, Exception) or first is sentinel:
        future.cancel()
        stats.record(False, error=str(first))
        return _error("Upstream synthesis failed before any audio was received", 502)

    def generate():
        sent = 0
        completed = False
        failure = "Client disconnected"
        try:
            item = first
            while item is not sentinel:
                if isinstance(item, Exception):
                    raise RuntimeError("Upstream audio stream failed") from item
                sent += len(item)
                yield item
                item = next_item()
            completed = True
        except queue.Empty as exc:
            failure = "Stream timed out"
            raise RuntimeError(failure) from exc
        except Exception as exc:
            failure = str(exc)
            raise  # After headers, abort the response instead of claiming full success.
        finally:
            future.cancel()
            stats.record(completed, len(params["text"]), sent, time.monotonic() - started,
                         error=None if completed else failure)
            log.info("[%s] Stream finished | bytes=%d complete=%s", req_id, sent, completed)

    resp = Response(stream_with_context(generate()), mimetype="audio/mpeg")
    resp.call_on_close(future.cancel)
    resp.headers["X-Char-Count"] = str(len(params["text"]))
    resp.headers["X-Voice"] = params["voice"]
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/tts/subtitles", methods=["POST", "GET"])
@require_api_key
@rate_limited
def tts_subtitles():
    """Return word timings + SRT/VTT subtitles (and optionally base64 audio)."""
    payload = _payload()
    if payload is None:
        return _error('Request body must be JSON: {"text": "..."}')
    params, err = _validate(payload)
    if err:
        return _error(err)
    include_audio = str(payload.get("include_audio", request.args.get("include_audio", ""))).lower() in ("1", "true")
    fmt = str(payload.get("format", request.args.get("format", "json"))).lower()
    if fmt not in {"json", "srt", "vtt"}:
        return _error("format must be json, srt or vtt")

    started = time.time()
    try:
        audio, duration_ms, words = runner.run(synthesize(**params, collect_words=True), timeout=SYNTH_TIMEOUT)
    except FutureTimeoutError:
        stats.record(False, error="Synthesis timed out")
        return _error("Synthesis timed out; please retry", 504)
    except RuntimeError as exc:
        stats.record(False, error=str(exc))
        return _error(str(exc), 502)
    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] Unexpected subtitle failure", g.request_id)
        stats.record(False, error=str(exc))
        return _error(f"Internal error: {type(exc).__name__}: {exc}", 500)

    srt, vtt = build_subtitles(words)
    stats.record(True, len(params["text"]), len(audio), time.time() - started)

    if fmt == "srt":
        return Response(srt, mimetype="text/plain; charset=utf-8",
                        headers={"X-Duration-Ms": str(duration_ms), "X-Voice": params["voice"]})
    if fmt == "vtt":
        return Response(vtt, mimetype="text/vtt; charset=utf-8",
                        headers={"X-Duration-Ms": str(duration_ms), "X-Voice": params["voice"]})

    body: Dict[str, Any] = {
        "voice": params["voice"],
        "char_count": len(params["text"]),
        "duration_ms": duration_ms,
        "word_count": len(words),
        "words": words,
        "srt": srt,
        "vtt": vtt,
    }
    if include_audio:
        body["audio_base64"] = base64.b64encode(audio).decode("ascii")
        body["audio_mime"] = "audio/mpeg"
    return jsonify(body)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    return _error("Not found", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return _error("Method not allowed", 405)


@app.errorhandler(413)
def too_large(_):
    return _error("Request body too large", 413)


@app.errorhandler(500)
def internal(_):
    return _error("Internal server error", 500)


# ---------------------------------------------------------------------------
# Startup: warm the voice catalogue in the background (non-blocking)
# ---------------------------------------------------------------------------
def _warm_voices() -> None:
    try:
        n = len(get_voices())
        log.info("Voice catalogue warmed: %d voices", n)
    except Exception as exc:  # noqa: BLE001
        log.warning("Voice warm-up failed: %s", exc)


_warmup_lock = threading.Lock()
_next_warmup = 0.0


def _schedule_voice_warmup():
    global _next_warmup
    with _warmup_lock:
        now = time.monotonic()
        if now < _next_warmup:
            return
        _next_warmup = now + (VOICE_CACHE_TTL if _voice_cache["data"] else 60)
    threading.Thread(target=_warm_voices, name="voice-warmup", daemon=True).start()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting Edge-TTS server v%s on port %d (auth=%s, rate_limit=%d/min, cache=%dMB)",
             VERSION, PORT, bool(API_KEY), RATE_LIMIT, CACHE_MAX_BYTES // (1024 * 1024))
    app.run(host="0.0.0.0", port=PORT, threaded=True)

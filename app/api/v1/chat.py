"""
app/api/v1/chat.py
───────────────────
WebSocket endpoint for real-time voice streaming.

Protocol
────────
  Client sends:
    • Binary frames  — raw PCM audio chunks (16-bit, 16 kHz, mono)
    • JSON frame     {"type": "end"}  — signals end of utterance

  Server sends:
    • {"type": "interim",  "text": "...", "is_final": false}
    • {"type": "final",    "text": "...", "language": "en"}
    • {"type": "reply",    "text": "...", "intent": "...", "session_id": "..."}
    • {"type": "error",    "message": "..."}

Pipeline per streaming session
───────────────────────────────
  1. Deepgram Nova-3 streaming ASR  (real-time interim results)
  2. On "end" signal: LID + NER + slot fill
  3. Claude agent turn → reply sent back
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent.dialogue_manager import get_dialogue_manager, get_or_create_session
from app.asr.language_detector import detect_language
from app.asr.router import get_asr_router
from app.core.logging import get_logger
from app.nlp.slot_filler import get_slot_filler

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


@router.websocket("/stream")
async def stream_chat(websocket: WebSocket):
    """
    Real-time streaming voice chatbot endpoint.
    One WebSocket connection = one conversation turn (send audio → get reply).
    Re-connect for next turn or reuse session_id across reconnects.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("ws_connected", session_id=session_id)

    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    accumulated_text: list[str] = []

    # ── Audio receiver coroutine ───────────────────────────────────────────────
    async def receive_audio() -> None:
        try:
            while True:
                msg = await websocket.receive()
                if "bytes" in msg:
                    await audio_queue.put(msg["bytes"])
                elif "text" in msg:
                    data = json.loads(msg["text"])
                    if data.get("type") == "end":
                        # Client signals end of utterance
                        await audio_queue.put(None)
                        break
                    elif data.get("type") == "session":
                        nonlocal session_id
                        session_id = data.get("session_id", session_id)
        except WebSocketDisconnect:
            await audio_queue.put(None)

    # ── Streaming ASR chunk generator ─────────────────────────────────────────
    async def audio_gen():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                return
            yield chunk

    # ── Main handler ───────────────────────────────────────────────────────────
    asyncio.create_task(receive_audio())

    try:
        asr = get_asr_router()
        async for chunk in asr.transcribe_stream(audio_gen()):
            if not chunk.is_final:
                await websocket.send_json({
                    "type": "interim",
                    "text": chunk.text,
                    "is_final": False,
                })
            else:
                accumulated_text.append(chunk.text)
                await websocket.send_json({
                    "type": "final",
                    "text": chunk.text,
                    "language": "en",   # will be updated after LID below
                })

        # ── Post-stream NLU + agent turn ──────────────────────────────────────
        full_text = " ".join(accumulated_text).strip()
        if not full_text:
            await websocket.send_json({"type": "error", "message": "No speech recognised."})
            return

        lang = detect_language(full_text)
        filler = get_slot_filler()
        parsed = filler.parse(full_text, language=lang)

        manager = get_dialogue_manager()
        session = get_or_create_session(session_id)
        reply = await manager.respond(session, parsed, full_text)

        await websocket.send_json({
            "type": "reply",
            "text": reply,
            "intent": parsed.intent,
            "missing_slots": parsed.missing_slots,
            "session_id": session_id,
        })

    except WebSocketDisconnect:
        logger.info("ws_disconnected", session_id=session_id)
    except Exception as exc:
        logger.exception("ws_error", error=str(exc))
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        logger.info("ws_closed", session_id=session_id)

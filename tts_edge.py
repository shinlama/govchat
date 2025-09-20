# app/services/tts_edge.py
import io
import edge_tts

async def synth_mp3_bytes(text: str, voice: str = "ko-KR-SunHiNeural") -> bytes:
    comm = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()
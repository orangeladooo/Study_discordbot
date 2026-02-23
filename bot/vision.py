import os
import logging
import base64      # To convert image bytes to base64 string for Groq's vision API
import tempfile    # For temporary video file storage

logger = logging.getLogger("vision")

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
SUPPORTED_PDFS   = {".pdf"}
SUPPORTED_VIDEOS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def get_file_type(filename: str) -> str:
    """Returns 'image', 'pdf', 'video', or 'unsupported' based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in SUPPORTED_IMAGES:
        return "image"
    elif ext in SUPPORTED_PDFS:
        return "pdf"
    elif ext in SUPPORTED_VIDEOS:
        return "video"
    return "unsupported"


async def extract_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts all text from a PDF using pymupdf (imported as fitz).
    Returns the raw text content, truncated to 8000 chars to stay within token limits.
    """
    try:
        import fitz
    except ImportError:
        return "❌ pymupdf not installed. Run: pip install pymupdf"

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            all_text.append(f"--- Page {page_num + 1} ---\n{text}")

    doc.close()

    if not all_text:
        return "⚠️ No extractable text found in PDF (might be a scanned image)."

    full_text = "\n\n".join(all_text)

    # Truncate at 8000 chars to avoid overwhelming the LLM context
    if len(full_text) > 8000:
        full_text = full_text[:8000] + "\n\n[...truncated...]"

    return full_text


async def describe_image_with_groq(image_bytes: bytes, user_query: str) -> str:
    """
    Sends an image directly to Groq's vision model (llama-3.2-11b-vision-preview).
    
    This is now fully working — no need to pull a local model!
    Groq handles it in the cloud at high speed.

    image_bytes → raw image bytes downloaded from Discord
    user_query  → what the user asked about the image
    """
    # Convert the raw image bytes to a base64 string
    # Groq's vision API requires images in base64 format
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Import here to avoid circular imports (agent imports memory, vision imports agent)
    import agent
    return agent.process_image_with_groq(image_b64, user_query)


async def extract_frames_from_video(file_bytes: bytes, filename: str, num_frames: int = 4) -> list:
    """
    Extracts evenly-spaced frames from a video as raw bytes.
    Returns a list of PNG image bytes, or empty list if it fails.
    """
    try:
        import cv2
    except ImportError:
        logger.error("cv2 not found — run: pip install opencv-python")
        return []

    suffix = os.path.splitext(filename)[1]
    tmp_path = None

    try:
        # Write to a temp file — OpenCV needs a file path, not raw bytes
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        logger.info(f"Temp video written to: {tmp_path} ({len(file_bytes)} bytes)")

        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            logger.error(f"OpenCV could not open the video file: {tmp_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video info: {total_frames} frames, {fps:.1f} fps, {width}x{height}")

        # If total_frames is 0 → likely audio-only (voice message, podcast clip etc.)
        if total_frames <= 0:
            logger.warning("No video frames found — file might be audio-only")
            cap.release()
            return []

        # Pick evenly-spaced frame positions
        positions = [
            int(total_frames * (i + 1) / (num_frames + 1))
            for i in range(num_frames)
        ]

        frames_bytes = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode(".png", frame)
                frames_bytes.append(buffer.tobytes())
                logger.info(f"  Extracted frame at position {pos}")
            else:
                logger.warning(f"  Could not read frame at position {pos}")

        cap.release()
        logger.info(f"Total frames extracted: {len(frames_bytes)}")
        return frames_bytes

    except Exception as e:
        logger.error(f"Frame extraction error: {e}")
        return []
    finally:
        # Always clean up the temp file, even if something crashed
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {tmp_path}: {e}")


async def process_attachment(file_bytes: bytes, filename: str, user_question: str) -> str:
    """
    Main entry point called by bot.py for $see command.
    Routes to the right processor based on file type.
    """
    file_type = get_file_type(filename)
    logger.info(f"Processing {file_type}: {filename} ({len(file_bytes)} bytes)")

    if file_type == "unsupported":
        supported = ", ".join(list(SUPPORTED_IMAGES) + list(SUPPORTED_PDFS) + list(SUPPORTED_VIDEOS))
        return f"❌ Unsupported file type. Supported: {supported}"

    elif file_type == "pdf":
        return await extract_from_pdf(file_bytes)

    elif file_type == "image":
        return await describe_image_with_groq(file_bytes, user_question)

    elif file_type == "video":
        frames = await extract_frames_from_video(file_bytes, filename)

        # No frames found — most likely an audio-only file (voice message, etc.)
        if not frames:
            return (
                "⚠️ **Couldn't extract video frames.**\n\n"
                "Possible reasons:\n"
                "• The file is a **voice/audio recording** (MP4 audio-only) — I can't hear audio, only see visuals.\n"
                "• The video codec is unsupported by OpenCV.\n\n"
                "If it's a voice message, please **type out what you sang/said** and I'll give feedback on that!"
            )

        # Describe each frame — catch per-frame errors so one bad frame doesn't kill the whole response
        descriptions = []
        for i, frame_bytes in enumerate(frames, 1):
            try:
                desc = await describe_image_with_groq(frame_bytes, f"Describe frame {i} of a video briefly. What do you see?")
                descriptions.append(f"**Frame {i}:** {desc}")
                logger.info(f"Frame {i} described successfully")
            except Exception as e:
                logger.error(f"Frame {i} description failed: {e}")
                descriptions.append(f"**Frame {i}:** ❌ Could not describe this frame (`{e}`).")

        if not descriptions:
            return "❌ Frame extraction worked but Groq vision failed on all frames. Check logs for details."

        return "\n\n".join(descriptions)

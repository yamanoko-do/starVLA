"""Quick chat with Qwen3.5-0.8B to test inference speed.
Usage:
  plain text       -> text chat
  img:path.jpg hi  -> load image and chat about it
  vid:path.mp4 hi  -> load video and chat about it
"""
import re, sys, time
from PIL import Image
import cv2
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from transformers.video_utils import VideoMetadata

# --- configurable ---
MAX_VIDEO_FRAMES = 32       # max frames to sample
INVERT_TIMESTAMPS = True    # count down from end: <3.2s> ... <0.2s> instead of up
FIBONACCI_SAMPLING = True   # sample frames by Fibonacci offsets from the end (dense near end)
# -------------------

def _fib_offsets(n):
    """Generate n Fibonacci offsets: 1, 2, 3, 5, 8, 13, ..."""
    if n <= 0:
        return []
    a, b = 1, 2
    seq = [a]
    for _ in range(n - 1):
        seq.append(b)
        a, b = b, a + b
    return seq

MODEL_PATH = "/mnt/workspace/yama/oss_yama/cache/hf_cache/hub/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"

# Set to a trained Zone checkpoint to load VLM from it instead.
# e.g. ZONE_CKPT = "playground/Checkpoints/foldclothes_zone/checkpoints/steps_80000_pytorch_model.pt"
ZONE_CKPT = "playground/Checkpoints/foldclothes_zone_unfreezeVLM/checkpoints/steps_80000_pytorch_model.pt"


print("Loading model...")
t0 = time.time()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH, attn_implementation="sdpa", torch_dtype=torch.bfloat16
).to("cuda")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

if ZONE_CKPT is not None:
    print(f"Loading VLM weights from Zone checkpoint: {ZONE_CKPT}")
    ckpt = torch.load(ZONE_CKPT, map_location="cpu")
    vlm_state = {}
    # ckpt:  qwen_vl_interface.model.model.xxx  (QwenZone wraps Qwen3_5ForConditionalGeneration)
    # model: model.xxx                          (raw Qwen3_5ForConditionalGeneration)
    prefix = "qwen_vl_interface.model."
    for k, v in ckpt.items():
        if k.startswith(prefix):
            vlm_state[k[len(prefix):]] = v
    # strict=True would raise on mismatch; we use strict=False but check manually
    model_state = model.state_dict()
    matched = [k for k in vlm_state if k in model_state]
    missing = [k for k in model_state if k not in vlm_state]
    print(f"  Checkpoint VLM keys: {len(vlm_state)}")
    print(f"  Matched & loaded:    {len(matched)}")
    print(f"  Missing from ckpt:   {len(missing)} (model has but ckpt lacks)")
    model.load_state_dict(vlm_state, strict=False)
    if len(missing) > 0 and len(missing) < 10:
        for k in missing:
            print(f"    missing: {k}")
# Monkey-patch _calculate_timestamps to use actual video frame indices
# (needed for non-uniform sampling like Fibonacci). The processor calls
# this with indices=[0..N-1] and effective_fps, but we override to compute
# exact timestamps from the real frame positions.
_orig_calc_ts = processor._calculate_timestamps
def _exact_ts(indices, video_fps, merge_size=2):
    # Use stored actual frame indices to get real timestamps
    real_indices = getattr(processor, '_sample_indices', None)
    real_fps = getattr(processor, '_real_fps', None)
    if real_indices is not None and real_fps is not None:
        # frame_ts[k] = real video time of frame indices[k]
        frame_ts = [real_indices[idx] / real_fps for idx in indices]
    else:
        frame_ts = [idx / video_fps for idx in indices]
    if INVERT_TIMESTAMPS:
        max_frame_time = frame_ts[-1]
        frame_ts = [max_frame_time - t for t in frame_ts]
    return [(frame_ts[i] + frame_ts[i + merge_size - 1]) / 2
            for i in range(0, len(frame_ts), merge_size)]
processor._calculate_timestamps = _exact_ts
if INVERT_TIMESTAMPS:
    print("  [WARN] Video timestamps inverted (countdown from end)")
print(f"Loaded in {time.time() - t0:.1f}s")

# Warmup
print("Warmup...")
t0 = time.time()
dummy = torch.randint(0, 1000, (1, 16)).to("cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    model(dummy)
print(f"Warmup done in {time.time() - t0:.1f}s")

def load_video_frames(path, max_frames=MAX_VIDEO_FRAMES):
    """Read a video file and return (frames, fps, total_frames, width, height, duration, indices)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0
    if total == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {path}")
    if FIBONACCI_SAMPLING:
        # Sample by Fibonacci offsets from the last frame.
        # fib=1 → last frame, fib=2 → 2nd-to-last, etc.
        # Closer to the end = denser sampling.
        offsets = [f for f in _fib_offsets(max_frames) if total - f >= 0]
        indices = sorted([total - f for f in offsets])  # chronological order
    else:
        indices = [int(i * total / max_frames) for i in range(min(max_frames, total))]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, fps, total, width, height, duration, indices


print("\nChat ready! Type 'quit' to exit.")
print("  img:<path> <question>  -- ask about an image")
print("  vid:<path> <question>  -- ask about a video\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break
    if user_input.lower() == "quit":
        break

    content = []
    video_metadata = None
    if user_input.startswith("img:"):
        # Format: img:/path/to/image.jpg describe this image
        parts = user_input[4:].strip().split(None, 1)
        img_path = parts[0]
        question = parts[1] if len(parts) > 1 else "描述这张图片"
        try:
            image = Image.open(img_path).convert("RGB")
            content.append({"type": "image", "image": image})
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            continue
        content.append({"type": "text", "text": question})
    elif user_input.startswith("vid:"):
        # Format: vid:/path/to/video.mp4 -- describe this video
        #          vid:/path/to/video.mp4 describe this video
        rest = user_input[4:].strip()
        if " -- " in rest:
            vid_path, question = rest.split(" -- ", 1)
        elif " --" in rest:
            vid_path, question = rest.split(" --", 1)
        elif "--" in rest:
            vid_path, question = rest.split("--", 1)
        else:
            parts = rest.split(None, 1)
            vid_path = parts[0]
            question = parts[1] if len(parts) > 1 else "描述这段视频"
        if "--" in user_input:
            vid_path = vid_path.strip()
            question = question.strip()
        try:
            frames, real_fps, total, width, height, duration, sample_indices = load_video_frames(
                vid_path, max_frames=MAX_VIDEO_FRAMES
            )
            # Store for monkey-patched _calculate_timestamps (exact timestamps
            # even with non-uniform sampling like Fibonacci).
            processor._sample_indices = sample_indices
            processor._real_fps = real_fps
            # effective_fps only used for VideoMetadata (to prevent processor
            # from resampling our pre-sampled frames). Timestamps come from
            # our monkey-patched _calculate_timestamps, not from this.
            effective_fps = len(frames) / duration
            video_metadata = [VideoMetadata(
                total_num_frames=len(frames),
                fps=effective_fps,
                frames_indices=list(range(len(frames))),
            )]
            print(f"  [video] {vid_path}")
            print(f"    fps={real_fps:.1f}  frames={total}  duration={duration:.1f}s  "
                  f"resolution={width}x{height}")
            spacing = "fib" if FIBONACCI_SAMPLING else "uniform"
            print(f"    sampled {len(frames)}/{total} frames ({spacing})")
            content.append({"type": "video", "video": frames, "fps": effective_fps})
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            continue
        content.append({"type": "text", "text": question})
    else:
        content.append({"type": "text", "text": user_input})

    messages = [{"role": "user", "content": content}]
    extra_kwargs = {}
    if video_metadata is not None:
        extra_kwargs["video_metadata"] = video_metadata
    inputs = processor.apply_chat_template(
        messages, tokenize=True, padding=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
        do_sample_frames=False,      # don't let processor resample our pre-sampled frames
        **extra_kwargs,
    ).to("cuda")

    # Fix: split video_grid_thw so each temporal chunk has its own row.
    # The processor creates one [T,H,W] per video but splits tokens into T
    # segments with timestamps between them. The model expects one row per segment.
    if "video_grid_thw" in inputs:
        vg = inputs["video_grid_thw"]
        new_rows = []
        for row in vg:
            T = int(row[0].item())
            for _ in range(T):
                new_rows.append([1, int(row[1].item()), int(row[2].item())])
        inputs["video_grid_thw"] = torch.tensor(
            new_rows, dtype=vg.dtype, device=vg.device
        )

        # Print chunk layout from decoded tokens
        decoded = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        timestamps = re.findall(r"<([\d.]+) seconds>", decoded)
        T_raw = sum(int(r[0].item()) for r in vg)
        tokens_per_chunk = int(vg[0, 2].item()) * int(vg[0, 1].item()) // 4
        print(f"    temporal chunks: {T_raw}")
        for ci in range(min(T_raw, len(timestamps))):
            print(f"      chunk {ci}: <{timestamps[ci]}s>  {tokens_per_chunk} tokens")

    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(**inputs, max_new_tokens=256)
    elapsed = time.time() - t0

    reply = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Bot: {reply}")
    print(f"[{elapsed:.2f}s, {len(outputs[0]) - inputs['input_ids'].shape[1]} tokens]")

import numpy as np

# JARVIS-VLA 51-Token Action Scheme
# 22 keyboard tokens (0-21) + 29 camera tokens (pitch 0-13, yaw 14-27, center 28)

KEYBOARD_TOKENS = {
    "ESC": 0, "back": 1, "drop": 2, "forward": 3,
    "hotbar.1": 4, "hotbar.2": 5, "hotbar.3": 6, "hotbar.4": 7,
    "hotbar.5": 8, "hotbar.6": 9, "hotbar.7": 10, "hotbar.8": 11,
    "hotbar.9": 12, "inventory": 13, "jump": 14, "left": 15,
    "right": 16, "sneak": 17, "sprint": 18, "swapHands": 19,
    "attack": 20, "use": 21,
}

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.w": "forward", "key.keyboard.s": "back",
    "key.keyboard.a": "left", "key.keyboard.d": "right",
    "key.keyboard.space": "jump", "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint", "key.keyboard.q": "drop",
    "key.keyboard.e": "inventory", "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2", "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4", "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6", "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8", "key.keyboard.9": "hotbar.9",
    "key.keyboard.escape": "ESC", "key.keyboard.f": "swapHands",
}

CAMERA_SCALER = 360.0 / 2400.0

NOOP_ACTION = {
    "ESC": 0, "back": 0, "drop": 0, "forward": 0,
    "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0,
    "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0,
    "hotbar.9": 0, "inventory": 0, "jump": 0, "left": 0,
    "right": 0, "sneak": 0, "sprint": 0, "swapHands": 0,
    "attack": 0, "use": 0, "pickItem": 0,
    "camera": np.array([0.0, 0.0], dtype=np.float32),
}


class CameraQuantizer:
    """μ-law quantization for mouse camera (pitch/yaw).
    From OpenAI VPT lib/actions.py.
    """
    def __init__(self, binsize=2, maxval=10, mu=10):
        self.binsize = binsize
        self.maxval = maxval
        self.mu = mu
        self.n_bins = int((maxval * 2) / binsize) + 1  # 21 bins

    def discretize(self, xy: np.ndarray) -> np.ndarray:
        xy = np.clip(xy, -self.maxval, self.maxval)
        xy = xy / self.maxval
        v_encode = np.sign(xy) * (
            np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu)
        )
        v_encode *= self.maxval
        return np.round(
            (v_encode + self.maxval) / self.binsize
        ).astype(np.int64)

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        return self.discretize(xy)


def json_action_to_env_action(step_data: dict) -> tuple[dict, bool]:
    """Convert VPT JSONL step data to MineRL action dict.
    From OpenAI run_inverse_dynamics_model.py.
    """
    env_action = NOOP_ACTION.copy()
    is_null = True

    for key in step_data.get("keyboard", {}).get("keys", []):
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null = False

    mouse = step_data.get("mouse", {})
    dx = mouse.get("dx", 0.0)
    dy = mouse.get("dy", 0.0)
    camera_action = np.array([dy * CAMERA_SCALER, dx * CAMERA_SCALER], dtype=np.float32)
    env_action["camera"] = camera_action

    buttons = mouse.get("buttons", [])
    if len(buttons) > 0 and buttons[0]:
        env_action["attack"] = 1
        is_null = False
    if len(buttons) > 1 and buttons[1]:
        env_action["use"] = 1
        is_null = False
    if len(buttons) > 2 and buttons[2]:
        env_action["pickItem"] = 1
        is_null = False

    return env_action, is_null


def env_action_to_token(env_action: dict, quantizer: CameraQuantizer) -> int:
    """Convert a factored MineRL action dict to a single 51-token ID."""
    # Priority: keyboard > camera > center
    for key, token_id in KEYBOARD_TOKENS.items():
        if env_action.get(key, 0) == 1:
            return token_id

    camera = env_action.get("camera", np.array([0.0, 0.0]))
    if np.abs(camera).max() > 0.5:
        bins = quantizer(camera)  # [pitch_bin, yaw_bin]
        # pitch_bin 0-20 → token 0-13
        if bins[0] != 10:
            return max(0, min(13, int(bins[0]) - 4))
        # yaw_bin 0-20 → token 14-27
        if bins[1] != 10:
            return max(14, min(27, int(bins[1]) + 14))

    return 28  # center / no-op


# ──────────────────────────────────────────
# Lumine action parser (TESS VLA dataset)
# ──────────────────────────────────────────

# TESS action format:
#   <|action_start|> mouse_x mouse_y mouse_z ; keys_50ms ; keys_50ms ; keys_50ms ; keys_50ms <|action_end|>
# Each keys_50ms is a space-separated list of key names (LMB, RMB, forward, back, etc.)

LUMINE_KEY_MAP = {
    "LMB": "attack",
    "RMB": "use",
    "forward": "forward",
    "back": "back",
    "left": "left",
    "right": "right",
    "jump": "jump",
    "sneak": "sneak",
    "sprint": "sprint",
    "drop": "drop",
    "inventory": "inventory",
    "swapHands": "swapHands",
    "ESC": "ESC",
    "1": "hotbar.1", "2": "hotbar.2", "3": "hotbar.3",
    "4": "hotbar.4", "5": "hotbar.5", "6": "hotbar.6",
    "7": "hotbar.7", "8": "hotbar.8", "9": "hotbar.9",
}

CAMERA_MOUSE_SCALE = 1.0  # TESS likely uses different scaling; tune empirically


def parse_lumine_action(action_str: str) -> int:
    """Convert a Lumine-format action string to a JARVIS-VLA 51-token ID.

    Format: <|action_start|> mx my mz ; k1 k2 ; k3 k4 ; k5 k6 ; k7 k8 <|action_end|>

    Priority: keyboard > camera > no-op.
    Returns a single token ID (0-28).
    """
    cleaned = (
        action_str.replace("<|action_start|>", "")
        .replace("<|action_end|>", "")
        .strip()
    )
    parts = [p.strip() for p in cleaned.split(";")]
    if len(parts) < 5:
        return 28  # no-op / center

    # Parse mouse
    mouse_parts = parts[0].strip().split()
    mouse_x = float(mouse_parts[0]) if len(mouse_parts) > 0 else 0.0
    mouse_y = float(mouse_parts[1]) if len(mouse_parts) > 1 else 0.0

    # Parse key chunks (4 × 50ms windows → combine to single frame action)
    keys_pressed = set()
    for chunk in parts[1:5]:
        for key_name in chunk.strip().split():
            key_name = key_name.strip()
            if key_name in LUMINE_KEY_MAP:
                keys_pressed.add(LUMINE_KEY_MAP[key_name])

    # Priority 1: Keyboard tokens
    for key, token_id in KEYBOARD_TOKENS.items():
        if key in keys_pressed:
            return token_id

    # Priority 2: Camera tokens
    if abs(mouse_x) > 0.5 or abs(mouse_y) > 0.5:
        quantizer = CameraQuantizer()
        camera_xy = np.array([mouse_y * CAMERA_MOUSE_SCALE, mouse_x * CAMERA_MOUSE_SCALE], dtype=np.float32)
        bins = quantizer(camera_xy)
        if bins[0] != 10:
            return max(0, min(13, int(bins[0]) - 4))
        if bins[1] != 10:
            return max(14, min(27, int(bins[1]) + 14))

    return 28  # center / no-op

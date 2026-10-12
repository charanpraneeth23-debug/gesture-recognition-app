# Gesture to Text/Command Mapping Dictionary
# This file contains all gesture-to-text translations for everyday communication

GESTURE_VOCABULARY = {
    # Hand Gestures
    "open_palm": {
        "text": "Hello",
        "command": "greet",
        "confidence_threshold": 0.7,
        "description": "Open hand with palm facing forward"
    },
    "wave": {
        "text": "Hi / Wave",
        "command": "wave",
        "confidence_threshold": 0.75,
        "description": "Hand moving side to side"
    },
    "thumbs_up": {
        "text": "Yes / Good / Agree",
        "command": "yes",
        "confidence_threshold": 0.8,
        "description": "Thumb pointing upward"
    },
    "thumbs_down": {
        "text": "No / Bad / Disagree",
        "command": "no",
        "confidence_threshold": 0.8,
        "description": "Thumb pointing downward"
    },
    "fist": {
        "text": "Stop / Hold",
        "command": "stop",
        "confidence_threshold": 0.75,
        "description": "Closed fist"
    },
    "peace_sign": {
        "text": "Peace / Victory / Two",
        "command": "peace",
        "confidence_threshold": 0.75,
        "description": "Index and middle finger up, others down"
    },
    "ok_sign": {
        "text": "OK / Good / Understood",
        "command": "ok",
        "confidence_threshold": 0.75,
        "description": "Thumb and index finger forming circle"
    },
    "point_left": {
        "text": "Left / That way",
        "command": "left",
        "confidence_threshold": 0.7,
        "description": "Index finger pointing left"
    },
    "point_right": {
        "text": "Right / Over there",
        "command": "right",
        "confidence_threshold": 0.7,
        "description": "Index finger pointing right"
    },
    "point_up": {
        "text": "Up / Above",
        "command": "up",
        "confidence_threshold": 0.7,
        "description": "Index finger pointing upward"
    },
    "point_down": {
        "text": "Down / Below",
        "command": "down",
        "confidence_threshold": 0.7,
        "description": "Index finger pointing downward"
    },
    "hands_together": {
        "text": "Please / Thank you / Prayer",
        "command": "please",
        "confidence_threshold": 0.75,
        "description": "Both palms pressed together"
    },
    "call_me": {
        "text": "Come here / Call me",
        "command": "come",
        "confidence_threshold": 0.75,
        "description": "Fingers curling inward"
    },
    "shrug": {
        "text": "I don't know / Maybe",
        "command": "maybe",
        "confidence_threshold": 0.7,
        "description": "Both shoulders up, hands open"
    },
    "hand_raise": {
        "text": "Attention / Question / Help",
        "command": "attention",
        "confidence_threshold": 0.7,
        "description": "Hand raised above head"
    },
    "crossed_arms": {
        "text": "Busy / Not now / Annoyed",
        "command": "busy",
        "confidence_threshold": 0.7,
        "description": "Both arms crossed over chest"
    },
    "both_hands_up": {
        "text": "Help / Emergency / Surrender",
        "command": "help",
        "confidence_threshold": 0.75,
        "description": "Both hands raised high"
    },
    "clapping": {
        "text": "Great / Applause / Well done",
        "command": "applause",
        "confidence_threshold": 0.75,
        "description": "Hands clapping together"
    },
    "thumbs_sideways": {
        "text": "So-so / Okay / Medium",
        "command": "so_so",
        "confidence_threshold": 0.75,
        "description": "Thumb sideways (like rocking)"
    },
    "hand_stop": {
        "text": "Stop / Wait / Halt",
        "command": "wait",
        "confidence_threshold": 0.75,
        "description": "Palm facing forward, arm extended"
    },
    "victory": {
        "text": "Victory / Success / Win",
        "command": "victory",
        "confidence_threshold": 0.75,
        "description": "Both hands with peace sign"
    },
    "finger_to_lips": {
        "text": "Shh / Quiet / Keep quiet",
        "command": "quiet",
        "confidence_threshold": 0.7,
        "description": "Index finger to lips"
    },
    "beckoning": {
        "text": "Come closer / Over here",
        "command": "come_here",
        "confidence_threshold": 0.75,
        "description": "Palm up, fingers curling inward"
    },
    
    # Body/Pose Gestures
    "arms_open_wide": {
        "text": "Welcome / Embrace / Open",
        "command": "welcome",
        "confidence_threshold": 0.7,
        "description": "Both arms wide open"
    },
    "nodding": {
        "text": "Yes / Agree / Understood",
        "command": "nod_yes",
        "confidence_threshold": 0.7,
        "description": "Head moving up and down"
    },
    "head_shake": {
        "text": "No / Disagree / Deny",
        "command": "shake_no",
        "confidence_threshold": 0.7,
        "description": "Head moving side to side"
    },
    "bow": {
        "text": "Respect / Thank you / Apologize",
        "command": "bow",
        "confidence_threshold": 0.7,
        "description": "Upper body bending forward"
    },
    "jumping": {
        "text": "Happy / Excited / Yes!",
        "command": "excited",
        "confidence_threshold": 0.7,
        "description": "Person jumping up and down"
    },
    "standing_still": {
        "text": "Attention / Ready / Waiting",
        "command": "ready",
        "confidence_threshold": 0.6,
        "description": "Person standing still"
    },
}

def get_gesture_text(gesture_name: str) -> str:
    """Get text translation for a gesture."""
    return GESTURE_VOCABULARY.get(gesture_name, {}).get("text", "Unknown gesture")

def get_gesture_command(gesture_name: str) -> str:
    """Get command for a gesture."""
    return GESTURE_VOCABULARY.get(gesture_name, {}).get("command", "unknown")

def get_confidence_threshold(gesture_name: str) -> float:
    """Get confidence threshold for a gesture."""
    return GESTURE_VOCABULARY.get(gesture_name, {}).get("confidence_threshold", 0.7)

def list_all_gestures():
    """Return list of all supported gestures."""
    return list(GESTURE_VOCABULARY.keys())

def get_gesture_description(gesture_name: str) -> str:
    """Get description of a gesture."""
    return GESTURE_VOCABULARY.get(gesture_name, {}).get("description", "No description")

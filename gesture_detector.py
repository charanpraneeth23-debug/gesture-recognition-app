import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from gesture_map import GESTURE_VOCABULARY, get_gesture_text, get_gesture_command

class GestureDetector:
    """
    Real-time gesture detection using MediaPipe for hands and pose.
    Translates detected gestures to text and commands for communication.
    """
    
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create hand and pose detectors
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.previous_gesture = None
        self.gesture_hold_frames = 0
        self.hold_threshold = 5  # Hold gesture for 5 frames to confirm
        
    def detect_gestures(self, frame: np.ndarray) -> Dict:
        """
        Main detection method - processes frame and returns detected gestures.
        
        Args:
            frame: Input image frame from camera
            
        Returns:
            Dictionary with detected gestures, text, and confidence scores
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        
        # Process hand landmarks
        hand_results = self.hands.process(rgb_frame)
        
        # Process pose landmarks
        pose_results = self.pose.process(rgb_frame)
        
        result = {
            "gesture": None,
            "text": "",
            "command": None,
            "confidence": 0.0,
            "detected_hands": 0,
            "pose_detected": False,
            "landmarks_hand": None,
            "landmarks_pose": None
        }
        
        # Detect hand gestures
        if hand_results.multi_hand_landmarks:
            result["detected_hands"] = len(hand_results.multi_hand_landmarks)
            hand_gesture, confidence = self._classify_hand_gesture(
                hand_results.multi_hand_landmarks[0],
                hand_results.multi_handedness[0]
            )
            result["landmarks_hand"] = hand_results.multi_hand_landmarks[0]
            
            if confidence > 0.6:
                result["gesture"] = hand_gesture
                result["confidence"] = confidence
        
        # Detect body/pose gestures
        if pose_results.pose_landmarks:
            result["pose_detected"] = True
            result["landmarks_pose"] = pose_results.pose_landmarks
            pose_gesture, pose_confidence = self._classify_pose_gesture(
                pose_results.pose_landmarks
            )
            
            # Prefer hand gesture if both detected, otherwise use pose
            if pose_confidence > 0.6 and result["confidence"] < pose_confidence:
                result["gesture"] = pose_gesture
                result["confidence"] = pose_confidence
        
        # Translate gesture to text and command
        if result["gesture"]:
            result["text"] = get_gesture_text(result["gesture"])
            result["command"] = get_gesture_command(result["gesture"])
        
        return result
    
    def _classify_hand_gesture(self, landmarks, handedness) -> Tuple[Optional[str], float]:
        """
        Classify hand gesture based on landmarks.
        Uses distance and angle calculations for robust detection.
        """
        # Extract key landmarks
        # 0: wrist, 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
        lm = landmarks.landmark
        
        # Calculate distances and angles
        thumb_tip = np.array([lm[4].x, lm[4].y])
        index_tip = np.array([lm[8].x, lm[8].y])
        middle_tip = np.array([lm[12].x, lm[12].y])
        ring_tip = np.array([lm[16].x, lm[16].y])
        pinky_tip = np.array([lm[20].x, lm[20].y])
        palm = np.array([lm[9].x, lm[9].y])
        wrist = np.array([lm[0].x, lm[0].y])
        
        # Count extended fingers
        fingers_extended = self._count_extended_fingers(lm)
        
        # Check for specific gestures
        # Thumbs up
        if self._is_thumbs_up(lm, thumb_tip, index_tip):
            return "thumbs_up", 0.85
        
        # Thumbs down
        if self._is_thumbs_down(lm, thumb_tip, index_tip):
            return "thumbs_down", 0.85
        
        # Open palm
        if fingers_extended >= 4:
            if self._is_open_palm(lm):
                return "open_palm", 0.8
        
        # Peace sign (2 fingers up)
        if fingers_extended == 2:
            if self._is_peace_sign(lm, index_tip, middle_tip):
                return "peace_sign", 0.8
        
        # Fist (all fingers closed)
        if fingers_extended == 0:
            return "fist", 0.85
        
        # OK sign
        if self._is_ok_sign(lm, thumb_tip, index_tip):
            return "ok_sign", 0.8
        
        # Pointing gestures
        if fingers_extended == 1:
            point_direction = self._detect_pointing_direction(lm, index_tip, palm)
            if point_direction:
                return point_direction, 0.75
        
        # Hand stop
        if self._is_hand_stop(lm):
            return "hand_stop", 0.75
        
        return None, 0.0
    
    def _classify_pose_gesture(self, landmarks) -> Tuple[Optional[str], float]:
        """
        Classify body/pose gesture based on body landmarks.
        """
        lm = landmarks.landmark
        
        # Get shoulder, elbow, wrist positions
        left_shoulder = np.array([lm[11].x, lm[11].y, lm[11].z])
        right_shoulder = np.array([lm[12].x, lm[12].y, lm[12].z])
        left_elbow = np.array([lm[13].x, lm[13].y, lm[13].z])
        right_elbow = np.array([lm[14].x, lm[14].y, lm[14].z])
        left_wrist = np.array([lm[15].x, lm[15].y, lm[15].z])
        right_wrist = np.array([lm[16].x, lm[16].y, lm[16].z])
        head = np.array([lm[0].x, lm[0].y, lm[0].z])
        
        # Check for both hands up
        if left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
            if abs(left_wrist[0] - left_shoulder[0]) > 0.2 and abs(right_wrist[0] - right_shoulder[0]) > 0.2:
                return "both_hands_up", 0.8
        
        # Check for arms open wide
        if self._is_arms_open_wide(left_wrist, right_wrist, left_shoulder, right_shoulder):
            return "arms_open_wide", 0.75
        
        # Check for crossed arms
        if self._is_crossed_arms(left_wrist, right_wrist, left_shoulder, right_shoulder):
            return "crossed_arms", 0.8
        
        # Check for bow
        if self._is_bowing(head, left_shoulder, right_shoulder):
            return "bow", 0.7
        
        # Check for hand raise (single hand)
        if left_wrist[1] < left_shoulder[1] - 0.15:
            return "hand_raise", 0.7
        if right_wrist[1] < right_shoulder[1] - 0.15:
            return "hand_raise", 0.7
        
        return None, 0.0
    
    def _count_extended_fingers(self, lm) -> int:
        """
        Count number of extended fingers.
        """
        fingers = []
        
        # Thumb (compare with index)
        if lm[4].x < lm[3].x:  # Thumb extended to the left
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (compare tip with PIP joint)
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if lm[tip].y < lm[pip].y:  # Tip above PIP means extended
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def _is_thumbs_up(self, lm, thumb_tip: np.ndarray, index_tip: np.ndarray) -> bool:
        """Check if thumbs up gesture."""
        # Thumb should be above index
        if thumb_tip[1] > index_tip[1]:
            return False
        # Other fingers should be closed
        extended = self._count_extended_fingers(lm)
        return extended == 1
    
    def _is_thumbs_down(self, lm, thumb_tip: np.ndarray, index_tip: np.ndarray) -> bool:
        """Check if thumbs down gesture."""
        # Thumb should be below index
        if thumb_tip[1] < index_tip[1]:
            return False
        # Other fingers should be closed
        extended = self._count_extended_fingers(lm)
        return extended == 1
    
    def _is_open_palm(self, lm) -> bool:
        """Check if open palm gesture."""
        # Check if palm is facing camera (relatively flat hand)
        fingers_up = sum([1 for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)] 
                         if lm[tip].y < lm[pip].y])
        return fingers_up >= 4
    
    def _is_peace_sign(self, lm, index_tip: np.ndarray, middle_tip: np.ndarray) -> bool:
        """Check if peace sign gesture."""
        # Index and middle should be up
        # Ring and pinky should be down
        if lm[8].y >= lm[6].y or lm[12].y >= lm[10].y:
            return False
        if lm[16].y < lm[14].y or lm[20].y < lm[18].y:
            return False
        return True
    
    def _is_ok_sign(self, lm, thumb_tip: np.ndarray, index_tip: np.ndarray) -> bool:
        """Check if OK sign gesture."""
        # Thumb and index close together
        distance = np.linalg.norm(thumb_tip - index_tip)
        return distance < 0.05
    
    def _detect_pointing_direction(self, lm, index_tip: np.ndarray, palm: np.ndarray) -> Optional[str]:
        """Detect pointing direction."""
        direction = index_tip - palm
        angle = np.arctan2(direction[1], direction[0])
        
        # Convert to degrees
        angle_deg = np.degrees(angle)
        
        if angle_deg > -45 and angle_deg < 45:
            return "point_right"
        elif angle_deg > 45 and angle_deg < 135:
            return "point_down"
        elif angle_deg > -135 and angle_deg < -45:
            return "point_up"
        else:
            return "point_left"
    
    def _is_hand_stop(self, lm) -> bool:
        """Check if hand stop gesture."""
        # Palm facing forward, fingers extended
        fingers_extended = self._count_extended_fingers(lm)
        return fingers_extended >= 4
    
    def _is_arms_open_wide(self, left_wrist: np.ndarray, right_wrist: np.ndarray,
                           left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> bool:
        """Check if arms are open wide."""
        # Distance between wrists should be large
        distance = abs(left_wrist[0] - right_wrist[0])
        return distance > 0.6
    
    def _is_crossed_arms(self, left_wrist: np.ndarray, right_wrist: np.ndarray,
                         left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> bool:
        """Check if arms are crossed."""
        # Wrists should be close to opposite shoulders
        left_near_right = abs(left_wrist[0] - right_shoulder[0]) < 0.1
        right_near_left = abs(right_wrist[0] - left_shoulder[0]) < 0.1
        return left_near_right and right_near_left
    
    def _is_bowing(self, head: np.ndarray, left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> bool:
        """Check if person is bowing."""
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        return head[1] > shoulder_y + 0.1
    
    def draw_landmarks(self, frame: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Draw detected landmarks on the frame.
        """
        output_frame = frame.copy()
        
        # Draw hand landmarks
        if detection_result["landmarks_hand"]:
            self.mp_drawing.draw_landmarks(
                output_frame,
                detection_result["landmarks_hand"],
                self.mp_hands.HAND_CONNECTIONS
            )
        
        # Draw pose landmarks
        if detection_result["landmarks_pose"]:
            self.mp_drawing.draw_landmarks(
                output_frame,
                detection_result["landmarks_pose"],
                self.mp_pose.POSE_CONNECTIONS
            )
        
        return output_frame

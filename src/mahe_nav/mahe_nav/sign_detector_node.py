"""
sign_detector_node.py
=====================
CV-based sign detection using HSV colour segmentation and shape analysis.
Replaces the old template-matching approach with robust colour + geometry
logic ported from CV_works.

Publishes:  /sign_detection        (mahe_nav_interfaces/msg/SignDetection)
            /sign_detection/debug_image  (sensor_msgs/msg/Image)
Subscribes: camera image topic (configurable, default /r1_mini/camera/image_raw)
"""

import numpy as np
import cv2
from collections import Counter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mahe_nav_interfaces.msg import SignDetection

# ── Physical constants for distance estimation ───────────────────────────
SIGN_PHYSICAL_WIDTH_M = 0.250
FOCAL_LENGTH_PX = 534.7

# ── HSV colour ranges (tuned from actual sign images) ────────────────────
BLUE_LOW    = np.array([100, 100, 80])
BLUE_HIGH   = np.array([130, 255, 255])

RED_LOW1    = np.array([0,   120, 70])    # red wraps around 0 in HSV
RED_HIGH1   = np.array([10,  255, 255])
RED_LOW2    = np.array([170, 120, 70])
RED_HIGH2   = np.array([180, 255, 255])

YELLOW_LOW  = np.array([20,  100, 100])
YELLOW_HIGH = np.array([35,  255, 255])

# ── Shape thresholds ─────────────────────────────────────────────────────
MIN_AREA = 500  # ignore tiny blobs


# ── Helper functions (from CV_works) ─────────────────────────────────────

def get_largest_blob(mask):
    """Return largest contour from mask, or None."""
    kernel  = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, cleaned
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_AREA:
        return None, cleaned
    return cnt, cleaned


def shape_metrics(cnt):
    """Compute circularity, solidity, and vertex count for a contour."""
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    circ  = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
    hull_a = cv2.contourArea(cv2.convexHull(cnt))
    solid  = area / hull_a if hull_a > 0 else 0
    approx = cv2.approxPolyDP(cnt, 0.04 * perim, True)
    return circ, solid, len(approx)


def classify_blue_signs(frame):
    """
    Detect blue directional signs: FORWARD, LEFT, RIGHT, INPLACE_ROTATION.
    Uses arrow tip direction and dot detection to disambiguate.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    valid = sorted(
        [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) > 300],
        reverse=True,
    )
    if not valid:
        return None, None, mask

    largest_area, largest_cnt = valid[0]
    bbox  = cv2.boundingRect(largest_cnt)
    perim = cv2.arcLength(largest_cnt, True)
    circ  = 4 * np.pi * largest_area / (perim ** 2) if perim > 0 else 0
    solid = largest_area / cv2.contourArea(cv2.convexHull(largest_cnt))

    # INPLACE_ROTATION — circular arc, low circularity & solidity
    if circ < 0.15 and solid < 0.35:
        return "INPLACE_ROTATION", bbox, mask

    # ── Arrow direction: is tip pointing UP or DOWN? ─────────────────
    pts = largest_cnt[:, 0, :]
    M   = cv2.moments(largest_cnt)
    cy  = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0

    top_y    = pts[:, 1].min()
    bottom_y = pts[:, 1].max()

    dist_top    = cy - top_y
    dist_bottom = bottom_y - cy
    pointing_up = dist_top > dist_bottom

    # ── Dot detection ────────────────────────────────────────────────
    has_dot = False
    if len(valid) >= 2:
        dot_area, dot_cnt = valid[1]
        dot_perim = cv2.arcLength(dot_cnt, True)
        dot_circ  = (4 * np.pi * dot_area / (dot_perim ** 2)
                     if dot_perim > 0 else 0)
        if dot_area < largest_area * 0.25 and dot_circ > 0.5:
            has_dot = True

    # ── Decision table ───────────────────────────────────────────────
    # RIGHT:   up arrow + dot
    # LEFT:    down arrow + dot
    # FORWARD: down arrow + no dot
    if has_dot and pointing_up:
        return "RIGHT", bbox, mask
    elif has_dot and not pointing_up:
        return "LEFT", bbox, mask
    elif not has_dot and not pointing_up:
        return "FORWARD", bbox, mask
    else:
        return "FORWARD", bbox, mask


def detect_stop(frame):
    """Detect STOP from red octagon."""
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
    mask2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
    mask  = cv2.bitwise_or(mask1, mask2)
    cnt, _ = get_largest_blob(mask)
    if cnt is None:
        return None, None, mask
    circ, solid, verts = shape_metrics(cnt)
    # Octagon: 6-10 vertices, high solidity
    if solid > 0.85 and 6 <= verts <= 10:
        return "STOP", cv2.boundingRect(cnt), mask
    return None, None, mask


def detect_goal(frame):
    """Detect GOAL from yellow star."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    cnt, _ = get_largest_blob(mask)
    if cnt is None:
        return None, None, mask
    circ, solid, verts = shape_metrics(cnt)
    # Star: low solidity (pointy), many vertices
    if solid < 0.75 and verts >= 8:
        return "GOAL", cv2.boundingRect(cnt), mask
    # Fallback: any yellow blob large enough = GOAL
    if cv2.contourArea(cnt) > 1000:
        return "GOAL", cv2.boundingRect(cnt), mask
    return None, None, mask


def detect(frame):
    """
    Run all detectors and return the highest-priority result.
    Priority: STOP > GOAL > blue directional signs.
    """
    blue_sign, blue_box, blue_mask = classify_blue_signs(frame)
    stop_sign, stop_box, stop_mask = detect_stop(frame)
    goal_sign, goal_box, goal_mask = detect_goal(frame)

    if stop_sign:
        return stop_sign, stop_box, stop_mask
    if goal_sign:
        return goal_sign, goal_box, goal_mask
    if blue_sign:
        return blue_sign, blue_box, blue_mask
        
    # If no sign detected, return the blue mask by default (or a combined mask)
    combined = cv2.bitwise_or(blue_mask, cv2.bitwise_or(stop_mask, goal_mask))
    return "NONE", None, combined


# ── ROS 2 Node ───────────────────────────────────────────────────────────

class SignDetectorNode(Node):
    def __init__(self):
        super().__init__('sign_detector')

        # Parameters
        self.declare_parameter('camera_topic', '/r1_mini/camera/image_raw')
        self.declare_parameter('consensus_frames', 5)
        camera_topic    = self.get_parameter('camera_topic').value
        self.consensus_n = self.get_parameter('consensus_frames').value

        self.bridge = CvBridge()
        self.detection_history = []
        self.last_logged_sign = 'NONE'

        # ROS 2 interfaces
        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(SignDetection, '/sign_detection', 10)
        self.debug_pub = self.create_publisher(Image, '/sign_detection/debug_image', 10)
        self.mask_pub = self.create_publisher(Image, '/sign_detection/mask_image', 10)
        self.sub = self.create_subscription(
            Image, camera_topic, self._image_cb, sensor_qos)

        self.get_logger().info(
            f'Sign Detector Active — HSV colour + shape detection '
            f'(camera: {camera_topic})')

    def _image_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        sign, box, mask = detect(cv_img)

        # ── Temporal consensus (majority vote over N frames) ─────────
        self.detection_history.append(sign)
        if len(self.detection_history) > self.consensus_n:
            self.detection_history.pop(0)

        counts = Counter(self.detection_history)
        top, cnt = counts.most_common(1)[0]
        final_sign = (top if (top != "NONE"
                              and cnt >= max(3, self.consensus_n // 2 + 1))
                      else "NONE")

        # ── Build and publish SignDetection message ──────────────────
        out = SignDetection()
        out.header = msg.header
        out.sign_type = final_sign

        # Create a debug image for RViz / rqt_image_view visualization
        debug_img = cv_img.copy()

        if box is not None:
            x, y, w, h = box
            out.image_x = float(x + w / 2)
            out.image_y = float(y + h / 2)
            out.distance_estimate = (
                (FOCAL_LENGTH_PX * SIGN_PHYSICAL_WIDTH_M) / w
                if w > 0 else 0.0)
            out.confidence = 1.0 if final_sign != "NONE" else 0.0

            # Draw on debug image
            color = (0, 255, 0) if final_sign != "NONE" else (0, 0, 255)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_img, f"{final_sign} ({out.distance_estimate:.2f}m)",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            out.image_x = 0.0
            out.image_y = 0.0
            out.distance_estimate = 0.0
            out.confidence = 0.0

        self.pub.publish(out)

        # ── Print detection status to terminal ───────────────────────
        if final_sign != 'NONE':
            self.get_logger().info(
                f'🟢 DETECTED: {final_sign} | dist={out.distance_estimate:.2f}m '
                f'| raw={sign} | confidence={out.confidence:.1f}')
            self.last_logged_sign = final_sign
        elif self.last_logged_sign != 'NONE':
            self.get_logger().info('🔴 Sign lost — scanning...')
            self.last_logged_sign = 'NONE'

        # Publish debug image to ROS
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, 'bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
            
            mask_msg = self.bridge.cv2_to_imgmsg(mask, 'mono8')
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

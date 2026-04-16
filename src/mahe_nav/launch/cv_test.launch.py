"""
cv_test.launch.py
=================
Minimal launch for testing CV sign detection with teleop.

Starts ONLY:
  1. Sign detector  — camera → /sign_detection (prints detections to terminal)
  2. Motor driver   — /cmd_vel → hardware PWM (so teleop can drive the robot)

Usage:
    Terminal 1:  ros2 launch mahe_nav cv_test.launch.py
    Terminal 2:  ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    log_level = LaunchConfiguration('log_level')

    # ── 1. Sign detector (HSV colour + shape analysis) ───────────────────
    sign_node = Node(
        package    = 'mahe_nav',
        executable = 'sign_detector',
        name       = 'sign_detector',
        output     = 'screen',
        parameters = [
            {'use_sim_time': False},
            {'camera_topic': '/r1_mini/camera/image_raw'},
            {'consensus_frames': 5},
        ],
        arguments  = ['--ros-args', '--log-level', log_level],
    )
    
    # ── 2. CV Viewer Node (pops up cv2 windows) ──────────────────────────
    viewer_node = Node(
        package    = 'mahe_nav',
        executable = 'cv_viewer',
        name       = 'cv_viewer',
        output     = 'screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level: debug, info, warn, error',
        ),
        sign_node,
        viewer_node,
    ])

"""
aruco_test.launch.py
====================
Minimal launch for testing ArUco marker detection with teleop and camera info.

Starts:
  1. ArUco detector — camera → /aruco/detections
  2. CV Viewer      — pops up a window showing ArUco hits

Usage:
    Terminal 1:  ros2 launch mahe_nav aruco_test.launch.py
    Terminal 2:  ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    log_level = LaunchConfiguration('log_level')

    # ── 1. ArUco detector ───────────────────────────────────────────────────
    aruco_node = Node(
        package    = 'mahe_nav',
        executable = 'aruco_detector',
        name       = 'aruco_detector',
        output     = 'screen',
        parameters = [
            {'use_sim_time': False},
            {'marker_size_m': 0.150},
            {'camera_topic': '/r1_mini/camera/image_raw'},
            {'info_topic': '/r1_mini/camera/camera_info'},
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
        aruco_node,
        viewer_node,
    ])

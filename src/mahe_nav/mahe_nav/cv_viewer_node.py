import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CVViewerNode(Node):
    def __init__(self):
        super().__init__('cv_viewer')
        self.bridge = CvBridge()
        
        self.create_subscription(Image, '/sign_detection/debug_image', self.debug_cb, 5)
        self.create_subscription(Image, '/sign_detection/mask_image', self.mask_cb, 5)
        self.create_subscription(Image, '/aruco/debug_image', self.aruco_cb, 5)
        
        self.get_logger().info('CV Viewer Node Active - Window will open when images are received')

    def debug_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imshow("Sign Detection [Bounding Boxes]", cv_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error viewing debug image: {e}")

    def mask_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            cv2.imshow("Sign Detection [HSV MASK]", cv_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error viewing mask image: {e}")

    def aruco_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imshow("ArUco Detection", cv_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error viewing ArUco image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CVViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

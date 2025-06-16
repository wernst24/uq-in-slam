import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lae5777/ros2_ws/src/uq-in-slam/install/test_new_pkg_name'

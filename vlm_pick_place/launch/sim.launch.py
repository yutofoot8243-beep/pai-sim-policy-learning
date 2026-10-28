"""Gazebo (Ignition) + CRANE+ V2 (カメラスタンド付き) + MoveIt を起動する.

crane_plus_gazebo/crane_plus_with_table.launch.py を元にしている. 講義の Docker イメージに
入っている crane_plus は古く、ワールド指定・カメラ引数が無いため自前で用意した.
カメラ画像はワールド側のセンサー (worlds/vlm_table.sdf の overhead_camera) が配信する.
"""

import os

from ament_index_python.packages import get_package_share_directory
from crane_plus_description.robot_description_loader import RobotDescriptionLoader
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    declare_world = DeclareLaunchArgument(
        'world_name',
        default_value=os.path.join(
            get_package_share_directory('vlm_pick_place'), 'worlds', 'vlm_table.sdf'))
    # 評価の連続実行用. GUI を出さずサーバーだけ動かす（カメラは EGL で X なしに描画する）
    declare_headless = DeclareLaunchArgument('headless', default_value='false')
    headless = LaunchConfiguration('headless')

    # PATHを追加で通さないとSTLファイルが読み込まれない
    env = {'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.environ['LD_LIBRARY_PATH'],
           'IGN_GAZEBO_RESOURCE_PATH': os.path.dirname(
               get_package_share_directory('crane_plus_description'))}
    gui_config = os.path.join(
        get_package_share_directory('crane_plus_gazebo'), 'gui', 'gui.config')
    # -r オプションで起動時にシミュレーションをスタートしないと、コントローラが起動しない
    ign_gazebo = ExecuteProcess(
        cmd=['ign gazebo -r', LaunchConfiguration('world_name'), '--gui-config', gui_config],
        output='screen',
        additional_env=env,
        shell=True,
        condition=UnlessCondition(headless))
    ign_gazebo_server = ExecuteProcess(
        cmd=['ign gazebo -s -r --headless-rendering', LaunchConfiguration('world_name')],
        output='screen',
        additional_env=env,
        shell=True,
        condition=IfCondition(headless))

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', '/robot_description', '-name', 'crane_plus',
                   '-z', '1.015', '-allow_renaming', 'true'])

    loader = RobotDescriptionLoader()
    loader.use_camera = 'true'   # カメラスタンドの形状と camera_color_optical_frame の tf
    loader.use_gazebo = 'true'
    loader.use_ignition = 'true'
    loader.gz_control_config_package = 'crane_plus_control'
    loader.gz_control_config_file_path = 'config/crane_plus_controllers.yaml'
    description = loader.load()

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('crane_plus_moveit_config'),
            '/launch/run_move_group.launch.py']),
        launch_arguments={'loaded_description': description}.items())

    # 起動直後は描画付きシミュレーションが重く、3つ同時に spawn すると activate が
    # タイムアウトすることがある. 1つずつ起動し、失敗したら activate だけやり直す.
    controllers = ['joint_state_broadcaster', 'crane_plus_arm_controller',
                   'crane_plus_gripper_controller']
    spawn_script = ' && '.join(
        f'(ros2 run controller_manager spawner {c} || '
        f'(for i in 1 2 3 4 5; do sleep 3; '
        f'ros2 control set_controller_state {c} active && break; done))'
        for c in controllers)
    spawn_controllers = ExecuteProcess(cmd=[spawn_script], shell=True, output='screen')

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                   '/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
                   '/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
                   '/side_camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image'],
        output='screen')

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        declare_world,
        declare_headless,
        ign_gazebo,
        ign_gazebo_server,
        spawn,
        move_group,
        spawn_controllers,
        bridge,
    ])

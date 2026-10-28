"""画面なしの Gazebo と評価ノードを起動し、評価が終わったら全部止める.

  ros2 launch vlm_pick_place eval.launch.py episodes:=20 agent:=oracle record:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue

ARGS = {   # 名前: (既定値, 型)
    'agent': ('oracle', str),
    'episodes': ('10', int),
    'episodes_per_run': ('0', int),
    'seed': ('0', int),
    'n_cubes': ('0', int),
    'yaw_jitter_deg': ('20.0', float),
    'physics_step': ('0.004', float),
    'max_steps': ('6', int),
    'record': ('false', bool),
    'output_dir': ('', str),
    'act_n_action_steps': ('10', int),
    'act_max_sec': ('40.0', float),
    'act_grip_snap_rad': ('0.05', float),
}


def generate_launch_description():
    pkg = get_package_share_directory('vlm_pick_place')
    declares = [DeclareLaunchArgument(name, default_value=default)
                for name, (default, _) in ARGS.items()]
    declares += [
        DeclareLaunchArgument('headless', default_value='true'),
        DeclareLaunchArgument(
            'world_name', default_value=os.path.join(pkg, 'worlds', 'vlm_table_eval.sdf')),
    ]

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={'world_name': LaunchConfiguration('world_name'),
                          'headless': LaunchConfiguration('headless')}.items())

    evaluator = Node(
        package='vlm_pick_place',
        executable='eval',
        output='screen',
        parameters=[os.path.join(pkg, 'config', 'params.yaml'),
                    {name: ParameterValue(LaunchConfiguration(name), value_type=t)
                     for name, (_, t) in ARGS.items()}])

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        *declares,
        sim,
        # コントローラと move_group が立ち上がるのを待ってから起動する
        TimerAction(period=15.0, actions=[evaluator]),
        RegisterEventHandler(OnProcessExit(
            target_action=evaluator,
            on_exit=[EmitEvent(event=Shutdown(reason='evaluation finished'))])),
    ])

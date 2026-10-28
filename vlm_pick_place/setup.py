from glob import glob

from setuptools import setup

package_name = 'vlm_pick_place'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ysRSpoteto',
    maintainer_email='noreply@example.com',
    description='Language-instructed pick and place for CRANE+ V2 using Gemini Robotics-ER',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_agent = vlm_pick_place.vlm_agent_node:main',
            'pick_test = vlm_pick_place.pick_test_node:main',
            'recorder = vlm_pick_place.recorder_node:main',
            'eval = vlm_pick_place.eval_node:main',
        ],
    },
)

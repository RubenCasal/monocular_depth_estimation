from setuptools import setup

package_name = 'depth_estimation_dac'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/your_launch_file.py']),  # optional
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='Depth estimation ROS 2 package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'depth_node = depth_estimation_dac.depth_node:main'
        ],
    },
)

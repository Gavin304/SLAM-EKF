from setuptools import find_packages, setup
from glob import glob
package_name = 'ekf_filter'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asrlab',
    maintainer_email='asrlab@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_filter= ekf_filter.ekf_filter:main',
            'comparison= ekf_filter.comparison:main',
            'slam_ekf= ekf_filter.slam_ekf:main',
        ],
    },
)

from setuptools import setup

package_name = 'ssd_detection'
submodules = "ssd_detection/utils"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seniorcar',
    maintainer_email='seniorcar@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "ssd_detection = ssd_detection.ros2_panorama_detection:main"
        ],
    },
)

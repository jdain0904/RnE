from setuptools import setup, find_packages
package_name = 'hpso_ik'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=['hpso_ik', 'hpso_ik.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', ['urdf/fourdof_arm.urdf']),
        ('share/' + package_name + '/config', ['config/controllers.yaml']),
        ('share/' + package_name + '/launch', ['launch/gazebo_arm.launch.py','launch/hpso_svm.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='HPSO+SVM IK demo with Gazebo',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'hpso_svm_node = hpso_ik.hpso_svm_node:main',
        ],
    },
)

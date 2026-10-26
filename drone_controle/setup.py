from setuptools import find_packages, setup

package_name = 'drone_controle'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lefri',
    maintainer_email='lefri@todo.todo',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mixer_node = drone_controle.mixer_node:main',
            'lqr_node = drone_controle.lqr_node:main',
        ],
    },
)

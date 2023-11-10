import sys
sys.path.append('./tocabirl')

__version__ = '0.0.1'

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tocabirl", 
    version= __version__,
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy>=1.15','scipy>=1.0.0','gym>=0.17.0,<=0.19.0','stable_baselines3<=1.3.0,>=1.3.0','mujoco-py<2.2,>=2.1', 'pyquaternion'],
    package_data={
        "tocabirl": ["cust_gym/assets/*.xml", "cust_gym/meshes/dyros_tocabi/*.STL"],
        # "tocabirl": ["cust_gym/meshes/*.STL"],
    },
    include_package_data=True,   
)

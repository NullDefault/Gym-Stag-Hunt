from setuptools import setup, find_packages

setup(
    name="gym_stag_hunt",
    version="0.0.1",
    author="David Nesterov-Rappoport",
    author_email="davisha999@gmail.com",
    description="Markov stag hunt environment for openai gym",
    long_description="This package is based on openai-gym and created for running experiments on Markov stag hunt "
    "games.",
    long_description_content_type="text/markdown",
    url="https://github.com/NullDefault/gym-stag-hunt",
    packages=find_packages(),
    install_requires=["gym", "pygame", "opencv-python", "pettingzoo"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

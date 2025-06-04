from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wifi-radar",
    version="0.1.0",
    author="WiFi-Radar Team",
    author_email="example@example.com",
    description="Human pose estimation through WiFi signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wifi-radar",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "tensorflow>=2.6.0",
        "dash>=2.0.0",
        "plotly>=5.3.0",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
        "scikit-learn>=1.0.0",
        "ffmpeg-python>=0.2.0",
        "websockets>=10.0",
        "pyrtmp>=0.1.0",
        "pywifi>=1.1.12",
        "dash-bootstrap-components>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "wifi-radar=scripts.start_wifi_radar:main",
        ],
    },
)

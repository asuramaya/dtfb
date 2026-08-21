from setuptools import setup

setup(
    name="dtfb",
    version="0.1.0",
    description="Automate vehicle listing posts from dealer websites to Facebook/Instagram/Threads",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.11",
    install_requires=[
        "pillow>=12",
        "playwright>=1.62",
        "requests>=2.34",
        "numpy>=2",
        "torch>=2.13",
        "torchvision>=0.28",
        "transformers>=5.14",
        "onnxruntime-gpu>=1.28",
        "open_clip_torch>=3.3",
        "rembg>=2.0",
        "imagehash>=4.3",
        "spandrel>=0.4",
    ],
    extras_require={
        "dev": ["pytest>=8", "pytest-xdist>=3"],
    },
    entry_points={
        "console_scripts": [
            "dtfb = dtfb:main",
            "compose = compose_cli:main",
            "hero-video = hero_video_cli:main",
            "posts = posts_cli:main",
            "recompose = recompose_cli:main",
            "inventory-sync = inventory_sync:main",
        ],
    },
)
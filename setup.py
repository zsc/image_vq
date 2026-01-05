from setuptools import setup

setup(
    name="open-magvit2-demo",
    version="0.1.0",
    description="Open-MAGVIT2 Inference Demo",
    py_modules=["main"],
    install_requires=[
        "numpy",
        "torch",
        "einops",
        "Pillow",
        "huggingface_hub",
        "pillow-avif-plugin",
    ],
    entry_points={
        "console_scripts": [
            "image_vq=main:main",
        ],
    },
)

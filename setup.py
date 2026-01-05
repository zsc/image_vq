from setuptools import setup

setup(
    name="image_vq",
    version="0.1.0",
    description="Image VQ: Open-MAGVIT2 Inference Tool",
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

"""
Transformer-PyTorch 安装脚本

基于 PyTorch 的 Transformer 架构实现包。

作者: shihom_wu
版本: 1.0.0
"""

from setuptools import setup, find_packages
import os

# 读取 README 文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# 读取版本信息
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'transformer_pytorch', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.0"

# 读取依赖
def get_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'tqdm>=4.60.0',
        'jupyter>=1.0.0',
        'pytest>=6.0.0',
        'black>=21.0.0',
        'flake8>=3.8.0',
    ]

setup(
    name="transformer-pytorch",
    version=get_version(),
    author="shihom_wu",
    author_email="transformer-pytorch@example.com",
    description="基于 PyTorch 的 Transformer 架构实现",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/transformer-pytorch/transformer-pytorch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "isort>=5.0.0",
            "mypy>=0.800",
            "pre-commit>=2.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.10.0",
            "myst-parser>=0.15.0",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "tensorboard>=2.5.0",
        ],
        "all": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "isort>=5.0.0",
            "mypy>=0.800",
            "pre-commit>=2.10.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.10.0",
            "myst-parser>=0.15.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "tensorboard>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "transformer-train=transformer_pytorch.scripts.train:main",
            "transformer-generate=transformer_pytorch.scripts.generate:main",
            "transformer-eval=transformer_pytorch.scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "transformer_pytorch": [
            "config/*.json",
            "data/*.txt",
            "examples/*.py",
        ],
    },
    zip_safe=False,
    keywords=[
        "transformer",
        "attention",
        "neural networks",
        "deep learning",
        "pytorch",
        "nlp",
        "machine learning",
        "artificial intelligence",
    ],
    project_urls={
        "Bug Reports": "https://github.com/transformer-pytorch/transformer-pytorch/issues",
        "Source": "https://github.com/transformer-pytorch/transformer-pytorch",
        "Documentation": "https://transformer-pytorch.readthedocs.io/",
    },
)

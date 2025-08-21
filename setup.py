from setuptools import setup, find_packages

setup(
    name="gbwm-reinforcement-learning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Goals-Based Wealth Management using Reinforcement Learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "pymoo>=0.6.0",
        "tqdm>=4.64.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0"
        ]
    }
)
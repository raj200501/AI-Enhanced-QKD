from setuptools import setup, find_packages

setup(
    name="ai-enhanced-qkd",
    version="1.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
    entry_points={
        "console_scripts": [
            "qkd=ai_qkd.cli:main",
            "qkd-preprocess=data_preprocessing:main",
            "qkd-anomaly=cnn_anomaly_detection:main",
            "qkd-error=rnn_error_correction:main",
            "qkd-rl=rl_key_distribution:main",
            "qkd-evaluate=evaluate_results:main",
        ],
    },
)

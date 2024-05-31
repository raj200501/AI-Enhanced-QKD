from setuptools import setup, find_packages

setup(
    name='AI-Enhanced-QKD',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'torch',
        'matplotlib',
        'seaborn',
        'gym',
        'jupyter',
    ],
    entry_points={
        'console_scripts': [
            'data_preprocessing=src.data_preprocessing:main',
            'cnn_anomaly_detection=src.cnn_anomaly_detection:main',
            'rnn_error_correction=src.rnn_error_correction:main',
            'rl_key_distribution=src.rl_key_distribution:main',
            'evaluate_results=src.evaluate_results:main',
        ],
    },
)

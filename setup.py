from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ml_cli',
    version='0.1',
    packages=find_packages(),
    install_requires=required,  # Directly reads dependencies from requirements.txt
    entry_points={
        'console_scripts': 'ml=ml_cli.cli:cli'
    },
    author='Atunrase Ayomide',
    author_email='atunraseayomide@gmail.com',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Ayo-Cyber/ml_cli.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)

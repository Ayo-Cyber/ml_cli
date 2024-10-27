from setuptools import setup, find_packages

setup(
    name='ml_cli',  # Replace with your package name
    version='0.1',            # Version number
    packages=find_packages(),  # Automatically find and include all packages in the directory
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'click',
        'pyyaml'
    ],
    entry_points={
        'console_scripts': 'ml=ml_cli.cli:cli'

    },
    author='Atunrase Ayomide',          # name
    author_email='atunraseayomide@gmail.com',  # email
    description='A brief description of your project',  # Short description of your project
    long_description=open('README.md').read(),  # Optional: Read the long description from a README file
    long_description_content_type='text/markdown',  # Specify the format of the long description (e.g., text/markdown)
    url='https://github.com/Ayo-Cyber/ml_cli.git',  # URL to your project (e.g., GitHub repo)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Specify your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version
)

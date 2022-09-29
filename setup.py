import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tlagtk',
    author='Jordi Aguilar Larruy',
    author_email='jaguilar@icmab.es',
    description='TLAG toolkit for data analysis of XRD data',
    keywords='data analysis, XRD, TLAG',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Jordi-aguilar/TLAGToolkit',
    project_urls={
        'Documentation': 'https://github.com/Jordi-aguilar/TLAGToolkit',
        'Source Code': 'https://github.com/Jordi-aguilar/TLAGToolkit',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={},
    packages=setuptools.find_packages(),
    classifiers=[

        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires='>=3.9',
    install_requires=[
        'pandas==1.5.0',
        'numpy==1.23.3',
        'matplotlib==3.6.0',
        'scipy==1.9.1',
        'pyyaml==6.0',
        'BaselineRemoval==0.0.8',
        'lmfit==1.0.3'
    ],
    
    entry_points={
        'console_scripts': [  # This can provide executable scripts
            'visualizer=tlagtk.visualizer:main',
            'peak_fitting=tlagtk.peak_fitting:main'
        ],
    },
)
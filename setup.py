import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tlagtk',
    version='0.0.1'
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
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires='>=3.9',
    install_requires=[
        'asteval==0.9.27',
        'baselineremoval==0.0.8',
        'colorama==0.4.5',
        'contourpy==1.0.5',
        'cycler==0.11.0',
        'fabio==0.14.0',
        'fonttools==4.37.3',
        'future==0.18.2',
        'joblib==1.2.0',
        'kiwisolver==1.4.4',      
        'lmfit==1.0.3',
        'matplotlib==3.6.0',
        'numpy==1.23.3',
        'opencv-python==4.6.0.66',
        'packaging==21.3',
        'pandas==1.5.0',
        'pillow==9.2.0',
        'pybaselines==0.8.0',
        'pyparsing==3.0.9',
        'pyqt5==5.15.7',
        'pyqt5-qt5==5.15.2',
        'pyqt5-sip==12.11.0',
        'pyqtgraph==0.13.1',
        'python-dateutil==2.8.2',
        'pytz==2022.2.1',
        'pyyaml==6.0',
        'scikit-learn==1.1.2',
        'scipy==1.9.1',
        'six==1.16.0',
        'threadpoolctl==3.1.0',
        'tqdm==4.64.1',
        'uncertainties==3.1.7',
    ],
    
    entry_points={
        'console_scripts': [  # This can provide executable scripts
            'visualizer=tlagtk.visualizer:main',
            'peak_fitting=tlagtk.peak_fitting:main'
        ],
    },
)

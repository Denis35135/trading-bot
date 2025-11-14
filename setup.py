#!/usr/bin/env python3
"""
Setup.py pour The Bot
Installation via pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lire le README
this_directory = Path(__file__).parent
long_description = (this_directory / "docs" / "README.md").read_text(encoding='utf-8')

# Lire les requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()

# Filtrer les commentaires et lignes vides
requirements = [req.strip() for req in requirements 
                if req.strip() and not req.strip().startswith('#')]

# Requirements optionnels
extras_require = {
    'dev': [
        'pytest>=7.3.1',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.0.0',
        'black>=23.3.0',
        'flake8>=6.0.0',
        'mypy>=1.3.0',
        'ipython>=8.12.0',
    ],
    'ml': [
        'tensorflow>=2.12.0',
        'keras>=2.12.0',
        'optuna>=3.1.0',
    ],
    'monitoring': [
        'streamlit>=1.22.0',
        'plotly>=5.14.0',
        'dash>=2.9.0',
    ],
    'notifications': [
        'python-telegram-bot>=20.2',
        'discord.py>=2.2.3',
    ],
    'all': [
        # Tout inclure
    ]
}

# Ajouter tout dans 'all'
extras_require['all'] = list(set(
    req for extra_reqs in extras_require.values() 
    for req in extra_reqs if isinstance(extra_reqs, list)
))

setup(
    # MÃƒÂ©tadonnÃƒÂ©es
    name='the-bot',
    version='1.0.0',
    description='Bot de Trading Algorithmique Haute Performance pour Cryptomonnaies',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='The Bot Team',
    author_email='contact@thebot.trading',
    url='https://github.com/yourusername/the-bot',
    license='Proprietary',
    
    # Packages
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'scripts']),
    include_package_data=True,
    
    # Python version
    python_requires='>=3.9',
    
    # DÃƒÂ©pendances
    install_requires=requirements,
    extras_require=extras_require,
    
    # Entry points
    entry_points={
        'console_scripts': [
            'thebot=main:main',
            'thebot-test=test_connection:main',
        ],
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Natural Language :: English',
    ],
    
    # Keywords
    keywords='trading bot cryptocurrency binance algorithmic-trading machine-learning',
    
    # Project URLs
    project_urls={
        'Documentation': 'https://github.com/yourusername/the-bot/docs',
        'Source': 'https://github.com/yourusername/the-bot',
        'Tracker': 'https://github.com/yourusername/the-bot/issues',
        'Changelog': 'https://github.com/yourusername/the-bot/blob/main/CHANGELOG.md',
    },
    
    # Package data
    package_data={
        '': ['*.txt', '*.md', '*.yml', '*.yaml'],
        'data': ['*.json'],
    },
    
    # Zip safe
    zip_safe=False,
)
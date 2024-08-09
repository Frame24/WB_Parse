from setuptools import setup, find_packages

setup(
    name='reviews_keywords',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'nltk',
        'spacy',
        'torch',
        'transformers',
        'pandas',
        'tqdm',
        'yake',
        'pymorphy2',
        'sumy',
        'emoji',
        'pyyaml'
    ],
    entry_points={
        'console_scripts': [
            'analyze_reviews=reviews_keywords.main:analyze_reviews',
        ],
    },
)

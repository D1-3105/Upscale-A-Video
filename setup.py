from setuptools import setup, find_packages

setup(
    name='vedio-upscaler',
    version='0.1.0',
    description='LLaVA + Video Inference tools for vision-language models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Munaza Ashraf',
    url='https://github.com/RunwareThirdParty/Upscale-A-Video',
    packages=find_packages(include=['llava', 'llava.*', 'models_video', 'models_video.*']),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.9',
)

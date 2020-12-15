from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
#     #print(long_description)


setup(
      name='ALG',
      version='0.0.1',
      author="Yang Xue",
      description='ALG environment for OpenAI Gym',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/AustinXY/adversarial-level-generation",
      install_requires=['gym>=0.2.3', 'numpy>=1.14.1', 'tqdm>=4.32.1',
                        'imageio>=2.3.0', 'stable_baselines3>=0.10.0', 'IPython>=6.4.0', 'torch>=1.7.1'],
      packages=find_packages(),
      package_data={
      'ALG': ['envs/*', 'envs/surface/*', 'envs/surface/*/*'],
      },
      include_package_data=True,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

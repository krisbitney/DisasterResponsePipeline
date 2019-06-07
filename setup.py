import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="DisasterResponsePipeline",
    version="0.0.1",
    author="Kris Bitney",
    description="Data Engineering Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krisbitney/DisasterResponsePipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
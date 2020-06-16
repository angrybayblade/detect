import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yodo-codekage", # Replace with your own username
    version="0.0.1",
    author="Viraj Patel",
    author_email="vptl185@gmail.com",
    description="You Only Detect One",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/code-kage/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA :: 10.1",
        "Development Status :: 3 - Alpha",
        

    ],
    python_requires='>=3.5',
)
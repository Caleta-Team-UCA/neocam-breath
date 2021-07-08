import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neocam-breath",
    use_scm_version=False,#True,
    version="0.0.0",
    author="Lionel Cervera",
    author_email="lionel.cervera@uca.es",
    description="Newborn breath monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Caleta-Team-UCA/neocam-breath",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"": ["*.toml"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "argcomplete==1.12.1",
        "blobconverter==0.0.9",
        "boto3==1.17.67",
        "botocore==1.20.67",
        "certifi==2020.12.5",
        "chardet==3.0.4",
        "cycler==0.10.0",
        "depthai==2.5.0.0",
        "ffmpeg==1.4",
        "idna==2.10",
        "jmespath==0.10.0",
        "kiwisolver==1.3.1",
        "matplotlib==3.4.1",
        "numpy==1.19.5",
        "opencv-python==4.5.1.48",
        "Pillow==8.2.0",
        "pyparsing==2.4.7",
        "PyQt5==5.15.4",
        "PyQt5-Qt5==5.15.2",
        "PyQt5-sip==12.8.1",
        "PyQt5-stubs==5.15.2.0",
        "python-dateutil==2.8.1",
        "PyYAML==5.3.1",
        "requests==2.24.0",
        "s3transfer==0.4.2",
        "scipy==1.6.3",
        "six==1.15.0",
        "urllib3==1.25.11",
    ],
)

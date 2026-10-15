import setuptools

setuptools.setup(
    packages=setuptools.find_packages(),
    package_data={
        "auditory_cortex": ["config.yml"]
    },
)
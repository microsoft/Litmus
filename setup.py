import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages()
print ("Installing packages: ", packages)

setuptools.setup(
    name="LitmusPredictor",
    version="0.0.1",
    author="Microsoft",
    description="AI Assistant for Building Reliable, High-performing and Fair Multilingual NLP Systems",
    long_description=long_description,
    packages=packages,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

from setuptools import setup, find_packages
import os


with open("README.md", "r", encoding="utf-8") as file:
    description = file.read()


def load_requirements(filename='requirements.txt'):
    if os.path.exists(filename):
        with open(filename) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []


setup(
    name="table2html",
    version="1.3.3",
    author="TraiPPN",
    license='Apache License 2.0',
    author_email="phamphungoctraivl@gmail.com",
    description="Detect and convert table image to html table",
    packages=find_packages(include=['table2html', 'table2html.*']),
    package_data={
        'table2html': [
            'models/*.pt',
            'models/*.onnx',
            'source/*.py',
        ],
    },
    include_package_data=True,
    install_requires=load_requirements(),
    python_requires='>=3.7',
    long_description=description,
    long_description_content_type="text/markdown"
)
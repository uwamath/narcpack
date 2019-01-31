from setuptools import setup, find_packages

setup(name='narcpack',
    version='0.1',
    description='Numerical Analysis Research Club Package',
    url='http://github.com/uwamath/narcpack',
    author='UW NARC',
    author_email='narc-admin@amath.washington.edu',
    license='GPLv3',
    packages=find_packages(),
    install_requires=['numpy'],
    test_suite='nose2.collector.collector',
    tests_require=['nose2'],
    zip_safe=False)

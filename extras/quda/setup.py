try:
    from lyncs_setuptools import setup, CMakeExtension
except:
    print(
        """
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Install first the requirements:
    pip install -r requirements.txt
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    """
    )

requirements = ["lyncs-cppyy"]

QUDA_CMAKE_ARGS = [
    "-DQUDA_BUILD_SHAREDLIB=ON",
]

setup(
    "lyncs_quda",
    exclude=["*.config"],
    ext_modules=[
        CMakeExtension(
            "lyncs_quda.lib",
            ".",
            ["-DQUDA_CMAKE_ARGS='%s'" % " ".join(QUDA_CMAKE_ARGS)],
        )
    ],
    data_files=[(".", ["config.py.in"])],
    install_requires=["lyncs-cppyy",],
    keywords=["Lyncs", "quda", "Lattice QCD", "python", "interface",],
)

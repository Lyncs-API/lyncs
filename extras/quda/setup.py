import sys
from pathlib import Path
import fileinput

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

requirements = [
    "cupy",
    "lyncs-cppyy",
]

QUDA_CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=RELEASE",
    "-DQUDA_BUILD_SHAREDLIB=ON",
    "-DQUDA_GPU_ARCH=sm_60",
    "-DQUDA_FORCE_GAUGE=ON",
    "-DQUDA_TEX=OFF",
]


def patch_include(builder, ext):
    'Replaces #include instances in header files that use <> with "" for relative includes'
    install_dir = builder.get_install_dir(ext + "/include")
    for path in Path(install_dir).rglob("*.h"):
        with fileinput.FileInput(str(path), inplace=True, backup=".bak") as fp:
            for fline in fp:
                line = str(fline)
                if line.strip().startswith("#include"):
                    include = line.split()[1]
                    if include[0] == "<" and include[-1] == ">":
                        include = include[1:-1]
                        if (path.parents[0] / include).exists():
                            print(line.replace(f"<{include}>", f'"{include}"'), end="")
                            continue
                print(line, end="")


setup(
    "lyncs_quda",
    exclude=["*.config"],
    ext_modules=[
        CMakeExtension(
            "lyncs_quda.lib",
            ".",
            ["-DQUDA_CMAKE_ARGS='%s'" % ";".join(QUDA_CMAKE_ARGS)],
            post_build=patch_include,
        )
    ],
    data_files=[(".", ["config.py.in"])],
    install_requires=["lyncs-cppyy",],
    keywords=["Lyncs", "quda", "Lattice QCD", "python", "interface",],
)

def capture_print(fnc, *args, **kwargs):
    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        fnc(*args, **kwargs)
    return out.getvalue()


def capture_error(fnc, *args, **kwargs):
    import io
    from contextlib import redirect_stderr

    out = io.StringIO()
    with redirect_stderr(out):
        fnc(*args, **kwargs)
    return out.getvalue()


def test_kwargs():
    from lyncs_setuptools import find_version, get_kwargs, print_keys
    from lyncs_setuptools import __version__ as version

    assert find_version() == version

    assert 'version: "' in capture_print(print_keys, [])

    assert capture_print(print_keys, ["author"]) == get_kwargs()["author"] + "\n"


def test_cmake():
    from lyncs_setuptools import CMakeExtension, CMakeBuild
    from distutils.dist import Distribution

    dist = Distribution()
    build = CMakeBuild(dist)
    build.extensions = [CMakeExtension("test", "tests", ["-DMESSAGE=test1234"])]
    build.build_lib = "tests"
    build.build_temp = "tests/tmp"

    assert "test1234" in capture_print(build.run)

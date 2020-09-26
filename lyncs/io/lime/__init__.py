from ...tunable import Tunable, computable, delayed

# List of available implementations
engines = [
    "pylime",
]

from . import lime as pylime

default_engine = "pylime"

try:
    from . import DDalphaAMG

    engines.append("DDalphaAMG")
except:
    pass
# from lyncs import config
# if config.clime_enabled:
#     import .clime
#     engines.append("clime")

# TODO: add more, e.g. lemon


@computable
def engine(engine=None):
    import sys

    engine = engine or default_engine
    self = sys.modules[__name__]
    return getattr(self, engine)


def is_compatible(filename):
    try:
        return pylime.is_lime_file(filename)
    except:
        return False


def get_lattice(records):
    assert "ildg-format" in records, "ildg-format not found"

    from lyncs import Lattice
    import xmltodict

    info = xmltodict.parse(records["ildg-format"])["ildgFormat"]

    return Lattice(
        dims={
            "t": int(info["lt"]),
            "x": int(info["lx"]),
            "y": int(info["ly"]),
            "z": int(info["lz"]),
        },
        dofs="QCD",
    )


def get_field_type(records):
    assert "ildg-format" in records, "ildg-format not found"

    import xmltodict

    info = xmltodict.parse(records["ildg-format"])["ildgFormat"]

    if info["field"] in [
        "su3gauge",
    ]:
        return "gauge_links"
    else:
        # TODO
        assert False, "To be implemented"


def get_type(filename, lattice=None, field_type=None, **kwargs):
    records = pylime.scan_file(filename)
    records = {
        r["lime_type"]: (r["data"] if "data" in r else r["data_length"])
        for r in records
    }

    assert "ildg-binary-data" in records, "ildg-binary-data not found"

    read_lattice = None
    try:
        read_lattice = get_lattice(records)
    except AssertionError:
        if not lattice:
            raise

    if lattice and read_lattice:
        assert (
            lattice == read_lattice
        ), "Given lattice not compatible with the one read from file"
    else:
        lattice = read_lattice

    read_field_type = None
    try:
        read_field_type = get_field_type(records)
    except AssertionError:
        if not field_type:
            raise

    if field_type and read_field_type:
        assert (
            field_type == read_field_type
        ), "Given field_type not compatible with the one read from file"
    else:
        field_type = read_field_type

    from lyncs import Field

    import xmltodict

    info = xmltodict.parse(records["ildg-format"])["ildgFormat"]

    field = Field(
        lattice=lattice,
        field_type=field_type,
        dtype="complex%d" % (int(info["precision"]) * 2),
    )

    assert (
        field.byte_size == records["ildg-binary-data"]
    ), """
        Size of deduced field (%s) is not compatible with size of data (%s).
        """ % (
        field.byte_size,
        records["ildg-binary-data"],
    )

    return field


def fixed_options(field, key):
    dims_order = ["t", "z", "y", "x"]

    if key == "axes_order":
        if field.field_type == "gauge_links":
            return dims_order + ["n_dims", "color", "color"]
    elif key == "color_order":
        return [0, 1]
    elif key == "dirs_order":
        return dims_order
    else:
        # TODO
        assert False, "To be implemented"


class file_manager(Tunable):
    def __init__(self, field, **kwargs):
        self.field = field
        from ...tunable import Choice

        self.add_option("lime_engine", Choice(engines))

        for key, val in kwargs.items():
            assert hasattr(self, key), "Attribute %s not found" % key
            setattr(self, key, val)

    @property
    def filename(self):
        return getattr(self, "_filename", None)

    @filename.setter
    def filename(self, value):
        assert isinstance(value, str), "Filename must be a string"
        from os.path import abspath

        self._filename = abspath(value)

    @property
    def engine(self):
        return engine(self.lime_engine)

    def read(self, **kwargs):
        from ...field import Field

        filename = kwargs.get("filename", self.filename)
        field = Field(self.field, zeros_init=True, **self.fixed_options)

        field.field = self.engine.read(
            filename,
            shape=field.field_shape,
            chunks=field.field_chunks,
        )
        return field

    @property
    def fixed_options(self):
        opts = {}
        opts["axes_order"] = fixed_options(self.field, "axes_order")

        if self.field.field_type == "gauge_links":
            opts["color_order"] = fixed_options(self.field, "color_order")
            opts["dirs_order"] = fixed_options(self.field, "dirs_order")
        else:
            # TODO
            assert False, "To be implemented"

        return opts

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token((type(self), self.filename))

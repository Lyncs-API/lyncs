from ...tunable import computable

# Constants
lime_header_size = 144
lime_magic_number = 1164413355
lime_file_version_number = 1
lime_type_length = 128


def read_type(file, type):
    import struct

    data = file.read(struct.calcsize(type))
    return struct.unpack_from(type, data)[0]


def is_lime_file(filename):
    with open(filename, "rb") as f:
        magic_number = read_type(f, ">l")
    return magic_number == lime_magic_number


def scan_file(filename):
    "Scans the content of a lime file and returns the list of records"

    def read_record_data(record):
        "Conditions when the data of a record should be read in this function"
        if record["lime_type"] in [
            "ildg-binary-data",
        ]:
            return False
        elif record["lime_type"] in [
            "ildg-format",
        ]:
            return True
        elif records[-1]["data_length"] < 1000:
            return True
        return False

    import os

    fsize = os.path.getsize(filename)
    records = []
    with open(filename, "rb") as f:
        pos = 0
        while pos + lime_header_size < fsize:
            f.seek(pos)
            records.append(
                dict(
                    pos=pos + lime_header_size,
                    magic_number=read_type(f, ">l"),
                    file_version_number=read_type(f, ">h"),
                    msg_bits=read_type(f, ">h"),
                    data_length=read_type(f, ">q"),
                    lime_type=read_type(f, "%ds" % (lime_type_length,))
                    .decode()
                    .split("\0")[0],
                )
            )
            assert records[-1]["magic_number"] == lime_magic_number
            records[-1]["MBbit"] = (records[-1]["msg_bits"] & (1 << 15)) >> 15
            records[-1]["MEbit"] = (records[-1]["msg_bits"] & (1 << 14)) >> 14
            if read_record_data(records[-1]):
                records[-1]["data"] = read_type(
                    f, "%ds" % (records[-1]["data_length"],)
                ).decode()
            pos = (
                records[-1]["pos"]
                + records[-1]["data_length"]
                + ((8 - records[-1]["data_length"] % 8) % 8)
            )

    return records


def read_chunk(filename, shape, dtype, data_offset, chunks=None, chunk_id=None):
    import numpy as np
    from itertools import product

    shape = np.array(shape)
    chunks = np.array(chunks or shape)
    chunk_id = np.array(chunk_id or np.zeros_like(shape))

    n_chunks = shape // chunks

    start = (
        [
            0,
        ]
        + list(np.where(n_chunks > 1)[0])
    )[-1]
    consecutive = np.prod(chunks[start:])
    n_reads = np.prod(chunks) // consecutive

    if n_reads == 1:
        offset = 0
        for i, l, L in zip(chunk_id, chunks, shape):
            offset = offset * L + i * l
        offset *= dtype.itemsize
        offset += data_offset
        return np.fromfile(
            filename, dtype=dtype, count=consecutive, offset=offset
        ).reshape(chunks)

    arr = np.ndarray(tuple(chunks[:start]) + (consecutive,), dtype=dtype)
    read_ids = list(product(*[range(l) for l in chunks[:start]]))
    assert len(read_ids) == n_reads

    for read_id in read_ids:
        offset = 0
        for i, j, l, L in zip(
            chunk_id,
            read_id + tuple(0 for i in range(len(shape) - start)),
            chunks,
            shape,
        ):
            offset = offset * L + i * l + j
        offset *= dtype.itemsize
        offset += data_offset

        arr[read_id] = np.fromfile(
            filename, dtype=dtype, count=consecutive, offset=offset
        )

    return arr.reshape(chunks)


@computable
def read(filename, shape, chunks):
    from dask.highlevelgraph import HighLevelGraph
    from dask.array.core import normalize_chunks, Array
    from itertools import product
    from ...tunable import delayed
    from numpy import prod, dtype
    import xmltodict

    records = scan_file(filename)
    records = {r["lime_type"]: r for r in records}

    data_record = records["ildg-binary-data"]
    data_offset = data_record["pos"]

    info = xmltodict.parse(records["ildg-format"]["data"])["ildgFormat"]
    dtype = dtype("complex%d" % (int(info["precision"]) * 2))

    assert data_record["data_length"] == prod(shape) * dtype.itemsize

    normal_chunks = normalize_chunks(chunks, shape=shape)
    chunks_id = list(product(*[range(len(bd)) for bd in normal_chunks]))

    reads = [
        delayed(read_chunk)(filename, shape, dtype, data_offset, chunks, chunk_id)
        for chunk_id in chunks_id
    ]

    keys = [(filename, *chunk_id) for chunk_id in chunks_id]
    vals = [read.key for read in reads]
    dsk = dict(zip(keys, vals))

    graph = HighLevelGraph.from_collections(filename, dsk, dependencies=reads)

    return Array(graph, filename, normal_chunks, dtype=dtype)

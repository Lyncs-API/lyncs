__all__ = [
    "Reader",
]

from .lib import lib


class Reader:
    __slots__ = ["filename", "_fp", "_reader", "max_bytes"]

    def __init__(self, filename, max_bytes=64000):
        import os

        assert os.path.isfile(filename)
        self.filename = os.path.abspath(filename)
        self.max_bytes = max_bytes
        self._fp = None
        self._reader = None

    def __del__(self):
        if self._reader is not None:
            self.close()

    @property
    def reader(self):
        if self._reader is None:
            self.open()
        return self._reader

    @property
    def record(self):
        record = dict(
            offset=self.reader.rec_start,
            nbytes=lib.limeReaderBytes(self.reader),
            lime_type=lib.limeReaderType(self.reader),
            bytes_pad=lib.limeReaderPadBytes(self.reader),
            MB_flag=lib.limeReaderMBFlag(self.reader),
            ME_flag=lib.limeReaderMEFlag(self.reader),
        )

        if record["nbytes"] < self.max_bytes:
            from array import array

            nbytes = record["nbytes"]
            arr = array("u", ["\0"] * nbytes)
            read_bytes = array("L", [nbytes])
            status = lib.limeReaderReadData(arr, read_bytes, self.reader)
            try:
                record["data"] = bytes(arr).decode().strip("\0")
            except:
                record["data"] = bytes(arr)

        return record

    def open(self):
        self._fp = lib.fopen(self.filename, "r")
        self._reader = lib.limeCreateReader(self._fp)

    def close(self):
        lib.limeDestroyReader(self._reader)
        lib.fclose(self._fp)
        self._fp = None
        self._reader = None

    def __enter__(self):
        if not self._reader is None:
            self.close()
        self.open()
        return self

    def __exit__(self, typ, value, tb):
        self.close()

    def __len__(self):
        offset = lib.limeGetReaderPointer(self.reader)
        lib.limeSetReaderPointer(self.reader, 0)
        count = 0
        status = lib.limeReaderNextRecord(self.reader)
        while status != lib.LIME_EOF:
            status = lib.limeReaderNextRecord(self.reader)
            count += 1
        status = lib.limeSetReaderPointer(self.reader, offset)
        return count

    def __iter__(self):
        lib.limeSetReaderPointer(self.reader, 0)
        return self

    def __next__(self):
        status = lib.limeReaderNextRecord(self.reader)
        if status != lib.LIME_EOF:
            return self.record
        else:
            raise StopIteration

    def __str__(self):
        rec = 0
        msg = 0
        first = True
        res = ""
        for record in self:
            if not first:
                res += "\n\n"
            if record["MB_flag"] == 1 or first:
                rec = 0
                msg += 1
                first = False
            rec += 1
            res += "Message:        %s\n" % msg
            res += "Record:         %s\n" % rec
            res += "Type:           %s\n" % record["lime_type"]
            res += "Data Length:    %s\n" % record["nbytes"]
            res += "Padding Length: %s\n" % record["bytes_pad"]
            res += "MB flag:        %s\n" % record["MB_flag"]
            res += "ME flag:        %s\n" % record["ME_flag"]
            if "data" not in record:
                res += "Data:           [Long record skipped]\n"
            elif type(record["data"]) is str:
                res += 'Data:           "%s"\n' % record["data"]
            else:
                res += "Data:           [Binary data]\n"
        return res


def main():
    import sys

    assert len(sys.argv) == 2, "Usage: %s <lime_file>" % sys.argv[0]
    return str(Reader(sys.argv[1]))

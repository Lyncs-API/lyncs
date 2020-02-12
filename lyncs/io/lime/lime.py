
# Constants
lime_header_size = 144
lime_magic_number = 1164413355
lime_file_version_number = 1
lime_type_length = 128

def read_type(file, type):
    import struct
    data=file.read(struct.calcsize(type))
    return struct.unpack_from(type,data)[0]


def is_lime_file(filename):
    with open(filename, "rb") as f:
        magic_number = read_type(f,'>l')
    return magic_number == lime_magic_number


def scan_file(filename):
    "Scans the content of a lime file and returns the list of records"
    
    def read_record_data(record):
        "Conditions when the data of a record should be read in this function"
        if record["lime_type"] == "ildg-binary-data":
            return False
        if records[-1]["data_length"] < 1000:
            return True

    import os
    fsize = os.path.getsize(filename)
    records = []
    with open(filename, "rb") as f:
        pos = 0
        while pos+lime_header_size < fsize:
            f.seek(pos)
            records.append( dict(
                pos = pos+lime_header_size,
                magic_number = read_type(f,'>l'),
                file_version_number = read_type(f,'>h'),
                msg_bits = read_type(f,'>h'),
                data_length = read_type(f,'>q'),
                lime_type = read_type(f,'%ds'%(lime_type_length,)).decode().split("\0")[0],
                ))
            assert records[-1]["magic_number"] == lime_magic_number
            records[-1]["MBbit"] = (records[-1]["msg_bits"] & (1 << 15)) >> 15
            records[-1]["MEbit"] = (records[-1]["msg_bits"] & (1 << 14)) >> 14
            if read_record_data(records[-1]):
                records[-1]["data"] = read_type(f,'%ds'%(records[-1]["data_length"],)).decode()
            pos = records[-1]["pos"] + records[-1]["data_length"] + ((8-records[-1]["data_length"]%8)%8)
                
    return records

# Test-fixture-only Compact Protocol rewrite: preserve pages and all other footer fields.
import struct


def rewrite_encoding_lists(data, replacement):
    offset = 0
    edits = []

    def byte():
        nonlocal offset
        value = data[offset]
        offset += 1
        return value

    def varint():
        value = 0
        shift = 0
        while True:
            b = byte()
            value |= (b & 127) << shift
            if not b & 128:
                return value
            shift += 7

    def skip(kind, context, field=False):
        nonlocal offset
        if kind in (1, 2):
            if not field:
                byte()
        elif kind == 3:
            byte()
        elif kind in (4, 5, 6):
            varint()
        elif kind == 7:
            offset += 8
        elif kind == 8:
            length = varint()
            offset += length
        elif kind in (9, 10):
            header = byte()
            count = header >> 4
            if count == 15:
                count = varint()
            for _ in range(count):
                skip(header & 15, context)
        elif kind == 11:
            count = varint()
            if count:
                types = byte()
                for _ in range(count):
                    skip(types >> 4, 0)
                    skip(types & 15, 0)
        elif kind == 12:
            structure(context)
        else:
            raise ValueError(kind)

    def structure(context):
        field_id = 0
        while header := byte():
            delta = header >> 4
            if delta:
                field_id += delta
            else:
                value = varint()
                field_id = (value >> 1) ^ -(value & 1)
            start = offset
            child = {(1, 4): 2, (2, 1): 3, (3, 3): 4}.get((context, field_id), 0)
            skip(header & 15, child, True)
            if context == 4 and field_id == 2:
                edits.append((start, offset))

    structure(1)
    assert offset == len(data)
    for start, end in reversed(edits):
        data = data[:start] + replacement + data[end:]
    assert edits
    return data

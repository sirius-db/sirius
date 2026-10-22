"""Test-only Compact Protocol footer edits; preserve every untouched byte."""


def _rewrite_fields(data, context, field_number, replacement, include_header=False):
    offset = 0
    edits = []
    child_context = {(1, 4): 2, (2, 1): 3, (3, 3): 4, (1, 2): 5}

    def byte():
        nonlocal offset
        value = data[offset]
        offset += 1
        return value

    def varint():
        value = 0
        shift = 0
        while True:
            current = byte()
            value |= (current & 127) << shift
            if not current & 128:
                return value
            shift += 7

    def skip(kind, child=0, field=False):
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
                skip(header & 15, child)
        elif kind == 11:
            count = varint()
            if count:
                types = byte()
                for _ in range(count):
                    skip(types >> 4)
                    skip(types & 15)
        elif kind == 12:
            structure(child)
        else:
            raise ValueError(kind)

    def structure(parent):
        field_id = 0
        while True:
            start = offset
            header = byte()
            if not header:
                break
            delta = header >> 4
            if delta:
                field_id += delta
            else:
                encoded = varint()
                field_id = (encoded >> 1) ^ -(encoded & 1)
            value_start = offset
            skip(header & 15, child_context.get((parent, field_id), 0), True)
            if parent == context and field_id == field_number:
                edits.append((start if include_header else value_start, offset))

    structure(1)
    assert offset == len(data) and edits, (offset, len(data), edits)
    for start, end in reversed(edits):
        data = data[:start] + replacement + data[end:]
    return data


def rewrite_encoding_lists(data, replacement):
    return _rewrite_fields(data, 4, 2, replacement)


def remove_raw_logical_annotation(data):
    # SchemaElement.logicalType is the last field, so removing its header and
    # value leaves the preceding compact field deltas intact.
    return _rewrite_fields(data, 5, 10, b"", include_header=True)

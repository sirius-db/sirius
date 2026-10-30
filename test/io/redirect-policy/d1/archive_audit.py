#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Read GNU ar members without extracting or executing them."""
import hashlib

def members(path):
    result = []
    names = b''
    with path.open('rb') as f:
        if f.read(8) != b'!<arch>\n':
            raise ValueError('expected regular archive')
        while (header := f.read(60)):
            if len(header) != 60 or header[58:] != b'`\n':
                raise ValueError('bad ar header')
            size = int(header[48:58])
            name = header[:16].decode().strip()
            data = f.read(size)
            if len(data) != size:
                raise ValueError('truncated ar member')
            if size % 2:
                f.read(1)
            if name == '//':
                names = data
                continue
            if name in ('/', '/SYM64/'):
                continue
            if name.startswith('/'):
                start = int(name[1:])
                name = names[start:names.index(b'/\n', start)].decode()
            else:
                name = name.rstrip('/')
            result.append((name, hashlib.sha256(data).hexdigest()))
    return result

def verify(original, instrumented):
    (a, b) = (members(original), members(instrumented))
    if [x[0] for x in a] != [x[0] for x in b]:
        raise ValueError('archive members changed')
    changed = [x[0] for (x, y) in zip(a, b) if x != y]
    if changed != ['rest_reactor.cpp.o']:
        raise ValueError('unexpected member differences: ' + repr(changed))
    return {'members': len(a), 'changed': changed, 'unchanged_members': len(a) - 1}

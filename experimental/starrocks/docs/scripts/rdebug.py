"""Parser for Rust {:#?} Debug dumps into Python objects.

struct  -> dict with '__t__' = type name
tuple struct Foo(x) -> {'__t__': 'Foo', '__v__': [x]}  (Some(x) -> x, None -> None)
map {k: v} -> dict (no '__t__')
"""
import re, sys, pickle, os
sys.setrecursionlimit(100000)
TOK = re.compile(r'\s+|("(?:[^"\\]|\\.)*")|([A-Za-z_][A-Za-z0-9_]*)|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|([{}()\[\],:])')

def tokenize(s):
    out = []
    pos = 0
    n = len(s)
    for m in TOK.finditer(s):
        if m.start() != pos:
            raise ValueError(f"bad char at {pos}: {s[pos:pos+40]!r}")
        pos = m.end()
        if m.group(1) is not None: out.append(('s', m.group(1)[1:-1]))
        elif m.group(2) is not None: out.append(('i', m.group(2)))
        elif m.group(3) is not None:
            t = m.group(3)
            out.append(('n', float(t) if ('.' in t or 'e' in t or 'E' in t) else int(t)))
        elif m.group(4) is not None: out.append(('p', m.group(4)))
    if pos != n: raise ValueError("trailing")
    return out

class P:
    def __init__(self, toks): self.t = toks; self.i = 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else (None, None)
    def nxt(self):
        x = self.t[self.i]; self.i += 1; return x
    def expect(self, p):
        k, v = self.nxt()
        if v != p: raise ValueError(f"expected {p} got {v} at {self.i}")
    def value(self):
        k, v = self.nxt()
        if k == 's' or k == 'n': return v
        if k == 'i':
            if v == 'None': return None
            if v == 'true': return True
            if v == 'false': return False
            k2, v2 = self.peek()
            if v2 == '{':
                self.nxt(); d = {'__t__': v}
                while self.peek()[1] != '}':
                    _, name = self.nxt(); self.expect(':'); d[name] = self.value()
                    if self.peek()[1] == ',': self.nxt()
                self.nxt(); return d
            if v2 == '(':
                self.nxt(); items = []
                while self.peek()[1] != ')':
                    items.append(self.value())
                    if self.peek()[1] == ',': self.nxt()
                self.nxt()
                if v == 'Some': return items[0]
                return {'__t__': v, '__v__': items}
            return {'__t__': v}  # unit variant
        if v == '[':
            items = []
            while self.peek()[1] != ']':
                items.append(self.value())
                if self.peek()[1] == ',': self.nxt()
            self.nxt(); return items
        if v == '{':
            d = {}; st = []
            while self.peek()[1] != '}':
                key = self.value()
                if self.peek()[1] == ':':
                    self.nxt(); d[key] = self.value()
                else:
                    st.append(key)
                if self.peek()[1] == ',': self.nxt()
            self.nxt()
            return st if st and not d else d
        if v == '(':
            items = []
            while self.peek()[1] != ')':
                items.append(self.value())
                if self.peek()[1] == ',': self.nxt()
            self.nxt(); return tuple(items)
        raise ValueError(f"unexpected {k} {v} at {self.i}")

def parse_file(path):
    with open(path) as f: s = f.read()
    return P(tokenize(s)).value()

def load_all(d, cache):
    if os.path.exists(cache):
        with open(cache, 'rb') as f: return pickle.load(f)
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.startswith('fragment-') and fn.endswith('.txt'):
            out[fn] = parse_file(os.path.join(d, fn))
    with open(cache, 'wb') as f: pickle.dump(out, f)
    return out

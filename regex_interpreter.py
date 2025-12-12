from dataclasses import dataclass, field
from typing import List, Optional, Set


# --- AST nodes for a very small regex subset ---
@dataclass
class Node:
    pass


@dataclass
class StartAnchor(Node):
    pass


@dataclass
class EndAnchor(Node):
    pass


@dataclass
class Literal(Node):
    text: str


@dataclass
class Dot(Node):
    """Matches any character except newline."""

    pass


@dataclass
class CharClass(Node):
    chars: List[str]
    negated: bool = False


@dataclass
class Sequence(Node):
    children: List["Node"] = field(default_factory=list)


@dataclass
class Quantified(Node):
    child: Node
    quant: str  # one of "?", "+", "*"


@dataclass
class Group(Node):
    child: Node
    capturing: bool
    group_index: Optional[int] = None


@dataclass
class NonCapturingGroup(Node):
    child: Node


# --- Very small regex parser for the subset ---
def _parse_char_class(src: str, i: int):
    assert src[i] == "["
    j = src.find("]", i + 1)
    if j == -1:
        raise ValueError("Unterminated character class")
    body = src[i + 1 : j]
    negated = False
    chars: List[str] = []
    if body.startswith("^"):
        negated = True
        body = body[1:]
    k = 0
    while k < len(body):
        c = body[k]
        if c == "\\" and k + 1 < len(body):
            chars.append(body[k + 1])
            k += 2
        else:
            chars.append(c)
            k += 1
    return CharClass(chars=chars, negated=negated), j + 1


def _parse_group_body(src: str) -> Sequence:
    """
    Parse the inside of (...) or (?:...) where, by assumption,
    there are no nested parentheses.
    """
    children: List[Node] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\\":
            if i + 1 >= n:
                raise ValueError("Dangling escape in group")
            node: Node = Literal(src[i + 1])
            i += 2
        elif c == ".":
            node = Dot()
            i += 1
        elif c == "[":
            node, i = _parse_char_class(src, i)
        else:
            node = Literal(c)
            i += 1

        if i < n and src[i] in "?+*":
            node = Quantified(child=node, quant=src[i])
            i += 1

        children.append(node)
    return Sequence(children)


def parse_pattern(src: str) -> Sequence:
    """
    Parse a limited regex into an AST.
    Supported pieces:
      ^, $
      literals and \"\\x\" escapes
      ., [..], [^x]
      (..), (?:..)
      ?, +, * with your placement restrictions
    """
    src = src[1:-1]  # strip outer parentheses

    i = 0
    n = len(src)
    children: List[Node] = []

    if i < n and src[i] == "^":
        children.append(StartAnchor())
        i += 1

    group_count = 0

    while i < n:
        c = src[i]

        if c == "$" and i == n - 1:
            children.append(EndAnchor())
            i += 1
            break

        if c == "\\":
            if i + 1 >= n:
                raise ValueError("Dangling escape in pattern")
            node: Node = Literal(src[i + 1])
            i += 2
        elif c == ".":
            node = Dot()
            i += 1
        elif c == "[":
            node, i = _parse_char_class(src, i)
        elif c == "(":
            j = src.find(")", i + 1)
            if j == -1:
                raise ValueError("Unterminated group")
            inner = src[i + 1 : j]
            if inner.startswith("?:"):
                body = inner[2:]
                child = _parse_group_body(body)
                node = NonCapturingGroup(child=child)
            else:
                group_count += 1
                body = inner
                child = _parse_group_body(body)
                node = Group(child=child, capturing=True, group_index=group_count)
            i = j + 1
        else:
            node = Literal(c)
            i += 1

        if i < n and src[i] in "?+*":
            node = Quantified(child=node, quant=src[i])
            i += 1

        children.append(node)

    return Sequence(children)


def _coalesce_literals(node: Node) -> Node:
    """
    Merge adjacent Literal nodes into one Literal so that
    we can emit nicer code like \"http\" instead of 'h','t','t','p'.
    """
    if isinstance(node, Sequence):
        new_children: List[Node] = []
        buf: List[str] = []
        for child in node.children:
            child = _coalesce_literals(child)
            if isinstance(child, Literal):
                buf.append(child.text)
            else:
                if buf:
                    new_children.append(Literal("".join(buf)))
                    buf = []
                new_children.append(child)
        if buf:
            new_children.append(Literal("".join(buf)))
        node.children = new_children
    elif isinstance(node, Group):
        node.child = _coalesce_literals(node.child)
    elif isinstance(node, NonCapturingGroup):
        node.child = _coalesce_literals(node.child)
    elif isinstance(node, Quantified):
        node.child = _coalesce_literals(node.child)
    return node


# --- C++/CUDA code generation ---
@dataclass
class CodegenContext:
    indent: int = 0
    lines: List[str] = field(default_factory=list)

    def emit(self, line: str = ""):
        self.lines.append(" " * (self.indent * 4) + line)

    def indent_more(self):
        self.indent += 1

    def indent_less(self):
        self.indent -= 1


def _collect_groups(node: Node, acc: Set[int]):
    if isinstance(node, Group) and node.capturing and node.group_index is not None:
        acc.add(node.group_index)
        _collect_groups(node.child, acc)
    elif isinstance(node, Sequence):
        for c in node.children:
            _collect_groups(c, acc)
    elif isinstance(node, NonCapturingGroup):
        _collect_groups(node.child, acc)
    elif isinstance(node, Quantified):
        _collect_groups(node.child, acc)


def _emit_mismatch(ctx: CodegenContext):
    ctx.emit("{")
    ctx.indent_more()
    ctx.emit("*out = url;")
    ctx.emit("return;")
    ctx.indent_less()
    ctx.emit("}")


def _emit_char_literal(ch: str) -> str:
    if ch == "\\":
        return "'\\\\'"
    if ch == "'":
        return "'\\''"
    return f"'{ch}'"


def _emit_charclass_match(ctx: CodegenContext, cls: CharClass):
    if cls.negated:
        if len(cls.chars) != 1:
            raise NotImplementedError("Only single-char negated classes are supported")
        ch = _emit_char_literal(cls.chars[0])
        ctx.emit(f"if (pos >= static_cast<int32_t>(len) || url[pos] == {ch})")
        _emit_mismatch(ctx)
        ctx.emit("++pos;")
    else:
        conds = [f"url[pos] == {_emit_char_literal(ch)}" for ch in cls.chars]
        cond_str = " || ".join(conds)
        ctx.emit(f"if (pos >= static_cast<int32_t>(len) || !({cond_str}))")
        _emit_mismatch(ctx)
        ctx.emit("++pos;")


def _generate_quantified(ctx: CodegenContext, q: Quantified):
    child = q.child
    quant = q.quant
    ctx.emit(f"// Quantifier {quant}")

    if quant == "?":
        ctx.emit("{")
        ctx.indent_more()
        ctx.emit("int32_t save_pos = pos;")

        if isinstance(child, Literal):
            text = child.text
            if len(text) == 1:
                ch = _emit_char_literal(text)
                ctx.emit(f"if (pos < static_cast<int32_t>(len) && url[pos] == {ch}) {{")
                ctx.indent_more()
                ctx.emit("++pos;")
                ctx.indent_less()
                ctx.emit("} else {")
                ctx.indent_more()
                ctx.emit("pos = save_pos;")
                ctx.indent_less()
                ctx.emit("}")
            else:
                n = len(text)
                ctx.emit(f"if (len - pos >= {n} && ")
                conds = [
                    f"url[pos + {i}] == {_emit_char_literal(ch)}"
                    for i, ch in enumerate(text)
                ]
                ctx.emit("    " + " && ".join(conds) + ") {")
                ctx.indent_more()
                ctx.emit(f"pos += {n};")
                ctx.indent_less()
                ctx.emit("} else {")
                ctx.indent_more()
                ctx.emit("pos = save_pos;")
                ctx.indent_less()
                ctx.emit("}")
        elif isinstance(child, NonCapturingGroup) and isinstance(child.child, Sequence):
            # e.g. (?:www\.)?
            lit = ""
            for c in child.child.children:
                if not isinstance(c, Literal):
                    raise NotImplementedError(
                        "Only literal non-capturing groups are supported for '?'"
                    )
                lit += c.text
            n = len(lit)
            ctx.emit(f"if (len - pos >= {n} && ")
            conds = [
                f"url[pos + {i}] == {_emit_char_literal(ch)}"
                for i, ch in enumerate(lit)
            ]
            ctx.emit("    " + " && ".join(conds) + ") {")
            ctx.indent_more()
            ctx.emit(f"pos += {n};")
            ctx.indent_less()
            ctx.emit("} else {")
            ctx.indent_more()
            ctx.emit("pos = save_pos;")
            ctx.indent_less()
            ctx.emit("}")
        else:
            raise NotImplementedError(
                "Unsupported child for '?' quantifier in this prototype"
            )

        ctx.indent_less()
        ctx.emit("}")

    elif quant in ("+", "*"):
        if isinstance(child, Dot):
            ctx.emit("auto newline_pos = url.find('\\n', pos);")
            if quant == "+":
                ctx.emit("if (pos >= static_cast<int32_t>(len) || newline_pos == pos)")
                _emit_mismatch(ctx)
            ctx.emit("if (newline_pos == cudf::string_view::npos) {")
            ctx.indent_more()
            ctx.emit("pos = static_cast<int32_t>(len);")
            ctx.indent_less()
            ctx.emit("} else {")
            ctx.indent_more()
            ctx.emit("pos = static_cast<int32_t>(newline_pos);")
            ctx.indent_less()
            ctx.emit("}")
        elif isinstance(child, CharClass) and child.negated and len(child.chars) == 1:
            ch = _emit_char_literal(child.chars[0])
            ctx.emit(f"auto stop_pos = url.find({ch}, pos);")
            if quant == "+":
                ctx.emit("if (pos >= static_cast<int32_t>(len) || stop_pos == pos)")
                _emit_mismatch(ctx)
            ctx.emit("if (stop_pos == cudf::string_view::npos) {")
            ctx.indent_more()
            ctx.emit("pos = static_cast<int32_t>(len);")
            ctx.indent_less()
            ctx.emit("} else {")
            ctx.indent_more()
            ctx.emit("pos = static_cast<int32_t>(stop_pos);")
            ctx.indent_less()
            ctx.emit("}")
        elif isinstance(child, Literal) and len(child.text) == 1:
            ch = _emit_char_literal(child.text)
            if quant == "+":
                ctx.emit(f"if (pos >= static_cast<int32_t>(len) || url[pos] != {ch})")
                _emit_mismatch(ctx)
            ctx.emit(f"auto next_pos = url.find_first_not_of({ch}, pos);")
            ctx.emit("if (next_pos == cudf::string_view::npos) {")
            ctx.indent_more()
            ctx.emit("pos = static_cast<int32_t>(len);")
            ctx.indent_less()
            ctx.emit("} else {")
            ctx.indent_more()
            ctx.emit("pos = static_cast<int32_t>(next_pos);")
            ctx.indent_less()
            ctx.emit("}")
        else:
            raise NotImplementedError(
                "Unsupported child for '+' or '*' in this prototype"
            )
    else:
        raise NotImplementedError(f"Unknown quantifier: {quant}")


def _generate_node(ctx: CodegenContext, node: Node):
    if isinstance(node, StartAnchor):
        ctx.emit("// ^ start anchor")
    elif isinstance(node, EndAnchor):
        ctx.emit("// $ end anchor")
        ctx.emit("if (pos != static_cast<int32_t>(len))")
        _emit_mismatch(ctx)
    elif isinstance(node, Literal):
        text = node.text
        n = len(text)
        ctx.emit(f'// Literal "{text}"')
        ctx.emit(f"if (!(len - pos >= {n} && ")
        conds = [
            f"url[pos + {i}] == {_emit_char_literal(ch)}" for i, ch in enumerate(text)
        ]
        ctx.emit("      " + " && ".join(conds) + "))")
        _emit_mismatch(ctx)
        ctx.emit(f"pos += {n};")
    elif isinstance(node, Quantified):
        _generate_quantified(ctx, node)
    elif isinstance(node, NonCapturingGroup):
        ctx.emit("// Non-capturing group")
        _generate_node(ctx, node.child)
    elif isinstance(node, Group):
        if not isinstance(node.child, Sequence):
            raise NotImplementedError(
                "Only sequence groups are supported in this prototype"
            )
        gid = node.group_index
        ctx.emit(f"// Capturing group {gid}")
        ctx.emit(f"g{gid}_start = pos;")
        for child in node.child.children:
            _generate_node(ctx, child)
        ctx.emit(f"g{gid}_end = pos;")
    elif isinstance(node, CharClass):
        ctx.emit("// Character class")
        _emit_charclass_match(ctx, node)
    elif isinstance(node, Dot):
        ctx.emit("// Dot")
        ctx.emit("if (pos >= static_cast<int32_t>(len) || url[pos] == '\\n')")
        _emit_mismatch(ctx)
        ctx.emit("++pos;")
    else:
        raise NotImplementedError(f"Unsupported node: {type(node)}")


def generate_cuda_udf(
    fn_name: str,
    pattern_ast: Sequence,
    replacement: str,
) -> str:
    """
    Generate a CUDA device function that applies the given regex
    and, on success, emits the first capture group for replacement '\\1'.
    For other replacement patterns, this just returns the original string
    (left as a TODO).
    """
    ctx = CodegenContext()

    ctx.emit(
        f"__device__ void {fn_name}(cudf::string_view* out, "
        f"cuda::std::optional<cudf::string_view> const url_opt) {{"
    )
    ctx.indent_more()
    ctx.emit("// Skip null")
    ctx.emit("if (!url_opt.has_value()) {")
    ctx.indent_more()
    ctx.emit("return;")
    ctx.indent_less()
    ctx.emit("}")
    ctx.emit("cudf::string_view url = url_opt.value();")
    ctx.emit("auto len = url.length();")
    ctx.emit("int32_t pos = 0;")

    group_ids: Set[int] = set()
    _collect_groups(pattern_ast, group_ids)
    for gid in sorted(group_ids):
        ctx.emit(f"int32_t g{gid}_start = -1;")
        ctx.emit(f"int32_t g{gid}_end = -1;")

    for node in pattern_ast.children:
        _generate_node(ctx, node)

    ctx.emit("// Build replacement on success")
    replacement = replacement[1:-1]  # strip outer parentheses
    if replacement == r"\1" and 1 in group_ids:
        ctx.emit("if (g1_start >= 0 && g1_end >= g1_start) {")
        ctx.indent_more()
        ctx.emit("*out = url.substr(g1_start, g1_end - g1_start);")
        ctx.indent_less()
        ctx.emit("} else {")
        ctx.indent_more()
        ctx.emit("*out = url;")
        ctx.indent_less()
        ctx.emit("}")
    else:
        ctx.emit("*out = url; // TODO: general replacement handling")

    ctx.indent_less()
    ctx.emit("}")
    return "\n".join(ctx.lines)


if __name__ == "__main__":
    pattern = "(^https?://(?:www\\.)?([^/]+)/.*$)"
    replacement = "(\\1)"
    print("pattern =", pattern)
    print("replacement =", replacement)
    ast = parse_pattern(pattern)
    # print("1) ast =", ast)
    ast = _coalesce_literals(ast)
    # print("2) ast =", ast)
    code = generate_cuda_udf("extract_domain", ast, replacement)
    print(code)

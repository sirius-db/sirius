# Header manifest v1: SHA256 of sorted records decimal-byte-length(name) : name
# lowercase-hex-SHA256(contents) newline The name length and fixed 64-character
# content digest delimit each record. Contents are normalized before BOTH
# embedding and hashing (C++ source newlines).
function(jit_normalize_header input output)
  string(REPLACE "\r\n" "\n" normalized "${input}")
  string(REPLACE "\r" "\n" normalized "${normalized}")
  set(${output}
      "${normalized}"
      PARENT_SCOPE)
endfunction()

function(jit_manifest_record name contents output)
  string(LENGTH "${name}" name_length)
  string(SHA256 content_digest "${contents}")
  set(${output}
      "${name_length}:${name}${content_digest}\n"
      PARENT_SCOPE)
endfunction()

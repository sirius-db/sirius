#pragma once

#include <cudf/types.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// ── bitextract / bitjoin spec parsing ─────────────────────────────────────────
//
// Bit-field layout convention (shared by bitextract and bitjoin):
//
//   Fields are packed MSB-first in the order they appear, both in the DSL
//   line and in a spec like `bitextract_1sign_8exponent_23mantissa`. The
//   first field occupies the top bits of the packed value; the next field
//   the bits immediately below, and so on. If the listed fields cover fewer
//   bits than the packed type's width, the unused high-end bits above the
//   first field are zero (bitjoin) or ignored (bitextract).
//
//   Concretely, for a packed type of width `W` and fields with widths
//   `b_0, b_1, ..., b_{n-1}` listed in DSL order, field `i` occupies bits
//   `[W - sum(b_0..b_i) , W - sum(b_0..b_i) + b_i - 1]` of the packed value.
//
//   Example: `input_3:0, input_7:4 -> bitjoin_u8 -> swapped` packs input
//   bits [3:0] into output bits [7:4] and input bits [7:4] into [3:0].
//
// Packed-type token (where the type lives in each operator's name):
//
//   - bitjoin's packed type is the OUTPUT and is required at the END of
//     the spec:           `bitjoin_<fields>_<type>` or `bitjoin_<type>`
//   - bitextract's packed type is the INPUT and is optional at the FRONT:
//                         `bitextract_[<type>_]<fields>` or alias `bitextract_f{16,32,64}`
//
//   The recognised type tokens are `u8`/`u16`/`u32`/`u64`, `i8`/`i16`/`i32`/`i64`,
//   `f32`/`f64` (with `uint8`/`int8`/... long forms accepted by bitjoin).
//   For bitextract the type token is unambiguous because field tokens always
//   start with a digit (`<bits><name>`).
//
//   When the bitextract type prefix is absent, the spec reconstructs as the
//   smallest unsigned int that fits the sum of field widths. This is enough
//   for symmetric round-trips on unsigned inputs, but loses signedness/width
//   information for signed inputs. Compression therefore auto-injects the
//   actual column type into the stored DSL (see `compress_with_plan`), so
//   `.hpln` files always carry an explicit type prefix and round-trips are
//   exact without any side-channel.

namespace simpatico {

struct bitfield_spec {
  uint32_t bits;
  std::string name;
};

/// Result of parsing a bitextract or bitjoin spec suffix.
struct bitextract_spec_result {
  std::vector<bitfield_spec> fields;
  cudf::data_type output_type{cudf::type_id::EMPTY};
};

/// Compressor-name prefix for the bitextract family ("bitextract_<spec>").
inline constexpr std::string_view kBitextractPrefix = "bitextract_";

/// If `name` starts with `kBitextractPrefix`, return the spec suffix; otherwise nullopt.
std::optional<std::string_view> strip_bitextract_prefix(std::string_view name);

/// IEEE-754 float aliases shared by parse_bitextract_spec and parse_bitjoin_spec.
std::optional<bitextract_spec_result> parse_float_alias(std::string_view suffix);

/// Map a packed-type token (`u8`/`i16`/`f32`/...) to a cudf::data_type.
/// Returns EMPTY if the token is unrecognised. Long forms (`uint8`) are also accepted.
cudf::data_type parse_packed_type_token(std::string_view tok);

/// Canonical "namespace" form of a compressor name, with any optional
/// packed-type prefix stripped from the bitextract spec. Float aliases are
/// preserved because they are part of the operator's identity.
///
/// This is what `parse_plan_dsl` uses to derive output paths, so a bitextract
/// step's downstream paths are stable across DSL forms — i.e. both
/// `bitextract_3hi_5lo` and `bitextract_i16_3hi_5lo` share the namespace
/// `bitextract_3hi_5lo.<field>`. That decouples the auto-injected type prefix
/// (stored in `step.compressor` for type reconstruction) from the path layout
/// (used by the leaves map).
std::string bitextract_canonical_name(std::string const& compressor);

/// Canonicalize a path token by applying `bitextract_canonical_name` to the
/// leftmost dot-separated segment. Lets users reference bitextract outputs
/// downstream in either typed or untyped form interchangeably.
std::string canonicalize_path(std::string const& path);

/// Parse the field-list portion of a bitextract spec ("3hi_5lo", "1sign_8exponent_23mantissa").
/// Each token must be `<digits><name>`. Returns empty fields on malformed input.
std::vector<bitfield_spec> parse_bitfield_list(std::string_view fields_suffix);

/// Parse a bitextract spec. Accepted forms:
///   - alias              "f32" / "f64"
///   - generic            "3hi_5lo"                 (output type defaults to smallest uintN that
///   fits)
///   - typed              "i16_3hi_5lo" / "u32_4a_4b_24c"
/// On failure returns a result with empty fields.
///
/// "f16" is deliberately not an accepted alias here (unlike parse_bitjoin_spec):
/// cudf has no native FLOAT16 storage, so there is no genuine 16-bit-wide
/// column to split — bitextract needs its input to actually be the width the
/// spec assumes, which only holds for f32/f64.
bitextract_spec_result parse_bitextract_spec(std::string_view suffix);

/// Parse a bitjoin spec. Accepted forms:
///   - alias              "f16" / "f32" / "f64"
///   - bare type          "u32"                       (fields provided via DSL bit-range tokens)
///   - typed fields       "1sign_8exponent_23mantissa_f32" / "3hi_5lo_u8"
/// The packed type is the LAST underscore-separated token. On failure returns empty fields.
bitextract_spec_result parse_bitjoin_spec(std::string_view suffix);

}  // namespace simpatico

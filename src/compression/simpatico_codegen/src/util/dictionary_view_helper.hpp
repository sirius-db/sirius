#ifndef SIMPATICO_DICTIONARY_VIEW_HELPER_HPP
#define SIMPATICO_DICTIONARY_VIEW_HELPER_HPP

#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

namespace simpatico {

/// View kind for dictionary column children. Used to get read-only column_view
/// over keys (strings column), keys offsets, keys chars, or indices.
enum class dictionary_view_kind : unsigned {
  Keys        = 0,  ///< Full keys column (strings column)
  KeysOffsets = 1,  ///< Offsets into the keys string data (INT32/size_type)
  KeysChars   = 2,  ///< Raw character data of the keys (UINT8)
  Indices     = 3,  ///< Indices into the dictionary (INT32)
};

/// Returns a read-only column_view for the given dictionary child.
/// \param dict_col Must be a DICTIONARY32 column_view.
/// \param view_kind Which child view to return (Keys, KeysOffsets, KeysChars, Indices).
/// \return column_view over the requested child; valid only while dict_col is valid.
///
/// Uses dictionary_column_view.keys() and .indices() to get the correct children,
/// since the internal child ordering varies between cuDF versions.
/// Uses strings_column_view for accessing offsets from the keys.
/// NOTE: KeysChars returns empty column_view - use get_dictionary_keys_chars_info() instead.
inline cudf::column_view get_dictionary_child_view(cudf::column_view const& dict_col,
                                                   dictionary_view_kind view_kind)
{
  if (dict_col.type().id() != cudf::type_id::DICTIONARY32) { return dict_col; }
  cudf::dictionary_column_view dict_view(dict_col);

  switch (view_kind) {
    case dictionary_view_kind::Keys: return dict_view.keys();
    case dictionary_view_kind::KeysOffsets: {
      // Use strings_column_view to properly access offsets
      cudf::strings_column_view strings_view(dict_view.keys());
      return strings_view.offsets();
    }
    case dictionary_view_kind::KeysChars: {
      // In modern cuDF, chars are not exposed as a column_view child.
      // Return empty view - caller should use get_dictionary_keys_chars_info() instead.
      return cudf::column_view();
    }
    case dictionary_view_kind::Indices: return dict_view.indices();
    default: return dict_col;
  }
}

/// Returns the keys chars data pointer and size for a dictionary column.
/// In modern cuDF, chars are accessed via strings_column_view::chars_begin() not as a column.
/// \param dict_col Must be a DICTIONARY32 column_view.
/// \param stream CUDA stream for accessing chars_size.
/// \return pair of (chars data pointer, chars size in bytes)
inline std::pair<char const*, int64_t> get_dictionary_keys_chars_info(
  cudf::column_view const& dict_col, rmm::cuda_stream_view stream)
{
  if (dict_col.type().id() != cudf::type_id::DICTIONARY32) { return {nullptr, 0}; }
  cudf::dictionary_column_view dict_view(dict_col);
  cudf::strings_column_view strings_view(dict_view.keys());
  return {strings_view.chars_begin(stream), strings_view.chars_size(stream)};
}

}  // namespace simpatico

#endif

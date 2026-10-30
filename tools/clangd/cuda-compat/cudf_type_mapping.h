/* nvcc accepts an unannotated explicit specialization of a __host__ __device__
   function template. Clang requires the specialization to carry the same target
   attributes. libcudf defines every mapping except `char` through this macro,
   and the header keeps the definition only when the macro is not already set. */
#ifndef CUDF_TYPE_MAPPING
#define CUDF_TYPE_MAPPING(Type, Id)                                    \
  template <>                                                          \
  __host__ __device__ constexpr inline type_id base_type_to_id<Type>() \
  {                                                                    \
    return Id;                                                         \
  }                                                                    \
  template <>                                                          \
  inline std::string type_to_name_impl::operator()<Type>()             \
  {                                                                    \
    return CUDF_STRINGIFY(Type);                                       \
  }                                                                    \
  template <>                                                          \
  struct id_to_type_impl<Id> {                                         \
    using type = Type;                                                 \
  };
#endif

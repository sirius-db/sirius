//! Construct and inspect immutable plans for local Parquet scans and filters.
//!
//! Supply the file's schema, build expressions from a scan's columns, then finish
//! the plan with [`Plan::from_root`]. This requires neither file access nor a GPU.
//! Construction checks types and input scopes. Binding the file and checking engine
//! support belong to preparation.

use crate::diagnostic::Diagnostic;
use sirius_sys::c_api as ffi;
use std::{
    ffi::CStr,
    path::Path,
    ptr::{self, NonNull},
};

/// An error from plan construction or inspection. Errors own their diagnostic text
/// and can be sent or shared between threads. Match the variant to handle a category
/// and use [`Display`](std::fmt::Display) for a user-facing message.
///
/// Fallible plan operations report [`AllocationFailure`](Self::AllocationFailure)
/// if allocation fails, [`Internal`](Self::Internal) for internal library errors,
/// and [`Unexpected`](Self::Unexpected) for an unrecognized status. Operations
/// accepting invalid inputs describe their [`InvalidArgument`](Self::InvalidArgument)
/// cases individually. Diagnostic text is best-effort and may be empty.
///
/// # Examples
///
/// ```
/// use sirius::{PlanError, Schema};
///
/// assert!(matches!(Schema::new(&[]), Err(PlanError::InvalidArgument(_))));
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum PlanError {
    /// Invalid field, local path, expression, scope, or operator argument.
    InvalidArgument(String),
    /// The bindings or native library could not allocate required storage.
    AllocationFailure(String),
    /// The native library reported an internal error.
    Internal(String),
    /// An unrecognized native status and its best-effort diagnostic.
    Unexpected {
        /// Status returned by the linked library.
        status: u32,
        /// Diagnostic supplied by the library, or an empty string when unavailable.
        message: String,
    },
}
impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (category, message) = match self {
            Self::InvalidArgument(s) => ("invalid plan argument", s),
            Self::AllocationFailure(s) => ("allocation failed", s),
            Self::Internal(s) => ("internal error", s),
            Self::Unexpected { status, message } => {
                write!(f, "Sirius status {status}")?;
                return if message.is_empty() {
                    Ok(())
                } else {
                    write!(f, ": {message}")
                };
            }
        };
        if message.is_empty() {
            f.write_str(category)
        } else {
            f.write_str(message)
        }
    }
}
impl std::error::Error for PlanError {}

fn check(status: u32, error: Diagnostic) -> Result<(), PlanError> {
    match status {
        ffi::SIRIUS_SUCCESS => Ok(()),
        ffi::SIRIUS_INVALID_ARGUMENT => {
            Err(PlanError::InvalidArgument(error.best_effort_message()))
        }
        ffi::SIRIUS_ALLOCATION_FAILURE => {
            Err(PlanError::AllocationFailure(error.best_effort_message()))
        }
        ffi::SIRIUS_INTERNAL_ERROR => Err(PlanError::Internal(error.best_effort_message())),
        status => Err(PlanError::Unexpected {
            status,
            message: error.best_effort_message(),
        }),
    }
}

// All factories clear the output slot before work and transfer one live reference on success.
fn create<T>(
    call: impl FnOnce(*mut *mut T, *mut *mut ffi::SiriusError) -> u32,
) -> Result<NonNull<T>, PlanError> {
    let mut out = ptr::null_mut();
    let mut error = Diagnostic(ptr::null_mut());
    let status = call(&mut out, &mut error.0);
    check(status, error)?;
    Ok(NonNull::new(out).expect("successful Sirius factory returned a null handle"))
}

macro_rules! immutable_handle {
    ($(#[$doc:meta])* $name:ident, $raw:ident, $retain:ident, $release:ident) => {
        $(#[$doc])*
        #[derive(Debug)]
        pub struct $name(NonNull<ffi::$raw>);
        // SAFETY: the C API permits transferring references and final destruction on any thread.
        unsafe impl Send for $name {}
        // SAFETY: concurrent reads and retains on a live immutable handle are supported.
        unsafe impl Sync for $name {}
        impl Clone for $name {
            fn clone(&self) -> Self {
                // SAFETY: self owns the live reference being retained.
                unsafe { ffi::$retain(self.0.as_ptr()) };
                Self(self.0)
            }
        }
        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: each wrapper releases its single owned reference once.
                unsafe { ffi::$release(self.0.as_ptr()) };
            }
        }
    };
}

immutable_handle!(
    /// The ordered, named fields produced by a plan node. A schema owns its field names.
    /// Cloning retains the same immutable schema. Threads may share and inspect it concurrently.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{DataType, Field, Schema};
    ///
    /// let schema = Schema::new(&[Field {
    ///     name: "id",
    ///     data_type: DataType::Int64,
    ///     nullable: false,
    /// }])?;
    /// assert_eq!(schema.field(0)?.name, "id");
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    Schema, SiriusSchema, sirius_schema_retain, sirius_schema_release);
immutable_handle!(
    /// One physical operator with an output schema and its inputs. Build a scan first,
    /// then attach filters. Each node owns the inputs it needs. Clones share the same
    /// operator and can be used to construct plans concurrently across threads.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{Expression, PlanNode, PlanNodeKind};
    /// # use sirius::{DataType, Field, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// let filtered = scan.filter(&Expression::literal_boolean(true)?)?;
    /// assert_eq!(filtered.kind(), PlanNodeKind::Filter);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    PlanNode, SiriusPlanNode, sirius_plan_node_retain, sirius_plan_node_release);
immutable_handle!(
    /// A typed scalar expression. Literals can be used with any input. A column
    /// retains the particular node it came from, even if another node has an identical
    /// schema. Clones share immutable data that may be used concurrently across threads.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{Comparison, Expression};
    /// # use sirius::{DataType, Field, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// let id = scan.column(0)?;
    /// let positive = id.compare(Comparison::Greater, &Expression::literal_int64(0)?)?;
    /// let filtered = scan.filter(&positive)?;
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    Expression, SiriusExpression, sirius_expression_retain, sirius_expression_release);
immutable_handle!(
    /// A single-root physical plan. The plan retains its operators, expressions, and
    /// metadata. Clones may be inspected concurrently on different threads.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{Plan, PlanNodeKind};
    /// # use sirius::{DataType, Field, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// let plan = Plan::from_root(&scan)?;
    /// assert_eq!(plan.root().kind(), PlanNodeKind::ParquetScan);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    Plan, SiriusPlan, sirius_plan_retain, sirius_plan_release);

/// Primitive type of a field or expression. Nullability is specified separately
/// by [`Field::nullable`] or [`Expression::nullable`]. Safe to send and share.
///
/// # Examples
///
/// ```
/// use sirius::{DataType, Expression};
///
/// assert_eq!(Expression::literal_boolean(true)?.data_type(), DataType::Boolean);
/// # Ok::<(), sirius::PlanError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
pub enum DataType {
    /// A SQL Boolean value.
    Boolean = ffi::SIRIUS_TYPE_BOOLEAN,
    /// A signed 64-bit integer.
    Int64 = ffi::SIRIUS_TYPE_INT64,
}
impl DataType {
    fn from_native(value: u32) -> Self {
        match value {
            ffi::SIRIUS_TYPE_BOOLEAN => Self::Boolean,
            ffi::SIRIUS_TYPE_INT64 => Self::Int64,
            _ => unreachable!("unexpected type in a plan constructed by these bindings"),
        }
    }
}

/// A field descriptor with a borrowed name. [`Schema::new`] copies the descriptor.
/// [`Schema::field`] borrows its name from the schema. A field may be sent or
/// shared across threads for the lifetime of its name.
///
/// # Examples
///
/// ```
/// use sirius::{DataType, Field, Schema};
///
/// let id = Field { name: "id", data_type: DataType::Int64, nullable: false };
/// let schema = Schema::new(&[id])?;
/// assert_eq!(schema.field(0)?, id);
/// # Ok::<(), sirius::PlanError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field<'a> {
    /// Nonempty UTF-8 name without NUL. Names are unique within a schema.
    pub name: &'a str,
    /// Field's primitive type.
    pub data_type: DataType,
    /// Whether this field can contain SQL NULL.
    pub nullable: bool,
}

impl Schema {
    /// Copy fields in declaration order into an owned schema. Field names are compared
    /// byte for byte and are case-sensitive.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError::InvalidArgument`] for an empty schema, duplicate names,
    /// or a name that is empty or contains NUL. Reserving the field descriptors
    /// can return [`PlanError::AllocationFailure`] before calling the native library.
    /// See [`PlanError`] for other failures.
    ///
    /// # Panics
    ///
    /// Panics if the native library reports success with a null handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{DataType, Field, Schema};
    ///
    /// let schema = Schema::new(&[Field {
    ///     name: "id",
    ///     data_type: DataType::Int64,
    ///     nullable: false,
    /// }])?;
    /// assert_eq!(schema.field_count(), 1);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn new(fields: &[Field<'_>]) -> Result<Self, PlanError> {
        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(fields.len())
            .map_err(|_| PlanError::AllocationFailure(String::new()))?;
        descriptors.extend(fields.iter().map(|f| ffi::sirius_field {
            name: f.name.as_ptr().cast(),
            name_size: f.name.len(),
            type_: f.data_type as u32,
            nullable: f.nullable.into(),
        }));
        // SAFETY: descriptors and borrowed names remain readable throughout the call.
        create(|out, err| unsafe {
            ffi::sirius_schema_create(descriptors.as_ptr(), descriptors.len(), out, err)
        })
        .map(Self)
    }
    /// Return the number of fields in declaration order.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{DataType, Field, Schema};
    ///
    /// let schema = Schema::new(&[Field {
    ///     name: "id",
    ///     data_type: DataType::Int64,
    ///     nullable: false,
    /// }])?;
    /// assert_eq!(schema.field_count(), 1);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn field_count(&self) -> usize {
        // SAFETY: self holds a live schema.
        unsafe { ffi::sirius_schema_field_count(self.0.as_ptr()) }
    }
    /// Borrow the field at a zero-based index. Its name is valid for the duration of
    /// the schema borrow.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError::InvalidArgument`] for an out-of-range index.
    /// See [`PlanError`] for other failures.
    ///
    /// # Panics
    ///
    /// Panics if the native library returns an invalid UTF-8 field name or an unknown type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sirius::{DataType, Field, Schema};
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// let id = schema.field(0)?;
    /// assert_eq!(id.name, "id");
    /// assert!(!id.nullable);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn field(&self, index: usize) -> Result<Field<'_>, PlanError> {
        let mut field = std::mem::MaybeUninit::uninit();
        let mut error = Diagnostic(ptr::null_mut());
        // SAFETY: the native function initializes the exclusive output on success.
        let status = unsafe {
            ffi::sirius_schema_field(self.0.as_ptr(), index, field.as_mut_ptr(), &mut error.0)
        };
        check(status, error)?;
        // SAFETY: the call succeeded and self retains the nonempty field name's storage.
        let field = unsafe { field.assume_init() };
        let bytes = unsafe { std::slice::from_raw_parts(field.name.cast(), field.name_size) };
        Ok(Field {
            name: std::str::from_utf8(bytes).expect("Sirius guarantees UTF-8 field names"),
            data_type: DataType::from_native(field.type_),
            nullable: field.nullable != 0,
        })
    }
}

/// Physical operator identified by [`PlanNode::kind`]. Safe to send and share.
///
/// # Examples
///
/// ```
/// use sirius::PlanNodeKind;
/// # use sirius::{DataType, Field, PlanNode, Schema};
///
/// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
/// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
/// assert_eq!(scan.kind(), PlanNodeKind::ParquetScan);
/// # Ok::<(), sirius::PlanError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanNodeKind {
    /// A scan of a single local Parquet file.
    ParquetScan,
    /// A predicate applied to one input node.
    Filter,
}

impl PlanNode {
    /// Describe a scan of one local Parquet file with the supplied output schema.
    /// The caller may obtain the schema from Parquet metadata or already know it.
    /// Construction copies the literal path, retains the schema and leaves the file unopened.
    /// Relative paths are resolved at preparation time. Glob characters are literal.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError::InvalidArgument`] for an empty path, a NUL byte, or a URI
    /// scheme prefix. Use `./data:1.parquet` for a local filename shaped like a URI.
    /// On non-Unix platforms the path must also be representable as UTF-8.
    /// See [`PlanError`] for other failures.
    ///
    /// # Panics
    ///
    /// Panics if the native library reports success with a null handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{DataType, Field, PlanNode, Schema};
    ///
    /// let schema = Schema::new(&[Field {
    ///     name: "id",
    ///     data_type: DataType::Int64,
    ///     nullable: false,
    /// }])?;
    /// let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// assert_eq!(scan.schema().field(0)?.name, "id");
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn scan_parquet(path: impl AsRef<Path>, schema: &Schema) -> Result<Self, PlanError> {
        let path = path.as_ref();
        #[cfg(unix)]
        let path = {
            use std::os::unix::ffi::OsStrExt;
            path.as_os_str().as_bytes()
        };
        #[cfg(not(unix))]
        let path = path
            .to_str()
            .ok_or_else(|| PlanError::InvalidArgument("path is not UTF-8".into()))?
            .as_bytes();
        // SAFETY: the immutable borrowed bytes and schema live through the call.
        create(|out, err| unsafe {
            ffi::sirius_plan_scan_parquet(
                path.as_ptr().cast(),
                path.len(),
                schema.0.as_ptr(),
                out,
                err,
            )
        })
        .map(Self)
    }
    /// Reference a zero-based output column of this node. The expression carries the
    /// column's type, nullability and input node identity.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError::InvalidArgument`] if the index is outside the output schema.
    /// See [`PlanError`] for other failures.
    ///
    /// # Panics
    ///
    /// Panics if the native library reports success with a null handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::DataType;
    /// # use sirius::{Field, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// let id = scan.column(0)?;
    /// assert_eq!(id.data_type(), DataType::Int64);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn column(&self, index: usize) -> Result<Expression, PlanError> {
        // SAFETY: self remains live while the C factory retains its scope.
        create(|out, err| unsafe {
            ffi::sirius_expression_column(self.0.as_ptr(), index, out, err)
        })
        .map(Expression)
    }
    /// Keep rows for which the predicate is SQL TRUE. FALSE and NULL rows are
    /// discarded. The resulting node retains this input and preserves its schema.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError::InvalidArgument`] for a non-Boolean predicate, a predicate
    /// referencing a different input node, or a node chain deeper than 256 levels.
    /// See [`PlanError`] for other failures.
    ///
    /// # Panics
    ///
    /// Panics if the native library reports success with a null handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{Comparison, Expression};
    /// # use sirius::{DataType, Field, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// let id = scan.column(0)?;
    /// let predicate = id.compare(Comparison::Greater, &Expression::literal_int64(10)?)?;
    /// let filtered = scan.filter(&predicate)?;
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn filter(&self, predicate: &Expression) -> Result<Self, PlanError> {
        // SAFETY: both borrowed inputs remain live during construction.
        create(|out, err| unsafe {
            ffi::sirius_plan_filter(self.0.as_ptr(), predicate.0.as_ptr(), out, err)
        })
        .map(Self)
    }
    /// Return the physical operator represented by this node.
    ///
    /// # Panics
    ///
    /// Panics if the native library returns an unknown operator kind.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::PlanNodeKind;
    /// # use sirius::{DataType, Field, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// assert_eq!(scan.kind(), PlanNodeKind::ParquetScan);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn kind(&self) -> PlanNodeKind {
        // SAFETY: self holds a live plan node.
        match unsafe { ffi::sirius_plan_node_get_kind(self.0.as_ptr()) } {
            ffi::SIRIUS_NODE_PARQUET_SCAN => PlanNodeKind::ParquetScan,
            ffi::SIRIUS_NODE_FILTER => PlanNodeKind::Filter,
            _ => unreachable!("unexpected node in a plan constructed by these bindings"),
        }
    }
    /// Return an owning reference to the input of a filter. A scan has no input.
    /// The returned node can outlive this one.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{Expression, PlanNodeKind};
    /// # use sirius::{DataType, Field, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// assert!(scan.input().is_none());
    /// let filtered = scan.filter(&Expression::literal_boolean(true)?)?;
    /// assert_eq!(filtered.input().map(|n| n.kind()), Some(PlanNodeKind::ParquetScan));
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn input(&self) -> Option<Self> {
        // SAFETY: self keeps the borrowed input alive through retain.
        unsafe {
            NonNull::new(ffi::sirius_plan_node_input(self.0.as_ptr())).map(|p| {
                ffi::sirius_plan_node_retain(p.as_ptr());
                Self(p)
            })
        }
    }
    /// Return an owning reference to this node's output schema. The schema remains
    /// available after the node is dropped.
    ///
    /// # Panics
    ///
    /// Panics if the native library returns a node without an output schema.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sirius::{DataType, Field, PlanNode, Schema};
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// let output = scan.schema();
    /// drop(scan);
    /// assert_eq!(output.field(0)?.name, "id");
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn schema(&self) -> Schema {
        // SAFETY: a live node owns its schema, which we retain before returning.
        unsafe {
            let p = NonNull::new(ffi::sirius_plan_node_schema(self.0.as_ptr()))
                .expect("node has a schema");
            ffi::sirius_schema_retain(p.as_ptr());
            Schema(p)
        }
    }
}

/// SQL comparison operator for [`Expression::compare`]. Both operands must have
/// the same type. Boolean operands support equality and inequality. Signed integers
/// support all six operators. Safe to send and share.
///
/// # Examples
///
/// ```
/// use sirius::{Comparison, Expression};
///
/// let ten = Expression::literal_int64(10)?;
/// let twelve = Expression::literal_int64(12)?;
/// let at_least_ten = twelve.compare(Comparison::GreaterEqual, &ten)?;
/// # Ok::<(), sirius::PlanError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
pub enum Comparison {
    /// SQL equality (`=`).
    Equal = ffi::SIRIUS_COMPARISON_EQUAL,
    /// SQL inequality (`<>`).
    NotEqual = ffi::SIRIUS_COMPARISON_NOT_EQUAL,
    /// Strictly less than (`<`).
    Less = ffi::SIRIUS_COMPARISON_LESS,
    /// Less than or equal to (`<=`).
    LessEqual = ffi::SIRIUS_COMPARISON_LESS_EQUAL,
    /// Strictly greater than (`>`).
    Greater = ffi::SIRIUS_COMPARISON_GREATER,
    /// Greater than or equal to (`>=`).
    GreaterEqual = ffi::SIRIUS_COMPARISON_GREATER_EQUAL,
}

impl Expression {
    /// Construct a non-null signed 64-bit literal that can be used with any input.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError`] if the native library cannot construct the expression.
    ///
    /// # Panics
    ///
    /// Panics if the native library reports success with a null handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{DataType, Expression};
    ///
    /// let threshold = Expression::literal_int64(10)?;
    /// assert_eq!(threshold.data_type(), DataType::Int64);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn literal_int64(value: i64) -> Result<Self, PlanError> {
        // SAFETY: create supplies exclusive writable outputs.
        create(|out, err| unsafe { ffi::sirius_expression_literal_int64(value, out, err) })
            .map(Self)
    }
    /// Construct a non-null Boolean literal that can be used with any input.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError`] if the native library cannot construct the expression.
    ///
    /// # Panics
    ///
    /// Panics if the native library reports success with a null handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{DataType, Expression};
    ///
    /// let always = Expression::literal_boolean(true)?;
    /// assert_eq!(always.data_type(), DataType::Boolean);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn literal_boolean(value: bool) -> Result<Self, PlanError> {
        // SAFETY: create supplies exclusive writable outputs.
        create(|out, err| unsafe { ffi::sirius_expression_literal_boolean(value.into(), out, err) })
            .map(Self)
    }
    /// Compare this expression (the left operand) with `right`. The result has
    /// Boolean type and evaluates to NULL if either operand is NULL. A literal can
    /// be compared with a column from any node.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError::InvalidArgument`] for different operand types, ordered
    /// Boolean comparisons, columns from different input nodes, or an expression tree
    /// deeper than 256 levels.
    /// See [`PlanError`] for other failures.
    ///
    /// # Panics
    ///
    /// Panics if the native library reports success with a null handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{Comparison, DataType, Expression};
    /// # use sirius::{Field, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// let id = scan.column(0)?;
    /// let predicate = id.compare(Comparison::Greater, &Expression::literal_int64(10)?)?;
    /// assert_eq!(predicate.data_type(), DataType::Boolean);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn compare(&self, op: Comparison, right: &Self) -> Result<Self, PlanError> {
        // SAFETY: both expression references remain live throughout the call.
        create(|out, err| unsafe {
            ffi::sirius_expression_compare(self.0.as_ptr(), op as u32, right.0.as_ptr(), out, err)
        })
        .map(Self)
    }
    /// Return the type produced by evaluating the expression.
    ///
    /// # Panics
    ///
    /// Panics if the native library returns an unknown expression type.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{DataType, Expression};
    ///
    /// let ten = Expression::literal_int64(10)?;
    /// assert_eq!(ten.data_type(), DataType::Int64);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn data_type(&self) -> DataType {
        // SAFETY: self holds a live expression.
        DataType::from_native(unsafe { ffi::sirius_expression_type(self.0.as_ptr()) })
    }
    /// Return whether this expression may produce SQL NULL. A column inherits the
    /// nullability declared in its input schema.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::Expression;
    ///
    /// assert!(!Expression::literal_int64(10)?.nullable());
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn nullable(&self) -> bool {
        // SAFETY: self holds a live expression.
        unsafe { ffi::sirius_expression_nullable(self.0.as_ptr()) != 0 }
    }
}

impl Plan {
    /// Finish a plan by retaining its root and creating a diagnostic description.
    /// The plan may outlive the root handle passed by the caller.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError`] if the native library cannot construct the plan.
    ///
    /// # Panics
    ///
    /// Panics if the native library reports success with a null handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::Plan;
    /// # use sirius::{DataType, Field, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// let plan = Plan::from_root(&scan)?;
    /// drop(scan);
    /// assert_eq!(plan.root().schema().field_count(), 1);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn from_root(root: &PlanNode) -> Result<Self, PlanError> {
        // SAFETY: root remains live throughout construction.
        create(|out, err| unsafe { ffi::sirius_plan_from_root(root.0.as_ptr(), out, err) })
            .map(Self)
    }
    /// Return an owning reference to the root operator, which remains valid after
    /// the plan is dropped.
    ///
    /// # Panics
    ///
    /// Panics if the native library returns a plan without a root.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::PlanNodeKind;
    /// # use sirius::{DataType, Field, Plan, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// # let plan = Plan::from_root(&scan)?;
    /// let root = plan.root();
    /// drop(plan);
    /// assert_eq!(root.kind(), PlanNodeKind::ParquetScan);
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn root(&self) -> PlanNode {
        // SAFETY: self keeps the borrowed root alive through retain.
        unsafe {
            let p = NonNull::new(ffi::sirius_plan_root(self.0.as_ptr())).expect("plan has a root");
            ffi::sirius_plan_node_retain(p.as_ptr());
            PlanNode(p)
        }
    }
    /// Borrow the plan's human-readable diagnostic text. Names and paths are escaped.
    /// The format can change between releases. Use [`Self::root`] for structured inspection.
    ///
    /// # Panics
    ///
    /// Panics if the native library supplies a description containing invalid UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::Plan;
    /// # use sirius::{DataType, Field, PlanNode, Schema};
    ///
    /// # let schema = Schema::new(&[Field { name: "id", data_type: DataType::Int64, nullable: false }])?;
    /// # let scan = PlanNode::scan_parquet("orders.parquet", &schema)?;
    /// let plan = Plan::from_root(&scan)?;
    /// println!("{}", plan.describe());
    /// # Ok::<(), sirius::PlanError>(())
    /// ```
    pub fn describe(&self) -> &str {
        // SAFETY: self owns the NUL-terminated description until the borrow ends.
        unsafe { CStr::from_ptr(ffi::sirius_plan_describe(self.0.as_ptr())) }
            .to_str()
            .expect("plan descriptions escape names and paths")
    }
}

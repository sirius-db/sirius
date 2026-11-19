//! Immutable physical plan descriptions. Construction copies the path and schema
//! and checks types and expression scopes without opening the file or using a GPU.

use crate::diagnostic::Diagnostic;
use sirius_sys::c_api as ffi;
use std::{
    ffi::CStr,
    path::Path,
    ptr::{self, NonNull},
};

/// A failure to construct or inspect a plan. May be sent or shared between threads.
#[derive(Debug)]
#[non_exhaustive]
pub enum PlanError {
    /// Invalid field, local path, expression, scope, or operator argument.
    InvalidArgument(String),
    /// Native allocation failed. The best-effort diagnostic may be empty.
    AllocationFailure(String),
    /// The native library reported an internal error.
    Internal(String),
    /// An unrecognized native status and its best-effort diagnostic.
    Unexpected { status: u32, message: String },
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
    /// An immutable ordered schema. Owning references can be sent or shared between threads.
    Schema, SiriusSchema, sirius_schema_retain, sirius_schema_release);
immutable_handle!(
    /// An immutable physical operator and its inputs. Can be sent or shared between threads.
    PlanNode, SiriusPlanNode, sirius_plan_node_retain, sirius_plan_node_release);
immutable_handle!(
    /// A typed immutable expression retaining its input scope. Can be sent or shared between threads.
    Expression, SiriusExpression, sirius_expression_retain, sirius_expression_release);
immutable_handle!(
    /// A complete immutable plan owning its root. Can be sent or shared between threads.
    Plan, SiriusPlan, sirius_plan_retain, sirius_plan_release);

/// Primitive type of a schema field or expression. Safe to send and share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
pub enum DataType {
    Boolean = ffi::SIRIUS_TYPE_BOOLEAN,
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

/// A borrowed field description. Safe to send and share while its name remains live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field<'a> {
    /// Nonempty UTF-8 field name.
    pub name: &'a str,
    /// Field's primitive type.
    pub data_type: DataType,
    /// Whether this field can contain SQL NULL.
    pub nullable: bool,
}

impl Schema {
    /// Copy a nonempty sequence of uniquely named fields into a schema.
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
    /// Number of fields in declaration order.
    pub fn field_count(&self) -> usize {
        // SAFETY: self holds a live schema.
        unsafe { ffi::sirius_schema_field_count(self.0.as_ptr()) }
    }
    /// Borrow a field at a zero-based index. The name remains valid for this schema's lifetime.
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

/// Physical operator kind. Safe to send and share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanNodeKind {
    ParquetScan,
    Filter,
}

impl PlanNode {
    /// Describe a local Parquet file with the caller's schema. No file is opened.
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
    /// Reference a zero-based column scoped to this specific input.
    pub fn column(&self, index: usize) -> Result<Expression, PlanError> {
        // SAFETY: self remains live while the C factory retains its scope.
        create(|out, err| unsafe {
            ffi::sirius_expression_column(self.0.as_ptr(), index, out, err)
        })
        .map(Expression)
    }
    /// Construct a filter that keeps rows where the Boolean predicate is true.
    pub fn filter(&self, predicate: &Expression) -> Result<Self, PlanError> {
        // SAFETY: both borrowed inputs remain live during construction.
        create(|out, err| unsafe {
            ffi::sirius_plan_filter(self.0.as_ptr(), predicate.0.as_ptr(), out, err)
        })
        .map(Self)
    }
    /// Inspect the physical operator.
    pub fn kind(&self) -> PlanNodeKind {
        // SAFETY: self holds a live plan node.
        match unsafe { ffi::sirius_plan_node_get_kind(self.0.as_ptr()) } {
            ffi::SIRIUS_NODE_PARQUET_SCAN => PlanNodeKind::ParquetScan,
            ffi::SIRIUS_NODE_FILTER => PlanNodeKind::Filter,
            _ => unreachable!("unexpected node in a plan constructed by these bindings"),
        }
    }
    /// Return an owned reference to the input of a filter, or None for a scan.
    pub fn input(&self) -> Option<Self> {
        // SAFETY: self keeps the borrowed input alive through retain.
        unsafe {
            NonNull::new(ffi::sirius_plan_node_input(self.0.as_ptr())).map(|p| {
                ffi::sirius_plan_node_retain(p.as_ptr());
                Self(p)
            })
        }
    }
    /// Return an owned reference to the output schema.
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

/// Typed SQL comparison. Safe to send and share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
pub enum Comparison {
    Equal = ffi::SIRIUS_COMPARISON_EQUAL,
    NotEqual = ffi::SIRIUS_COMPARISON_NOT_EQUAL,
    Less = ffi::SIRIUS_COMPARISON_LESS,
    LessEqual = ffi::SIRIUS_COMPARISON_LESS_EQUAL,
    Greater = ffi::SIRIUS_COMPARISON_GREATER,
    GreaterEqual = ffi::SIRIUS_COMPARISON_GREATER_EQUAL,
}

impl Expression {
    /// Create an unscoped signed 64-bit literal.
    pub fn literal_int64(value: i64) -> Result<Self, PlanError> {
        // SAFETY: create supplies exclusive writable outputs.
        create(|out, err| unsafe { ffi::sirius_expression_literal_int64(value, out, err) })
            .map(Self)
    }
    /// Create an unscoped Boolean literal.
    pub fn literal_boolean(value: bool) -> Result<Self, PlanError> {
        // SAFETY: create supplies exclusive writable outputs.
        create(|out, err| unsafe { ffi::sirius_expression_literal_boolean(value.into(), out, err) })
            .map(Self)
    }
    /// Compare two expressions of compatible type and scope.
    pub fn compare(&self, op: Comparison, right: &Self) -> Result<Self, PlanError> {
        // SAFETY: both expression references remain live throughout the call.
        create(|out, err| unsafe {
            ffi::sirius_expression_compare(self.0.as_ptr(), op as u32, right.0.as_ptr(), out, err)
        })
        .map(Self)
    }
    /// Result type of this expression.
    pub fn data_type(&self) -> DataType {
        // SAFETY: self holds a live expression.
        DataType::from_native(unsafe { ffi::sirius_expression_type(self.0.as_ptr()) })
    }
    /// Whether this expression may return SQL NULL.
    pub fn nullable(&self) -> bool {
        // SAFETY: self holds a live expression.
        unsafe { ffi::sirius_expression_nullable(self.0.as_ptr()) != 0 }
    }
}

impl Plan {
    /// Finish constructing a plan by retaining its root.
    pub fn from_root(root: &PlanNode) -> Result<Self, PlanError> {
        // SAFETY: root remains live throughout construction.
        create(|out, err| unsafe { ffi::sirius_plan_from_root(root.0.as_ptr(), out, err) })
            .map(Self)
    }
    /// Return an owned reference to the root.
    pub fn root(&self) -> PlanNode {
        // SAFETY: self keeps the borrowed root alive through retain.
        unsafe {
            let p = NonNull::new(ffi::sirius_plan_root(self.0.as_ptr())).expect("plan has a root");
            ffi::sirius_plan_node_retain(p.as_ptr());
            PlanNode(p)
        }
    }
    /// Borrow the diagnostic description. Its format can change between releases.
    pub fn describe(&self) -> &str {
        // SAFETY: self owns the NUL-terminated description until the borrow ends.
        unsafe { CStr::from_ptr(ffi::sirius_plan_describe(self.0.as_ptr())) }
            .to_str()
            .expect("plan descriptions escape names and paths")
    }
}

//! Owned handles for the public plan and schema functions. Cloning retains a native
//! reference. Borrowed metadata lives as long as its owning handle.

use super::{Error, checked, factory};
use crate::c_api;
use std::{ffi::CStr, ptr::NonNull};

immutable_handle!(
    /// An owning reference to an immutable ordered field schema. Safe to send and share.
    #[derive(Debug)]
    Schema,
    c_api::SiriusSchema,
    c_api::sirius_schema_retain,
    c_api::sirius_schema_release
);

immutable_handle!(
    /// An owning reference to an immutable plan operator. Safe to send and share.
    #[derive(Debug)]
    PlanNode,
    c_api::SiriusPlanNode,
    c_api::sirius_plan_node_retain,
    c_api::sirius_plan_node_release
);

immutable_handle!(
    /// An owning reference to an immutable typed expression. Safe to send and share.
    #[derive(Debug)]
    Expression,
    c_api::SiriusExpression,
    c_api::sirius_expression_retain,
    c_api::sirius_expression_release
);

immutable_handle!(
    /// An owning reference to an immutable single-root plan. Safe to send and share.
    #[derive(Debug)]
    Plan,
    c_api::SiriusPlan,
    c_api::sirius_plan_retain,
    c_api::sirius_plan_release
);

/// A borrowed field descriptor. The type code uses the public C ABI constants.
#[derive(Debug, Clone, Copy)]
pub struct Field<'a> {
    /// Name copied on creation or borrowed from the inspected schema.
    pub name: &'a str,
    /// Native primitive type identifier.
    pub data_type: u32,
    /// Whether the field permits SQL NULL.
    pub nullable: bool,
}

impl Schema {
    /// Copy borrowed descriptors into an owned schema.
    pub fn new<'a>(fields: impl ExactSizeIterator<Item = Field<'a>>) -> Result<Self, Error> {
        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(fields.len())
            .map_err(|_| Error {
                status: c_api::SIRIUS_ALLOCATION_FAILURE,
                message: String::new(),
            })?;
        descriptors.extend(fields.map(|f| c_api::sirius_field {
            name: f.name.as_ptr().cast(),
            name_size: f.name.len(),
            type_: f.data_type,
            nullable: f.nullable.into(),
        }));
        // SAFETY: descriptors and borrowed names remain readable throughout the call.
        factory(|out, err| unsafe {
            c_api::sirius_schema_create(descriptors.as_ptr(), descriptors.len(), out, err)
        })
        .map(Self)
    }

    /// Return the number of fields.
    pub fn field_count(&self) -> usize {
        // SAFETY: self holds a live schema.
        unsafe { c_api::sirius_schema_field_count(self.0.as_ptr()) }
    }

    /// Borrow a field whose name is backed by this schema.
    pub fn field(&self, index: usize) -> Result<Field<'_>, Error> {
        let mut field = std::mem::MaybeUninit::uninit();
        // SAFETY: the native function initializes the exclusive field output on success.
        checked(|error| unsafe {
            c_api::sirius_schema_field(self.0.as_ptr(), index, field.as_mut_ptr(), error)
        })?;
        // SAFETY: the call succeeded and self retains the nonempty field name's storage.
        let field = unsafe { field.assume_init() };
        // SAFETY: the schema owns a non-null name of this length and is borrowed by the result.
        let bytes = unsafe { std::slice::from_raw_parts(field.name.cast(), field.name_size) };
        Ok(Field {
            name: std::str::from_utf8(bytes).expect("Sirius guarantees UTF-8 field names"),
            data_type: field.type_,
            nullable: field.nullable != 0,
        })
    }
}

impl PlanNode {
    /// Create a local Parquet scan using a native-byte path and an explicit schema.
    pub fn scan_parquet(path: &[u8], schema: &Schema) -> Result<Self, Error> {
        // SAFETY: the immutable borrowed bytes and schema live through the call.
        factory(|out, err| unsafe {
            c_api::sirius_plan_scan_parquet(
                path.as_ptr().cast(),
                path.len(),
                schema.0.as_ptr(),
                out,
                err,
            )
        })
        .map(Self)
    }

    /// Reference a zero-based output column.
    pub fn column(&self, index: usize) -> Result<Expression, Error> {
        // SAFETY: self remains live while the C factory retains its scope.
        factory(|out, err| unsafe {
            c_api::sirius_expression_column(self.0.as_ptr(), index, out, err)
        })
        .map(Expression)
    }

    /// Create a filter that retains its input and predicate.
    pub fn filter(&self, predicate: &Expression) -> Result<Self, Error> {
        // SAFETY: both borrowed inputs remain live during construction.
        factory(|out, err| unsafe {
            c_api::sirius_plan_filter(self.0.as_ptr(), predicate.0.as_ptr(), out, err)
        })
        .map(Self)
    }

    /// Return the native operator kind identifier.
    pub fn kind(&self) -> u32 {
        // SAFETY: self holds a live plan node.
        unsafe { c_api::sirius_plan_node_get_kind(self.0.as_ptr()) }
    }

    /// Retain the input node, if present.
    pub fn input(&self) -> Option<Self> {
        // SAFETY: self keeps the borrowed input alive through retain.
        unsafe {
            NonNull::new(c_api::sirius_plan_node_input(self.0.as_ptr())).map(|p| {
                c_api::sirius_plan_node_retain(p.as_ptr());
                Self(p)
            })
        }
    }

    /// Retain the output schema.
    pub fn schema(&self) -> Schema {
        // SAFETY: a live node owns its schema, which we retain before returning.
        unsafe {
            let p = NonNull::new(c_api::sirius_plan_node_schema(self.0.as_ptr()))
                .expect("node has a schema");
            c_api::sirius_schema_retain(p.as_ptr());
            Schema(p)
        }
    }
}

impl Expression {
    /// Create a signed 64-bit literal.
    pub fn literal_int64(value: i64) -> Result<Self, Error> {
        // SAFETY: create supplies exclusive writable outputs.
        factory(|out, err| unsafe { c_api::sirius_expression_literal_int64(value, out, err) })
            .map(Self)
    }

    /// Create a Boolean literal.
    pub fn literal_boolean(value: bool) -> Result<Self, Error> {
        // SAFETY: create supplies exclusive writable outputs.
        factory(|out, err| unsafe {
            c_api::sirius_expression_literal_boolean(value.into(), out, err)
        })
        .map(Self)
    }

    /// Compare two expressions using a native comparison identifier.
    pub fn compare(&self, op: u32, right: &Self) -> Result<Self, Error> {
        // SAFETY: both expression references remain live throughout the call.
        factory(|out, err| unsafe {
            c_api::sirius_expression_compare(self.0.as_ptr(), op, right.0.as_ptr(), out, err)
        })
        .map(Self)
    }

    /// Return the native result type identifier.
    pub fn data_type(&self) -> u32 {
        // SAFETY: self holds a live expression.
        unsafe { c_api::sirius_expression_type(self.0.as_ptr()) }
    }

    /// Return whether the expression may produce SQL NULL.
    pub fn nullable(&self) -> bool {
        // SAFETY: self holds a live expression.
        unsafe { c_api::sirius_expression_nullable(self.0.as_ptr()) != 0 }
    }
}

impl Plan {
    /// Create a plan retaining its root.
    pub fn from_root(root: &PlanNode) -> Result<Self, Error> {
        // SAFETY: root remains live throughout construction.
        factory(|out, err| unsafe { c_api::sirius_plan_from_root(root.0.as_ptr(), out, err) })
            .map(Self)
    }

    /// Retain the root operator.
    pub fn root(&self) -> PlanNode {
        // SAFETY: self keeps the borrowed root alive through retain.
        unsafe {
            let p =
                NonNull::new(c_api::sirius_plan_root(self.0.as_ptr())).expect("plan has a root");
            c_api::sirius_plan_node_retain(p.as_ptr());
            PlanNode(p)
        }
    }

    /// Borrow diagnostic text for the lifetime of the plan.
    pub fn describe(&self) -> &str {
        // SAFETY: self owns the NUL-terminated description until the borrow ends.
        unsafe { CStr::from_ptr(c_api::sirius_plan_describe(self.0.as_ptr())) }
            .to_str()
            .expect("plan descriptions escape names and paths")
    }
}

use doris_thrift::exprs::TExprNodeType;
use doris_thrift::plan_nodes::TPlanNodeType;
use doris_thrift::types::{TPrimitiveType, TTypeNodeType};

/// Translator result alias.
pub type Result<T> = std::result::Result<T, TranslateError>;

/// Structured failures emitted by the Doris-to-Substrait translator.
///
/// Marked `#[non_exhaustive]`: as the supported plan/expression/type surface
/// grows, new variants can be added without a breaking change, so downstream
/// matches must include a wildcard arm.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TranslateError {
    /// A Thrift field required for this conversion slice was absent.
    #[error("{context} missing required field {field}")]
    MissingField {
        /// Thrift struct or translation context where the field was expected.
        context: &'static str,
        /// Required field name.
        field: &'static str,
    },
    /// The input plan or expression is internally inconsistent.
    #[error("malformed plan: {0}")]
    MalformedPlan(String),
    /// Descriptor-table lookup or consistency failure.
    #[error("descriptor error: {0}")]
    Descriptor(String),
    /// A Doris plan node is outside the supported translation slice.
    #[error("unsupported plan node {node_type:?} at node {node_id}: {reason}")]
    UnsupportedPlanNode {
        /// Doris plan-node id.
        node_id: i32,
        /// Doris plan-node type.
        node_type: TPlanNodeType,
        /// Short unsupported reason.
        reason: &'static str,
    },
    /// A file scan range is outside the supported slice (e.g. a non-parquet
    /// format, or byte-range splits that do not tile their file).
    #[error("unsupported scan range at node {node_id}: {reason}")]
    UnsupportedScanRange {
        /// Doris plan-node id of the scan.
        node_id: i32,
        /// Short unsupported reason.
        reason: &'static str,
    },
    /// A Doris expression node is outside the supported translation slice.
    #[error("unsupported expression node {node_type:?}: {reason}")]
    UnsupportedExpression {
        /// Doris expression-node type.
        node_type: TExprNodeType,
        /// Short unsupported reason.
        reason: &'static str,
    },
    /// A Doris type is outside the supported Substrait mappings.
    #[error("unsupported type primitive={primitive:?} node={node_type:?}: {reason}")]
    UnsupportedType {
        /// Primitive type when the failure comes from a scalar type.
        primitive: Option<TPrimitiveType>,
        /// Type-node kind when available.
        node_type: Option<TTypeNodeType>,
        /// Short unsupported reason.
        reason: &'static str,
    },
}

impl TranslateError {
    /// Builds a malformed-plan error from owned or borrowed text.
    pub(crate) fn malformed(message: impl Into<String>) -> Self {
        Self::MalformedPlan(message.into())
    }
}

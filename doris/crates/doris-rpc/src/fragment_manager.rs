//! Fragment lifecycle management.
//!
//! Tracks active fragment executions, handles concurrent submissions,
//! cancellation, and status reporting.

use std::sync::Arc;

use dashmap::DashMap;
use tracing::{info, warn};

/// Unique identifier for a fragment instance (query_id + fragment_instance_id).
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct FragmentInstanceId {
    pub query_hi: i64,
    pub query_lo: i64,
    pub instance_hi: i64,
    pub instance_lo: i64,
}

/// Current state of a fragment execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentState {
    Preparing,
    Running,
    Finished,
    Cancelled,
    Failed,
}

/// Per-fragment execution context.
pub struct FragmentContext {
    pub id: FragmentInstanceId,
    pub state: FragmentState,
    /// Serialized Substrait plan bytes (produced by plan_translator).
    pub substrait_plan: Vec<u8>,
}

/// Manages all active fragment executions.
pub struct FragmentManager {
    fragments: DashMap<FragmentInstanceId, Arc<FragmentContext>>,
}

impl FragmentManager {
    pub fn new() -> Self {
        Self {
            fragments: DashMap::new(),
        }
    }

    /// Register a new fragment for execution.
    pub fn submit(
        &self,
        id: FragmentInstanceId,
        substrait_plan: Vec<u8>,
    ) -> Arc<FragmentContext> {
        let ctx = Arc::new(FragmentContext {
            id: id.clone(),
            state: FragmentState::Preparing,
            substrait_plan,
        });
        self.fragments.insert(id, ctx.clone());
        ctx
    }

    /// Cancel a fragment by query/instance id.
    pub fn cancel(&self, id: &FragmentInstanceId) -> bool {
        if let Some((_, _ctx)) = self.fragments.remove(id) {
            info!(?id, "cancelled fragment");
            true
        } else {
            warn!(?id, "cancel: fragment not found");
            false
        }
    }

    /// Cancel all fragments for a given query.
    pub fn cancel_query(&self, query_hi: i64, query_lo: i64) {
        self.fragments.retain(|id, _| {
            id.query_hi != query_hi || id.query_lo != query_lo
        });
    }

    /// Get the number of active fragments.
    pub fn active_count(&self) -> usize {
        self.fragments.len()
    }
}

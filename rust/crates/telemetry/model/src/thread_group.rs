//! ThreadGroup entity: a generic resource group for bucketing threads.
//!
//! Used for the per-(device, thread type) buckets under each
//! [`crate::gpu_device`] group (instance names `executor_thread`,
//! `task_manager_loop_thread`) and for the engine-level `shared` group that
//! holds threads with no single GPU (e.g. the task scheduler thread).

use quent_model::serde::{Deserialize, Serialize};
use quent_model::{Attributes, entity};
use uuid::Uuid;

#[derive(Debug, Attributes, Serialize, Deserialize)]
#[serde(crate = "quent_model::serde")]
pub struct Declaration {
    /// Human-readable group label shown in the viewer, e.g. "executor_thread".
    pub instance_name: String,
    /// The id of the resource group this bucket is nested under
    /// (a gpu_device group or the engine).
    pub parent_group_id: Uuid,
}

entity! {
    ThreadGroup: ResourceGroup {
        declaration: declaration,
        events: {
            declaration: Declaration,
        },
    }
}

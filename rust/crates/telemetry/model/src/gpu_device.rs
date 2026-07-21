//! GpuDevice entity: a per-GPU resource group nested under the engine.
//!
//! One instance is declared per GPU at engine startup (instance names
//! `gpu-0`, `gpu-1`, ...). Per-thread-type [`crate::thread_group`] buckets and
//! device-scoped resources (task queues) are parented under it so the viewer
//! renders an `engine -> gpu-N -> thread type -> threads` collapsible tree.

use quent_model::serde::{Deserialize, Serialize};
use quent_model::{Attributes, entity};
use uuid::Uuid;

#[derive(Debug, Attributes, Serialize, Deserialize)]
#[serde(crate = "quent_model::serde")]
pub struct Declaration {
    /// Human-readable group label shown in the viewer, e.g. "gpu-0".
    pub instance_name: String,
    /// The id of the resource group this device belongs to (the engine).
    pub parent_group_id: Uuid,
    /// CUDA device ordinal of this GPU.
    pub ordinal: u32,
}

entity! {
    GpuDevice: ResourceGroup {
        declaration: declaration,
        events: {
            declaration: Declaration,
        },
    }
}

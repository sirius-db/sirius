// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capability inventory for finalized typed storage. Ingestion and semantic
//! finalization deliberately remain explicit in the model builder.
use quent_analyzer::{Entity, RefTreeEntity, resource::Resource};
use uuid::Uuid;

use crate::{boilerplate::*, model::SiriusModel};

macro_rules! with_families {
    ($consumer:ident) => {
        $consumer! {
            resources {
                activity_tracks: QueryActivityTrack,
                memories: Memory,
                channels: Channel,
                task_queues: TaskQueue,
                task_manager_loop_threads: TaskManagerLoopThread,
                executor_threads: ExecutorThread,
            }
            groups { gpu_devices: GpuDevice, thread_groups: ThreadGroup, }
            entities { workers: Worker, query_groups: QueryGroup, queries: Query,
                plans: Plan, operators: Operator, ports: Port, }
            fsms { tasks: Task => Task(sirius_telemetry_store::TaskEvent::TYPE_NAME),
                data_batches: DataBatch => DataBatch(sirius_telemetry_store::DataBatchEvent::TYPE_NAME),
                activities: QueryActivity => Activity(sirius_telemetry_store::QueryActivityEvent::TYPE_NAME), }
        }
    };
}
pub(crate) use with_families;

macro_rules! model_families {
    (resources { $($resource:ident: $rt:ident,)* }
     groups { $($group:ident: $gt:ident,)* }
     entities { $($entity:ident: $et:ident,)* }
     fsms { $($fsm:ident: $ft:ident => $variant:ident($name:path),)* }) => {
        impl SiriusModel {
            pub(crate) fn sirius_resources(&self) -> impl Iterator<Item = &dyn Resource> {
                std::iter::empty() $(.chain(self.$resource.values().map(|r| r as &dyn Resource)))*
            }
            pub(crate) fn sirius_resource(&self, id: Uuid) -> Option<&dyn Resource> {
                $(if let Some(r) = self.$resource.get(&id) { return Some(r); })*
                None
            }
            pub(crate) fn resource_instance_name(&self, id: Uuid) -> Option<&str> {
                $(if let Some(r) = self.$resource.get(&id) { return Some(r.instance_name()); })*
                None
            }
            pub(crate) fn registered_group(&self, id: Uuid) -> Option<&dyn RefTreeEntity> {
                $(if let Some(g) = self.$group.get(&id) { return Some(g); })*
                None
            }
            pub(crate) fn resource_scope_instance_name(&self, id: Uuid) -> Option<&str> {
                self.query_groups.get(&id).and_then(|g| g.data().instance_name.as_deref())
                    $(.or_else(|| self.$group.get(&id).map($gt::instance_name)))*
            }
            pub(crate) fn registered_ref_entities(&self) -> impl Iterator<Item = &dyn RefTreeEntity> {
                std::iter::once(&self.engine as &dyn RefTreeEntity)
                    $(.chain(self.$resource.values().map(|e| e as &dyn RefTreeEntity)))*
                    $(.chain(self.$group.values().map(|e| e as &dyn RefTreeEntity)))*
                    $(.chain(self.$entity.values().map(|e| e as &dyn RefTreeEntity)))*
                    $(.chain(self.$fsm.values().map(|e| e as &dyn RefTreeEntity)))*
            }
            pub(crate) fn registered_ref_entity(&self, id: Uuid) -> Option<&dyn RefTreeEntity> {
                if self.engine.id() == id { return Some(&self.engine); }
                $(if let Some(e) = self.$resource.get(&id) { return Some(e); })*
                $(if let Some(e) = self.$group.get(&id) { return Some(e); })*
                $(if let Some(e) = self.$entity.get(&id) { return Some(e); })*
                $(if let Some(e) = self.$fsm.get(&id) { return Some(e); })*
                None
            }
        }
    };
}
with_families!(model_families);

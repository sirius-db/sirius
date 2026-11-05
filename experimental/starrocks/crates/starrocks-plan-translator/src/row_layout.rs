use crate::descriptor_table::{DescriptorTable, SlotKey};
use crate::error::{Result, TranslateError};

/// One entry per emitted column; `None` denotes a column without a StarRocks binding.
#[derive(Clone, Debug)]
pub(crate) struct RowLayout(Vec<Option<SlotKey>>);

impl RowLayout {
    pub(crate) fn new(columns: impl IntoIterator<Item = Option<SlotKey>>) -> Self {
        Self(columns.into_iter().collect())
    }

    pub(crate) fn from_tuples(desc: &DescriptorTable, tuples: &[i32]) -> Result<Self> {
        let mut columns = Vec::new();
        for &tuple_id in tuples {
            columns.extend(
                desc.materialized_slot_ids(tuple_id)?
                    .into_iter()
                    .map(|slot_id| Some(SlotKey::new(tuple_id, slot_id))),
            );
        }
        Ok(Self(columns))
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn columns(&self) -> impl Iterator<Item = Option<SlotKey>> + '_ {
        self.0.iter().copied()
    }

    pub(crate) fn resolve(&self, key: SlotKey) -> Result<usize> {
        let mut matches = self
            .0
            .iter()
            .enumerate()
            .filter_map(|(index, binding)| (*binding == Some(key)).then_some(index));
        match (matches.next(), matches.next()) {
            (Some(index), None) => Ok(index),
            (None, _) => Err(TranslateError::descriptor(format!(
                "slot {} (tuple {}) is not part of the row layout",
                key.slot_id, key.tuple_id
            ))),
            _ => Err(TranslateError::descriptor(format!(
                "slot {} (tuple {}) is ambiguous in the row layout",
                key.slot_id, key.tuple_id
            ))),
        }
    }

    pub(crate) fn concat(&self, right: &Self) -> Self {
        Self::new(self.columns().chain(right.columns()))
    }

    pub(crate) fn append(&mut self, binding: Option<SlotKey>) {
        self.0.push(binding);
    }

    pub(crate) fn select(&self, mapping: &[i32]) -> Result<Self> {
        mapping
            .iter()
            .map(|&index| {
                usize::try_from(index)
                    .ok()
                    .and_then(|index| self.0.get(index))
                    .copied()
                    .ok_or_else(|| {
                        TranslateError::malformed(format!(
                            "column {index} is outside row layout of width {}",
                            self.len()
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()
            .map(Self)
    }

    pub(crate) fn output_names(&self, desc: &DescriptorTable) -> Result<Vec<String>> {
        self.columns()
            .enumerate()
            .map(|(index, binding)| match binding {
                Some(key) => Ok(desc.slot(key.tuple_id, key.slot_id)?.output_name()),
                None => Ok(format!("expr_{index}")),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporary_columns_count_toward_join_offsets_and_follow_selection() {
        let left_key = SlotKey::new(0, 1);
        let right_key = SlotKey::new(1, 1);
        let mut left = RowLayout::new([Some(left_key)]);
        left.append(None);
        let joined = left.concat(&RowLayout::new([Some(right_key), None]));
        assert_eq!(joined.len(), 4);
        assert_eq!(joined.resolve(right_key).unwrap(), 2);
        let selected = joined.select(&[2, 0]).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected.resolve(right_key).unwrap(), 0);
        assert_eq!(selected.resolve(left_key).unwrap(), 1);
        assert!(joined.select(&[-1]).is_err());
        assert!(joined.select(&[4]).is_err());
    }

    #[test]
    fn repeated_bindings_are_ambiguous_and_anonymous_columns_do_not_bind() {
        let key = SlotKey::new(0, 1);
        let layout = RowLayout::new([Some(key), None]);
        let repeated = layout.select(&[0, 0, 1]).unwrap();
        assert_eq!(repeated.len(), 3);
        assert!(
            repeated
                .resolve(key)
                .unwrap_err()
                .to_string()
                .contains("ambiguous")
        );
        assert!(
            layout
                .resolve(SlotKey::new(0, 2))
                .unwrap_err()
                .to_string()
                .contains("not part of the row layout")
        );
    }
}

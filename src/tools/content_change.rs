use std::ops::Range;

use ropey::Rope;

#[derive(Debug, PartialEq, Eq)]
pub enum Operation {
    Retain(usize),
    Delete(usize),
    Insert(String),
}

#[derive(Debug, PartialEq, Eq)]
pub struct ChangeSet {
    operations: Vec<Operation>,
    len_before: usize,
    len_after: usize,
}

impl ChangeSet {
    pub fn replace(len_before: usize, replacement: String) -> Self {
        let mut changes = Self {
            operations: Vec::with_capacity(2),
            len_before,
            len_after: 0,
        };
        changes.insert(replacement);
        changes.delete(len_before);
        changes
    }

    pub fn from_replacements(
        original: &str,
        replacements: impl IntoIterator<Item = (Range<usize>, String)>,
    ) -> Result<Self, String> {
        let mut changes = Self {
            operations: Vec::new(),
            len_before: original.chars().count(),
            len_after: 0,
        };
        let mut position = 0;

        for (range, replacement) in replacements {
            if range.start > range.end
                || range.end > original.len()
                || !original.is_char_boundary(range.start)
                || !original.is_char_boundary(range.end)
            {
                return Err("replacement range is outside the original contents".into());
            }
            if range.start < position {
                return Err("replacement ranges overlap or are out of order".into());
            }

            changes.retain(original[position..range.start].chars().count());
            if original[range.clone()] == replacement {
                changes.retain(original[range.clone()].chars().count());
            } else {
                changes.insert(replacement);
                changes.delete(original[range.clone()].chars().count());
            }
            position = range.end;
        }

        changes.retain(original[position..].chars().count());
        Ok(changes)
    }

    /// Apply this changeset to a Rope using character offsets.
    pub fn apply(&self, text: &mut Rope) -> bool {
        if text.len_chars() != self.len_before {
            return false;
        }

        let mut position = 0;
        for operation in &self.operations {
            match operation {
                Operation::Retain(length) => position += length,
                Operation::Delete(length) => text.remove(position..position + length),
                Operation::Insert(value) => {
                    text.insert(position, value);
                    position += value.chars().count();
                }
            }
        }

        debug_assert_eq!(text.len_chars(), self.len_after);
        true
    }

    pub fn is_changed(&self) -> bool {
        self.operations
            .iter()
            .any(|operation| matches!(operation, Operation::Delete(_) | Operation::Insert(_)))
    }

    fn retain(&mut self, length: usize) {
        if length == 0 {
            return;
        }
        self.len_after += length;
        match self.operations.last_mut() {
            Some(Operation::Retain(previous)) => *previous += length,
            _ => self.operations.push(Operation::Retain(length)),
        }
    }

    fn delete(&mut self, length: usize) {
        if length == 0 {
            return;
        }
        match self.operations.last_mut() {
            Some(Operation::Delete(previous)) => *previous += length,
            _ => self.operations.push(Operation::Delete(length)),
        }
    }

    fn insert(&mut self, text: String) {
        if text.is_empty() {
            return;
        }
        self.len_after += text.chars().count();
        let new_last = match self.operations.as_mut_slice() {
            [.., Operation::Insert(previous)]
            | [.., Operation::Insert(previous), Operation::Delete(_)] => {
                previous.push_str(&text);
                return;
            }
            [.., last @ Operation::Delete(_)] => std::mem::replace(last, Operation::Insert(text)),
            _ => Operation::Insert(text),
        };
        self.operations.push(new_last);
    }
}

#[cfg(test)]
mod tests {
    use ropey::Rope;

    use super::{ChangeSet, Operation};

    #[test]
    fn applies_disjoint_replacements_in_one_pass() {
        let changes =
            ChangeSet::from_replacements("abcdef", [(1..2, "X".into()), (4..6, "YZ".into())])
                .unwrap();
        let mut text = Rope::from("abcdef");

        assert!(changes.apply(&mut text));
        assert_eq!(text.to_string(), "aXcdYZ");
    }

    #[test]
    fn combines_adjacent_operations_and_omits_no_op_replacements() {
        let changes = ChangeSet::from_replacements(
            "abcdef",
            [(1..2, "b".into()), (2..3, "X".into()), (3..4, "Y".into())],
        )
        .unwrap();

        assert_eq!(
            changes.operations,
            [
                Operation::Retain(2),
                Operation::Insert("XY".into()),
                Operation::Delete(2),
                Operation::Retain(2)
            ]
        );
        let mut text = Rope::from("abcdef");
        assert!(changes.apply(&mut text));
        assert_eq!(text.to_string(), "abXYef");
    }

    #[test]
    fn rejects_overlapping_and_out_of_order_replacements() {
        assert!(
            ChangeSet::from_replacements("abcdef", [(2..4, "x".into()), (3..5, "y".into())],)
                .is_err()
        );
        assert!(ChangeSet::from_replacements("abcdef", [(0..7, "x".into())]).is_err());
        assert!(ChangeSet::from_replacements("é", [(1..2, "x".into())]).is_err());
    }

    #[test]
    fn full_replacement_uses_character_lengths() {
        let changes = ChangeSet::replace(3, "replacement".into());
        let mut text = Rope::from("é🙂a");

        assert!(changes.apply(&mut text));
        assert_eq!(text.to_string(), "replacement");

        let changes = ChangeSet::replace(0, "new file".into());
        let mut text = Rope::new();
        assert!(changes.apply(&mut text));
        assert_eq!(text.to_string(), "new file");
    }

    #[test]
    fn tracks_multibyte_inserted_characters() {
        let changes = ChangeSet::from_replacements("before", [(0..6, "é".into())]).unwrap();
        let mut text = Rope::from("before");

        assert_eq!(changes.len_after, 1);
        assert!(changes.apply(&mut text));
        assert_eq!(text.to_string(), "é");
    }
}

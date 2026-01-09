//! Field selection helpers for CSV records.

/// A mask that indicates which field indices should be kept.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldMask {
    mask: Vec<bool>,
}

impl FieldMask {
    /// Create a mask from a set of indices that should be kept.
    ///
    /// Indices outside the range `0..len` are ignored.
    pub fn from_indices<I>(len: usize, indices: I) -> FieldMask
    where
        I: IntoIterator<Item = usize>,
    {
        let mut mask = vec![false; len];
        for index in indices {
            if let Some(entry) = mask.get_mut(index) {
                *entry = true;
            }
        }
        FieldMask { mask }
    }

    /// Create a mask by evaluating a predicate for each index.
    pub fn from_predicate<F>(len: usize, mut predicate: F) -> FieldMask
    where
        F: FnMut(usize) -> bool,
    {
        let mut mask = Vec::with_capacity(len);
        for index in 0..len {
            mask.push(predicate(index));
        }
        FieldMask { mask }
    }

    /// Return a new mask with each selection inverted.
    pub fn invert(&self) -> FieldMask {
        FieldMask {
            mask: self.mask.iter().map(|selected| !selected).collect(),
        }
    }

    /// Return true if the given index is selected by this mask.
    pub(crate) fn keeps(&self, index: usize) -> bool {
        self.mask.get(index).copied().unwrap_or(false)
    }
}

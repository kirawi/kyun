//! Tables resume from the first cell of the current column group and a wrapped-line number.
//! Cell node IDs select independent boundary caches. No table display-row sequence is retained.

use unicode_width::UnicodeWidthStr;

use super::{
    ColumnCache, RowCache,
    tree::{Element, Tree},
};

pub const MIN_COLUMN_WIDTH: usize = 3;
pub const SEPARATOR: &str = " │ ";
pub const SEPARATOR_WIDTH: usize = 3;

// TP1: For a given table, we can track its state for rendering by simply knowing the first cell that appears for a row.
// The first cell identifies the source row/group; its sibling cells supply their own cached wrapping positions.
// TP2: There needs to be row separators; it's a bit visually confusing right now. This is also possible with the cell
// model if the separators, which can share a border with neighboring cells, are part of each cell.
// These additional source-row separators remain outside this replacement's scope.
impl Tree {
    pub fn table_rows(&self, table: usize) -> impl Iterator<Item = usize> {
        self.children(table)
            .flat_map(|section| self.children(section))
            .filter(|&row| !self.is_open_container(row))
    }

    pub fn cell_location(&self, cell: usize) -> (usize, usize, usize) {
        let path = self.path(cell);
        let table = path
            .iter()
            .copied()
            .find(|&id| matches!(self.nodes[id].data, Element::Table(_)))
            .unwrap();
        let row = path
            .iter()
            .copied()
            .find(|&id| matches!(self.nodes[id].data, Element::TableRow))
            .unwrap();
        (
            table,
            row,
            self.children(row).position(|id| id == cell).unwrap(),
        )
    }

    pub fn columns_per_group(&self, table: usize, width: usize) -> usize {
        let content_width = width.saturating_sub(self.prefix(&self.path(table), &[], None).width());
        if content_width < MIN_COLUMN_WIDTH {
            1
        } else {
            ((content_width + SEPARATOR_WIDTH) / (MIN_COLUMN_WIDTH + SEPARATOR_WIDTH)).max(1)
        }
    }

    pub fn columns<'a>(
        &self,
        source: &str,
        table: usize,
        width: usize,
        cache: &'a mut ColumnCache,
    ) -> &'a [usize] {
        // Possible approach: maintain natural column maxima as rows become visible, then compare assigned widths.
        // With immutable completed rows, appending rows cannot reduce a natural maximum. Allocating limited space
        // can still shrink another column, so growing maxima do not imply that earlier cell wrapping stays valid.
        // Reuse boundaries only where text, assigned width, and prefix context remain unchanged; include row closures.
        cache.entry(table).or_insert_with(|| {
            let Element::Table(alignments) = &self.nodes[table].data else {
                unreachable!()
            };
            let mut natural = vec![MIN_COLUMN_WIDTH; alignments.len()];
            for row in self.table_rows(table) {
                for (column, cell) in self.children(row).enumerate() {
                    natural[column] =
                        natural[column].max(self.content(source, cell).natural_width(0));
                }
            }
            let content_width =
                width.saturating_sub(self.prefix(&self.path(table), &[], None).width());
            let group_size = self.columns_per_group(table, width);
            let mut columns = vec![MIN_COLUMN_WIDTH; natural.len()];
            for start in (0..natural.len()).step_by(group_size) {
                let end = (start + group_size).min(natural.len());
                let mut remaining = content_width
                    .saturating_sub(SEPARATOR_WIDTH * (end - start).saturating_sub(1))
                    .saturating_sub(MIN_COLUMN_WIDTH * (end - start));
                // Minimum-width columns share extra space until their natural widths are reached.
                while remaining > 0 {
                    let active = (start..end)
                        .filter(|&column| columns[column] < natural[column])
                        .collect::<Vec<_>>();
                    if active.is_empty() {
                        break;
                    }
                    let share = (remaining / active.len()).max(1);
                    for column in active {
                        let amount = share.min(natural[column] - columns[column]).min(remaining);
                        columns[column] += amount;
                        remaining -= amount;
                    }
                }
            }
            columns
        })
    }

    pub fn table_row_height(
        &self,
        source: &str,
        first_cell: usize,
        width: usize,
        rows: &mut RowCache,
        columns: &mut ColumnCache,
    ) -> usize {
        // Possible approach for forward navigation: advance the current group's cells together and retain only the
        // boundaries requested. At fixed column widths, this group's height is the maximum wrapped cell height;
        // drawing its first row need not compute that maximum. Backward entry at the last row or an exact height
        // request may still require finishing the group's wrapping. Keep that need separate from boundary retention.
        let (table, row, start) = self.cell_location(first_cell);
        let group_size = self.columns_per_group(table, width);
        let widths = self.columns(source, table, width, columns);
        self.children(row)
            .enumerate()
            .skip(start)
            .take(group_size)
            .map(|(column, cell)| self.cache_rows(source, cell, widths[column], rows).len())
            .max()
            .unwrap()
    }

    pub fn table_height(
        &self,
        source: &str,
        table: usize,
        width: usize,
        rows: &mut RowCache,
        columns: &mut ColumnCache,
    ) -> usize {
        let group_size = self.columns_per_group(table, width);
        let widths = self.columns(source, table, width, columns);
        let mut height = 0;
        for start in (0..widths.len()).step_by(group_size) {
            if start > 0 {
                height += 1;
            }
            for row in self.table_rows(table) {
                height += self
                    .children(row)
                    .enumerate()
                    .skip(start)
                    .take(group_size)
                    .map(|(column, cell)| self.cache_rows(source, cell, widths[column], rows).len())
                    .max()
                    .unwrap();
            }
        }
        height
    }
}

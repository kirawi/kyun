An Algorithm For Soft-Wrapping Text
===================================

General Definitions
-------------------

1.  Throughout this document, width is defined according to `unicode-width (=0.1.12)` (which uses [UAX #11](https://www.unicode.org/reports/tr11/) among other things; but basically for rendering to a terminal; no library can get it "correct" enough but *trust me* it's not really feasible without basically creating a new universe) where for a string `S`
    $$
    \operatorname{width}(S) \;=\; \sum_{g \in \operatorname{graphemes}(S)} w(g)
    $$
2.  Line endings are defined as CR, LF, and CRLF.
3.  Indentation refers to any leading sequence of tabs or spaces, where tabs add some amount of spaces to snap to the next tab stop.
4.  A word refers to any maximal contiguous sequence of graphemes that does not contain a word separator.
5.  A separator run refers to any maximal contiguous sequence of word separators: Unicode whitespace or punctuation, or the `_` character. This set is based on common terminal-based editors for the purposes of primarily working on programming languages (though it may, in the future, be extended to the Unicode segmentation rules or a limited subset for CJK similar to Vim).

---

Let `L` be a language whose tokens provide:
- a sequence of zero or more graphemes to be rendered;
- a rendered width equal to that of — or greater than — the width of the text.

Let `N(L)` be a normalized derivative of language `L`, representing the vocabulary accepted by the wrapping algorithm. `N(L)` excludes `{ \r \n \r\n }` and distinguishes two structural tokens:
- `HardBreak`: representing an explicit line break from the source.
- `SoftBreak`: representing a virtual line break inserted or recognized by the soft-wrapping transformation.

Let `norm(S)` be a deterministic finite-state transduction that maps an input sequence `S` in `L` to a sequence in `N(L)` by: converting all source line endings (`\r`, `\n`, `\r\n`) into `HardBreak` tokens.

Let:
- a real line be a maximal contiguous sequence of tokens in `N(L)` that does not contain a `HardBreak`;
- a virtual line be one of the rendered line segments created when a real line contains one or more `SoftBreak`s.

Let:
- `A` be an input in `N(L)`;
- `B` be a soft-wrapped output;
- `M` be the maximum accepted width of a rendered line;
- `T_M` be the width-aware transduction for a given `M` upon some `A` to produce some `B`;
- `P_M` be the predicate that states that an input is correctly soft-wrapped for width `M`.

**Note 1:** For simplicity, the transformation is intentionally left undefined for `M < 3` (due to continuation indentation) and for any language that contains any token whose excess width (i.e. the rendered width beyond that derived from its text) exceeds `M` (due to need to insert `SoftBreak` which would be impossible with a 100 width excess and M=80).


Definition of `P_M`
-------------------

`P_M(B)` is true IFF all of the following conditions hold:
1.  No rendered line in `B` exceeds width `M`.
2.  For any sequence of virtual lines, all virtual lines excluding the first one must begin with exactly two literal spaces of indentation.
3.  Every `SoftBreak` is necessary, i.e.
    - removing it (and therefore its continuation indentation) would produce a rendered line exceeding `M`;
    - if the text following it begins with a word, then moving that word before the `SoftBreak` would produce a preceding rendered line exceeding `M`.
4.
    - No `SoftBreak` has split a word that is no wider than 40 and would fit on an otherwise empty continuation line.
    - If a word is wider than 40 or would not fit on an otherwise empty continuation line, and placing it in full on the rendered line where it begins would make that line exceed `M`, then a `SoftBreak` occurs inside that word.
5.  If a `SoftBreak` occurs inside a word, then moving the next grapheme of that word before the `SoftBreak` would make the preceding rendered line exceed `M`.

(Note: rules for separator runs are intentionally not explicitly stated for the sake of a focused document, but they are essentially just written until `M` is met.)

Let `Fix(T_M) = { A | T_M(A) = A }`, the set of fixed points of `T_M`.

`P_M` and `T_M` are related by:

```text
P_M(A)  ⟺  A ∈ Fix(T_M)
```

i.e. the inputs satisfying `P_M` are exactly the fixed points of `T_M`

`T_M` is idempotent by the following derivation:
1.  By construction, every output `T_M(A)` satisfies `P_M` (`T_M` only emits correctly wrapped text, per the rules in "General Idea").
2.  By the equivalence above, `P_M(T_M(A))` implies `T_M(A) ∈ Fix(T_M)`.
3.  By definition of `Fix(T_M)`:

```
T_M(T_M(A)) = T_M(A)
```

(Note that T_M is not formally defined, so treat it as a rough sketch. Rules above are given by the informal description of T_M.)


General Idea
------------

Given `A = "hello, world"` and `M = 8`:

```
 hello   |   ,<space>    |   world
WORD_RUN | SEPARATOR_RUN | WORD_RUN
```

where `hello` followed by `,<space>` still fits on the first line while `world` does not, resulting in a final render of:

```
hello,<space>
  world
```

The algorithm *prefers* to consume as many words that can fit on the current line rather than just how many graphemes fit.

For example,

```
hello,<space>
  world
```

is preferred over:

```
hello, w
  orld
```

with the exception of words that are too long and thus must be chunked.

---

A `SoftBreak` may be inserted into the middle of a word (i.e. split) if the entire word does not fit in the remaining space on the current rendered line and either:
- the word is wider than 40, where `40` was chosen arbitrarily;
- the word would not fit on an otherwise empty continuation line.

In such a case:
1. Greedily place graphemes on the line until adding the next one would make the rendered line exceed `M`.
2. Insert a `SoftBreak`.
3. Continue the procedure with the remaining content.


Closing
-------

This algorithm can be most optimally implemented as a left-to-right O(N) iterator with O(N) auxiliary memory (for zero-width graphemes, so usually O(1) in practice) for buffering. Implementation behavior should be derived from the rules given above; this specification is provided as-is in so far as it is concerned with the intended use case of `kyun` (e.g. `L` is underconstrained). This is not intended to be a formal, rigorous proof. It was simply the result of a process in which I separated problem solving from implementation. I'm not a mathematician, and this was really just a fun learning exercise to see what it's sort-of like to formalize an algorithm!

A generic language `L` is used since markdown needs to be rendered.

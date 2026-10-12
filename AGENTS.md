`kyun` is an agentic harness.

**Rules:**
- Follow the user's instructions explicitly and unconditionally.
- Run `cargo fmt` only when prompted. Do not try to format code yourself.
- For informational queries, clearly begin with a succinct and clear answer. Only then may you expand upon it.
- Practice clear, straightforward communication and explain your reasoning. Avoid colorful verbs/nouns (e.g. "minted," "fixture," "orchestrate"), obscure/uncommon metaphors for the given context (e.g., "escape hatch," "algorithm shape"), "quirky personality," and literary prose (e.g., "the honest ledger," "charter", "logical spine"). Exclude also technical terminology that don't relate to the domain occupied by this project (e.g. "provenance", "contract"). Do not invent your own terminology. Prioritize terms like "create" or "use" appropriate for contemporary, standard communication. This applies to both code and chat. To be precise: follow the ASD-STE100 style
- When there is friction between you and the user, clearly state your rationale and stick to evidence-backed claims. Do NOT treat questions by the user as authoritative "hints" or "directions" as to what to do. They are genuine questions prompted to get you to think for yourself. You are required to disagree with the user when the evidence pushes back against the user's claims.
- Reflow any of your added *code comments* to a width of 120 columns. Code will be reflowed separately by the user and is not limited to 120 columns.
- Abstractions should be introduced only when emperical evidence supports their existence (i.e. how code paths develop based on actual requirements) **OR** for the sake of long-term maintanence. Consider design patterns.
- Avoid redundant tests that are vacuously true by construction of what they are testing (e.g. you don't test if `Vec::push` appends an element to a vector). Try to avoid adding permanent tests that are trivial and would require a lot of editing if, say, a style changes. Such tests are generally not useful.
- Prefer `.unwrap()` over `.expect()`.
- Use `pub` instead of `pub(crate)`.
- Retain existing comments; and update them as necessary. Comments should capture what a new reader would reasonably believe or want to know.
- Collect emperical evidence to support theories when reasoning alone does not reveal clear invariants, insights, or generally proofs of adequacy (e.g. by running experiments at `/tmp`, temporary tests) while respecting constraints (e.g. read-only). Temporary tests MUST be prefixed with `temp_`
- Trust that the code is completely owned, i.e. it is not depended upon by an external application. Search for insights or invariants (existing or potentially new ones to establish) to utilize for understanding existing code or writing new code, avoiding overly defensive code.
- Tests DO NOT decide the design of APIs (e.g. should not be based on unreachable application conditions that only exist in module tests).
- Avoid circular reasoning to justify decisions. Never engage in motivated reasoning to defend a preferred solution nor engage in status quo bias to defend existing architecture.
- NEVER make any assumptions about design or feedback. You MUST **stop** and consult the user when there are competing choices. Never silently avoid making a decision either. The user must be kept informed at all times.

**Coding Style Rules:**
- Create functions only when they are deservably used multiple times. Never create one-off functions (i.e. functions called at exactly one location) UNLESS: a) it would be at least 10 lines long and the caller's body is already 40+ lines long, b) it enforces an important and necessary encapsulation boundary between distinct modules. This ensures that state behavior is not obfuscated by distance to where it actually exists. One-off functions, if used, should generally also cover an invariant or abstraction.
- Place complex abstractions in new files if that abstraction is used in multiple files or if the abstraction is >250 lines of code.
- Ensure variables are initialized only when needed. This is to avoid unnecessarily wasting CPU registers.
- Limit single-use variables (i.e. variables used in only one location) *unless:* necessary for correctness/logic (e.g. snapshot), the call site would otherwise be too complex, or the purpose of the inlined variable content would be unclear (i.e. the variable serves a documentation purpose). Try to err on the side of caution about using alias variables (e.g. shortening field access is not a valid reason unless it's used a lot and is significantly shorter).

# Coding conventions

## Python

- Write code that is clearly correct and self-documenting.
- Use type hints to assert the intended input types.
- Use assertions for guards against edge cases that may happen, with an explanatory message: `assert condition, "error message"`.
- When several operations can produce similar errors, use a single or a few focused `try` blocks with a shared error message or exception object carried through the relevant code. Set that message or exception in a variable before the code it describes so the failure context documents the operations below.

## TypeScript

- Avoid monolithic files. Separate components and responsibilities logically.
- When prop drilling becomes an issue, use a Context Provider. Consider other state management libraries for more complex cases.

## Formatting

- Whitespace and specific formatting are handled by a formatter; formatter settings will be supplied separately.
- Do not change the compactness of otherwise appropriate code merely for formatting preferences. Prioritize readability, clean structure, and clear correctness.

---
name: test-gen
description: Generate comprehensive unit tests for functions, classes, and modules
version: 1.0.0
tags: [testing, code, quality]
requires: [read_file, write_file]
---

# Test Generation

Generate comprehensive unit tests that verify correctness and catch edge cases.

## Instructions

When generating tests:

1. **Analyze the code under test** - Understand inputs, outputs, and side effects
2. **Identify test cases**:
   - Happy path (normal operation)
   - Edge cases (empty input, max values, boundary conditions)
   - Error cases (invalid input, exceptions)
   - Integration points (mocks for external dependencies)
3. **Write tests following best practices**:
   - One assertion per test (when practical)
   - Descriptive test names
   - Arrange-Act-Assert pattern
   - Proper setup and teardown
4. **Match the project's testing framework** (pytest, unittest, jest, etc.)

## Output Format

Generate tests in the appropriate format for the language:

Python (pytest):
```python
def test_function_name_scenario():
    # Arrange
    input_data = ...

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected
```

JavaScript (Jest):
```javascript
describe('FunctionName', () => {
    it('should handle normal input', () => {
        // Arrange
        const input = ...;

        // Act
        const result = functionUnderTest(input);

        // Assert
        expect(result).toBe(expected);
    });
});
```

## Examples

- Generate tests for a string parsing function
- Create tests for a database repository class
- Write tests for an API endpoint handler
- Test async functions with mocked dependencies

## Guidelines

- Cover at least 80% of code paths
- Test error handling, not just success paths
- Use meaningful test data, not random values
- Mock external dependencies (databases, APIs, file system)
- Keep tests independent and isolated
- Follow the project's existing test patterns

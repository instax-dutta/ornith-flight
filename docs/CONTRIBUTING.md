# Contributing to Ornith-Flight

Thank you for your interest in contributing! This project aims to make large MoE models accessible on consumer hardware.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/ornith-flight.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test thoroughly: `python3 research/test_suite.py --test all`
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## Code Style

This project follows [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### Key Points
- Use type hints for all functions
- Write comprehensive docstrings (Args, Returns, Side effects)
- Maximum line length: 100 characters
- Use f-strings for formatting
- Extract magic numbers to constants

### Example
```python
def calculate_hit_rate(hits: int, misses: int) -> float:
    """Calculate cache hit rate from hit and miss counts.
    
    Args:
        hits: Number of cache hits
        misses: Number of cache misses
    
    Returns:
        Hit rate as a fraction between 0.0 and 1.0
    
    Raises:
        ValueError: If hits or misses are negative
    """
    if hits < 0 or misses < 0:
        raise ValueError("Hits and misses must be non-negative")
    
    total = hits + misses
    return hits / total if total > 0 else 0.0
```

## Testing

All changes must include tests:

```bash
# Run all tests
python3 research/test_suite.py --test all

# Run specific test category
python3 research/test_suite.py --test cache
python3 research/test_suite.py --test edge
```

### Writing Tests
- Test edge cases (empty inputs, zero values, boundary conditions)
- Use descriptive test names
- Add docstrings explaining what's being tested
- Ensure tests are deterministic

## Code Review Process

We follow [Google's Code Review Guidelines](https://google.github.io/eng-practices/review/):

### For Authors
1. Keep changes small and focused (one logical change per PR)
2. Write clear PR descriptions explaining the "why"
3. Respond promptly to reviewer comments
4. Don't take feedback personally - it's about the code

### For Reviewers
1. Respond within one business day
2. Focus on design, functionality, complexity, tests, naming
3. Be kind and explain your reasoning
4. Approve once the change improves code health (doesn't need to be perfect)

## Areas for Contribution

### High Priority
- **C Implementation (Phase 1):** Metal backend for M2
- **CUDA Backend (Phase 2):** NVIDIA GPU support
- **Documentation:** Tutorials, examples, architecture diagrams

### Medium Priority
- **Per-Layer Caching:** Improve hit rates with layer-specific caches
- **Expert Pruning:** Remove least-used experts to reduce total count
- **Better Quantization:** Explore 3-bit, 2-bit with group quantization

### Low Priority
- **Performance Profiling:** Real hardware benchmarks
- **Model Support:** Adapt to other MoE architectures
- **UI/Tools:** Dashboard for monitoring inference

## Commit Messages

Follow the conventional commit format:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `perf:` Performance improvements
- `chore:` Maintenance tasks

**Example:**
```
feat: add per-layer cache strategy

Implement layer-specific LRU caches to improve hit rates from 12% to
18% on M2. Each layer now has a small dedicated cache (128MB) plus
access to the global hot-store.

Closes #42
```

## Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] All existing tests pass
- [ ] New tests added for changes
- [ ] Manually tested on target hardware

## Checklist
- [ ] Code follows style guidelines
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] Documentation updated if needed
- [ ] No breaking changes (or documented)
```

## Questions?

Open an issue or start a discussion. We're happy to help!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

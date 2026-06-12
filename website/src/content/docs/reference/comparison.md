---
title: Comparison ops
description: The 25 comparison operations, generated from the op table.
order: 22
section: "Reference"
---

Generated from the binaries by `scripts/gen-ops-reference.ts` — do not edit by
hand. Every op also documents itself: `torch <op> --help`.

### eq

elementwise equality (returns a Bool tensor)

```
usage: torch eq <t1> <t2>
```

### allclose

true if all elements are close (returns a JSON bool)

```
usage: torch allclose <t1> <t2> [--rtol <Float>] [--atol <Float>]
```

### sort

sort along --dim (default last); returns values and indices

```
usage: torch sort <t1> [--dim <Int>] [--descending]
```

### gt

elementwise a > b (Bool, broadcasting)

```
usage: torch gt <t1> <t2>
```

### lt

elementwise a < b (Bool, broadcasting)

```
usage: torch lt <t1> <t2>
```

### ge

elementwise a >= b (Bool, broadcasting)

```
usage: torch ge <t1> <t2>
```

### le

elementwise a <= b (Bool, broadcasting)

```
usage: torch le <t1> <t2>
```

### ne

elementwise a != b (Bool, broadcasting)

```
usage: torch ne <t1> <t2>
```

### logical_and

elementwise logical AND (Bool, broadcasting)

```
usage: torch logical_and <t1> <t2>
```

### logical_or

elementwise logical OR (Bool, broadcasting)

```
usage: torch logical_or <t1> <t2>
```

### logical_xor

elementwise logical XOR (Bool, broadcasting)

```
usage: torch logical_xor <t1> <t2>
```

### isclose

elementwise closeness (Bool; --rtol/--atol)

```
usage: torch isclose <t1> <t2> [--rtol <Float>] [--atol <Float>]
```

### isnan

elementwise NaN test (Bool)

```
usage: torch isnan <t1>
```

### isinf

elementwise infinity test (Bool)

```
usage: torch isinf <t1>
```

### isfinite

elementwise finiteness test (Bool)

```
usage: torch isfinite <t1>
```

### isposinf

elementwise +inf test (Bool)

```
usage: torch isposinf <t1>
```

### isneginf

elementwise -inf test (Bool)

```
usage: torch isneginf <t1>
```

### logical_not

elementwise logical NOT (Bool)

```
usage: torch logical_not <t1>
```

### equal

whole-tensor equality (returns a JSON bool)

```
usage: torch equal <t1> <t2>
```

### topk

top-k values+indices (--smallest = PyTorch largest=False, a NuTorch-ism)

```
usage: torch topk <t1> <k> [--dim <Int>] [--smallest]
```

### argsort

indices that would sort along --dim (default last)

```
usage: torch argsort <t1> [--dim <Int>] [--descending]
```

### searchsorted

insertion indices: searchsorted(sorted_seq, values)

```
usage: torch searchsorted <t1> <t2>
```

### bucketize

bucket indices: bucketize(values, boundaries)

```
usage: torch bucketize <t1> <t2>
```

### msort

sort along the first dimension (values only)

```
usage: torch msort <t1>
```

### unique

sorted unique values

```
usage: torch unique <t1>
```

---
title: Shape ops
description: The 23 shape operations, generated from the op table.
order: 25
section: "Reference"
---

Generated from the binaries by `scripts/gen-ops-reference.ts` — do not edit by
hand. Every op also documents itself: `torch <op> --help`.

### cat

concatenate tensors along --dim (default 0)

```
usage: torch cat <t1>... (at least 2) [--dim <Int>]
```

### reshape

reshape to the given shape (-1 infers one dim)

```
usage: torch reshape <t1> <shape>
```

### permute

permute dimensions

```
usage: torch permute <t1> <dims>
```

### transpose

swap two dimensions

```
usage: torch transpose <t1> <dim0> <dim1>
```

### t

transpose a 2-D tensor

```
usage: torch t <t1>
```

### squeeze

drop size-1 dims (all, or --dim)

```
usage: torch squeeze <t1> [--dim <Int>]
```

### unsqueeze

insert a size-1 dim

```
usage: torch unsqueeze <t1> <dim>
```

### flatten

flatten dims (--start_dim/--end_dim)

```
usage: torch flatten <t1> [--start_dim <Int>] [--end_dim <Int>]
```

### stack

stack tensors along a NEW --dim (default 0)

```
usage: torch stack <t1>... (at least 2) [--dim <Int>]
```

### split

split into chunks of split_size along --dim

```
usage: torch split <t1> <split_size> [--dim <Int>]
```

### chunk

split into N chunks along --dim

```
usage: torch chunk <t1> <chunks> [--dim <Int>]
```

### gather

gather values along --dim using an int64 index tensor

```
usage: torch gather <t1> <t2> [--dim <Int>]
```

### index_select

select rows/cols along --dim by an int64 index tensor

```
usage: torch index_select <t1> <t2> [--dim <Int>]
```

### masked_select

select by mask (numeric mask cast via != 0, a NuTorch-ism)

```
usage: torch masked_select <t1> <t2>
```

### where

cond ? x : y (numeric cond cast via != 0, a NuTorch-ism)

```
usage: torch where <t1> <t2> <t3>
```

### narrow

slice: length elements from start along dim

```
usage: torch narrow <t1> <dim> <start> <length>
```

### flip

reverse along the given dims

```
usage: torch flip <t1> <dims>
```

### roll

roll elements by shifts (optionally along --dims)

```
usage: torch roll <t1> <shifts> [--dims <IntList>]
```

### repeat

tile the tensor by repeats per dim

```
usage: torch repeat <t1> <repeats>
```

### repeat_interleave

repeat each element N times (optionally along --dim)

```
usage: torch repeat_interleave <t1> <repeats> [--dim <Int>]
```

### movedim

move a dim to a new position

```
usage: torch movedim <t1> <source> <destination>
```

### take_along_dim

gather along --dim with a broadcastable int64 index

```
usage: torch take_along_dim <t1> <t2> [--dim <Int>]
```

### scatter

non-inplace scatter: input, int64 index, src along --dim

```
usage: torch scatter <t1> <t2> <t3> [--dim <Int>]
```

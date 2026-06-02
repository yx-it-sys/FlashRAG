# Sample Count Report

This report uses `intermediate_data.json` only, matching the LJ / correctness split used in the current figures.

## InternVL3.5-8B

### Totals

| Setting | Samples | Correct | Incorrect |
|---|---:|---:|---:|
| Original | 1500 | 193 | 1307 |
| Disturb | 905 | 101 | 804 |
| Oracle | 694 | 75 | 619 |

### Per-task counts

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 300 | 133 | 148 |
| Single-hop Attribute Query | 300 | 147 | 217 |
| Multi-hop | 300 | 153 | 158 |
| Comparison | 300 | 176 | 143 |
| Subproblem Aggregation | 300 | 296 | 28 |

### Per-task counts by correctness

#### Correct

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 80 | 16 | 44 |
| Single-hop Attribute Query | 10 | 3 | 15 |
| Multi-hop | 2 | 0 | 2 |
| Comparison | 1 | 0 | 1 |
| Subproblem Aggregation | 100 | 82 | 13 |

#### Incorrect

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 220 | 117 | 104 |
| Single-hop Attribute Query | 290 | 144 | 202 |
| Multi-hop | 298 | 153 | 156 |
| Comparison | 299 | 176 | 142 |
| Subproblem Aggregation | 200 | 214 | 15 |

## Qwen2.5 / Qwen2.5-vl-7B

### Totals

| Setting | Samples | Correct | Incorrect |
|---|---:|---:|---:|
| Original | 1500 | 427 | 1073 |
| Disturb | 1019 | 296 | 723 |
| Oracle | 460 | 104 | 356 |

### Per-task counts

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 300 | 99 | 79 |
| Single-hop Attribute Query | 300 | 170 | 141 |
| Multi-hop | 300 | 229 | 125 |
| Comparison | 300 | 237 | 99 |
| Subproblem Aggregation | 300 | 284 | 16 |

### Per-task counts by correctness

#### Correct

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 114 | 29 | 31 |
| Single-hop Attribute Query | 39 | 35 | 22 |
| Multi-hop | 36 | 39 | 21 |
| Comparison | 47 | 44 | 18 |
| Subproblem Aggregation | 191 | 149 | 12 |

#### Incorrect

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 186 | 70 | 48 |
| Single-hop Attribute Query | 261 | 135 | 119 |
| Multi-hop | 264 | 190 | 104 |
| Comparison | 253 | 193 | 81 |
| Subproblem Aggregation | 109 | 135 | 4 |

## Qwen3-vl-4B

### Totals

| Setting | Samples | Correct | Incorrect |
|---|---:|---:|---:|
| Original | 1500 | 583 | 917 |
| Disturb | 1263 | 446 | 817 |
| Oracle | 383 | 135 | 248 |

### Per-task counts

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 300 | 174 | 106 |
| Single-hop Attribute Query | 300 | 244 | 134 |
| Multi-hop | 300 | 267 | 64 |
| Comparison | 300 | 285 | 65 |
| Subproblem Aggregation | 300 | 293 | 14 |

### Per-task counts by correctness

#### Correct

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 145 | 70 | 55 |
| Single-hop Attribute Query | 69 | 65 | 32 |
| Multi-hop | 67 | 59 | 11 |
| Comparison | 95 | 84 | 30 |
| Subproblem Aggregation | 207 | 168 | 7 |

#### Incorrect

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 155 | 104 | 51 |
| Single-hop Attribute Query | 231 | 179 | 102 |
| Multi-hop | 233 | 208 | 53 |
| Comparison | 205 | 201 | 35 |
| Subproblem Aggregation | 93 | 125 | 7 |

## Qwen3-vl-8b

### Totals

| Setting | Samples | Correct | Incorrect |
|---|---:|---:|---:|
| Original | 1500 | 556 | 944 |
| Disturb | 1191 | 428 | 763 |
| Oracle | 379 | 121 | 258 |

### Per-task counts

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 300 | 147 | 94 |
| Single-hop Attribute Query | 300 | 234 | 142 |
| Multi-hop | 300 | 263 | 67 |
| Comparison | 300 | 255 | 63 |
| Subproblem Aggregation | 300 | 292 | 13 |

### Per-task counts by correctness

#### Correct

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 117 | 49 | 39 |
| Single-hop Attribute Query | 75 | 67 | 41 |
| Multi-hop | 68 | 71 | 16 |
| Comparison | 87 | 74 | 17 |
| Subproblem Aggregation | 209 | 167 | 8 |

#### Incorrect

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 183 | 98 | 55 |
| Single-hop Attribute Query | 225 | 167 | 101 |
| Multi-hop | 232 | 192 | 51 |
| Comparison | 213 | 181 | 46 |
| Subproblem Aggregation | 91 | 125 | 5 |

## Qwen3-vl-32B

### Totals

| Setting | Samples | Correct | Incorrect |
|---|---:|---:|---:|
| Original | 1500 | 610 | 890 |
| Disturb | 1195 | 466 | 729 |
| Oracle | 103 | 43 | 60 |

### Per-task counts

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 300 | 118 | 31 |
| Single-hop Attribute Query | 300 | 238 | 28 |
| Multi-hop | 300 | 276 | 27 |
| Comparison | 300 | 274 | 13 |
| Subproblem Aggregation | 300 | 289 | 4 |

### Per-task counts by correctness

#### Correct

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 146 | 44 | 20 |
| Single-hop Attribute Query | 93 | 80 | 10 |
| Multi-hop | 64 | 67 | 5 |
| Comparison | 88 | 79 | 5 |
| Subproblem Aggregation | 219 | 196 | 3 |

#### Incorrect

| Task | Original | Disturb | Oracle |
|---|---:|---:|---:|
| Entity Recognition | 154 | 74 | 11 |
| Single-hop Attribute Query | 207 | 158 | 18 |
| Multi-hop | 236 | 209 | 22 |
| Comparison | 212 | 195 | 8 |
| Subproblem Aggregation | 81 | 93 | 1 |


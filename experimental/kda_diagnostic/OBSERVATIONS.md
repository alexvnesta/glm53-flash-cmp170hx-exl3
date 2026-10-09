# Bounded native observation, 2026-10-09

One guarded LS/Q8/DFlash2 arithmetic request sampled loaded KDA layers 0, 1, and 2 on GPU0. The saved synthetic request is `projection_fixture.json` (SHA256 `7f8aeb457247b0fdeb0b7291a1fd2978e4342404eed0e67008b99bafa80807b7`). That model/tokenizer rendered 36 prompt tokens; the successful request returned 22 completion tokens. The token count is an observation, not a tokenizer-independent fixture contract. Both GEMV environment flags were set to 0 before imports and recorded as 0 at probe time.

Each probe repeated one actual FP16 `s_caof` row of width 8192 into eight rows and used the loaded projection to produce 4096 FP32 values per row. The projection had no bias. Newly owned input/output/scratch estimates were 442368 bytes per sample. Automatic Q1 and Q8 calls returned tags 4 and 3. Forced calls both returned tag 4 with a requested one-SM setting.

| Sample layer | Automatic first-row max absolute difference | Automatic differing elements / 4096 | Common tag 4 + requested SMS 1 first row |
| --- | ---: | ---: | --- |
| 0 | 2.0489096641540527e-08 | 3978 | Bitwise equal |
| 1 | 2.0489096641540527e-08 | 3903 | Bitwise equal |
| 2 | 1.3969838619232178e-08 | 3901 | Bitwise equal |

All compared outputs were finite. Recorded checks found protected storage/contents and owned inputs unchanged in all three samples. Both row counts passed native shape-compatibility checks. The native binding did not return measured SM-count telemetry; the pinned dispatch clamps the positive one-SM request to one.

This supplies bounded live evidence for these direct native calls beyond the CPU doubles. It does not reproduce CUDA graph scheduling or the native bias add, and it compares the first output row of a repeated input. The joint shape and SM setting changed together, so it is not proof that shape selection alone explains the automatic differences. It is not evidence of full-model parity, a lossless speculative-decoding fix, general numerical equivalence, or speed improvement. No automatic production change follows from this diagnostic.

The public tree contains the synthetic fixture and portable tools, not raw traces, model data, process addresses, or host receipts. The private full trace was reviewed separately. Broader conclusions require separate controlled model tests.

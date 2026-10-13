| experiment | median ms |
|---|---|
| per-channel mean, HWC | 3.699 |
| per-channel mean, CHW (contiguous) | 0.135 |
| per-channel mean, CHW (view) | 3.691 |
| HWC -> CHW copy | 0.212 |
| x*0.5+1, float32 | 2.678 |
| x*0.5+1, float16 | 7.363 |

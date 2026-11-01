# Third-party notices

## HabanaLabs QMAN ARC packet & SCAL headers (MIT License)

The following headers under `include/gaudirt/scal/` originate from HabanaLabs
open-source QMAN ARC packet and SCAL interface definitions and retain their
upstream license notices:

- `include/gaudirt/scal/gaudi2_arc_common_packets.h` — Copyright (C) 2020 HabanaLabs Ltd. (`SPDX-License-Identifier: MIT`)
- `include/gaudirt/scal/gaudi2_arc_eng_packets.h` — Copyright (C) 2020 HabanaLabs Ltd. (`SPDX-License-Identifier: MIT`)
- `include/gaudirt/scal/gaudi2_arc_sched_packets.h` — Copyright (C) 2020 HabanaLabs Ltd. (`SPDX-License-Identifier: MIT`)
- `include/gaudirt/scal/scal.h` — HabanaLabs SCAL C interface definitions (`https://github.com/HabanaAI/scal`, MIT License)

```text
MIT License

Copyright (c) 2020 HabanaLabs Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## External SDK, runtime, and model dependencies

This repository does not bundle or relicense Intel Gaudi drivers, firmware,
Synapse (`libSynapse.so`, `libSynapseMme.so`), HCL (`libhcl.so`), or PyTorch.
Those components must be installed separately under their respective licenses.

Model checkpoints (such as `MiMo-V2-Flash`), private evaluation corpora, and raw
trace/benchmark receipts are not distributed in this repository and are governed
by their respective distributors' terms.

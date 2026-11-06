"""An FX program retains valid bucket recipes across pool and device lifetimes."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    program = gaudirt._C.BucketProgram([1, 1], [gaudirt._C.FxNode.binary(gaudirt._C.FxOp.add, 0, 1)], [2])
    for generation in range(2):
        gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
        for rows in (3, 17):
            left_cpu = torch.full((rows, 64), generation + 1, dtype=torch.bfloat16)
            right_cpu = torch.full((rows, 64), rows, dtype=torch.bfloat16)
            left, right = left_cpu.to("gaudirt:0"), right_cpu.to("gaudirt:0")
            for _ in range(4):
                actual = program.run([left, right])[0]
                assert torch.equal(actual.cpu(), left_cpu + right_cpu)
                del actual
                # The unpadded input exercises graph staging. Release and
                # recreate its pool without invalidating the bucket recipe.
                gaudirt.empty_cache()
            del left, right
        stats = gaudirt.recipe_cache_stats()
        assert stats[2] == 2 and sum(stats[:2]) == 2, stats
        gaudirt.empty_cache()
        gaudirt.shutdown()
        print("context", generation, "recipe stats", stats, flush=True)
    print("program recipe cache survives pool release and device reinitialization", flush=True)


if __name__ == "__main__":
    main()

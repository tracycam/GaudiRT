"""Offline GNU-binutils audit of the two explicitly selected BFM artifacts."""
from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess

base = Path(__file__).resolve().parent
engine = base / 'binaries/libeng_mme_cmpt_bfm.so'
scheduler = base / 'binaries/libsched_cmpt_media_bfm.so'
raw = engine.read_bytes()
assert raw[:6] == b'\x7fELF\x02\x01'
phoff = struct.unpack_from('<Q', raw, 32)[0]
phentsize, phnum = struct.unpack_from('<HH', raw, 54)
segments = [struct.unpack_from('<IIQQQQQQ', raw, phoff+i*phentsize) for i in range(phnum)]

def read_va(addr, size):
    for typ, flags, off, va, pa, filesz, memsz, align in segments:
        if typ == 1 and va <= addr and addr+size <= va+filesz:
            return raw[off+addr-va:off+addr-va+size]
    raise ValueError(hex(addr))

def run(*args):
    return subprocess.check_output(args, text=True)

relocations = run('readelf', '-rW', str(engine))
(base/'bfm-relocations.txt').write_text(relocations)
symbols = {}
for line in relocations.splitlines():
    parts = line.split()
    if len(parts) >= 5 and parts[2] == 'R_X86_64_JUMP_SLOT':
        symbols[int(parts[0], 16)] = run('c++filt', parts[4]).strip()

def resolve_plt(addr):
    code = read_va(addr, 10)
    assert code[:6] == bytes.fromhex('f30f1efaff25')
    got = addr+10+struct.unpack_from('<i', code, 6)[0]
    return {'plt': hex(addr), 'got': hex(got), 'symbol': symbols[got]}

dispatch = []
for opcode, rel in enumerate(struct.unpack('<9i', read_va(0x6f5c0, 36))):
    target = 0x6f5c0+rel
    code = read_va(target, 5)
    assert code[0] == 0xe8
    call_target = target+5+struct.unpack_from('<i', code, 1)[0]
    dispatch.append({'opcode': opcode, 'case_address': hex(target), **resolve_plt(call_target)})

functions = [
    ('bfm-static-dispatch.asm', 0x63200, 0x5ec),
    ('bfm-dynamic-dispatch.asm', 0x604e0, 0x66c),
]
for name, start, size in functions:
    (base/name).write_text(run('objdump', '-d', '-C', f'--start-address={hex(start)}',
                             f'--stop-address={hex(start+size)}', str(engine)))
sched_symbols = run('nm', '-D', '-C', str(scheduler))
(base/'scheduler-dynamic-symbols.txt').write_text(sched_symbols)
handlers = [line for line in sched_symbols.splitlines() if 'SchedulerBFM::cmd_process_' in line]
assert len(handlers) == 14
summary = {
    'method': 'Offline GNU objdump/readelf/nm; REA open identified the artifact, but its native analysis provider was unavailable; no REA decompilation evidence is claimed',
    'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (engine,scheduler)},
    'architecture': 'ELF64 little-endian x86-64 BFM, not the ARC machine code image',
    'dynamic_dispatch_address': '0x604e0',
    'dynamic_jump_table_address': '0x6f5c0',
    'dynamic_dispatch': dispatch,
    'static_opcode8_call': {'instruction': '0x6337a', **resolve_plt(0x1f980)},
    'bit4': {
        'tests': ['static 0x632fa', 'dynamic 0x605de'],
        'observed_effect': 'When the other ECB list is eligible, set the current-list selection and return via a yielding path',
        'referenced_messages': ['yielding to static ecb list', 'yielding to dynamic ecb list'],
        'header_corroboration': 'archived HCL gaudi2_arc_eng_packets.h:1017 declares bit 4 as yield:1',
        'limit': 'Not a direct proof that bit 4 calls config_and_arm_monitor; these are distinct operations',
    },
    'scheduler_exported_command_handler_count': len(handlers),
    'scheduler_exported_command_handlers': handlers,
    'limitations': [
        'Only the selected dispatch and exported-symbol claims were audited; this is not a complete recovered protocol',
        'Same installed package does not by itself prove every BFM behavior matches the ARC firmware',
        'Host x86-64 C++ object offsets are not automatically ARC ABI or wire-structure offsets',
        'No custom CCB was submitted, and no runtime or firmware was patched',
    ],
}
(base/'bfm-audit.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({'opcode8': dispatch[8], 'scheduler_handlers': len(handlers), 'bit4': 'yield'}, indent=2))

#!/usr/bin/env python3
"""Disassemble a function in libSynapse.so and resolve direct/PLT call targets."""
import bisect, re, subprocess, sys
LIB = sys.argv[1]
names = sys.argv[2:]
out = subprocess.run(['nm', '-DC', LIB], capture_output=True, text=True).stdout
syms = []
for l in out.splitlines():
    m = re.match(r'([0-9a-f]+) ([TtWw]) (.*)', l)
    if m:
        syms.append((int(m.group(1), 16), m.group(3)))
syms.sort()
addrs = [a for a, _ in syms]
# PLT map via relocations
plt = {}
rel = subprocess.run(['objdump', '-d', '-j', '.plt.sec', LIB], capture_output=True, text=True).stdout
if not rel.strip():
    rel = subprocess.run(['objdump', '-d', '-j', '.plt', LIB], capture_output=True, text=True).stdout
relocs = subprocess.run(['readelf', '-rW', LIB], capture_output=True, text=True).stdout
got = {}
for l in relocs.splitlines():
    p = l.split()
    if len(p) >= 5 and 'JUMP_SLOT' in l:
        got[int(p[0], 16)] = p[4]
cur = None
for l in rel.splitlines():
    m = re.match(r'\s*([0-9a-f]+):', l)
    h = re.match(r'([0-9a-f]+) <', l)
    if h:
        cur = int(h.group(1), 16)
    m2 = re.search(r'jmp\s+\*0x([0-9a-f]+)\(%rip\)\s+# ([0-9a-f]+)', l)
    if m2 and m:
        target = int(m2.group(2), 16)
        start = int(m.group(1), 16) & ~0xf
        if target in got:
            plt[start] = got[target] + '@plt'
def name(a):
    if a in plt:
        return plt[a]
    i = bisect.bisect_right(addrs, a) - 1
    if i >= 0:
        off = a - syms[i][0]
        return syms[i][1][:120] + (f'+{off:#x}' if off else '')
    return hex(a)
for q in names:
    if q.startswith('0x'):
        start = int(q, 16)
    else:
        cands = [a for a, n in syms if n.startswith(q)]
        start = cands[0]
    i = bisect.bisect_right(addrs, start)
    while i < len(addrs) and addrs[i] == start:
        i += 1
    end = addrs[i] if i < len(addrs) else start + 0x2000
    print(f'==== {name(start)} [{start:#x}-{end:#x}]')
    d = subprocess.run(['objdump', '-d', '--no-show-raw-insn', f'--start-address={start:#x}',
                        f'--stop-address={end:#x}', LIB], capture_output=True, text=True).stdout
    for l in d.splitlines():
        m = re.match(r'\s*([0-9a-f]+):\s+(call|jmp)\s+([0-9a-f]+)', l)
        if m and m.group(2) == 'call':
            print(f'  {m.group(1)} call {name(int(m.group(3), 16))}')
        elif 'call   *' in l:
            print('  ' + l.strip() + '   [indirect]')

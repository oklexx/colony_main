import struct, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
exe = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\SkhClny3.exe"
out = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\resources_out"
os.makedirs(out, exist_ok=True)
raw = open(exe, "rb").read()

pe_off = struct.unpack_from("<I", raw, 0x3C)[0]
nsec = struct.unpack_from("<H", raw, pe_off+6)[0]
opt_off = pe_off + 20
magic2 = struct.unpack_from("<H", raw, opt_off)[0]
so = opt_off + (240 if magic2==0x20b else 224)
secs = []
for i in range(nsec):
    o = so + i*40
    vsize = struct.unpack_from("<I", raw, o+8)[0]
    va = struct.unpack_from("<I", raw, o+12)[0]
    rsize = struct.unpack_from("<I", raw, o+16)[0]
    rptr = struct.unpack_from("<I", raw, o+20)[0]
    secs.append((va, vsize, rptr, rsize))

def rva2off(rva):
    for va, vsize, rptr, rsize in secs:
        if va <= rva < va + max(vsize, rsize) and rsize > 0:
            return rptr + (rva - va)
    return None

res_rva = struct.unpack_from("<I", raw, opt_off + 96 + 2*8)[0]
res_size = struct.unpack_from("<I", raw, opt_off + 96 + 2*8 + 4)[0]
res_off = rva2off(res_rva)

def parse_dir(o, path):
    num_named = struct.unpack_from("<H", raw, o+8)[0]
    num_id = struct.unpack_from("<H", raw, o+10)[0]
    entries = num_named + num_id
    entries_off = o + 16
    for i in range(entries):
        eo = entries_off + i*8
        name_off = struct.unpack_from("<I", raw, eo)[0]
        data_off = struct.unpack_from("<I", raw, eo+4)[0]
        if name_off & 0x80000000:
            s_off = res_off + (name_off & 0x7FFFFFFF)
            slen = struct.unpack_from("<H", raw, s_off)[0]
            nm = raw[s_off+2:s_off+2+slen*2].decode("utf-16-le", errors="replace")
        else:
            nm = str(name_off)
        newpath = path + [nm]
        if data_off & 0x80000000:
            sub_off = res_off + (data_off & 0x7FFFFFFF)
            parse_dir(sub_off, newpath)
        else:
            de_off = res_off + data_off
            codepage = struct.unpack_from("<I", raw, de_off)[0]
            data_rva = struct.unpack_from("<I", raw, de_off+4)[0]
            size = struct.unpack_from("<I", raw, de_off+8)[0]
            d_off = rva2off(data_rva)
            data = raw[d_off:d_off+size]
            p = "_".join(newpath)
            fn = os.path.join(out, p + ".bin")
            with open(fn, "wb") as f:
                f.write(data)
            print("%-70s %9d bytes (codepage=%d)" % (p, size, codepage))

parse_dir(res_off, [])

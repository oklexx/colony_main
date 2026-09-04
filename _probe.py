import struct, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
base = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main"
for f in ["SkhClny3.exe","sakhalin_colony.exe","sakhalin_colony_gui.exe"]:
    data = open(base+"\\"+f,"rb").read()
    print("====", f, len(data), "bytes")
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    nsec = struct.unpack_from("<H", data, pe_off+6)[0]
    opt_off = pe_off + 20
    magic2 = struct.unpack_from("<H", data, opt_off)[0]
    so = opt_off + (240 if magic2==0x20b else 224)
    for i in range(nsec):
        o = so + i*40
        name = bytes(data[o:o+8]).replace(b"\x00",b" ").decode("latin1").strip()
        vsize = struct.unpack_from("<I", data, o+8)[0]
        va = struct.unpack_from("<I", data, o+12)[0]
        rsize = struct.unpack_from("<I", data, o+16)[0]
        rptr = struct.unpack_from("<I", data, o+20)[0]
        print(f"  {i}: {name:14s} vsize={vsize:#x} va={va:#x} rsize={rsize:#x} rptr={rptr:#x}")
    print("  optmagic:", hex(magic2))

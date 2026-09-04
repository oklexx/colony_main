import pefile, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
exe = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\SkhClny3.exe"
out = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\extracted3"
os.makedirs(out, exist_ok=True)
pe = pefile.PE(exe)
raw = open(exe, "rb").read()
RT = {1:"CURSOR",2:"BITMAP",3:"ICON",4:"MENU",5:"DIALOG",6:"STRING",7:"FONTDIR",8:"FONT",
      9:"ACCEL",10:"RCDATA",11:"MSTABLE",12:"GCURSOR",14:"GICON",16:"VERSION",23:"MANIFEST"}

def nm(e):
    if e.id == 0xFFFF:
        try:
            return str(e.name)
        except Exception:
            return "name"
    return str(e.id)

def walk(dire, path):
    for e in dire.entries:
        n = nm(e)
        if hasattr(e, "directory") and e.directory:
            walk(e.directory, path + [n])
        else:
            de = e.data
            off = de.struct.OffsetToData
            ln = de.struct.Size
            data = raw[off:off+ln]
            p = "_".join(path)
            fn = os.path.join(out, p + ".bin")
            with open(fn, "wb") as f:
                f.write(data)
            print("%-70s %9d bytes" % (p, ln))

for t in pe.DIRECTORY_ENTRY_RESOURCE.entries:
    tname = RT.get(t.id, str(t.id))
    if t.directory:
        walk(t.directory, [tname])
    else:
        de = t.data
        data = raw[de.struct.OffsetToData : de.struct.OffsetToData + de.struct.Size]
        with open(os.path.join(out, tname + ".bin"), "wb") as f:
            f.write(data)
        print("%-70s %9d bytes" % (tname, len(data)))

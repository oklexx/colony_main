import pefile, os, sys, struct
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
exe = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\SkhClny3.exe"
out = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\extracted2"
os.makedirs(out, exist_ok=True)
pe = pefile.PE(exe)
raw = pe.data

RT = {1:"CURSOR",2:"BITMAP",3:"ICON",4:"MENU",5:"DIALOG",6:"STRING",7:"FONTDIR",8:"FONT",
      9:"ACCEL",10:"RCDATA",11:"MSTABLE",12:"GCURSOR",14:"GICON",16:"VERSION",23:"MANIFEST",
      24:"PLUGPLAY",25:"VXD",26:"ANICURSOR",27:"ANICON",4095:"VJS"}

def walk(dire, path, depth=0):
    for e in dire.entries:
        if e.id == 0xFFFF:
            name = str(e.name)
        else:
            name = str(e.id)
        if hasattr(e, "directory") and e.directory:
            walk(e.directory, path + [name], depth+1)
        else:
            de = e.data
            off = de.struct.OffsetToData
            ln = de.struct.DataSize
            data = raw[off:off+ln]
            p = ".".join(path)
            fn = os.path.join(out, p.replace("\\","_").replace(" ","_") + ".bin")
            with open(fn,"wb") as f:
                f.write(data)
            print("%-60s %8d bytes" % (p, ln))

if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
    for t in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        tname = RT.get(t.id, str(t.id))
        if t.directory:
            walk(t.directory, [tname])
        else:
            de = t.data
            data = raw[de.struct.OffsetToData : de.struct.OffsetToData+de.struct.DataSize]
            with open(os.path.join(out, tname+".bin"),"wb") as f:
                f.write(data)
            print("%-60s %8d bytes" % (tname, len(data)))

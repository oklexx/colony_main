import pefile, sys, struct, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
exe = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\SkhClny3.exe"
pe = pefile.PE(exe)
print("=== FILE INFO ===")
print("Machine:", hex(pe.FILE_HEADER.Machine), "Sections:", pe.FILE_HEADER.NumberOfSections)
print("Subsystem:", pe.OPTIONAL_HEADER.Subsystem)
print("ImageBase:", hex(pe.OPTIONAL_HEADER.ImageBase))
print("Entry:", hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint))
print("DllCharacteristics:", hex(pe.OPTIONAL_HEADER.DllCharacteristics))
print("SizeOfImage:", hex(pe.OPTIONAL_HEADER.SizeOfImage))
print("SizeOfHeaders:", hex(pe.OPTIONAL_HEADER.SizeOfHeaders))
print("NumberOfRvaAndSizes:", pe.OPTIONAL_HEADER.NumberOfRvaAndSizes)
print()
print("=== SECTIONS (pefile) ===")
for s in pe.sections:
    print("  %-14s vsize=0x%08x va=0x%08x rsize=0x%08x rptr=0x%08x" % (
        s.Name.rstrip(b"\x00").decode("latin1").strip(), s.Misc_VirtualSize, s.VirtualAddress,
        s.SizeOfRawData, s.PointerToRawData))
print()
print("=== DIRECTORIES ===")
for i, name in enumerate(["EXPORT","IMPORT","RESOURCE","EXCEPTION","SECURITY","Basereloc","Debug","Architecture","GlobalPtr","TLS","LoadConfig","BoundImport","IAT","DelayImport","CLR","Reserved2","Reserved3"]):
    try:
        dd = pe.OPTIONAL_HEADER.DATA_DIRECTORY[i]
        print("  %-14s va=0x%08x size=0x%08x" % (name, dd.VirtualAddress, dd.Size))
    except Exception as e:
        break
print()
print("=== IMPORTS ===")
if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
    for mod in pe.DIRECTORY_ENTRY_IMPORT:
        print("  module:", mod.dll.decode())
        for imp in list(mod.imports)[:10]:
            nm = imp.name.decode() if imp.name else None
            print("    ", hex(imp.ordinal) if imp.ordinal and not nm else (nm or hex(imp.ordinal)))
else:
    print("  (none)")
print()
print("=== RESOURCES ===")
if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
    for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        t = entry.id
        print("  type", t, "name:", str(entry.name) if entry.name else None)
        if entry.directory:
            for e2 in entry.directory.entries:
                print("    name/ID:", str(e2.name) if e2.name else e2.id)
                if e2.directory:
                    for e3 in e2.directory.entries:
                        d = e3.data
                        print("      lang/data: size=%d offset=0x%x" % (len(bytes(d.data)), d.struct.OffsetToData))
else:
    print("  (none)")
print()
data = open(exe, "rb").read()
print("=== STRINGS SCAN ===")
# utf-16 strings with cyrillic
pat = re.compile(rb"(?:[\x20-\x7e][\x00]){5,}")
cyr = {}
for m in pat.finditer(data):
    s = m.group().decode("utf-16-le", errors="ignore")
    if any(0x400 <= ord(c) <= 0x4FF for c in s):
        cyr.setdefault(s, m.start())
print("utf16 cyrillic count:", len(cyr))
for s, off in list(cyr.items())[:100]:
    print("  0x%08x %r" % (off, s))
# ascii strings
pat2 = re.compile(rb"[\x20-\x7e]{8,}")
asc = {}
for m in pat2.finditer(data):
    s = m.group().decode("latin1")
    asc.setdefault(s, m.start())
print("ascii count:", len(asc))
for s, off in list(asc.items())[:120]:
    print("  0x%08x %s" % (off, s[:80]))

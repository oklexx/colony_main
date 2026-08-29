import pefile
import os

def extract_resources(exe_path, out_dir):
    pe = pefile.PE(exe_path)
    os.makedirs(out_dir, exist_ok=True)
    
    extracted_count = 0
    res = pe.DIRECTORY_ENTRY_RESOURCE
    
    for typ in res.entries:
        type_name = typ.name.string if typ.name else str(typ.id)
        if isinstance(type_name, bytes):
            type_name = type_name.decode('latin-1')
        type_dir = os.path.join(out_dir, type_name)
        os.makedirs(type_dir, exist_ok=True)
        
        for name_entry in typ.directory.entries:
            name_str = name_entry.name.string if name_entry.name else str(name_entry.id)
            if isinstance(name_str, bytes):
                name_str = name_str.decode('latin-1')
            
            for lang_entry in name_entry.directory.entries:
                lang_id = lang_entry.id
                if hasattr(lang_entry, 'data'):
                    data_entry = lang_entry.data
                    size = data_entry.struct.Size
                    rva = data_entry.struct.OffsetToData
                    data = pe.get_data(rva, size)
                    
                    filename = f"{name_str}_{lang_id}.bin"
                    filepath = os.path.join(type_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(data)
                    
                    extracted_count += 1
                    print(f"Extracted: {filepath} ({len(data)} bytes)")
    
    print(f"\nTotal resources extracted: {extracted_count}")
    print(f"Output directory: {out_dir}")

if __name__ == '__main__':
    exe_path = r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\SkhClny3.exe'
    out_dir = r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\resources_out'
    extract_resources(exe_path, out_dir)

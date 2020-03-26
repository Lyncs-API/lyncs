__all__ = [
    "find_description",
    ]

_ALREADY_RUNNING = False

def find_description(readme=None):
    import codecs, os
    from itertools import product
    from .data_files import add_to_data_files
    
    base = ["README", "readme", "description"]
    ext = ["", ".txt", ".md", ".rst"]
    options = ["".join(parts) for parts in product(base, ext)]
    
    if readme:
        assert os.path.isfile(readme), "Given readme does not exist"
    else:
        for filename in options:
            if os.path.isfile(filename):
                readme = filename
                break
            
    assert readme, """
    Couldn't find a compatible filename. 
    Options are %s""" % ", ".join(options)
    
    if readme.endswith(".md"):
        dtype = "text/markdown"
    elif readme.endswith(".rst"):
        dtype = "text/x-rst"
    else:
        dtype = "text/plain"
        
    with codecs.open(readme, encoding='utf-8') as f:
        add_to_data_files(readme, directory=".")
        dlong = f.read().split("\n")

    for i,line in enumerate(dlong):
        if line.startswith("[lyncs_setuptools]: "):
            global _ALREADY_RUNNING

            if not _ALREADY_RUNNING:
                _ALREADY_RUNNING = True
                line=": ".join(line.split(":")[1:])
                import io
                from contextlib import redirect_stdout
                
                f = io.StringIO()
                with redirect_stdout(f):
                    for line in line.split(";"):
                        exec(line.strip())
                dlong[i] = f.getvalue()#.replace("\\n", "\n")
                
                _ALREADY_RUNNING = False
            else:
                dlong[i] = ""
                
    dshort = ""
    for line in dlong:
        if line.split():
            dshort = line
            break

    if "markdown" in dtype:
        while dshort.startswith("#"):
            dshort = dshort[1:]
    

    return dshort.strip(), "\n".join(dlong), dtype
        
        
    
    

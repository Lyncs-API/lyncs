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
        dlong = f.read()
        
    dshort = ""
    for line in dlong.split("\n"):
        if line.split():
            dshort = line
            break

    if "markdown" in dtype:
        while dshort.startswith("#"):
            dshort = dshort[1:]
    

    return dshort.strip(), dlong, dtype
        
        
    
    

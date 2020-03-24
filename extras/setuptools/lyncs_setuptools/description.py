
def find_long_description(readme=None):
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
    
    with codecs.open(readme, encoding='utf-8') as f:
        add_to_data_files(readme, directory=".")
        return f.read()
    
    

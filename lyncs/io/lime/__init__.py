
# List of available implementations
engines = [
    "pylime",
    ]

from . import lime as pylime
default = "pylime"

# from lyncs import config
# if config.clime_enabled:
#     import .clime
#     engines.append("clime")

# TODO: add more, e.g. lemon

def is_compatible(filename):
    try:
        return pylime.is_lime_file(filename)
    except:
        return False
    

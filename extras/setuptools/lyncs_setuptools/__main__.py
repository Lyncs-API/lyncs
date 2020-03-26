from . import main, switcher

import sys

assert len(sys.argv)<=2, "Maximum one argument is allowed."

sys.exit(print(main(sys.argv[1] if len(sys.argv)==2 else "all")))

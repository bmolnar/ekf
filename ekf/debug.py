import sys

def info(type, value, tb):
   if hasattr(sys, 'ps1') or not sys.stderr.isatty(  ):
      # You are in interactive mode or don't have a tty-like
      # device, so call the default hook
      sys.__excepthook__(type, value, tb)
   else:
      import traceback, pdb
      # You are NOT in interactive mode; print the exception...
      traceback.print_exception(type, value, tb)
      print
      # ...then start the debugger in post-mortem mode
      pdb.pm(  )

sys.excepthook = info

"""Load the frozen runtime and install independently tested input repairs."""
from pathlib import Path
import sys
RUNTIME=Path(__file__).resolve().parent/'runtime'
sys.path.insert(0,str(RUNTIME))
import refresh
from live_inputs import install
install(refresh)
if __name__=='__main__':refresh.main()

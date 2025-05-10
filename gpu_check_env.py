#!/usr/bin/env python3
"""
GPU environment check wrapper (for backwards compatibility)
"""

import sys
from modules.utils.gpu_check import check_system_gpu, main

if __name__ == "__main__":
    # For backwards compatibility, we'll run the same main function
    sys.exit(main())
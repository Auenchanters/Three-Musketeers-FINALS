"""
Pytest configuration — adds project root to sys.path so all tests
can import models, engine, and data without sys.path hacking.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fsp import fsp
import pytest

def test_fsp():
    assert fsp() == "hello fsp"

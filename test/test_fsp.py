from src.fsp import Fsp
import pytest

def test_fsp():
    fsp = Fsp()
    assert fsp.helloFsp() == "hello fsp"

from src.fsp import Fsp

def test_fsp():
    fsp = Fsp()
    assert fsp.helloFsp() == "hello fsp"

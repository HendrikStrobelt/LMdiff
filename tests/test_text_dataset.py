import pytest
import path_fixes as pf
from analysis.text_dataset import TextDataset

EXAMPLES = pf.TESTS / "examples"

def test_properties():
    tds = TextDataset.load(EXAMPLES / "SmallGoodTest1.txt")
    tds2 = TextDataset.load(EXAMPLES / "SmallGoodTest2.txt")
    assert tds.name == "SmallGoodTest1"
    assert tds.type == "human_created"
    assert tds.type == "machine_generated"
    assert len(tds.content) == 5
    assert isinstance(tds.checksum, str)

@pytest.mark.now
def test_bad():
    with pytest.raises(ValueError):
        tds3 = TextDataset.load(EXAMPLES / "SmallBadTest1.txt")


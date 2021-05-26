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

def test_bad():
    with pytest.raises(ValueError):
        tds3 = TextDataset.load(EXAMPLES / "SmallBadTest1.txt")

def test_save_checksum(tmp_path):
    fname = EXAMPLES / "SmallGoodTest1.txt"

    fout1 = tmp_path / "bonkers.txt"
    fout2 = tmp_path / "bonkers2.txt"
    fout3 = tmp_path / "bonkers3.txt"

    tds = TextDataset.load(fname)
    tds.content += ["Another line for the record"]
    tds.save(fout1)

    tds2 = TextDataset.load(fname)
    tds2.content += ["Another line for the record"]
    tds2.content += ["Yet another, change the sum"]
    tds2.save(fout2)

    tds3 = TextDataset.load(fname)
    tds3.content += ["Another line for the record"]
    tds3.content += ["Yet another, change the sum"]
    tds3.save(fout3)

    assert TextDataset.load(fout1).checksum != TextDataset.load(fout2).checksum
    assert TextDataset.load(fout2).checksum == TextDataset.load(fout3).checksum
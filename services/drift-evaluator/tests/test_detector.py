import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator.detector import DriftDetector

def test_starts_in_warmup():
    assert DriftDetector(cooldown_sec=300).state == "WARMUP"

def test_no_drift_is_ok():
    d = DriftDetector(cooldown_sec=300)
    assert d.update(False, now=0) is False
    assert d.state == "OK"

def test_one_drift_check_is_only_suspected():
    d = DriftDetector(cooldown_sec=300)
    assert d.update(True, now=0) is False     
    assert d.state == "SUSPECTED"


def test_two_drift_checks_in_a_row_confirm():
    d = DriftDetector(cooldown_sec=300)
    d.update(True, now=0)
    assert d.update(True, now=15) is True      
    assert d.state == "CONFIRMED"


def test_drift_interrupted_by_normal_check_resets():
    d = DriftDetector(cooldown_sec=300)
    d.update(True, now=0)
    d.update(False, now=15)                    
    assert d.update(True, now=30) is False     
    assert d.state == "SUSPECTED"


def test_cooldown_blocks_new_events():
    d = DriftDetector(cooldown_sec=300)
    d.update(True, now=0)
    d.update(True, now=15)                     
    assert d.update(True, now=30) is False
    assert d.update(True, now=45) is False
    assert d.state == "COOLDOWN"
    assert d.cooldown_remaining(now=45) == 270


def test_after_cooldown_detection_works_again():
    d = DriftDetector(cooldown_sec=300)
    d.update(True, now=0)
    d.update(True, now=15)                    
    d.update(True, now=400)                  
    assert d.update(True, now=415) is True  
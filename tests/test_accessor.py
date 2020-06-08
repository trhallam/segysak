
import pytest
import numpy as np

from test_fixtures import (
    empty2d,
    empty2d_gath,
    empty3d,
    empty3d_gath,
    empty3d_twt,
    empty3d_depth,
    zeros3d
)

def test_is_2d(empty2d):
    assert empty2d.seis.is_2d()
    assert not empty2d.seis.is_3d()
    assert not empty2d.seis.is_3dgath()
    assert not empty2d.seis.is_2dgath()

def test_is_3d(empty3d):
    assert empty3d.seis.is_3d()
    assert not empty3d.seis.is_2d()
    assert not empty3d.seis.is_3dgath()
    assert not empty3d.seis.is_2dgath()

def test_is_2d_gath(empty2d_gath):
    assert empty2d_gath.seis.is_2dgath()
    assert not empty2d_gath.seis.is_3d()
    assert not empty2d_gath.seis.is_3dgath()
    assert not empty2d_gath.seis.is_2d()

def test_is_3d_gath(empty3d_gath):
    assert empty3d_gath.seis.is_3dgath()
    assert not empty3d_gath.seis.is_3d()
    assert not empty3d_gath.seis.is_2d()
    assert not empty3d_gath.seis.is_2dgath()

def test_is_twt(empty3d_twt):
    assert empty3d_twt.seis.is_twt()
    assert not empty3d_twt.seis.is_depth()

def test_is_depth(empty3d_depth):
    assert empty3d_depth.seis.is_depth()
    assert not empty3d_depth.seis.is_twt()

def test_is_empty2d(empty2d):
    assert empty2d.seis.is_empty()

def test_is_empty3d(empty3d):
    assert empty3d.seis.is_empty()

def test_is_empty3d_gath(empty3d_gath):
    assert empty3d_gath.seis.is_empty()

def test_is_empty2d_gath(empty2d_gath):
    assert empty2d_gath.seis.is_empty()

def test_is_empty_zeros3d(zeros3d):
    assert not zeros3d.seis.is_empty()

def test_zeros_like(zeros3d):
    assert 'data' in zeros3d.variables
    assert np.all(zeros3d['data'] == 0.0)
    assert zeros3d['data'].sum() == 0.0


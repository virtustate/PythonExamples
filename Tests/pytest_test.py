def test_fail():
    assert False


def test_pass():
    assert True


def test_add():
    assert add(5, 6) == 11


def add(x, y):
    return 'blah'

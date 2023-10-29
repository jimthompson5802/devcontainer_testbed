import cppmodule

def test_print_message():
    echo_string = cppmodule.echo_message("Hello, world!")
    assert echo_string == "echo " + "Hello, world!"

def test_add_numbers_int():
    result = cppmodule.add_numbers(2, 3)
    assert result == 5

def test_add_numbers_float():
    result = cppmodule.add_numbers(1.0, 2.5)
    assert result == 3.5


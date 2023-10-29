import cppmodule as cm

print(cm.echo_message("Hello from Python using C++!"))
print(cm.echo_message("Hello again from Python using C++!"))
print(cm.echo_message("Hello again again from Python using C++!"))

for a, b in zip([1, 2, 3], [4, 5, 6]):
    answer = cm.add_numbers(a, b)
    print(f"{a} + {b} answer is: " + str(answer))

try:
    print(f"floats: {cm.add_numbers(1.0, 2.5)}")
except TypeError as e:
    print(f"Error: {e}")
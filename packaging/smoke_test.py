import math

from numba_enzyme import differentiable
from numba_enzyme.types import Float64


@differentiable
def f(x: Float64, y: Float64) -> Float64:
    return math.sin(x) * y


primal = f(1.0, 2.0)
grad = f.grad(1.0, 2.0)
jvp = f.jvp((1.0, 2.0), (1.0, 0.0))

expected_primal = math.sin(1.0) * 2.0
expected_grad = (math.cos(1.0) * 2.0, math.sin(1.0))

print("primal:", primal, "expected:", expected_primal)
print("grad:", grad, "expected:", expected_grad)
print("jvp:", jvp, "expected:", expected_grad[0])

assert abs(primal - expected_primal) < 1e-9
assert abs(grad[0] - expected_grad[0]) < 1e-9
assert abs(grad[1] - expected_grad[1]) < 1e-9
assert abs(jvp - expected_grad[0]) < 1e-9
print("PASS")

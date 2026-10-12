import numpy as np

# (shape_a, shape_b, your_guess)
cases = [
    ((3,),          (3,)),
    ((4, 3),        (3,)),
    ((4, 3),        (4,)),
    ((4, 1),        (1, 3)),
    ((2, 3, 4),     (4,)),
    ((2, 3, 4),     (3, 1)),
    ((480, 640, 3), (3,)),
    ((480, 640, 3), (3, 1, 1)),
    ((1, 3, 480, 640), (3, 1, 1)),
    ((8, 1, 5),     (1, 7, 1)),
]

for i, (sa, sb) in enumerate(cases, 1):
    try:
        result = (np.zeros(sa) + np.zeros(sb)).shape
    except ValueError as e:
        result = f"ERROR: {e}"
    print(f"{i:2}. {sa} + {sb} -> {result}")
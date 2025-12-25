# Solution for Problem 1: Matrix-Vector Dot Product

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Check if dimensions match: number of columns in 'a' should equal length of 'b'
    if len(a[0]) != len(b):
        return -1
    
    # Compute dot product for each row
    result = []
    for row in a:
        dot_product = sum(row[i] * b[i] for i in range(len(b)))
        result.append(dot_product)
    
    return result

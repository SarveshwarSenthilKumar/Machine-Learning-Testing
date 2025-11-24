# Python file to calculate the finite differences of a given function

def calculate_finite_differences(y_values):
    """
    Calculate finite differences of a sequence until the differences become constant.
    Returns a list of difference sequences and the degree of the polynomial.
    """
    differences = [y_values]
    degree = 0
    
    while True:
        current_level = differences[-1]
        next_level = []
        
        # Calculate differences between consecutive elements
        for i in range(1, len(current_level)):
            next_level.append(current_level[i] - current_level[i-1])
        
        differences.append(next_level)
        degree += 1
        
        # Check if all differences are the same (within floating point tolerance)
        if len(set(round(d, 10) for d in next_level)) == 1 or len(next_level) == 1:
            break
            
    return differences, degree

def print_differences(differences):
    """Print the difference table in a readable format."""
    max_len = len(differences[0])
    
    print("\nFinite Differences Table:")
    print("-" * 50)
    
    for i, level in enumerate(differences):
        # Add padding to align the differences properly
        padding = '  ' * i
        values = [f"{x:8.2f}" for x in level]
        print(f"Level {i}: {padding}{' '.join(values)}")

# Example usage
if __name__ == "__main__":
    # Example data points (squares of integers)
    y_values = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    
    print("Example: y = x^2")
    print("Y values:", y_values)
    
    differences, degree = calculate_finite_differences(y_values)
    print_differences(differences)
    print(f"\nThe function is a polynomial of degree: {degree}")
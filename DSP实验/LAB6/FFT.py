from cmath import exp, pi
from math import ceil, log2

def fft(x, level=0):
    """
    Compute the FFT of the input sequence `x` using the Cooley-Tukey algorithm.
    Displays steps during computation.

    Parameters:
    x (list): Input sequence.
    level (int): Current recursion depth (used for display).

    Returns:
    list: FFT of the input sequence.
    """
    N = len(x)
    print(" " * (2 * level) + f"Level {level}: Processing {x}")
    if N <= 1:
        return x
    even = fft(x[0::2], level + 1)
    odd = fft(x[1::2], level + 1)
    T = [exp(-2j * pi * k / N) * odd[k] for k in range(N // 2)]
    result = ([even[k] + T[k] for k in range(N // 2)] +
              [even[k] - T[k] for k in range(N // 2)])
    print(" " * (2 * level) + f"Level {level}: Result {result}")
    return result

def pad_to_power_of_two(x):
    """
    Zero-pad the input sequence `x` to the nearest power of 2.

    Parameters:
    x (list): Input sequence.

    Returns:
    list: Zero-padded sequence.
    """
    target_length = 2 ** ceil(log2(len(x)))
    return x + [0] * (target_length - len(x))

# Input sequence
input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Zero-pad the input to a power of 2
padded_input = pad_to_power_of_two(input_data)

# Compute FFT
print('FFT of x[n] : ' + ' '.join("%5.3f" % abs(f) for f in fft(padded_input)))

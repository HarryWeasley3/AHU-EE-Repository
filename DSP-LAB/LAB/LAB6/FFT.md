### FFT code

```python
from cmath import pi, exp

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

# Test the function and display the FFT result.
print('FFT of x[n] : ' + ' '.join("%5.3f" % abs(f) for f in fft([1,2,3,4,5,6,7,8])))
```

### Output
```python
Level 0: Processing [1, 2, 3, 4, 5, 6, 7, 8]
  Level 1: Processing [1, 3, 5, 7] # 开始计算x[n]的even段,即{x[0],x[2],x[4],x[6]}
    Level 2: Processing [1, 5]  # 开始计算x[n]的even段的even段,即x{[0],x[4]}
      Level 3: Processing [1]  # even = x[0]
      Level 3: Processing [5]  # odd = x[4]
    Level 2: Result [(6+0j), (-4+0j)]  # 返回第二层,计算出{x[0],x[4]}的fft
    Level 2: Processing [3, 7]  # 开始计算x[n]的even段的odd段,即{x[2],x[6]}
      Level 3: Processing [3] # even = x[2]
      Level 3: Processing [7] # odd = x[6]
    Level 2: Result [(10+0j), (-4+0j)]   # 返回第二层,计算出{x[2],x[6]}的fft
  Level 1: Result [(16+0j), (-4+4j), (-4+0j), (-3.9999999999999996-4j)]  # 结束计算x[n]的even段
  Level 1: Processing [2, 4, 6, 8]  # 开始计算x[n]的odd段,即{x[1],x[3],x[5],x[7]}
    Level 2: Processing [2, 6]  # 开始计算x[n]的odd段的even段,即x{[1],x[5]}
      Level 3: Processing [2]  # even = x[1]
      Level 3: Processing [6]  # odd = x[5]
    Level 2: Result [(8+0j), (-4+0j)]  # 返回第二层,计算出{x[1],x[5]}的fft
    Level 2: Processing [4, 8]  # 开始计算x[n]的odd段的odd段,即{x[3],x[7]}
      Level 3: Processing [4] # even = x[3]
      Level 3: Processing [8] # odd = x[7]
    Level 2: Result [(12+0j), (-4+0j)] # 返回第二层,计算出{x[3],x[7]}的fft
  Level 1: Result [(20+0j), (-4+4j), (-4+0j), (-3.9999999999999996-4j)]  # 出计算x[n]odd段的fft
Level 0: Result [(36+0j), (-4+9.65685424949238j), (-4+4j), (-4+1.6568542494923797j), (-4+0j), (-4-1.6568542494923806j), (-3.9999999999999996-4j), (-3.9999999999999987-9.65685424949238j)] # x[n]的fft
FFT of x[n] : 36.000 10.453 5.657 4.330 4.000 4.330 5.657 10.453
```

### Pad Zero

```python
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

```

### Output

```python
Level 0: Processing [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0]
  Level 1: Processing [1, 3, 5, 7, 9, 0, 0, 0]
    Level 2: Processing [1, 5, 9, 0]
      Level 3: Processing [1, 9]
        Level 4: Processing [1]
        Level 4: Processing [9]
      Level 3: Result [(10+0j), (-8+0j)]
      Level 3: Processing [5, 0]
        Level 4: Processing [5]
        Level 4: Processing [0]
      Level 3: Result [(5+0j), (5+0j)]
    Level 2: Result [(15+0j), (-8-5j), (5+0j), (-8+5j)]
    Level 2: Processing [3, 7, 0, 0]
      Level 3: Processing [3, 0]
        Level 4: Processing [3]
        Level 4: Processing [0]
      Level 3: Result [(3+0j), (3+0j)]
      Level 3: Processing [7, 0]
        Level 4: Processing [7]
        Level 4: Processing [0]
      Level 3: Result [(7+0j), (7+0j)]
    Level 2: Result [(10+0j), (3.0000000000000004-7j), (-4+0j), (2.9999999999999996+7j)]
  Level 1: Result [(25+0j), (-10.82842712474619-12.071067811865476j), (5+4j), (-5.171572875253809-2.0710678118654737j), (5+0j), (-5.17157287525381+2.0710678118654755j), (5-4j), (-10.828427124746192+12.071067811865474j)]      
  Level 1: Processing [2, 4, 6, 8, 0, 0, 0, 0]
    Level 2: Processing [2, 6, 0, 0]
      Level 3: Processing [2, 0]
        Level 4: Processing [2]
        Level 4: Processing [0]
      Level 3: Result [(2+0j), (2+0j)]
      Level 3: Processing [6, 0]
        Level 4: Processing [6]
        Level 4: Processing [0]
      Level 3: Result [(6+0j), (6+0j)]
    Level 2: Result [(8+0j), (2.0000000000000004-6j), (-4+0j), (1.9999999999999996+6j)]
    Level 2: Processing [4, 8, 0, 0]
      Level 3: Processing [4, 0]
        Level 4: Processing [4]
        Level 4: Processing [0]
      Level 3: Result [(4+0j), (4+0j)]
      Level 3: Processing [8, 0]
        Level 4: Processing [8]
        Level 4: Processing [0]
      Level 3: Result [(8+0j), (8+0j)]
    Level 2: Result [(12+0j), (4.000000000000001-8j), (-4+0j), (3.9999999999999996+8j)]
  Level 1: Result [(20+0j), (-0.8284271247461894-14.485281374238571j), (-4+4j), (4.828427124746191-2.4852813742385695j), (-4+0j), (4.82842712474619+2.4852813742385713j), (-3.9999999999999996-4j), (-0.8284271247461916+14.48528137423857j)]
Level 0: Result [(45+0j), (-17.13707118454409-25.136697460629243j), (5+9.65685424949238j), (-5.619914404421773-7.483028813327444j), (5+4j), (-4.723231346085844-3.3408931895964944j), (4.999999999999999+1.6568542494923797j), (-4.519783064948289-0.9945618368982903j), (5+0j), (-4.51978306494829+0.994561836898292j), (5-1.6568542494923806j), (-4.7232313460858455+3.340893189596496j), (5-4j), (-5.619914404421777+7.483028813327445j), (5.000000000000001-9.65685424949238j), (-17.137071184544094+25.136697460629236j)]
FFT of x[n] : 45.000 30.423 10.875 9.358 6.403 5.785 5.267 4.628 5.000 4.628 5.267 5.785 6.403 9.358 10.875 30.423
```


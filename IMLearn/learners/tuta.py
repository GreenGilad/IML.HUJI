import math

x = 0
for i in range(9):
    # print(x)
    x += (3 ** i) / (math.factorial(i) * math.factorial(8 - i))

print(x)
print(1 / x)
p_4 = (1 / x) * (3 ** 4) / (math.factorial(4) * math.factorial(8 - 4))
print(p_4)

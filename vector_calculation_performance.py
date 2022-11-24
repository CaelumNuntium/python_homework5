import numpy
import scipy
import time
import math
import random
from matplotlib import pyplot as plt


class MyList(list):
    def __init__(self, lst):
        super().__init__(lst)

    def __mul__(self, other):
        return MyList([self[_] * other[_] for _ in range(len(self))])


def make_list(dim, length):
    if dim == 1:
        return MyList([random.uniform(0, 1) for _ in range(math.ceil(length))])
    else:
        return MyList([make_list(dim - 1, length) for _ in range(math.ceil(length))])


def average(lst):
    return sum(lst) / len(lst)


def sigma(lst):
    avg = average(lst)
    return math.sqrt(average([(lst[_] - avg) ** 2 for _ in range(len(lst))]))


def run_test(repeat_times, a, b):
    m_times = []
    for k in range(repeat_times):
        t1 = time.time()
        a * b
        t2 = time.time()
        m_times.append(t2 - t1)
        print(f"    #{k + 1}: {t2 - t1} s")
    return [average(m_times), sigma(m_times)]


def approx(f, x, y, n):
    args, _ = scipy.optimize.curve_fit(f, x, y)
    lb = min(x)
    rb = max(x)
    h = (rb - lb) / n
    return [[lb + j * h for j in range(n)], [f(lb + i * h, *(__ for __ in args)) for i in range(n)]]


maxsize = 10000000
repeat_times = 7
num_measurements = 10
max_lengths = [maxsize, math.ceil(math.sqrt(maxsize)), math.ceil(math.pow(maxsize, 1 / 3))]
steps = [max_lengths[_] / num_measurements for _ in range(3)]
c = ["#FF0000", "#00FF00", "#0000FF"]
for i in range(3):
    plt.subplot(1, 2, 1)
    plt.xlabel("list length")
    plt.ylabel("t, s")
    plt.title("List multiplication")
    times = []
    sigmas = []
    sizes = []
    print(f"{i + 1}-dimension list:")
    for j in range(num_measurements):
        sizes.append(math.ceil((j + 1) * steps[i]) ** (i + 1))
        a = make_list(i + 1, (j + 1) * steps[i])
        b = make_list(i + 1, (j + 1) * steps[i])
        print(f"  list size {sizes[j]}")
        res = run_test(repeat_times, a, b)
        times.append(res[0])
        sigmas.append(res[1])
        print(f"  average: {res[0]} s\n  error: {res[1]}\n")
    plt.errorbar(sizes, times, yerr=sigmas, fmt="o", ecolor=c[i], capsize=5, color=c[i], label=f"{i + 1}-dimension list")
    plt.legend(fontsize=10)
    curve = approx((lambda x, k, m: k * x + m), sizes, times, 1000)
    plt.plot(curve[0], curve[1], color=c[i])
for i in range(3):
    plt.subplot(1, 2, 2)
    plt.xlabel("array length")
    plt.ylabel("t, s")
    plt.title("Numpy array multiplication")
    times = []
    sigmas = []
    sizes = []
    print(f"{i + 1}-dimension array:")
    for j in range(num_measurements):
        sizes.append(math.ceil((j + 1) * steps[i]) ** (i + 1))
        a = numpy.array(make_list(i + 1, (j + 1) * steps[i]))
        b = numpy.array(make_list(i + 1, (j + 1) * steps[i]))
        print(f"  array size {sizes[j]}")
        res = run_test(repeat_times, a, b)
        times.append(res[0])
        sigmas.append(res[1])
        print(f"  average: {res[0]} s\n  error: {res[1]}\n")
    plt.errorbar(sizes, times, yerr=sigmas, fmt="o", ecolor=c[i], capsize=5, color=c[i], label=f"{i + 1}-dimension array")
    plt.legend(fontsize=10)
    curve = approx((lambda x, k, m: k * x + m), sizes, times, 1000)
    plt.plot(curve[0], curve[1], color=c[i])
fig = plt.gcf()
fig.set_size_inches(12, 5.5)
#fig.savefig("test.png", dpi=250)
plt.show()

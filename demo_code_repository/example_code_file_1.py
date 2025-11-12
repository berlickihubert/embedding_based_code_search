def add_numbers(a, b):
    return a + b

def subtract_numbers(a, b):
    return a - b

def multiply_numbers(a, b):
    return a * b

def divide(a, b):
    return a / b if b != 0 else None

def square(n):
    return n ** 2

def cube(n):
    return n ** 3

def is_even(n):
    return n % 2 == 0

def is_odd(n):
    return n % 2 != 0

def absolute(n):
    return n if n >= 0 else -n

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def is_palindrome(s):
    return s == s[::-1]

def reverse_string(s):
    return s[::-1]

def count_vowels(s):
    return sum(1 for c in s.lower() if c in "aeiou")

def max_of_three(a, b, c):
    return max(a, b, c)

def min_of_three(a, b, c):
    return min(a, b, c)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

def sum_list(lst):
    return sum(lst)

def average_list(lst):
    return sum(lst) / len(lst) if lst else 0

def max_in_list(lst):
    return max(lst) if lst else None

def min_in_list(lst):
    return min(lst) if lst else None

def remove_duplicates(lst):
    return list(set(lst))

def count_occurrences(lst, item):
    return lst.count(item)

def flatten_once(lst):
    return [x for sub in lst for x in sub]


def fibonacci(n):
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

def starts_with_vowel(s):
    return s[0].lower() in "aeiou" if s else False

def c_to_f(c):
    return (c * 9/5) + 32

def f_to_c(f):
    return (f - 32) * 5/9

def word_count(s):
    return len(s.split())

def capitalize_words(s):
    return ' '.join(word.capitalize() for word in s.split())

def is_sorted(lst):
    return lst == sorted(lst)

def merge_lists(a, b):
    return a + b

def intersection(a, b):
    return list(set(a) & set(b))

def union(a, b):
    return list(set(a) | set(b))

def unique_ordered(lst):
    seen = set()
    result = []
    for x in lst:
        if x not in seen:
            result.append(x)
            seen.add(x)
    return result

def second_largest(lst):
    unique_sorted = sorted(set(lst))
    return unique_sorted[-2] if len(unique_sorted) >= 2 else None

def count_words(lst):
    return {word: lst.count(word) for word in set(lst)}

def underscore_spaces(s):
    return s.replace(' ', '_')

def remove_punctuation(s):
    import string
    return ''.join(c for c in s if c not in string.punctuation)

def list_to_string(lst):
    return ', '.join(map(str, lst))

def sqrt(n):
    return n ** 0.5

def is_perfect_square(n):
    return int(n ** 0.5) ** 2 == n

def char_frequency(s):
    return {c: s.count(c) for c in set(s)}

def swap_dict(d):
    return {v: k for k, v in d.items()}

def middle_char(s):
    return s[len(s)//2] if s else ''

def has_duplicates(lst):
    return len(lst) != len(set(lst))

def repeat_string(s, n):
    return s * n

def common_prefix(a, b):
    prefix = ""
    for x, y in zip(a, b):
        if x == y:
            prefix += x
        else:
            break
    return prefix

def reverse_list(lst):
    return lst[::-1]

def count_digits(n):
    return len(str(abs(n)))

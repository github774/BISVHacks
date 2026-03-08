def cantor_pair(a, b):
    return (a + b) * (a + b + 1) // 2 + b

def cantor_unpair(c):
    w = int(((8*c + 1)**0.5 - 1)//2)
    t = w*(w+1)//2
    b = c - t
    a = w - b
    return a, b

if __name__ == "__main__":
    print("Cantor pairing: enter 2 integers (a b)")
    while True:
        try:
            s = input("> ")
            if not s.strip():
                continue
            a, b = map(int, s.split())
            c = cantor_pair(a, b)
            a2, b2 = cantor_unpair(c)
            print(f"  encode: ({a}, {b}) -> {c}")
            print(f"  decode: {c} -> ({a2}, {b2})")
        except (ValueError, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                break
            print("  enter two integers separated by space")

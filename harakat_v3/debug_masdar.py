#!/usr/bin/env python3
import sys
import os
import io

def main():
    if sys.platform == 'win32':
        os.system('chcp 65001 > nul 2>&1')
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except:
            pass

    HARAKAT = 'ًٌٍَُِّْ'
    HARAKAT_SET = set(HARAKAT)

    def strip_harakat(text):
        return ''.join(c for c in text if c not in HARAKAT_SET)

    word = 'تَعْظِيمَ'
    base = strip_harakat(word)

    print(f"Word: {word}")
    print(f"Base: {base}")
    print(f"Length: {len(base)}")
    print(f"Starts with ت: {base.startswith('ت')}")
    print(f"Ends with يم: {base.endswith('يم')}")
    print(f"Chars: {[c for c in base]}")

    # Check each character code point
    print(f"\nCharacter code points:")
    for i, c in enumerate(base):
        print(f"  {i}: '{c}' = U+{ord(c):04X}")

    # Check last 2 characters
    if len(base) >= 2:
        last2 = base[-2:]
        print(f"\nLast 2 chars: '{last2}'")
        print(f"Expected 'يم': U+064A U+0645")
        print(f"Actual: U+{ord(last2[0]):04X} U+{ord(last2[1]):04X}")

if __name__ == '__main__':
    main()

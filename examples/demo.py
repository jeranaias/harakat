#!/usr/bin/env python3
"""
Harakat Demo Script
Demonstrates Arabic diacritization with impressive examples.
Use this script to record a terminal GIF demo with tools like:
- asciinema (https://asciinema.org)
- termtosvg (https://github.com/nbedos/termtosvg)
- VHS (https://github.com/charmbracelet/vhs)
"""

import sys
import time
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harakat import diacritize

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def slow_print(text, delay=0.02):
    """Print text character by character for demo effect."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def demo_section(title):
    """Print a section header."""
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title:^60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print()
    time.sleep(0.5)

def demo_example(arabic_text, description):
    """Demonstrate diacritization of a text."""
    print(f"{Colors.YELLOW}{description}{Colors.END}")
    print()
    print(f"{Colors.BLUE}Input:  {Colors.END}{arabic_text}")
    time.sleep(0.3)

    result = diacritize(arabic_text)

    print(f"{Colors.GREEN}Output: {Colors.END}{result}")
    print()
    time.sleep(1)

def main():
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Title
    print()
    print(f"{Colors.BOLD}{Colors.GREEN}")
    print("  _    _                 _         _   ")
    print(" | |  | |               | |       | |  ")
    print(" | |__| | __ _ _ __ __ _| | ____ _| |_ ")
    print(" |  __  |/ _` | '__/ _` | |/ / _` | __|")
    print(" | |  | | (_| | | | (_| |   < (_| | |_ ")
    print(" |_|  |_|\\__,_|_|  \\__,_|_|\\_\\__,_|\\__|")
    print(f"{Colors.END}")
    print()
    print(f"{Colors.CYAN}High-Accuracy Arabic Diacritization{Colors.END}")
    print(f"{Colors.YELLOW}2.29% DER | 99.997% Quran Accuracy | 6.7 MB{Colors.END}")
    print()
    time.sleep(2)

    # Quran Demo
    demo_section("Quranic Text - 99.997% Accuracy")

    demo_example(
        "بسم الله الرحمن الرحيم",
        "Basmala (Opening phrase)"
    )

    demo_example(
        "الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين",
        "Al-Fatiha (Opening chapter)"
    )

    demo_example(
        "قل هو الله احد الله الصمد لم يلد ولم يولد ولم يكن له كفوا احد",
        "Al-Ikhlas (Sincerity)"
    )

    # Modern Arabic Demo
    demo_section("Modern Standard Arabic - 2.29% DER")

    demo_example(
        "ذهب الطالب الى الجامعة ليدرس العلوم الحديثة",
        "Education sentence"
    )

    demo_example(
        "يعمل المهندسون على تطوير التكنولوجيا المتقدمة",
        "Technology sentence"
    )

    demo_example(
        "السلام عليكم ورحمة الله وبركاته",
        "Arabic greeting"
    )

    # Proverbs Demo
    demo_section("Arabic Proverbs")

    demo_example(
        "العلم نور والجهل ظلام",
        "Knowledge is light, ignorance is darkness"
    )

    demo_example(
        "من جد وجد ومن زرع حصد",
        "He who strives, finds; he who sows, reaps"
    )

    demo_example(
        "الصبر مفتاح الفرج",
        "Patience is the key to relief"
    )

    # Performance Demo
    demo_section("Performance")

    print(f"{Colors.YELLOW}Processing speed test...{Colors.END}")
    print()

    long_text = "ذهب الطالب الى المدرسة في الصباح الباكر ليتعلم العلوم المفيدة والمعارف الجديدة التي تساعده على النجاح في حياته العلمية والعملية"

    start = time.time()
    for _ in range(100):
        diacritize(long_text)
    elapsed = time.time() - start

    print(f"{Colors.BLUE}Text length: {Colors.END}{len(long_text)} characters")
    print(f"{Colors.BLUE}Iterations:  {Colors.END}100")
    print(f"{Colors.GREEN}Total time:  {Colors.END}{elapsed:.2f} seconds")
    print(f"{Colors.GREEN}Per text:    {Colors.END}{elapsed*10:.1f} ms")
    print()

    # Final
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}{'Visit: github.com/jeranaias/harakat':^60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print()

if __name__ == "__main__":
    main()

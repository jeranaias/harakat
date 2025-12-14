#!/usr/bin/env python3
"""
Batch processing example for Harakat Arabic diacritization.
Demonstrates how to process multiple texts efficiently.
"""

import sys
import time
sys.path.insert(0, '..')

from harakat import diacritize

def process_file(input_path: str, output_path: str) -> dict:
    """
    Process an entire file and write diacritized output.

    Returns statistics about the processing.
    """
    stats = {
        "lines": 0,
        "words": 0,
        "time_seconds": 0
    }

    start_time = time.time()

    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                line = line.strip()
                if line:
                    diacritized = diacritize(line)
                    f_out.write(diacritized + '\n')
                    stats["lines"] += 1
                    stats["words"] += len(line.split())
                else:
                    f_out.write('\n')

    stats["time_seconds"] = time.time() - start_time
    return stats


def process_batch(texts: list) -> list:
    """
    Process a batch of texts.

    Args:
        texts: List of undiacritized Arabic texts

    Returns:
        List of diacritized texts
    """
    return [diacritize(text) for text in texts]


if __name__ == "__main__":
    # Example: Process a list of texts
    sample_texts = [
        "الكتاب على الطاولة",
        "ذهب الولد الى المدرسة",
        "قرأت الجريدة صباحا",
        "الطقس جميل اليوم",
        "تعلمت اللغة العربية",
    ]

    print("Batch Processing Example")
    print("=" * 50)

    start = time.time()
    results = process_batch(sample_texts)
    elapsed = time.time() - start

    for original, diacritized in zip(sample_texts, results):
        print(f"Original:    {original}")
        print(f"Diacritized: {diacritized}")
        print()

    print(f"Processed {len(sample_texts)} texts in {elapsed:.3f} seconds")
    print(f"Average: {elapsed/len(sample_texts)*1000:.1f} ms per text")

/* Build: gcc -O2 -Wall -Wextra -std=c11 pp_demo.c -o pp_demo
 * Run: ./pp_demo [0|1]. See README.txt for cache geometry.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#ifndef WAYS
#define WAYS 8
#endif
#ifndef LINE_SIZE
#define LINE_SIZE 64
#endif
#ifndef STRIDE
#define STRIDE 4096
#endif
#define TRIALS 5000

// Keep the volatile memory read between the TSC timestamps.
static uint64_t time_load(volatile uint8_t *address) {
    unsigned int cpu_id;
    _mm_lfence();
    uint64_t start = __rdtsc();
    _mm_lfence();
    (void)*address;
    _mm_lfence();
    uint64_t end = __rdtscp(&cpu_id);
    _mm_lfence();
    return end - start;
}

// STRIDE selects another line in the same cache set.
static void prime(volatile uint8_t *base) {
    for (int i = 0; i < WAYS; ++i) {
        (void)base[i * STRIDE];
    }
}

static uint64_t probe(volatile uint8_t *base) {
    uint64_t ticks = 0;
    for (int i = 0; i < WAYS; ++i) {
        ticks += time_load(base + i * STRIDE);
    }
    return ticks;
}

__attribute__((noinline))
static void victim(int bit, volatile uint8_t *base) {
    prime(base + bit * LINE_SIZE);
}

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "0") != 0 &&
                                strcmp(argv[1], "1") != 0)) {
        fprintf(stderr, "Usage: %s [0|1]\n", argv[0]);
        return 2;
    }
    int bit = argc == 2 ? atoi(argv[1]) : 1;

    // The first half holds attacker lines; the second holds victim lines.
    const size_t bytes = 2 * (size_t)WAYS * STRIDE;
    uint8_t *memory = aligned_alloc(STRIDE, bytes);
    if (memory == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }
    memset(memory, 0, bytes); // Fault in the pages before any trial.
    int correct = 0;
    uint64_t total_a = 0, total_b = 0;
    puts("trial,score_a_ticks,score_b_ticks,guess");

    for (int trial = 0; trial < TRIALS; ++trial) {
        prime(memory);
        prime(memory + LINE_SIZE);
        victim(bit, memory + WAYS * STRIDE);
        // Alternate the first set probed on each trial.
        int order = trial % 2;
        uint64_t first = probe(memory + order * LINE_SIZE);
        uint64_t second = probe(memory + (1 - order) * LINE_SIZE);
        uint64_t time_a = order == 0 ? first : second;
        uint64_t time_b = order == 0 ? second : first;
        int guess = time_b > time_a; // Ties predict 0.
        correct += guess == bit;
        total_a += time_a;
        total_b += time_b;
        printf("%d,%" PRIu64 ",%" PRIu64 ",%d\n", trial, time_a, time_b, guess);
    }

    fprintf(stderr, "Expected bit: %d; correct guesses: %d/%d (%.1f%%)\n",
            bit, correct, TRIALS, 100.0 * correct / TRIALS);
    fprintf(stderr, "Mean set A: %.1f ticks; mean set B: %.1f ticks\n",
            (double)total_a / TRIALS, (double)total_b / TRIALS);
    free(memory);
    return 0;
}

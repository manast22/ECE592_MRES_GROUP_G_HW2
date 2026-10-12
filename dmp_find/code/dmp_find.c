#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include "find_threshold.h"

#define PAGE_SIZE 4096
#define LINE_SIZE 64
#define POOL_PAGES 1024 
#define ARRAY_ENTRIES 64 
#define STREAMED_ENTRIES 32 
#define MAX_DIST (ARRAY_ENTRIES - STREAMED_ENTRIES)
#define TRIALS 20000
#define COLD_SAMPLES 4000

//pointer + garbage is dmp test and sw prefetch are controls that stay cahced
enum dmp_condition { MODE_POINTER, MODE_GARBAGE, MODE_SW_PREFETCH, MODE_SW_PREFETCH_T2 };

static uint8_t target_pool[POOL_PAGES * PAGE_SIZE] __attribute__((aligned(4096)));
static uint64_t ptr_array[ARRAY_ENTRIES] __attribute__((aligned(4096)));

//offsets so cant do dmp directly
static uint32_t target_offsets[ARRAY_ENTRIES];

//xorshift to make random numbers from stackoverlfow
static uint32_t rng_state;
static uint32_t next_rand(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rng_state = x;
}

static volatile uint64_t sink; //keeps the stream sum from being optimized out

//for qsort smallest first
static int cmp_cycles(const void *first_ptr, const void *second_ptr) {
    uint64_t first = *(const uint64_t *)first_ptr;
    uint64_t second = *(const uint64_t *)second_ptr;
    return (first > second) - (first < second);
}

//time flushed reads of random lines to sanity check calibration
static void check_cold_miss(void) {
    static uint64_t cold_samples[COLD_SAMPLES];

    rng_state = 88675123u;
    for (int i = 0; i < COLD_SAMPLES; i++) {
        uint32_t page = next_rand() % POOL_PAGES;
        uint32_t line = next_rand() % (PAGE_SIZE / LINE_SIZE);
        uint8_t *rand_line = &target_pool[page * PAGE_SIZE + line * LINE_SIZE];
        _mm_clflush(rand_line);
        _mm_mfence();
        cold_samples[i] = access_time(rand_line);
    }

    qsort(cold_samples, COLD_SAMPLES, sizeof(uint64_t), cmp_cycles);
    printf("random-line miss median/5p : %llu / %llu cyc\n",
           (unsigned long long)cold_samples[COLD_SAMPLES / 2],
           (unsigned long long)cold_samples[(int)(5.0 / 100.0 * (COLD_SAMPLES - 1))]);
}

//runs one mode and fills per-distance hit rates then returns overall hit rate
static double run(const char *label, enum dmp_condition mode, double *rate_by_dist) {
    int hits_by_dist[MAX_DIST + 1] = {0};
    int trials_by_dist[MAX_DIST + 1] = {0};
    int total_hits = 0;

    rng_state = 2463534242u;

    for (int trial = 0; trial < TRIALS; trial++) {
        //new random page and line for every target
        for (int entry = 0; entry < ARRAY_ENTRIES; entry++) {
            uint32_t page = next_rand() % POOL_PAGES;
            uint32_t line = next_rand() % (PAGE_SIZE / LINE_SIZE);
            target_offsets[entry] = page * PAGE_SIZE + line * LINE_SIZE;
            ptr_array[entry] = (uint64_t)(uintptr_t)&target_pool[target_offsets[entry]];
        }

        //probe one entry past the streamed half if distance 1 = first unread entry
        int dist = 1 + next_rand() % MAX_DIST;
        int probed_entry = STREAMED_ENTRIES - 1 + dist;
        uint8_t *probed_target = &target_pool[target_offsets[probed_entry]];

        //garbage control so only the probed entry stops looking like a pointer
        if (mode == MODE_GARBAGE)
            ptr_array[probed_entry] = 0xdead000000000000ull | (uint64_t)trial;

        //flush every target and the whole pointer array
        _mm_mfence();
        for (int entry = 0; entry < ARRAY_ENTRIES; entry++)
            _mm_clflush(&target_pool[target_offsets[entry]]);
        for (int line = 0; line < ARRAY_ENTRIES * 8 / LINE_SIZE; line++)
            _mm_clflush(&ptr_array[line * LINE_SIZE / 8]);
        _mm_mfence();

        //read the first half as plain numbers and keep referenced
        volatile uint64_t *stream = ptr_array;
        uint64_t sum = 0;
        for (int entry = 0; entry < STREAMED_ENTRIES; entry++)
            sum += stream[entry];
        sink = sum;

        //prefetch into L1, maks ure it can see it
        if (mode == MODE_SW_PREFETCH)
            _mm_prefetch((const char *)probed_target, _MM_HINT_T0);

        //refetch into outer caches now
        if (mode == MODE_SW_PREFETCH_T2)
            _mm_prefetch((const char *)probed_target, _MM_HINT_T2);

        for (volatile int delay = 0; delay < 1000; delay++) {} //give any prefetch time to land, had issues wout this

        trials_by_dist[dist]++;
        if (access_time(probed_target) < threshold) {
            hits_by_dist[dist]++;
            total_hits++;
        }
    }

    for (int dist = 1; dist <= MAX_DIST; dist++)
        rate_by_dist[dist] = trials_by_dist[dist]
            ? 100.0 * hits_by_dist[dist] / trials_by_dist[dist]
            : 0.0;

    double rate = 100.0 * total_hits / TRIALS;
    printf("%-12s cached-after-flush: %6.2f%%\n", label, rate);
    return rate;
}

int main(void) {
    static double prefetch_by_dist[MAX_DIST + 1];
    static double prefetch_t2_by_dist[MAX_DIST + 1];
    static double pointer_by_dist[MAX_DIST + 1];
    static double garbage_by_dist[MAX_DIST + 1];

    //output to file
    if (freopen("dmp_results.txt", "w", stdout) == NULL) {
        perror("dmp_results.txt");
        return 1;
    }

    memset(target_pool, 1, sizeof target_pool); //give every page real memory

    if (calibrate() != 0)
        printf("error with calibation\n");
    check_cold_miss();

    double s = run("sw-prefetch", MODE_SW_PREFETCH, prefetch_by_dist);
    run("sw-prefetch-t2", MODE_SW_PREFETCH_T2, prefetch_t2_by_dist);
    double p = run("pointerlike", MODE_POINTER, pointer_by_dist);
    double g = run("garbage", MODE_GARBAGE, garbage_by_dist);

    //track gap, useful for debug
    printf("\ndistance  pointerlike  garbage\n");
    double best_delta = -100.0;
    int best_dist = 1;
    for (int dist = 1; dist <= MAX_DIST; dist++) {
        printf("%8d  %10.2f%%  %6.2f%%\n", dist,
               pointer_by_dist[dist], garbage_by_dist[dist]);
        double delta = pointer_by_dist[dist] - garbage_by_dist[dist];
        if (delta > best_delta) {
            best_delta = delta;
            best_dist = dist;
        }
    }

    printf("\nDelta (pointerlike - garbage): %.2f%%\n", p - g);
    printf("Best distance %d delta: %.2f%%\n", best_dist, best_delta);

    //check controls for baseline then judge real numbers on the best distance
    if (s < 90.0)
        puts("=> invalid, probe can't see a software prefetch");
    else if (g > 10.0)
        puts("=> invalid baseline too high"); 
    else
        puts(best_delta > 10.0
             ? "=> DMP behavior likely due to cache behavior"
             : "=> DMP unlikey due to cache behavior..");
    return 0;
}

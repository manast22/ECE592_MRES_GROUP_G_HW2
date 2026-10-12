#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include "find_threshold.h"

#define SAMPLES 4000
#define MISS_BUF_SIZE (4 * 1024 * 1024)
#define MISS_LINES (MISS_BUF_SIZE / 64)

uint64_t threshold = 0; //below this = cache hit
uint64_t timer_ovh = 0; //cycles the timing code costs by itself

//one cache line on its own page so nothing else shares it
static uint8_t hit_line[64] __attribute__((aligned(4096)));

//misses come from random lines in here, same as the real probes
static uint8_t miss_buf[MISS_BUF_SIZE] __attribute__((aligned(4096)));

//for qsort, smallest first
static int cmp_cycles(const void *first_ptr, const void *second_ptr)
{
    uint64_t first = *(const uint64_t *)first_ptr;
    uint64_t second = *(const uint64_t *)second_ptr;
    return (first > second) - (first < second);
}

//cycles to read one byte, minus timer overhead
uint64_t access_time(volatile uint8_t *addr)
{
    unsigned int core_id; //rdtscp needs this

    _mm_mfence(); //let earlier loads/flushes finish
    uint64_t start = __rdtscp(&core_id);
    (void)*addr;
    _mm_lfence(); //wait for the load before stopping the timer
    uint64_t end = __rdtscp(&core_id);

    uint64_t elapsed = end - start;

    //clamp to 0 so for unsigned op
    if (elapsed > timer_ovh)
        return elapsed - timer_ovh;
    return 0;
}

int calibrate(void)
{
    static uint64_t ovh_samples[SAMPLES];
    static uint64_t hit_samples[SAMPLES];
    static uint64_t miss_samples[SAMPLES];

    volatile uint8_t *hit_addr = hit_line;
    unsigned int core_id;

    //timer overhead for same fencees
    timer_ovh = 0;
    for (int i = 0; i < SAMPLES; i++) {
        _mm_mfence();
        uint64_t start = __rdtscp(&core_id);
        _mm_lfence();
        uint64_t end = __rdtscp(&core_id);
        ovh_samples[i] = end - start;
    }
    qsort(ovh_samples, SAMPLES, sizeof(uint64_t), cmp_cycles);
    timer_ovh = ovh_samples[SAMPLES / 2]; // median

    //hits line touch right before timing 
    (void)*hit_addr;
    for (int i = 0; i < SAMPLES; i++) {
        (void)*hit_addr;
        hit_samples[i] = access_time(hit_addr);
    }

    //misses flush random line right before timing
    //added this in because reusing one line gave me miss times that were all over the place
    memset(miss_buf, 1, sizeof miss_buf); //give every page real memory
    uint32_t rng = 2463534242u;
    for (int i = 0; i < SAMPLES; i++) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        uint32_t line_idx = rng % MISS_LINES;
        volatile uint8_t *miss_addr = &miss_buf[line_idx * 64];
        _mm_clflush((void *)miss_addr);
        _mm_mfence();
        miss_samples[i] = access_time(miss_addr);
    }

    qsort(hit_samples, SAMPLES, sizeof(uint64_t), cmp_cycles);
    qsort(miss_samples, SAMPLES, sizeof(uint64_t), cmp_cycles);

    int p95_idx = (int)(95.0 / 100.0 * (SAMPLES - 1));
    int p5_idx = (int)(5.0 / 100.0 * (SAMPLES - 1));

    uint64_t hit_median = hit_samples[SAMPLES / 2];
    uint64_t miss_median = miss_samples[SAMPLES / 2];
    uint64_t hit_p95 = hit_samples[p95_idx]; // slow end of hits
    uint64_t miss_p5 = miss_samples[p5_idx]; // fast end of misses

    printf("timer overhead : %llu cyc\n", (unsigned long long)timer_ovh);
    printf("hit   median/95p : %llu / %llu cyc\n",
           (unsigned long long)hit_median,
           (unsigned long long)hit_p95);
    printf("miss  median/5p : %llu / %llu cyc\n",
           (unsigned long long)miss_median,
           (unsigned long long)miss_p5);

    //hits and misses were overlapping so making fallback to midpoint of medians
    if (miss_p5 <= hit_p95) {
        printf("error with test, retry "
               "(miss_5p=%llu <= hit_95p=%llu). "
               "error, retry.\n",
               (unsigned long long)miss_p5,
               (unsigned long long)hit_p95);
        threshold = (hit_median + miss_median) / 2;
        return -1;
    }

    //go to midgap of hits+ misses
    threshold = (hit_p95 + miss_p5) / 2;
    printf("=> threshold set to %llu cyc "
           "(between hit_95p and miss_5p)\n",
           (unsigned long long)threshold);
    return 0;
}

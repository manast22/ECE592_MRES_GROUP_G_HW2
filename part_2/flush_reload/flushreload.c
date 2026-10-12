#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdlib.h>

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

unsigned long probe (char * adrs) {
    volatile unsigned long time;
    __asm__ __volatile__ (
        " mfence            \n"
        " lfence            \n"
        " rdtsc             \n"
        " lfence            \n"
        " movl %%eax, %%esi \n"
        " movl (%1), %%eax  \n"
        " lfence            \n"
        " rdtsc             \n"
        " subl %%esi, %%eax \n"
        " clflush 0(%1)     \n"
        : "=a" (time)
        : "c" (adrs)
        : "%esi", "%edx", "memory");
    return time;
}

int compare (const void * a, const void * b) {
    int aint = *(int*)a;
    int bint = *(int*)b;

    if (aint < bint) return -1;
    if (aint > bint) return 1;
    return 0;
}

int main(void) {
    alignas(64) static char target[64] = "test";

    int nwarm = 1000;


    unsigned long hits[1000] = {0};
    unsigned long misses[1000] = {0};
    uint32_t seed = 1234;
    uint32_t touch;
    int j = 0;
    int k = 0;


    for (int i = 0; i < nwarm; i++) {
        touch = xorshift32(&seed) % 2;
        if (touch) {
            volatile char c = *(volatile char *)target;
            (void)c;
            hits[j] = probe(target);
            j++;
        } else {
            misses[k] = probe(target);
            k++;
        }
    }

    unsigned long hit_count = 0;
    unsigned long miss_count = 0;

    for (int i = 0; i < nwarm; i++) {
        if (hits[i] != 0) {
            hit_count++;
        }
        if (misses[i] != 0) {
            miss_count++;
        }
    }

    qsort(hits, nwarm, sizeof(unsigned long), compare);
    qsort(misses, nwarm, sizeof(unsigned long), compare);

    int hit_med = hits[nwarm - (hit_count/2)];
    int miss_med = misses[nwarm - (miss_count/2)];


    printf("Hit Median: %lu\tMiss Median: %lu\n", hit_med, miss_med);


    unsigned long threshhold = (miss_med + hit_med)/2;

    printf("Threshhold: %lu\n", threshhold);

    unsigned long time;
    int pred_touch;

    FILE *csv = fopen("output.csv", "w");
    FILE *txt = fopen("output.txt", "w");

    if (csv == NULL) {
        printf("Failed to open file output.csv\n");
    } else {
        printf("File Opened\n");
    }

    fprintf(csv, "latency,predicted hit,actual hit\n");

    unsigned long times[20000];

    uint32_t touches[20000];

    for (int i = 0; i < 20000; i++) {
        touch = xorshift32(&seed) % 2;
        touches[i] = touch;
        if (touch) {
            volatile char c = *(volatile char *)target;
            (void)c;
            times[i] = probe(target);
        } else {
            times[i] = probe(target);
        }
    }

    long correct = 0;
    long wrong = 0;
    float acc = 0;
    unsigned long hit_sum = 0;
    unsigned long miss_sum = 0;
    for (int i = 0; i < 20000; i++) {
        pred_touch = times[i] < threshhold;
        if (touches[i]) {
            hit_count++;
            hit_sum += times[i];
        } else {
            miss_count++;
            miss_sum += times[i];
        }

        if (pred_touch == touches[i]) {
            correct++;
        } else {
            wrong++;
        }

        acc = (float)correct / ((float)wrong + (float)correct);

        fprintf(csv, "%lu,%d,%d\n", times[i], pred_touch, touches[i]);
    }
    fprintf(txt, "Average hit: %d\tAverage miss: %d\n", hit_sum/hit_count, miss_sum/miss_count);
    fprintf(txt, "Correct: %d\tWrong: %d\tAccuracy: %lf", correct, wrong, acc);

    fclose(csv);
    fclose(txt);
    return 0;
}
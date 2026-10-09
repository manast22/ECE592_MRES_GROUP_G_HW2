// ff_mthread.c Flush+Flush with separate threads
//
// Build: gcc -O0 -march=native -pthread ff_mthread.c -o ff_thread
// Run:   ./ff_thread


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <x86intrin.h>

volatile char target = 85; //target byte that we are trying to observe with ff

#define N      1000  //samples and warmup
#define WARMUP  1000  


static pthread_barrier_t round_start; //barriers to ensure ordering of threads
static pthread_barrier_t victim_done;


typedef struct {
    uint64_t *times;   // attacker writes here
    int      *touched; // victim writes here
    int       n; //samples
    int       warmup; //calibration/jitter
    uint32_t  seed;    // victim uses this; attacker ignores it
} thread_args;

uint64_t time_flush(volatile char *addr) { //basic function for running ff
    _mm_mfence(); _mm_lfence(); //set up memory and time fence for any previous instructions
    uint64_t start = __rdtsc(); //use x86 rdtsc for starting timing measurement
    _mm_lfence(); //one more lfence to make sure timing function finishes
    _mm_clflush((const void *)addr); //actual clflush for attack
    _mm_mfence(); _mm_lfence(); //time/memory fence to ensure singularity
    uint64_t end = __rdtsc(); //end timer
    return end - start; //calculate delta of flush instruction
}

void sort_u64(uint64_t *a, int n) { //sorting collected times to find median (insertion sort)
    for (int i = 1; i < n; i++) {
        uint64_t key = a[i]; //sorting head
        int j = i - 1; //secondary index
        while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; } //slide array[i] left until it is smallest
        a[j + 1] = key; //increment key
    }
}

uint64_t median_sorted(const uint64_t *sorted, int n) { //find median from sorted array, decided on median after sensitivity of mean
    if (n % 2) return sorted[n / 2]; //if array elements are odd, return exact middle (stepped)
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2; //else return middle of two middle values
}

void learn(const uint64_t *times, const int *touched, int n,
           uint64_t *threshold, int *cached_is_slower, FILE *file) { //function to learn from calibration set for future test
    uint64_t cached[n], uncached[n]; //split training data into cached and uncached
    int n_c = 0, n_u = 0; //number of hits index
    for (int i = 0; i < n; i++) {
        if (touched[i]) cached[n_c++]   = times[i]; //actually placing timing data into arrays
        else            uncached[n_u++] = times[i];
    }
    sort_u64(cached, n_c);
    sort_u64(uncached, n_u); //sort

    uint64_t med_c = median_sorted(cached, n_c); //find median
    uint64_t med_u = median_sorted(uncached, n_u); //find median
    *threshold = (med_c + med_u) / 2; //find threshold by citing middle ground between cached and uncached
    *cached_is_slower = med_c > med_u; //is cached time to flush slower than uncached (NO(0) is usually the right sign)
    printf("TRAINING METRICS: median cached=%llu  median uncached=%llu  " //print all collected statistics that we are measuring against
           "threshold=%llu  cached_is_slower=%d\n",
           (unsigned long long)med_c, (unsigned long long)med_u,
           (unsigned long long)*threshold, *cached_is_slower);
    fprintf(file, "TRAINING METRICS: median cached=%llu  median uncached=%llu  "
           "threshold=%llu  cached_is_slower=%d\n",
           (unsigned long long)med_c, (unsigned long long)med_u,
           (unsigned long long)*threshold, *cached_is_slower);
}

int score(const uint64_t *times, const int *touched, int n,
          uint64_t threshold, int cached_is_slower, FILE *file) { //actual prediction function, takes in metrics from training, against real data
    int correct = 0; //how many guesses correct
    for (int i = 0; i < n; i++) {
        int guess = cached_is_slower ? (times[i] > threshold) //ternary to decide if cache is slower, guess accurately based on timing threshold
                                      : (times[i] < threshold);
        correct += (guess == touched[i]); //use this "known state" array to grade guess
        fprintf(file, "%lu,%d,%d\n", times[i], guess, touched[i]); //print to csv
    }
    return correct; //return number of correct guesses
}


void *victim(void *arg) {
    thread_args *a = (thread_args *)arg; //set arguments for victim thread based on set params (have to pass in as void and recast based on documentation for ptthread)
    uint32_t seed = a->seed; //set seed

    for (int round = 0; round < a->warmup + a->n; round++) {
        pthread_barrier_wait(&round_start); // sync for victim is about to act event

        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        int touch = seed & 1; // flip a coin -- attacker doesn't know this
        if (touch) { volatile char c = target; (void)c; }

        pthread_barrier_wait(&victim_done); // signal that victim has acted and now attacker can time

        if (round >= a->warmup) // start recording after warmup (may avoid early issues)
            a->touched[round - a->warmup] = touch; //fill out touched array to check guesses later
    }
    return NULL;
}


void *attacker(void *arg) {
    thread_args *a = (thread_args *)arg; //again pass args for attacker based on params

    for (int round = 0; round < a->warmup + a->n; round++) {
        pthread_barrier_wait(&round_start); // sync for victim is about to act event
        pthread_barrier_wait(&victim_done); // wait for victim, has now acted and can time flush

        uint64_t t = time_flush(&target); // probe shared memory, and time the clflush

        if (round >= a->warmup) // start recording after warmup (may avoid early issues)
            a->times[round - a->warmup] = t; //fill out time for each access
    }
    return NULL;
}

// Run one phase (train or test): init barriers, start both threads,
// wait for both to finish, then destroy barriers.
static void run_phase(uint64_t *times, int *touched, uint32_t victim_seed) {
    pthread_barrier_init(&round_start, NULL, 2); // 2 threads must arrive for barrier to release
    pthread_barrier_init(&victim_done, NULL, 2); // same for victim doen as well

    pthread_t t_victim, t_attacker; //establish processes for victim and attacker
    thread_args args = { times, touched, N, WARMUP, victim_seed }; //set args for struct defined earlier
    pthread_create(&t_victim,   NULL, victim,   &args); //create addresses for attacker and victim threads and call processes
    pthread_create(&t_attacker, NULL, attacker, &args);
    pthread_join(t_victim,   NULL); //join these, so threads must wait on each other to move on
    pthread_join(t_attacker, NULL);

    pthread_barrier_destroy(&round_start); //frees barriers at reference address, so process can move on to next iter
    pthread_barrier_destroy(&victim_done);
}

int main(void) {

    FILE *file     = fopen("output_ff.csv", "w"); //set output files (data raw csv and measured txt)
    FILE *file_txt = fopen("output_ff.txt", "w");

    if (file == NULL) { //error collection for issue faced
        printf("Failed to open file output_ff.csv\n");
    } else {
        printf("File Opened\n");
    }

    fprintf(file, "latency,prediction,actual\n"); // set csv headers

    uint64_t train_times[N]; int train_touched[N]; //data sets for training times and predetermined cached/uncached
    uint64_t test_times[N];  int test_touched[N];  //declare new data sets for actual test

    uint64_t train_start = __rdtsc(); //simple timing prim for testing train vs run (train should take longer bc of calibration)
    run_phase(train_times, train_touched, 98765); // set run with seed

    uint64_t threshold; int cached_is_slower; //declare threshold vars
    learn(train_times, train_touched, N, &threshold, &cached_is_slower, file_txt); //set threshold vars based on learning function
    int train_correct = score(train_times, train_touched, N,
                              threshold, cached_is_slower, file); //baseline check on data WE KNOW IS GOOD
    uint64_t train_end = __rdtsc(); //timing prim end
    uint64_t train_tot = train_end - train_start; //timing delta

    
    uint64_t test_start = __rdtsc(); //timing prim for test phase
    run_phase(test_times, test_touched, 11111); //start run

    int test_correct = score(test_times, test_touched, N,
                             threshold, cached_is_slower, file); //score WITHOUT re-learning to test if calibration holds
    uint64_t test_end = __rdtsc();
    uint64_t test_tot = test_end - test_start; //timing primitive delta

    // Results 
    printf("\ntrain accuracy: %d/%d = %.1f%%\n", train_correct, N, 100.0 * train_correct / N);
    printf("test accuracy:  %d/%d = %.1f%%\n",  test_correct,  N, 100.0 * test_correct  / N);
    printf("train time: %llu cycles, test time: %llu cycles\n",
           (unsigned long long)train_tot, (unsigned long long)test_tot); //print results to terminal

    fprintf(file_txt, "\ntrain accuracy: %d/%d = %.1f%%\n", train_correct, N, 100.0 * train_correct / N);
    fprintf(file_txt, "test accuracy:  %d/%d = %.1f%%\n",  test_correct,  N, 100.0 * test_correct  / N);
    fprintf(file_txt, "train time: %llu cycles, test time: %llu cycles\n",
           (unsigned long long)train_tot, (unsigned long long)test_tot); //print results to file

    fclose(file);
    fclose(file_txt);//close files
    return 0;
}
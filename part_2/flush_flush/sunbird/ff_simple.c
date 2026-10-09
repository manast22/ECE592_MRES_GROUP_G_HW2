

#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

volatile char target = 85; //target byte that we are trying to observe with ff

uint64_t time_flush(volatile char *addr) { //basic function for running ff
    _mm_mfence(); _mm_lfence(); //set up memory and time fence for any previous instructions
    uint64_t start = __rdtsc(); //use x86 rdtsc for starting timing measurement
    _mm_lfence(); //one more lfence to make sure timing function finishes
    _mm_clflush((const void *)addr); //actual clflush for attack
    _mm_mfence(); _mm_lfence(); //time/memory fence to ensure singularity
    uint64_t end = __rdtsc(); //end timer
    return end - start; //calculate delta of flush instruction
}

uint64_t trial(int touch) { //trial function for finding baseline times
    if (touch) { volatile char c = target; (void)c; } //if touched previously, set this as target pull into cache
    return time_flush(&target); //flush attack target for present and not present- get time
}


void collect(uint64_t *times, int *touched, int n, int warmup) {
    for (int i = 0; i < warmup; i++) trial(i & 1);   // warm up, discard
   for (int i = 0; i < n; i++) {
        touched[i] = i & 1; //alternate even and odd (so 50/50 split between cached and not)
        times[i] = trial(touched[i]); //initialize and time each trial set by N, i
        //basically this function calls flush and flush N-1 times to learn overall sets

   }
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
    uint64_t cached[n], uncached[n]; //split training data into cached and uncaches
    int n_c = 0, n_u = 0; //number of hits index
    for (int i = 0; i < n; i++) {
        if (touched[i]) cached[n_c++]   = times[i]; //actually placing timing data into arrays
        else            uncached[n_u++] = times[i];
    }
    sort_u64(cached, n_c);
    sort_u64(uncached, n_u); //sort 
 
    uint64_t med_c = median_sorted(cached, n_c); //find median
    uint64_t med_u = median_sorted(uncached, n_u);//find mediam
    *threshold = (med_c + med_u) / 2; //find threshold by citing middle ground between cached and uncached
    *cached_is_slower = med_c > med_u; //is cached time to flush slower than uncached (NO(0) is usually the right sign)
    printf("TRAINING METRICS: median cached=%llu  median uncached=%llu  " //print all collected statistics that we are measuring against
           "threshold=%llu  cached_is_slower=%d\n",
           (unsigned long long)med_c, (unsigned long long)med_u,
           (unsigned long long)*threshold, *cached_is_slower);
    fprintf(file, "TRAINING METRICS: median cached=%llu  median uncached=%llu  " //print all collected statistics that we are measuring against
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
        fprintf(file, "%llu,%d, %d\n", times[i], guess, touched[i]);
    }
    
    return correct; //return number of correct guesses

}

int main(void) {

    FILE *file = fopen("output_ff.csv", "w");
    FILE *file_txt = fopen("output_ff.txt", "w");

    if (file == NULL) {
        printf("Failed to open file output.csv\n");
    } else {
        printf("File Opened\n");
    }

    fprintf(file, "latency,prediction,actual\n");


    const int N = 10, WARMUP = 10; //set training samples and test samples
    uint64_t train_start = __rdtsc();
    uint64_t train_times[N]; int train_touched[N]; //est data sets for training times and predetermined cached/uncached
    collect(train_times, train_touched, N, WARMUP); //set these data sets for usage
 
    uint64_t threshold; int cached_is_slower; //declare threshold vars
    learn(train_times, train_touched, N, &threshold, &cached_is_slower, file_txt); //set threshold vars based on learning function
    int train_correct = score(train_times, train_touched, N, threshold, cached_is_slower, file);
    uint64_t train_end = __rdtsc();
    uint64_t train_tot = train_end - train_start;
    //baseline check to make sure scoring is accruate on data WE KNOW IS GOOD
 
        uint64_t test_start = __rdtsc();

    uint64_t test_times[N]; int test_touched[N]; //declare new data sets for actual test
    collect(test_times, test_touched, N, WARMUP); //set these data sets for usage
 
    int test_correct = score(test_times, test_touched, N,threshold, cached_is_slower, file);
    //NOW we can score WITHOUT learning to see if we can use previous calibration to do F + F
         uint64_t test_end = __rdtsc();
    uint64_t test_tot = test_end - test_start;
    printf("\ntrain accuracy: %d/%d = %.1f%%\n", train_correct, N, 100.0 * train_correct / N); //print out training results
    printf("test accuracy:  %d/%d = %.1f%%\n", test_correct, N, 100.0 * test_correct / N); //print out real test results
    printf("train time: %llu, test time: %llu\n", (unsigned long long)train_tot, (unsigned long long)test_tot);

    fprintf(file_txt,"\ntrain accuracy: %d/%d = %.1f%%\n", train_correct, N, 100.0 * train_correct / N); //print out training results
    fprintf(file_txt,"test accuracy:  %d/%d = %.1f%%\n", test_correct, N, 100.0 * test_correct / N); //print out real test results
    fprintf(file_txt,"train time: %llu, test time: %llu\n", (unsigned long long)train_tot, (unsigned long long)test_tot);

    return 0;
}


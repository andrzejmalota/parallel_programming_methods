#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>


void fill_array(int* source_array, long int problem_size) {
     int* array = source_array;
     int id = omp_get_thread_num();
     int i;
     unsigned int seed = time(NULL)*id;

    #pragma omp for schedule(static)
         for (i = 0; i < problem_size; i++) {
             array[i] = rand_r(&seed) % problem_size;
         }
}

void find_buckets_sizes(int* buckets_start_idxs, int buckets_num, int** buckets, int* thread_start_idxs, int* buckets_sizes, int* array, int bucket_range, omp_lock_t* writelocks) {
    int id = omp_get_thread_num();
    int i, value;
    for(int i=thread_start_idxs[id]; i < thread_start_idxs[id+1]; i++) {
        value = array[i];
        int bucket_idx;
        bucket_idx = value/bucket_range;
        
        omp_set_lock(&(writelocks[bucket_idx]));
        buckets_sizes[bucket_idx]++;
        omp_unset_lock(&(writelocks[bucket_idx]));
    }

    for(i=0; i<buckets_num; i++) {
        buckets[i] = (int*)malloc(buckets_sizes[i] * sizeof(int));
    }

    #pragma omp master 
    {
        if(buckets_num > 1) {
            for(i=1; i<buckets_num; i++) {
                buckets_start_idxs[i] = buckets_start_idxs[i-1] + buckets_sizes[i-1];
            }
        }
    }
}

void split_numbers_into_buckets(int* thread_start_idxs, int* bucket_current_idx, int** buckets, int* array, int bucket_range, omp_lock_t* writelocks) {
    int id = omp_get_thread_num();
    int i;

    for(i=thread_start_idxs[id]; i < thread_start_idxs[id+1]; i++) {
        int value, tmp_bucket_current_idx, bucket_idx;
        value = array[i];
        bucket_idx = value/bucket_range;
        
        omp_set_lock(&(writelocks[bucket_idx]));
        tmp_bucket_current_idx = bucket_current_idx[value/bucket_range];
        buckets[value/bucket_range][tmp_bucket_current_idx] = value;
        bucket_current_idx[value/bucket_range]++;
        omp_unset_lock(&(writelocks[bucket_idx]));
    }
}

int my_compare (const void * a, const void * b)
{
    int _a = *(int*)a;
    int _b = *(int*)b;
    if(_a < _b) return -1;
    else if(_a == _b) return 0;
    else return 1;
}

int compare (const void * a, const void * b){
    return ( *(int*)a - *(int*)b );
}

void sort_buckets(int buckets_num, int* buckets_sizes, int** buckets) {
    int i;
    #pragma omp for schedule(dynamic) private(i)
        for(i=0; i<buckets_num; i++) 
            qsort(buckets[i], buckets_sizes[i], sizeof(int), my_compare);
}


void move_numbers_from_buckets_to_array(int buckets_num, int* buckets_sizes, int** buckets, int* buckets_start_idxs, int* array) {
    int i, start_idx;
    #pragma omp for schedule(static)
        for(i=0; i<buckets_num; i++) {
            start_idx = buckets_start_idxs[i];
            memcpy(&(array[start_idx]), buckets[i], buckets_sizes[i] * sizeof(int));
        }
}




int main(int argc, char *argv[]) {
    bool verbose = false;
    int threads_num = atoi(argv[1]);
    omp_set_num_threads(threads_num);
    int buckets_per_thread = atoi(argv[2]);
    long int problem_size = atoi(argv[3]); 
    float t1, t2, t3, t4, t_all, t_start, t_start_all;
    int i, thread_idx_range;
    int buckets_num = buckets_per_thread * threads_num;
    long int range = (long int) problem_size;
    long int tmp_bucket_range = (long int)(range / buckets_num);
    long int bucket_range = range % buckets_num == 0 ? tmp_bucket_range : tmp_bucket_range + 1;
    int* array = (int*)malloc(problem_size * sizeof(int));
    int** buckets = (int**)malloc(buckets_num * sizeof(int*));;
    int* buckets_sizes = (int*)malloc(buckets_num * sizeof(int));    
    memset(buckets_sizes, 0, buckets_num * sizeof(int));
    int* buckets_start_idxs = (int*)malloc((buckets_num + 1) * sizeof(int));
    memset(buckets_start_idxs, 0, buckets_num * sizeof(int)); 
    int* bucket_current_idx = (int*)malloc(buckets_num * sizeof(int));  
    memset(bucket_current_idx, 0, buckets_num * sizeof(int)); 
    int* thread_start_idxs = (int*)malloc((threads_num + 1)* sizeof(int));
    thread_idx_range = (int)problem_size / threads_num;

    for(i=0; i<threads_num; i++) {
        thread_start_idxs[i] = thread_idx_range * i;
    }

    
    thread_start_idxs[threads_num] = problem_size;


    omp_lock_t* writelocks = (omp_lock_t*)malloc(buckets_num * sizeof(omp_lock_t));
    
    int b;
    for(b=0; b <buckets_num; b++) {        
        omp_init_lock(&(writelocks[b]));
    }

    
    // main algorithm functions

    t_start = omp_get_wtime();
    t_start_all = omp_get_wtime();

    #pragma omp parallel 
    {
        fill_array(array, problem_size);

        #pragma omp master
            t1 = omp_get_wtime() - t_start;
    }

    #pragma omp parallel shared(writelocks) 
        {
        find_buckets_sizes(buckets_start_idxs, buckets_num, buckets, thread_start_idxs, buckets_sizes, array, bucket_range, writelocks);
        #pragma omp barrier



    #pragma omp master
        t_start = omp_get_wtime();
        split_numbers_into_buckets(thread_start_idxs, bucket_current_idx, buckets, array, bucket_range, writelocks); 
        t2 = omp_get_wtime() - t_start;

    #pragma omp barrier
    #pragma omp master
        t_start = omp_get_wtime();
        sort_buckets(buckets_num, buckets_sizes, buckets); 
        t3 = omp_get_wtime() - t_start;
        

    #pragma omp barrier
    #pragma omp master
        t_start = omp_get_wtime();
        move_numbers_from_buckets_to_array(buckets_num, buckets_sizes, buckets, buckets_start_idxs, array); 
        t4 = omp_get_wtime() - t_start;
        t_all = omp_get_wtime() - t_start_all;
    }

    for(b=0; b <buckets_num; b++)
    {        
        omp_destroy_lock(&(writelocks[b]));
    }

    // print sorted buckets
    if(verbose){
        int i, j;
        printf("\n");
        for(i=0; i<buckets_num; i++)
        {
            printf("Bucket nr: %d: ", i);
            for(j=0; j<buckets_sizes[i]; j++)
                printf("%d, ", buckets[i][j]);
            printf("\n");
        }
        printf("\n");
    }

    // print sorted array
    if(verbose){
        int i;
        printf("\n");
        for(i=0; i<problem_size; i++)
            printf("%d, ", array[i]);
        printf("\n");
    }
    
    printf("problem_size: %ld, threads_num: %d, buckets_per_thread: %d, t1: %f, t2: %f, t3: %f, t4: %f, t_all: %f \n", problem_size, threads_num, buckets_per_thread, t1, t2, t3, t4, t_all);

    free(buckets);
    buckets = NULL;   
   
    free(array);
    array = NULL;

    free(buckets_start_idxs);
    buckets_start_idxs = NULL;
   
    free(bucket_current_idx);
    bucket_current_idx = NULL;
   
    free(thread_start_idxs);
    thread_start_idxs = NULL;
   
    free(buckets_sizes);
    buckets_sizes = NULL;

    return 0;

}

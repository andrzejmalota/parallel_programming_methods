#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>

#define MAX_RAND_VAL 10

void print_array(int* array, int length, bool verbose)
{
    if(verbose){
    int i;
    printf("\n");
    for(i=0; i<length; i++)
        printf("%d, ", array[i]);
    printf("\n");
    }
}

int compare (const void * a, const void * b){
    return ( *(int*)a - *(int*)b );
}

bool evaluate(int* array, int len){
    bool correct = true;
    int i;
#pragma omp for schedule(guided)
        for(i=0; i<len-1; i++) {
            if (array[i] > array[i+1])
                correct = false;
        }
    return correct;
}


typedef struct{
    long int bucket_range;
    int threads_num;
    int buckets_num;
    long int array_len;
    int* source_array;
    int* histogram;
    int** buckets;
    int* buckets_sizes;
    int* buckets_start_idxs; //indexes from where i-th bucket has to be pasted into source_array, last element has a value of array_length, so size of buckets_start_idxs is buckets_num+1
    int* bucket_current_idx; //indexes in i-th bucket to which we wil write
    int* thread_start_idxs; //indexes from which i=th thread is reading, last element has a value of array_length, so size of thread_start_idxs is threads_num+1
}Config;

void initConfig(Config* config, int threads_num, int buckets_per_th, long int array_len){  
    int i, thread_idx_range;
    
    config->buckets_num = buckets_per_th * threads_num;
    config->array_len = array_len;
    config->threads_num = threads_num;
    omp_set_num_threads(config->threads_num);

    long int range = (long int) config->array_len; //po testach zmienić na RAND_MAX + 1
    long int tmp_bucket_range = (long int)(range / config->buckets_num);
    config->bucket_range = range % config->buckets_num == 0 ? tmp_bucket_range : tmp_bucket_range + 1;

    config->source_array = (int*)malloc(config->array_len * sizeof(int));
    config->histogram = (int*)malloc(MAX_RAND_VAL * sizeof(int));
    memset(config->histogram, 0, sizeof(int) * MAX_RAND_VAL);
    
    config->buckets = (int**)malloc(config->buckets_num * sizeof(int*));
    //po przejściu raz przez source_array zdecydujemy o wielkości buckets[i]
    
    config->buckets_sizes = (int*)malloc(config->buckets_num * sizeof(int));    
    memset(config->buckets_sizes, 0, config->buckets_num * sizeof(int));
    
    config->buckets_start_idxs = (int*)malloc((config->buckets_num + 1) * sizeof(int));
    memset(config->buckets_start_idxs, 0, config->buckets_num * sizeof(int));
    config->buckets_start_idxs[config->buckets_num] = config->array_len;
     
    config->bucket_current_idx = (int*)malloc(config->buckets_num * sizeof(int));  
    memset(config->bucket_current_idx, 0, config->buckets_num * sizeof(int));
    
    config->thread_start_idxs = (int*)malloc((config->threads_num + 1)* sizeof(int));
    thread_idx_range = config->array_len / config->threads_num;
    for(i=0; i<config->threads_num; i++)
    {
        config->thread_start_idxs[i] = thread_idx_range * i;
    } //dla 11 elem i 2 wątków wartości: 0, 6
    config->thread_start_idxs[config->threads_num] = config->array_len;
}
void freeConfig(Config *c){
    //uncomment free when buckets will be used
   int i;
   for(i=0; i<c->buckets_num; i++){
        int* ptr = c->buckets[i];
        free(ptr);
        ptr = NULL;
    }
   
   free(c->source_array);
   c->source_array = NULL;

   free(c->buckets_start_idxs);
   c->buckets_start_idxs = NULL;
   
   free(c->bucket_current_idx);
   c->bucket_current_idx = NULL;
   
   free(c->thread_start_idxs);
   c->thread_start_idxs = NULL;
   
   free(c->source_array);
   c->source_array = NULL;

   c->bucket_range = c->threads_num = c->buckets_num = c->array_len = 0;
}

void rand_fill_array(Config* config){
     int i, id;
     unsigned int seed;
     int* array = config->source_array;
     id = omp_get_thread_num();
     seed = time(NULL)*id;
#pragma omp for schedule(static)
     for (i = 0; i < config->array_len; i++)
     {
         array[i] = rand_r(&seed) % MAX_RAND_VAL;
     }
}

//allocates memory for each bucket to avoid dynamic extension/reallcoation
void prepare_buckets(Config* config){
    int id = omp_get_thread_num();
    int i, val;
    for(i=config->thread_start_idxs[id]; i < config->thread_start_idxs[id+1]; i++)
    {
        val = config->source_array[i];
        config->buckets_sizes[val/config->bucket_range]++;
    }
    for(i=0; i<config->buckets_num; i++)
    {
        config->buckets[i] = (int*)malloc(config->buckets_sizes[i] * sizeof(int));
    }
    
#pragma omp master
{
    if(config->buckets_num > 1)
    {
        //printf("ID in prepare: %d\n", id);
        for(i=1; i<config->buckets_num; i++)
        {
            config->buckets_start_idxs[i] = config->buckets_start_idxs[i-1] + config->buckets_sizes[i-1];
        }
    }
}
}

void fill_buckets(Config* config, omp_lock_t* writelock_ptr)
{
    int id = omp_get_thread_num();
    int i;
    for(i=config->thread_start_idxs[id]; i < config->thread_start_idxs[id+1]; i++)
    {
        int val, bucket_current_idx;
        omp_unset_lock(writelock_ptr);
        val = config->source_array[i]; //tu nie ma problemu
        bucket_current_idx = config->bucket_current_idx[val/config->bucket_range]; //tu jest problem
        config->buckets[val/config->bucket_range][bucket_current_idx] = val; //tu jest problem
        config->bucket_current_idx[val/config->bucket_range]++;
        //printf("ID: %d, val: %d, b_idx: %ld, idx: %d\n", id, val, val/config->bucket_range, bucket_current_idx);
        omp_unset_lock(writelock_ptr);
    }
}


void in_buckets_sort(Config* config){
    int i;
#pragma omp for schedule(static) private(i)
    for(i=0; i<config->buckets_num; i++)
        qsort(config->buckets[i], config->buckets_sizes[i], sizeof(int), compare);
}

void print_buckets(Config* config, bool verbose){
    if(verbose){
    int i, j;
    printf("\n");
    for(i=0; i<config->buckets_num; i++)
    {
        printf("Bucket nr: %d: ", i);
        for(j=0; j<config->buckets_sizes[i]; j++)
            printf("%d, ", config->buckets[i][j]);
        printf("\n");
    }
    printf("\n");
    }
}


void histogram(Config* config){
    int i, val;
    for(i=0; i<config->array_len; i++)
    {
        val = config->source_array[i];
        config->histogram[val]++;
    }
}

void transfer_data(Config* config){
    int i, start_idx;
    
#pragma omp for schedule(static) private(i)
    for(i=0; i<config->buckets_num; i++)
    {
        start_idx = config->buckets_start_idxs[i];
        memcpy(&(config->source_array[start_idx]), config->buckets[i], config->buckets_sizes[i] * sizeof(int));
        //printf("ID: %d, s_i: %d, i: %d\n", omp_get_thread_num(), start_idx, i);
    }
}

int main(int argc, char *argv[]) {

    bool verbose = false;
    omp_lock_t writelock;
    omp_init_lock(&writelock);
    int th = atoi(argv[1]);
    int buckets_per_th = atoi(argv[2]);
    long int problem_size = atoi(argv[3]); 
    Config* config = malloc(sizeof(Config));
    initConfig(config, th, buckets_per_th, problem_size); //25000000
    printf("Configuration: %d threads, %d buckets.\n", config->threads_num, config->buckets_num);
    float t1, t2, t3, t4, t5, total_timer1;
    bool print_histogram = false;
    
    total_timer1 = omp_get_wtime();
#pragma omp parallel
{
    rand_fill_array(config);
#pragma omp master
    t1 = omp_get_wtime();
}
    print_array(config->source_array, problem_size, verbose);
    if(print_histogram)
    {
        histogram(config);
        print_array(config->histogram, MAX_RAND_VAL, verbose);
    }

#pragma omp parallel shared(writelock)
{
    prepare_buckets(config);
#pragma omp master
    t2 = omp_get_wtime();

#pragma omp barrier
    fill_buckets(config, &writelock); //brakuje działającego locka
#pragma omp master
    t3 = omp_get_wtime();
    
    in_buckets_sort(config);
#pragma omp master
    t4 = omp_get_wtime();
    
    transfer_data(config);
#pragma omp master
    t5 = omp_get_wtime();
}
omp_destroy_lock(&writelock);

    if(verbose){
        printf("range: %ld", config->bucket_range);
    }
    
    print_array(config->source_array, problem_size, verbose);
    print_array(config->thread_start_idxs, th + 1, verbose);
    print_array(config->buckets_sizes, config->buckets_num, verbose);
    print_array(config->buckets_start_idxs, config->buckets_num + 1, verbose);
    print_buckets(config, verbose);

    printf("%ld, %d, %d, %f, %f, %f, %f, %f, %f\n", problem_size, th, buckets_per_th, t5-total_timer1, t1 - total_timer1, t2-t1, t3-t2, t4-t3, t5-t4);

    freeConfig(config);
    
    return 0;
}

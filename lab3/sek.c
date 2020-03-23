#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "math.h"

double calc_mean(long double data[], double N)
{
    double sum = 0.0, mean;
    for (int i = 0; i < N; ++i) { sum += data[i];}
    mean = sum / N;
    return mean;
}

double calc_std(long double data[], double N)
{
    double SD = 0.0, mean;
    mean = calc_mean(data, N);

    for (int i= 0; i < N; ++i)
        SD += pow(data[i] - mean, 2);

    return sqrt((double)(SD / N));
}

int main(int argc, char ** argv) {
    const double R = 1.00;
    const int N = 1;
    long long unsigned int num_pts_square = 0;
    long long unsigned int num_pts_circle = 0;
    long double x, y, d, pi, results[N], pi_mean, pi_std;

    if (argc > 0) { num_pts_square = atoi(argv[1]);}
    else { exit(-1);}

    for (int j=0; j<N; j++)
    {
        num_pts_circle = 0;
        srand((unsigned int)time(NULL));

        for (int i=0; i<num_pts_square; i++)
        {
            x = (double)rand() / RAND_MAX;
            y = (double)rand() / RAND_MAX;

            d = sqrt((x*x + y*y));
            if (d <= R) { num_pts_circle++;}
        }

        pi = (long double)4.0 * ((long double)num_pts_circle / (long double)num_pts_square);
        results[j] = pi;
    }

    pi_mean = calc_mean(results, N);
    pi_std = calc_std(results, N);
    printf("MEAN PI: %Lf \n", pi_mean);
    printf("STD PI: %Lf \n", pi_std);

    return 0;
}




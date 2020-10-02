#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include "darknet.h"
#include "list.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TIME(a) \
    do { \
    double start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

double what_time_is_it_now();

int int_index(int *a, int val, int n);
int constrain_int(int a, int min, int max);

char *fgetl(FILE *fp);
void malloc_error();
void file_error(char *s);

void strip(char *s);

char *copy_string(char *s);

#ifdef __cplusplus
}
#endif

#endif


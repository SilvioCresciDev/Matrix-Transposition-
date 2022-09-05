#include <stdio.h>
#include<time.h>
#include<omp.h>
#include <errno.h>   // for errno
#include <limits.h>  // for INT_MAX, INT_MIN
#include <stdlib.h>  // for strtol
#include <immintrin.h>


void trasposta(int **matrix, int num_colonne);
void matrix_switch(int **matrix, int size);

static inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi32(va, vb);
    *vh = _mm256_unpackhi_epi32(va, vb);
}

static inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi64(va, vb);
    *vh = _mm256_unpackhi_epi64(va, vb);
}

static inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0));
    *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1));
}

static void Transpose_8_8(
    __m256i *v0,
    __m256i *v1,
    __m256i *v2,
    __m256i *v3,
    __m256i *v4,
    __m256i *v5,
    __m256i *v6,
    __m256i *v7)
{
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;
    __m256i x0, x1, x2, x3, x4, x5, x6, x7;

    _mm256_merge_epi32(*v0, *v1, &w0, &w1);
    _mm256_merge_epi32(*v2, *v3, &w2, &w3);
    _mm256_merge_epi32(*v4, *v5, &w4, &w5);
    _mm256_merge_epi32(*v6, *v7, &w6, &w7);
        
    _mm256_merge_epi64(w0, w2, &x0, &x1);
    _mm256_merge_epi64(w1, w3, &x2, &x3);
    _mm256_merge_epi64(w4, w6, &x4, &x5);
    _mm256_merge_epi64(w5, w7, &x6, &x7);

    _mm256_merge_si128(x0, x4, v0, v1);
    _mm256_merge_si128(x1, x5, v2, v3);
    _mm256_merge_si128(x2, x6, v4, v5);
    _mm256_merge_si128(x3, x7, v6, v7);

}
 
int main (int argc, char **argv){

    clock_t begin = clock();

char *p;
int size;

errno = 0;
long conv = strtol(argv[1], &p, 10);

// Check for errors: e.g., the string does not represent an integer
// or the integer is larger than int
if (errno != 0 || *p != '\0' || conv > INT_MAX || conv < INT_MIN) {
    // Put here the handling of the error, like exiting the program with
    // an error message
} else {
    // No error
    size = conv;    
}

int **matrix;

matrix = (int**) aligned_alloc (32, size * sizeof(int*));

for (int r = 0; r<size; r++){
    matrix[r] = (int *) aligned_alloc (32, size * sizeof(int));
}
/*
matrix = (int**) malloc ( size * sizeof(int*));

for (int r = 0; r<size; r++){
    matrix[r] = (int *) malloc ( size * sizeof(int));
}
*/
for (int i = 0; i< size; i++){
    for (int j = 0; j < size; j++){
        matrix[i][j] = j + (size * i);
    }
}
/*
for (int i = 0; i< size; i++){
    for (int j = 0; j < size; j++){
        printf("%4d",matrix[i][j]); 
    }
    printf("\n");
}
*/
   trasposta(matrix, size);
    printf("\n");
    printf("\n");
   matrix_switch(matrix,size);

/*for (int i = 0; i< size; i++){
    for (int j = 0; j < size; j++){
        printf("%4d",matrix[i][j]); 
    }
    printf("\n");
}
*/
clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

printf("\n\nThe elapsed time is %f seconds\n\n", time_spent);

}

void matrix_switch(int **matrix, int size){
    int i,j, temp;
    int matrix_tmp[8][8];
    int tileSize = 8;
    for (int ih = 0; ih <size; ih += tileSize){
         if(ih<size-tileSize){
            for(int jh = ih; jh < size; jh += tileSize){
                if(jh<size-tileSize){
                    if(ih!=jh){
                        for(i=0;i<tileSize;i++){
                            #pragma omp simd aligned(matrix:32) 
                            for(j=0;j<tileSize;j++){
                                matrix_tmp[i][j] = matrix[i+ih][j+jh];
                                matrix[i+ih][j+jh] = matrix[i+jh][j+ih];
                                matrix[i+jh][j+ih] = matrix_tmp[i][j];

                            }
                        }

                    }
                }
            }
        } else {
                for(int i = ih; i< size; i++){
                    for(int j = 0; j < i; j++){
                        temp = matrix[i][j];
                        matrix[i][j] = matrix[j][i];
                        matrix[j][i] = temp;
                    }
                }
            
        }
    }
}

void trasposta(int **matrix, int size){
    int i,j,ih,jh,temp;
    int tileSize = 8;
    int32_t matrix8[8][8] __attribute__ ((aligned(32)));
    
    for( ih = 0; ih< size; ih += tileSize){
        if(ih<size-tileSize){
            for( jh = 0; jh < size; jh += tileSize){
                if(jh<size-tileSize){
                    //printf("%d\n",jh);
                    for (i=0;i<tileSize;i++){
                        #pragma omp simd aligned(matrix:32)
                        for (j=0;j<tileSize;j++){     
                            matrix8[i][j] = matrix[i+ih][j+jh];      
                        }
                    }
                    Transpose_8_8((__m256i *)matrix8[0], (__m256i *)matrix8[1], (__m256i *)matrix8[2], (__m256i *)matrix8[3], (__m256i *)matrix8[4], (__m256i *)matrix8[5], (__m256i *)matrix8[6], (__m256i *)matrix8[7]);
                    
                    for (i=0;i<tileSize;i++){
                        //#pragma omp parallel simd for
                        for (j=0;j<tileSize;j++){     
                            matrix[i+ih][j+jh] = matrix8[i][j];    
                        }
                    }
                } 
            }

        } 
    }
}
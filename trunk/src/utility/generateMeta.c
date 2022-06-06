#include <stdio.h>
#include <stdlib.h>
#include "../include/common.h"

int main(int argc, char **argv)
{
    FILE *fp;
    int outFd;
    long outSize;
    long offset, tupleOffset;
    int blockTotal;
    struct columnHeader header;
    int size_of_element;
    int *hash_table;
    int number_of_distinct_values = 0;
    int number_of_records = 0;
    int min = 0x7FFFFFFF, max = 0;
    
//    unsigned char *buffer;
    int inputValue;
    unsigned int hash_value = 0;
    // size_of_element = atoi(argv[2]);
//    buffer = (char *)calloc(size_of_element+1, sizeof(char));
    hash_table = (int *)calloc(65536, sizeof(int));
    fp = fopen(argv[1],"rb");
    fread(&header, sizeof(struct columnHeader), 1, fp);

    blockTotal = header.blockTotal;
    number_of_records = header.totalTupleNum;

    offset = 0, tupleOffset = 0;

    /* Each block has one separate column header */
    int i;
    for (i = 0; i < blockTotal; i++)
    {
        offset = i * sizeof(struct columnHeader) + tupleOffset * sizeof(int);
        fseek(fp, offset, SEEK_SET);
        fread(&header, sizeof(struct columnHeader), 1, fp);

        offset += sizeof(struct columnHeader);
        outSize =header.tupleNum * sizeof(int);
        
        // printf("outSize: %ld\n", outSize);
        while (fread(&inputValue, sizeof(int), 1, fp)) {
            hash_value = (0xFFFF) & inputValue;
            if (inputValue > max) {
                max = inputValue;
            }
            if (inputValue < min) {
                // printf("update min %d\n", min);
                min = inputValue;
            }
            if(hash_table[hash_value] == 0)
                number_of_distinct_values++;
            hash_table[hash_value]++;
        }

    }
    fprintf(stderr, "#records %d #distinct %d min %d max %d\n", number_of_records, number_of_distinct_values, min, max);

    fclose(fp);

    return 0;
}
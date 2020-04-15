#ifndef __GPU_CUDAHASH__
#define __GPU_CUDAHASH__

__device__ static unsigned int StringHash(const char* s)
{
    unsigned int hash = 0;
    int c;
 
    while((c = *s++))
    {
        hash = ((hash << 5) + hash) ^ c;
    }
 
    return hash;
}

#endif

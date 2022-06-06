all: generateMeta loader

loader: load.c
	gcc -o gpuDBLoader load.c

generateMeta: generateMeta.c
	gcc -o generateMeta generateMeta.c

sort: columnSort.c
	gcc -o columnSort columnSort.c -std=c99 

rle: rle.c
	gcc -std=c99 rle.c -o rleCompression

dict: dict.c
	gcc -std=c99 dict.c -o dictCompression

clean:
	rm -rf *.o gpuDBLoader generateMeta columnSort rleCompression dictCompression 

      

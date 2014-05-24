all:
	nvcc -arch sm_20 rmap.cu -o rmap
clean:
	rm ./rmap ./rmap.exp ./rmap.lib

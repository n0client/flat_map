# flat_map


Designed to be a header only hashmap implementation. I still have yet to add the fancy stl-like stuff and other important functionality. 

Uses linear probing and robin-hood. 

It does require arm_neon.h. I have not yet explored the equivalent sse intrinsics. 

A benchmark I ran a couple times on my laptop (apple m1 pro, 16gb ram, 10 (8 perf, 2 eff) cores, (16 gpu cores, but they shouldn't really do much))

inserted 100000000 uint64_t pairs in 5286 ms
looked up 100000000 uint64_t pairs in 2015 ms
checksum 100000000
removed 100000000 uint64_t pairs in 2673 ms
Checksum 100000000
cleared 100000000 uint64_t pairs in 41 ms

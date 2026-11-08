[13-june-2026] 

research log on running inference of GLM based model on local system.

Server: Mac M1 Pro 16 GB RAM with 512 GB unified memory
Usage: At max load I need to find a way to measure the
usage of one prompt.

cli:

```
llama-cli -m GLM-4.6V-Flash-Q4_K_M.gguf -ngl 40 --ctx-size 8192
```

from the parameter, I'm using GLM 4.6 Flash with Q4 KM - GGUF with context size 8192 and interestingly, I'm tunning
which size of the offloading from system's RAM to the M1's GPU VRAM. Currently, it's set to 40. What's 40 means?

Let's break it down.
NGL is abvr from number of GPU layers, which define how much layers will be offloaded from the system's RAM to the GPU
VRAM. 

With above parameters, I got this result:

[ Prompt: 43.9 t/s | Generation: 12.2 t/s ]

Not fast enough. Let's increase the NGL number to 60

[ Prompt: 47.8 t/s | Generation: 15.8 t/s ]

Okay, now we are talking. I got 15.8 t/s :)

Let's increase it to 80

[ Prompt: 45.1 t/s | Generation: 15.5 t/s ]

Hmm, there's no much changes - drastic even, now we got .5 slowdown of performance. 

Increased to 99.

[ Prompt: 45.9 t/s | Generation: 15.4 t/s ]
I think we have hit the wall.




1. [ ] Train and evaluate only
2. [ ] prepare a readme.md for code initiate instructions A-Z
3. [ ] prepare pyproject.toml + poetry .lock file (no previous knowledge on this have to study a bit)
4. [ ] else, requirements.txt will do (have knowledge on this)
5. [ ] jupyter notebooks for visualization and ~~evaluation~~ only.

### FOCUS

1. code skills (linter, format, struct etc)
2. DL knowledge
3. presentation and comm

### DATASET CREATION

1. [X] img sz 224x224x3
2. [X] rand rect, ellips
3. [X] wxh betn 20-40 px
4. [X] 3 different colors shapes
5. [X] multi (different) shapes can share same color
6. [ ] show tiled format dataset
7. [ ] acc doesnt matter, matter most is insights, coding struct and above FOCUS

### IMPLEMENT ISN

1. python
2. torch
3. OpenCV, Pillow
4. segment only rects
5. train
6. evaluate
7. readme
8. github

### PRESENTATION

1. 30-50 mins presentation, discussion, Q&A
2. discuss challenges and solutions
3. discuss potential improvements
4. occlusion (have to prep data with occlusion as well)
5. comment results on occlusions
6. with code remove them from dataset, if so, how did that. (I believe delete them considering noisy data !!!)

### PROCESS

1. trials on image generations
   1. shapes generation
   2. color generations
   3. randomize them
   4. fit in size constraints
   5. overlap, occlusions creations (img_gen2.py)
   6. masks creation (image_mask.py)
   7. multiple data creation at once
   8. final prep dataset
2. prepare training code
3. prepare valid/eval code
4. 


### CAN BE DONE


1. in code - arg parser generation for the arguments [done]
2. datasert_generator code is meant for running one time. it can be run multiple time as well but it will going to replace previous data. It's not handled to append instead of replace

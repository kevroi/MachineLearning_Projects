# Learning to Walk

This follow's [Phil Tabor's Implementation](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/TD3) of TD3 for the Bipedal Walker environment from [OpenAI Gym](https://www.gymlibrary.dev/environments/box2d/bipedal_walker/). We then extend this with the FORK (FORward looKing) Actor modification, by [Wei & Ying 2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9683288).

We record video capturesof the agent's progress as well as a text log of the reward, and the relevant moving averages to compare TD3 with its FORK modification

We observe the performance of our agent in the normal setting, and then try hardcore for fun :)

Results in normal:
![Normal Bipedal Walker video](./results/normal_episode%3D750_score%3D294.mp4)

Results in Hardcore:
![Hardcore Bipedal Walker video](./results/hardcore_episode%3D250.mp4) 

Blooper where it thought standing still and waiting out the time limit I set was its best bet:
![Blooper Bipedal Walker video](./results/blooper_episode%3D350.mp4)
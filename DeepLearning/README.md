## 1. Getting Started 🎬
You can choose to either run the code for the topics we explore on your local machine or on Google Colab (a python notebook server by Google that you can use for free). If, like me, your local machine does not have a particularly good GPU, Google Colab offers the chance to run large deep learning (DL) training sessions on their GPUs (with some time restrictions and cooldown periods I discovered the hard way).

### 1.1 Using Colab 🌐
I will include Colab versions of all scripts added here.

### 1.2 Using your own computer 💻
Your own computer will very likely be able to run a lot of the initial scripts included here, however the further your progress and the larger your Neural Network (NN) gets, I will slowly convince you to use Colab for time, heating and electricity bill related benefits :)

To build NNs will often need to leverage software libraries written by much smarter people, to save yourself from reinventing the wheel each time. There are many different python libraries out there, most notably `pytorch`, `tensorflow`, `theano` and `JAX`. We'll be using `pytorch` and a few other python packages, all loaded into a virtual environment. This is like a container for your python script and all its python dependencies - it makes organising what version is used where much easier, without interfering with your computer's operating system. To create a virtual environemnt go into the directory you wish to have all your DL projects in, and enter the following into your Command Line Interface (CLI, thats Terminal for Mac or Linux and Command Prompt or PowerShell for Windows):
```
python3 -m venv venv
```
The second `venv` can be replaced with your virtual environemnt name. I usually just call it venv
This should create a `venv` folder in your directory, which can be activated by:
```
source venv/bin/activate
```

You should now see a `(venv)` in the beginning of your command line! Congrats! You're inside a virtual environment, You can now install all the libraries and packages you'd like using `pip install`, and when you leave your environment using the `deactivate` command, your computer goes back to the standard python libraries and anything you installed earlier outside the `venv`. Lucky for you, I've listed all the python libraries I installed when making this in `requirements.txt`, which `pip` is smart enough to read with the command:
```
pip install -r requirements.txt
```

In short, `pip` is your package manager, `venv` is your virtual environment manager.

Note that you may not have access to `pip` if you are not the `sudo` (super user) for your Mac/Linux machine. In that case, the nextbest package manager I would recommend is Miniconda.

## 2. Why PyTorch? 🔥
You probably noticed that I mentioned we'll be relying on `pytorch` on all our DL projects. The main reason that drew me towards it was its dynamic graph feature. This means that as you code a NN architecture, PyTorch generates a graph structure on the fly - which as we'll come to see is incredibly useful for NN architectures and functiones which need to change as we run them. Tensorflow was a bit late on the bandwagon for this, but Google's other DL library, JAX, is gaining popularity for its performance.

Most of the DL research community and industry use PyTorch. I may have a go at writing some JAX tutorials if time permits.



## [Optional] Accessing a more powerful computer 🕹
If you have a computer with a nice GPU on your WiFi or other network, you can SSH into it from the machine you're on. A good GPU is by no means a pre-requisite, and Google Colab works just fine for most of your use cases.

Not that this is a general procedure that can be used to SSH into a computer on any network you are authorised to, not just for deep learning: from your GPU cluster to your gaming PC or even a raspberry pi!

Check if the device (named `devicename`) is on your network by typing the following command into your CLI
```
ping devicename.local
```
this. should reveal your devices IP address. You can exit out of the command by pressing `Cmd`+`C`

To SSH into the computer, enter the following into your CLI:
```
ssh name@IPAddress
```
You will now be asked to authorise your access by logging into the machine you're hoping to access with a password. you should now see a terminal window within the nicer computer you've accessed! You can leave your SSH session using the `exit` command.

## [Optional] `tmux` and servers 🎚
Will add a section explaining how to use `tmux` to leave terminal sessions running on your other computers as wella s the use of `ipython`. These are useful tools for responsibly using shared resources like a GPU cluster.

## [Optional] Data Visualisation 🎨
If you're tired of `matplotlib`, come back here later for more info on `visdom` and Weights and Biases
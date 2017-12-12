# Jupyter nootebooks for tutorial

Jupyter is an interactive computing environment that works across many programming languages.  IPython is very similar, except specific for python (and now replaced by Jupyter, I think)

Jupyter documentation is available http://jupyter.org/documentation.html

To install jupyter on your own machine, go here: http://jupyter.org/install.html

To use jupyter on a remote machine, typically you need ssh port forwarding.  An example on a GPU tower might go like this:

```
ssh user@gputower.university.edu
[success]
[setup environment (root, python, larcv, etc)]
jupyter notebook
```

You will get output like this:
```
[I 15:33:30.480 NotebookApp] Serving notebooks from local directory: /home/user
[I 15:33:30.480 NotebookApp] 0 active kernels 
[I 15:33:30.480 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/?token=98734813965897315687436589135143 (I replaced this with random numbers)
[I 15:33:30.480 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 15:33:30.481 NotebookApp] 
    
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=98734813965897315687436589135143 (I replaced this with random numbers)
```

If you open a browser and try to copy/paste that link (http://localhost:8888 ....) it doesn't work yet, because that's a port on the remote machine (8888) and your localhost on your laptop is not the localhost on the remote machine.  You can make your laptop listen to that remote port if you open an ssh tunnel to do so (in a different terminal):
```
ssh -L 8888:localhost:8888 user@gputower.university.edu
```
This will give you another log in, and now if you copy/paste the above link into your browser you will see the interactive jupyter session.

When you open the jupyter interface, navigate to this directory and open "JupyterBasics" to see some simple examples of how to use jupyter.

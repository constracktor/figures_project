# figures_project

Repository that contains the measured data and figures of the non-linear system
identification with Gaussian processes minimum working examples used for
"Scalability of Gaussian Processes using Asynchronous Tasks:
A Comparison between HPX and PETSc"

## Run code (requires nmupy and matplotlib)
`python3 plot.py`

## Run code in docker container

Build image:
`sudo docker build . -f docker/Dockerfile -t plotting_image`

Run container:
`sudo docker run -it --rm --mount type=bind,source="$(pwd)",target=/usr/src/python_workspace plotting_image`

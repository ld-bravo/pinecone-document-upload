## Python project initialization:

### Create new project_folder

    mkdir new_project
    cd new_project

### Create a python virtual environment

    python -m venv "venv"

### Activate the virtual environment

  Windows

    venv/Scripts/activate

  Mac OS/Linux
  
    source venv/bin/activate

### Install dependencies

    pip install django, requests, ...

### After installing all the packages create the requirements.txt

    pip freeze > requirements.txt

### If there's a requirements.txt file available

    python -m pip install -r requirements.txt

Version control:

- Create a git repo via IDE or 'git init'. 
- Create a .gitignore and add folders/files you don't want to commit like venv, output, __pycache, *remotes like github create those with common folders to exclude publish on remote.

Links:

Difference between couple of venvs: 

https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe

Gitignore creation tool: 

https://www.toptal.com/developers/gitignore

VSC interpreter selection: 

https://code.visualstudio.com/docs/python/python-tutorial#_select-a-python-interpreter

Python venv documentation: 

https://docs.python.org/3/library/venv.html
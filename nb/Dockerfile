FROM jupyter/base-notebook
WORKDIR /usr/src/app

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt
ENV JUPYTER_ENABLE_LAB=yes

CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
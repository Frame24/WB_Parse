# Используйте базовый образ RAPIDS AI с CUDA 12.2 и Python 3.10
FROM nvcr.io/nvidia/rapidsai/notebooks:24.06-cuda12.2-py3.10

# Установите дополнительные пакеты и библиотеки
USER root
RUN apt-get update && \
    apt-get install -y \
        openjdk-11-jdk \
        tzdata \
        software-properties-common \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Установите Miniconda, если она не установлена
RUN if [ ! -d /opt/conda ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
        bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
        rm Miniconda3-latest-Linux-x86_64.sh; \
    fi

# Добавьте Miniconda в PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Обновите conda
RUN conda update -n base -c defaults conda

# Установите дополнительные пакеты через conda
RUN conda install --yes jupyter ipywidgets pandas scikit-learn tqdm \
    matplotlib seaborn spacy

# Обновите pip, setuptools и wheel
RUN pip install --upgrade pip setuptools wheel

# Установите PyTorch с поддержкой CUDA 12.2
RUN pip install torch torchvision torchaudio

# Установите другие библиотеки
RUN pip install keras textblob language-tool-python pyspellchecker gdown emoji transformers

# Скачайте русский языковой пакет для spaCy
RUN python -m spacy download ru_core_news_lg

# Создайте рабочую директорию
WORKDIR /workspace

# Откройте порты Jupyter и другие необходимые порты
EXPOSE 8888 8787 8786 8010-8030

# Создайте конфигурационный файл Jupyter с отключенной аутентификацией и минимальными мерами безопасности
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_remote_access = True" >> /root/.jupyter/jupyter_notebook_config.py

# Применение изменений к пакетам и настройкам
RUN dpkg --configure -a && apt-get update -y

# Проверка установки Java
RUN java -version

# Команда по умолчанию для запуска Jupyter
CMD ["jupyter", "notebook"]
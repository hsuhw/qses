ARG PYTHON_IMG_VERSION
FROM python:${PYTHON_IMG_VERSION}

# Add system level dependencies
RUN apt-get update && apt-get install -y \
    vim less unzip g++ cmake uuid-dev default-jre \
 && rm -rf /var/lib/apt/lists/*

# Setup a common shell environment
ADD https://gist.githubusercontent.com/hsuhw/36a711277a448085e0ac/raw/.bash_common /root
ADD https://gist.githubusercontent.com/hsuhw/36a711277a448085e0ac/raw/.common.vim /root
ADD https://gist.github.com/hsuhw/3313a3f752bf718c5c78/raw/.gitconfig /root
RUN echo "\nsource ~/.bash_common" >> ~/.bashrc \
 && echo "\nsource ~/.common.vim" >> ~/.vimrc

# Install OpenFst
ENV OPENFST_VERSION 1.6.8

RUN wget -O openfst.tar.gz "http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-${OPENFST_VERSION}.tar.gz" \
 && mkdir -p /usr/src/openfst \
 && tar -xzC /usr/src/openfst --strip-components=1 -f openfst.tar.gz \
 && rm -rf openfst.tar.gz \
 && cd /usr/src/openfst \
 && ./configure && make && make install \
 && cd ..; rm -rf /usr/src/openfst

# Install ANTLR v4
ENV ANTLR_VERSION 4.7.1

RUN cd /usr/local/lib \
 && wget "https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar" \
 && export CLASSPATH=".:/usr/local/lib/antlr-${ANTLR_VERSION}-complete.jar:${CLASSPATH}" \
 && alias antlr4="java -jar /usr/local/lib/antlr-${ANTLR_VERSION}-complete.jar" \
 \
 && wget -O antlr4.zip "http://www.antlr.org/download/antlr4-cpp-runtime-${ANTLR_VERSION}-source.zip" \
 && unzip antlr4.zip -d /usr/src/antlr4 \
 && rm -rf antlr4.zip \
 && cd /usr/src/antlr4 \
 && cmake . && make && make install \
 && rm -rf /usr/src/antlr4

# Install Z3
ENV Z3_VERSION 4.7.1

RUN wget -O z3.tar.gz "https://github.com/Z3Prover/z3/archive/z3-${Z3_VERSION}.tar.gz" \
 && mkdir -p /usr/src/z3 \
 && tar -xzC /usr/src/z3 --strip-components=1 -f z3.tar.gz \
 && rm -rf z3.tar.gz \
 && cd /usr/src/z3 \
 && python scripts/mk_make.py && cd build; make && make install \
 && cd ..; rm -rf /usr/src/z3

# Install Trau
WORKDIR /usr/share/workspace

RUN git clone https://github.com/diepbp/FAT.git \
 && rm -rf FAT/build/src/*.d FAT/build/src/*.o \
 && rm -rf FAT/build/src/antlrParser/*.d FAT/build/src/antlrParser/*.o

COPY build.makefile.fixed FAT/build/makefile
COPY build.src.subdir.mk.fixed FAT/build/src/subdir.mk
COPY build.src.antlrParser.subdir.mk.fixed FAT/build/src/antlrParser/subdir.mk

RUN cd /usr/share/workspace/FAT \
 && unzip z3-z3-4.4.1.0.zip && mv z3-z3-4.4.1.0 /usr/share/workspace/custom-z3 \
 && cd /usr/share/workspace/custom-z3 \
 && python scripts/mk_make.py --prefix=/usr/share/workspace/custom-z3 \
 && cd build; sed -i -e 's/CXXFLAGS= /CXXFLAGS= -std=c++03 /' config.mk \
 && make && make install

ARG PROJECT_NAME

ENV LD_LIBRARY_PATH='/usr/share/workspace/custom-z3/lib:/usr/local/lib'

RUN cd /usr/share/workspace/FAT/build \
 && make clean && make \
 && mv Trau /usr/share/workspace/

# Set container default behavior
WORKDIR "/usr/share/workspace/${PROJECT_NAME}"

CMD ["python"]
